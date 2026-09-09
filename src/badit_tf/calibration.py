"""Per-sequence BADIT-TF q/Fisher collection and finite interventions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .core import TFAbilityLinear, iter_ability_layers


def task_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prompt_for_record(record: dict[str, Any]) -> str:
    payload = task_payload(record["source_file"])
    definition = payload.get("Definition", "")
    if isinstance(definition, list):
        definition = definition[0] if definition else ""
    return (
        f"{str(definition).strip()}\n\n"
        "Now complete the following example -\n"
        f"Input: {record['instance']['input']}\nOutput: "
    )


def target_for_record(record: dict[str, Any]) -> str:
    output = record["instance"]["output"]
    if isinstance(output, list):
        if not output:
            raise ValueError(f"{record['sample_id']} has no target")
        return str(output[0])
    return str(output)


def supervised_tensors(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_length: int,
    max_target_length: int = 50,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(
        prompt_for_record(record), add_special_tokens=True
    )["input_ids"]
    target_ids = tokenizer(
        target_for_record(record) + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"][:max_target_length]
    if not target_ids:
        raise ValueError(f"{record['sample_id']} has an empty target")
    # Reserve target space before truncating the prompt.  The upstream
    # ``(prompt + target)[:max_length]`` path can silently remove every target
    # token on long summarization inputs.
    prompt_budget = max_length - len(target_ids)
    if prompt_budget <= 0:
        target_ids = target_ids[: max_length - 1]
        prompt_budget = max_length - len(target_ids)
    prompt_ids = prompt_ids[:prompt_budget]
    input_ids = prompt_ids + target_ids
    retained_target = len(target_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones(
            1, len(input_ids), dtype=torch.long, device=device
        ),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def supervised_route_attention_mask(
    batch: dict[str, torch.Tensor], *, scope: str
) -> torch.Tensor:
    """Return the sequence positions that may condition the shared router.

    The supervised attention mask includes the answer tokens because the
    causal LM needs them for teacher forcing.  The deployed route c(x),
    however, is conditioned on the instruction x and cannot see the target y.
    Keeping these masks separate prevents target-label leakage into the
    sequence-pooled router while leaving the token NLL unchanged.

    ``full_supervised_sequence`` is retained only to reproduce frozen legacy
    runs.  New table-facing runs must explicitly request ``prompt_only``.
    """

    attention_mask = batch["attention_mask"]
    if scope == "full_supervised_sequence":
        return attention_mask
    if scope != "prompt_only":
        raise ValueError(f"unsupported route pooling scope: {scope}")
    labels = batch["labels"]
    if labels.shape != attention_mask.shape:
        raise ValueError("labels and attention_mask must have the same shape")
    route_mask = ((labels == -100) & attention_mask.bool()).to(attention_mask.dtype)
    if torch.any(route_mask.sum(dim=-1) <= 0):
        raise ValueError("prompt-only route mask contains an empty sequence")
    if torch.any(route_mask.bool() & (labels != -100)):
        raise AssertionError("prompt-only route mask leaked a target token")
    return route_mask


def prompt_tensors(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompt_for_record(record),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def freeze_for_calibration(model: torch.nn.Module) -> list[tuple[str, TFAbilityLinear]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = list(iter_ability_layers(model))
    if not layers:
        raise ValueError("model has no injected ability layers")
    for _, layer in layers:
        layer.enable_calibration()
        layer.reset_virtual_gates()
    return layers


def collect_gate_gradient(
    model: torch.nn.Module,
    layers: list[tuple[str, TFAbilityLinear]],
    tensors: dict[str, torch.Tensor],
) -> tuple[float, int, np.ndarray]:
    outputs = model(**tensors, use_cache=False)
    labels = tensors["labels"]
    target_tokens = int((labels[:, 1:] != -100).sum().item())
    if target_tokens <= 0:
        raise ValueError("sequence contains no predicted target token")
    loss = outputs.loss
    gates = [layer.primitive_gates for _, layer in layers]
    gradients = torch.autograd.grad(
        loss, gates, retain_graph=False, create_graph=False, allow_unused=False
    )
    array = torch.stack([gradient.detach().flatten().float().cpu() for gradient in gradients])
    if not torch.isfinite(array).all():
        raise FloatingPointError("gate gradient contains NaN/Inf")
    return float(loss.detach().item()), target_tokens, array.numpy()


def sample_fisher_target(
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_length: int,
    max_new_tokens: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    prompt = prompt_tensors(
        tokenizer, record, max_length=max_length - max_new_tokens, device=device
    )
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    with torch.no_grad():
        generated = model.generate(
            **prompt,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = int(prompt["input_ids"].shape[1])
    target_ids = generated[0, prompt_length:].tolist()
    if not target_ids:
        raise ValueError(f"{record['sample_id']} generated an empty Fisher target")
    labels = generated.clone()
    labels[:, :prompt_length] = -100
    return {
        "input_ids": generated,
        "attention_mask": torch.ones_like(generated),
        "labels": labels,
    }, target_ids


def collect_fisher_score(
    model: torch.nn.Module,
    layers: list[tuple[str, TFAbilityLinear]],
    tensors: dict[str, torch.Tensor],
) -> tuple[float, int, np.ndarray]:
    loss, target_tokens, gradients = collect_gate_gradient(model, layers, tensors)
    # log p gradient / sqrt(|y|) = -sqrt(|y|) * grad(mean NLL)
    score = -math.sqrt(target_tokens) * gradients
    return loss, target_tokens, score


def per_sequence_nll(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = []
    for sequence_logits, sequence_labels in zip(
        shift_logits, shift_labels, strict=True
    ):
        count = (sequence_labels != -100).sum()
        if int(count.item()) == 0:
            raise ValueError("an intervention example has no target tokens")
        token_sum = F.cross_entropy(
            sequence_logits.float(),
            sequence_labels,
            ignore_index=-100,
            reduction="sum",
        )
        losses.append(token_sum / count)
    return torch.stack(losses)


@torch.no_grad()
def intervention_losses(
    model: torch.nn.Module,
    layer: TFAbilityLinear,
    tensors: dict[str, torch.Tensor],
    gate_vectors: np.ndarray,
) -> np.ndarray:
    gates = torch.as_tensor(
        gate_vectors,
        dtype=layer.primitive_gates.dtype,
        device=layer.primitive_gates.device,
    ).reshape(-1, layer.num_experts, layer.rank)
    batch = gates.shape[0]
    repeated = {
        key: value.repeat(batch, 1)
        for key, value in tensors.items()
    }
    layer.set_virtual_gate_override(gates)
    try:
        outputs = model(
            input_ids=repeated["input_ids"],
            attention_mask=repeated["attention_mask"],
            use_cache=False,
        )
        losses = per_sequence_nll(outputs.logits, repeated["labels"])
    finally:
        layer.set_virtual_gate_override(None)
    return losses.float().cpu().numpy()


def tied_gate_vector(
    q: np.ndarray,
    fisher: np.ndarray,
    labels: np.ndarray,
    *,
    eta: float,
    num_experts: int,
) -> tuple[np.ndarray, float]:
    free = q / fisher
    tied = np.empty_like(free)
    for expert in range(num_experts):
        selected = labels == expert
        coefficient = q[selected].sum() / fisher[selected].sum()
        tied[selected] = coefficient
    regret = 0.5 * np.sum(fisher * np.square(tied - free))
    return 1.0 + eta * tied, float(regret)


def free_gate_vector(
    q: np.ndarray, fisher: np.ndarray, *, eta: float
) -> np.ndarray:
    return 1.0 + eta * (q / fisher)
