#!/usr/bin/env python3
"""Eight-GPU DeepSpeed runner for the matched P2 TF/GG mixed pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
import time
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deepspeed
from deepspeed.utils import nvtx as deepspeed_nvtx
import numpy as np
import torch
import torch.distributed as dist
import yaml
from nltk.stem import porter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

# DeepSpeed 0.19.2 enables NVTX annotations by default.  The pinned Primus
# image provides an NVTX binding whose DummyDomain does not accept the newer
# ``domain=`` keyword, so annotations fail before the first model forward.
# They are profiling-only and cannot affect numerical training semantics.
deepspeed_nvtx.enable_nvtx = False

from badit_tf.calibration import (
    prompt_for_record,
    supervised_route_attention_mask,
    supervised_tensors,
)
from badit_tf.core import (
    advance_router_step,
    inject_ability_layers,
    iter_ability_layers,
)
from badit_tf.training import (
    balanced_gradient_geometry_assignment,
    canonical_json_sha256,
    file_sha256,
    losses_are_finite_and_stable,
    nonzero_cosine_warmup_multiplier,
    residual_router_optimizer_groups,
    resolve_assignment_labels,
    resolve_code_commit,
)
from badit_tf.runtime import (
    apply_layer_assignments,
    assert_canonical_layer_assignments,
    inject_from_config,
    load_initial_bank_checkpoint,
)
from badit_tf.sequential import state_sha256, trainable_and_ability_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=("tf", "gg"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_record(record: dict[str, Any], cache: dict[str, dict]) -> dict[str, Any]:
    source = record["source_file"]
    if source not in cache:
        cache[source] = json.loads(Path(source).read_text(encoding="utf-8"))
    item = dict(record)
    item["instance"] = cache[source]["Instances"][int(record["instance_index"])]
    return item


def set_route_masks(model: torch.nn.Module, mask: torch.Tensor) -> None:
    for _, layer in iter_ability_layers(model):
        layer.set_attention_mask(mask)


def clear_route_overrides(model: torch.nn.Module) -> None:
    for _, layer in iter_ability_layers(model):
        layer.set_route_code_override(None)


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    encoded = tokenizer(
        prompt_for_record(record),
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    clear_route_overrides(model)
    set_route_masks(model, attention_mask)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    layers = list(iter_ability_layers(model))
    for _, layer in layers:
        layer.set_route_code_override(layer._last_route["actual"])
    generated: list[int] = []
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past = outputs.past_key_values
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(1, 1, dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    clear_route_overrides(model)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(no_punctuation.split())


class RougeTokenizer:
    """Exact frozen Google ROUGE default tokenization used by upstream."""

    _non_alphanumeric = re.compile(r"[^a-z0-9]+")
    _spaces = re.compile(r"\s+")
    _valid = re.compile(r"^[a-z0-9]+$")

    def __init__(self) -> None:
        self._stemmer = porter.PorterStemmer()

    def tokenize(self, text: str) -> list[str]:
        normalized = self._non_alphanumeric.sub(" ", text.lower())
        tokens = self._spaces.split(normalized)
        tokens = [
            self._stemmer.stem(token) if len(token) > 3 else token
            for token in tokens
        ]
        return [token for token in tokens if self._valid.match(token)]


def rouge_fmeasure(
    target: str, prediction: str, metric: str, tokenizer: RougeTokenizer
) -> float:
    target_tokens = tokenizer.tokenize(target)
    prediction_tokens = tokenizer.tokenize(prediction)
    if not target_tokens or not prediction_tokens:
        return 0.0
    if metric == "rouge1":
        overlap = sum(
            (Counter(target_tokens) & Counter(prediction_tokens)).values()
        )
    elif metric == "rougeL":
        previous = [0] * (len(prediction_tokens) + 1)
        for target_token in target_tokens:
            current = [0]
            for index, prediction_token in enumerate(prediction_tokens, start=1):
                if target_token == prediction_token:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current
        overlap = previous[-1]
    else:
        raise ValueError(metric)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = RougeTokenizer()
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        prediction = row["prediction"]
        references = row["references"]
        candidates = []
        for reference in references:
            candidates.append(
                {
                    "exact_match": float(
                        normalize_answer(prediction) == normalize_answer(reference)
                    ),
                    "rouge1": rouge_fmeasure(
                        reference, prediction, "rouge1", tokenizer
                    ),
                    "rougeL": rouge_fmeasure(
                        reference, prediction, "rougeL", tokenizer
                    ),
                }
            )
        by_task[row["task"]].append(
            {
                key: max(item[key] for item in candidates)
                for key in ("exact_match", "rouge1", "rougeL")
            }
        )
    per_task = {}
    for task, values in sorted(by_task.items()):
        per_task[task] = {
            key: 100.0 * float(np.mean([item[key] for item in values]))
            for key in ("exact_match", "rouge1", "rougeL")
        }
        per_task[task]["n"] = len(values)
    macro = {
        key: float(np.mean([item[key] for item in per_task.values()]))
        for key in ("exact_match", "rouge1", "rougeL")
    }
    micro = {
        key: 100.0
        * float(
            np.mean(
                [
                    item[key]
                    for values in by_task.values()
                    for item in values
                ]
            )
        )
        for key in ("exact_match", "rouge1", "rougeL")
    }
    return {"macro": macro, "micro": micro, "per_task": per_task}


def gradient_geometry_vectors(
    layer: Any, name: str, gradient_cache: dict[str, torch.Tensor]
) -> torch.Tensor:
    gradient_a = gradient_cache.get(f"{name}.lora_A")
    gradient_b = gradient_cache.get(f"{name}.lora_B")
    if gradient_a is None or gradient_b is None:
        raise RuntimeError("GG requires final-backward A/B gradients")
    gradients_a = gradient_a.float()
    gradients_b = gradient_b.float()
    vectors = torch.empty(
        layer.total_primitives,
        layer.in_features + layer.out_features,
        dtype=torch.float32,
        device=gradients_a.device,
    )
    for expert in range(layer.num_experts):
        for index in range(layer.rank):
            canonical = int(layer.primitive_indices[expert, index].item())
            vectors[canonical] = torch.cat(
                [gradients_a[expert, index], gradients_b[expert, :, index]]
            )
    dist.all_reduce(vectors, op=dist.ReduceOp.SUM)
    vectors.div_(dist.get_world_size())
    return vectors


def regroup_gg(
    model: torch.nn.Module,
    rank: int,
    final_gradient_vectors: dict[str, torch.Tensor],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    assignments: dict[str, list[int]] = {}
    audit: dict[str, Any] = {}
    for layer_index, (name, layer) in enumerate(iter_ability_layers(model)):
        vectors = final_gradient_vectors[name].to(layer.lora_A.device)
        labels_tensor = torch.empty(
            layer.total_primitives, dtype=torch.long, device=vectors.device
        )
        layer_audit = None
        if rank == 0:
            labels, layer_audit = balanced_gradient_geometry_assignment(
                vectors,
                num_experts=layer.num_experts,
                rank=layer.rank,
                seed=layer_index,
            )
            labels_tensor.copy_(torch.from_numpy(labels).to(labels_tensor.device))
        dist.broadcast(labels_tensor, src=0)
        layer.apply_assignment(labels_tensor)
        indices = layer.primitive_indices.detach().cpu().flatten().tolist()
        if sorted(indices) != list(range(layer.total_primitives)):
            raise AssertionError(f"{name}: regroup omitted/duplicated a primitive")
        assignments[name] = labels_tensor.detach().cpu().tolist()
        if rank == 0:
            audit[name] = layer_audit
    return assignments, audit


def trainable_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_deltas(
    model: torch.nn.Module, initial: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {
        name: float(
            torch.linalg.vector_norm(
                parameter.detach().float().cpu() - initial[name]
            ).item()
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_trainable_checkpoint(
    model: torch.nn.Module, path: Path, metadata: dict[str, Any]
) -> None:
    model_state = trainable_and_ability_state(model)
    metadata = {**metadata, "model_state_sha256": state_sha256(model_state)}
    torch.save({"model_state": model_state, "metadata": metadata}, path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size != int(config["world_size"]):
        raise RuntimeError(f"world size {world_size} != config {config['world_size']}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(int(config["seed"]))

    output_dir = Path(config["output_root"]) / args.run_id
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    manifest = json.loads(Path(config["order_manifest"]).read_text(encoding="utf-8"))
    assignments_payload = json.loads(
        Path(config["tf_assignment_path"]).read_text(encoding="utf-8")
    )
    # GG starts from the canonical warm-start bank.  It nevertheless needs the
    # frozen TF labels as a layer/capacity integrity reference; ``gg`` is a
    # training variant, not an assignment payload family.  Resolving the
    # config literal ``gg`` here used to fail before the first forward.
    assignment_method = str(config.get("assignment_method", "tf"))
    resolved_labels = resolve_assignment_labels(
        assignments_payload,
        method="tf" if assignment_method == "gg" else assignment_method,
        epsilon=float(config["tf_assignment_epsilon"]),
        num_experts=int(config["num_experts"]),
        rank=int(config["rank"]),
        random_index=int(config.get("assignment_random_index", 0)),
        allow_unequal_capacity=bool(config.get("allow_unequal_capacity", False)),
    )
    tf_assignments = {
        name: torch.tensor(labels, dtype=torch.long)
        for name, labels in resolved_labels.items()
    }
    assignments = tf_assignments if args.variant == "tf" else {}

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = getattr(torch, str(config["torch_dtype"]))
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=dtype,
        attn_implementation=config["attention_implementation"],
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # A warm-start checkpoint is always stored in canonical primitive order.
    # Load it before applying TF so checkpoint tensor slots cannot be mistaken
    # for already-regrouped slots.
    injection = inject_from_config(
        model,
        config,
        assignments=None if config.get("initial_bank_checkpoint") else assignments,
    )
    initial_bank = load_initial_bank_checkpoint(model, config)
    if initial_bank is not None:
        if args.variant == "tf":
            # TF is defined by the frozen, task-factorized assignment.
            apply_layer_assignments(model, tf_assignments)
        else:
            # GG begins from the identical canonical bank and changes grouping
            # only after collecting its final training gradients.  Applying
            # the empty non-TF map here was a pre-training implementation bug.
            assert_canonical_layer_assignments(model)
    for _, layer in iter_ability_layers(model):
        layer.disable_calibration()
    seed_everything(int(config["seed"]) + rank)

    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    train_order = manifest["train_order"]
    rank_records = train_order[rank::world_size]
    configured_steps = len(rank_records)
    total_steps = (
        min(configured_steps, args.max_steps)
        if args.max_steps is not None
        else configured_steps
    )
    router_lr_multiplier = float(config.get("router_lr_multiplier", 1.0))
    expert_only_warmstart = bool(config.get("expert_only_warmstart", False))
    if expert_only_warmstart:
        optimizer_groups = [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and ".residual_router." not in name
                ],
                "lr": float(config["learning_rate"]),
                "group_name": "expert",
            }
        ]
    else:
        optimizer_groups = residual_router_optimizer_groups(
            model,
            learning_rate=float(config["learning_rate"]),
            router_lr_multiplier=router_lr_multiplier,
        )
    optimizer = AdamW(
        optimizer_groups,
        betas=tuple(config["adam_betas"]),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler_total_steps = int(config.get("scheduler_total_steps", total_steps))
    if scheduler_total_steps < total_steps:
        raise ValueError("scheduler_total_steps cannot be below executed steps")
    warmup_steps = max(
        1, int(math.ceil(scheduler_total_steps * float(config["warmup_ratio"])))
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: nonzero_cosine_warmup_multiplier(
            step, warmup_steps=warmup_steps, total_steps=scheduler_total_steps
        ),
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": world_size,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
        "gradient_clipping": float(config["gradient_clipping"]),
        "steps_per_print": 1000000,
        "wall_clock_breakdown": False,
    }
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=[
            parameter
            for group in optimizer_groups
            for parameter in group["params"]
        ],
        config=ds_config,
    )
    initial = trainable_snapshot(engine.module) if rank == 0 else {}
    layer_names = [name for name, _ in iter_ability_layers(engine.module)]
    num_layers = len(layer_names)
    num_experts = int(config["num_experts"])
    cumulative_grad = torch.zeros(num_layers, num_experts, device=device)
    route_load = torch.zeros(num_layers, num_experts, device=device)
    route_abs_raw_sum = torch.zeros(num_layers, device=device)
    route_observations = torch.zeros(num_layers, device=device)
    loss_curve = []
    route_trace = {}
    first_backward_router_grad = {}
    first_update_router_delta = {}
    last_backward_available = False
    gg_last_vectors: dict[str, torch.Tensor] = {}
    gradient_cache: dict[str, torch.Tensor] = {}
    gradient_hooks = []
    for parameter_name, parameter in engine.module.named_parameters():
        if not parameter.requires_grad:
            continue

        def cache_gradient(
            gradient: torch.Tensor, *, key: str = parameter_name
        ) -> torch.Tensor:
            gradient_cache[key] = gradient.detach().clone()
            return gradient

        gradient_hooks.append(parameter.register_hook(cache_gradient))
    cache: dict[str, dict] = {}
    # Measure SFT only: model/tokenizer loading, adapter injection and
    # DeepSpeed initialization are intentionally outside this interval.
    torch.cuda.synchronize(device)
    started = time.monotonic()
    engine.train()
    for step, record_meta in enumerate(rank_records[:total_steps]):
        record = load_record(record_meta, cache)
        batch = supervised_tensors(
            tokenizer,
            record,
            max_length=int(config["max_sequence_length"]),
            max_target_length=int(config["max_target_length"]),
            device=device,
        )
        set_route_masks(
            engine.module,
            supervised_route_attention_mask(
                batch,
                scope=str(
                    config.get("route_pooling_scope", "full_supervised_sequence")
                ),
            ),
        )
        outputs = engine(**batch, use_cache=False)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        gradient_cache.clear()
        engine.backward(loss)
        layers = list(iter_ability_layers(engine.module))
        for layer_index, (name, layer) in enumerate(layers):
            gradient_a = gradient_cache.get(f"{name}.lora_A")
            gradient_b = gradient_cache.get(f"{name}.lora_B")
            if gradient_a is not None and gradient_b is not None:
                grad_a = torch.linalg.vector_norm(
                    gradient_a.float(), dim=(1, 2)
                )
                grad_b = torch.linalg.vector_norm(
                    gradient_b.float(), dim=(1, 2)
                )
                cumulative_grad[layer_index] += torch.sqrt(grad_a.square() + grad_b.square())
            route = layer._last_route
            actual = route["actual"]
            route_load[layer_index] += (actual != 0).float().sum(dim=0)
            route_abs_raw_sum[layer_index] += route["raw"].abs().mean()
            route_observations[layer_index] += 1
            if step in (0, 1) and rank == 0:
                route_trace.setdefault(str(step), {})[name] = {
                    key: value[0].float().cpu().tolist()
                    for key, value in route.items()
                }
            if step == 0 and rank == 0:
                router_norm = math.sqrt(
                    sum(
                        float(torch.sum(gradient.square()).item())
                        for key in (
                            f"{name}.residual_router.weight",
                            f"{name}.residual_router.bias",
                        )
                        if (gradient := gradient_cache.get(key)) is not None
                    )
                )
                first_backward_router_grad[name] = router_norm
        if args.variant == "gg" and step == total_steps - 1:
            last_backward_available = all(
                f"{name}.lora_A" in gradient_cache
                and f"{name}.lora_B" in gradient_cache
                for name, _ in layers
            )
            if last_backward_available:
                gg_last_vectors = {
                    name: gradient_geometry_vectors(
                        layer, name, gradient_cache
                    ).cpu()
                    for name, layer in layers
                }
        reduced_loss = loss.detach().float().clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss.div_(world_size)
        engine.step()
        advance_router_step(engine.module)
        if step == 0 and rank == 0:
            deltas = parameter_deltas(engine.module, initial)
            for name, _ in layers:
                prefix = f"{name}.residual_router."
                first_update_router_delta[name] = math.sqrt(
                    sum(value * value for key, value in deltas.items() if key.startswith(prefix))
                )
        if rank == 0:
            loss_curve.append(
                {
                    "step": step,
                    "loss": float(reduced_loss.item()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )
            if step % int(config["log_every_steps"]) == 0 or step + 1 == total_steps:
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "variant": args.variant,
                            "step": step,
                            "steps": total_steps,
                            "loss": float(reduced_loss.item()),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    torch.cuda.synchronize(device)
    training_wall_seconds_local = time.monotonic() - started
    training_wall_seconds_max = torch.tensor(
        training_wall_seconds_local, dtype=torch.float64, device=device
    )
    dist.all_reduce(training_wall_seconds_max, op=dist.ReduceOp.MAX)

    gg_assignments: dict[str, list[int]] = {}
    gg_audit: dict[str, Any] = {}
    if args.variant == "gg":
        if not last_backward_available:
            raise RuntimeError("GG final backward gradients were not retained")
        gg_assignments, gg_audit = regroup_gg(
            engine.module, rank, gg_last_vectors
        )
    dist.all_reduce(cumulative_grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(route_load, op=dist.ReduceOp.SUM)
    dist.all_reduce(route_abs_raw_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(route_observations, op=dist.ReduceOp.SUM)

    prediction_path = output_dir / f"predictions_rank{rank}.jsonl"
    if not args.skip_eval:
        engine.eval()
        model_for_eval = engine.module
        model_for_eval.config.use_cache = True
        with prediction_path.open("w", encoding="utf-8") as handle:
            for record_meta in manifest["test_records"][rank::world_size]:
                record = load_record(record_meta, cache)
                prediction = greedy_generate(
                    model_for_eval,
                    tokenizer,
                    record,
                    max_prompt_length=int(config["max_sequence_length"])
                    - int(config["max_new_tokens"]),
                    max_new_tokens=int(config["max_new_tokens"]),
                    device=device,
                )
                output = record["instance"]["output"]
                references = (
                    [str(item) for item in output]
                    if isinstance(output, list)
                    else [str(output)]
                )
                handle.write(
                    json.dumps(
                        {
                            "sample_id": record["sample_id"],
                            "task": record["task"],
                            "prediction": prediction,
                            "references": references,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    dist.barrier()

    local_model_state_sha = state_sha256(trainable_and_ability_state(engine.module))
    rank_model_state_shas: list[str] = [""] * world_size
    dist.all_gather_object(rank_model_state_shas, local_model_state_sha)
    model_state_consensus = len(set(rank_model_state_shas)) == 1

    if rank == 0:
        deltas = parameter_deltas(engine.module, initial)
        expert_parameter_delta = {}
        for name, _ in iter_ability_layers(engine.module):
            values = []
            for expert in range(num_experts):
                keys = (
                    f"{name}.lora_A",
                    f"{name}.lora_B",
                )
                squared = 0.0
                for key in keys:
                    current = dict(engine.module.named_parameters())[key].detach().float().cpu()
                    difference = current[expert] - initial[key][expert]
                    squared += float(torch.sum(difference.square()).item())
                values.append(math.sqrt(squared))
            expert_parameter_delta[name] = values
        all_predictions = []
        if not args.skip_eval:
            for prediction_file in sorted(output_dir.glob("predictions_rank*.jsonl")):
                all_predictions.extend(
                    json.loads(line)
                    for line in prediction_file.read_text(encoding="utf-8").splitlines()
                    if line
                )
        metrics = score_predictions(all_predictions) if all_predictions else {}
        losses = np.array([item["loss"] for item in loss_curve], dtype=np.float64)
        stable = losses_are_finite_and_stable(
            losses,
            explosion_factor=float(config["loss_explosion_factor"]),
            burn_in_steps=32,
        )
        cumulative = cumulative_grad.cpu().numpy()
        param_delta_matrix = np.array(
            [expert_parameter_delta[name] for name in layer_names]
        )
        step0_dense = all(
            sum(abs(value) > 0 for value in item["actual"]) == num_experts
            for item in route_trace.get("0", {}).values()
        )
        expected_step0_active = (
            num_experts if int(config["dense_steps"]) > 0 else int(config["top_k"])
        )
        step0_schedule_exact = all(
            sum(abs(value) > 0 for value in item["actual"])
            == expected_step0_active
            for item in route_trace.get("0", {}).values()
        )
        expected_step1_active = (
            num_experts if int(config["dense_steps"]) > 1 else int(config["top_k"])
        )
        step1_schedule_exact = total_steps < 2 or all(
            sum(abs(value) > 0 for value in item["actual"])
            == expected_step1_active
            for item in route_trace.get("1", {}).values()
        )
        def follows_stable_topk(item: dict[str, list[float]]) -> bool:
            dense = np.asarray(item["dense"], dtype=np.float64)
            selected = np.flatnonzero(np.asarray(item["actual"]) != 0)
            # ``mergesort`` is stable, hence equal scores retain ascending
            # expert ID.  It mirrors TFAbilityLinear.route exactly.
            expected = np.argsort(-dense, kind="mergesort")[: int(config["top_k"])]
            return np.array_equal(np.sort(selected), np.sort(expected))

        step1_ties_broken = (
            total_steps < 2
            or int(config["dense_steps"]) > 1
            or all(follows_stable_topk(item) for item in route_trace.get("1", {}).values())
        )
        assertions = {
            "world_size_8": world_size == int(config["world_size"]),
            "all_losses_finite_and_stable": stable,
            "first_backward_router_gradient_nonzero": bool(
                expert_only_warmstart
                or (
                    first_backward_router_grad
                    and all(value > 0 for value in first_backward_router_grad.values())
                )
            ),
            "first_update_router_parameter_changed": bool(
                expert_only_warmstart
                or (
                    first_update_router_delta
                    and all(value > 0 for value in first_update_router_delta.values())
                )
            ),
            # Preserve the historical assertion name for dense-first formal
            # runs.  A source-faithful LoRAMoE timing control starts sparse;
            # in that case density is not required and the explicit schedule
            # assertion below is the operative invariant.
            "step0_route_dense": step0_dense if int(config["dense_steps"]) > 0 else True,
            "step0_route_matches_schedule": step0_schedule_exact,
            "step1_route_matches_schedule": step1_schedule_exact,
            "ties_broken_before_topk": step1_ties_broken,
            "all_experts_received_gradient": bool((cumulative > 0).all()),
            "all_experts_parameter_changed": bool((param_delta_matrix > 0).all()),
            "test_records_complete": bool(
                args.skip_eval
                or (
                    len(all_predictions) == int(manifest["test_count"])
                    and len({row["sample_id"] for row in all_predictions})
                    == int(manifest["test_count"])
                )
            ),
            "gg_regroup_exact_capacity": bool(
                args.variant != "gg"
                or all(
                    np.array_equal(
                        np.bincount(labels, minlength=num_experts),
                        np.full(num_experts, int(config["rank"])),
                    )
                    for labels in gg_assignments.values()
                )
            ),
            "rank_model_state_consensus": model_state_consensus,
        }
        route_load_np = route_load.cpu().numpy()
        load_prob = route_load_np / np.maximum(
            route_load_np.sum(axis=1, keepdims=True), 1e-12
        )
        entropy = -np.sum(
            np.where(load_prob > 0, load_prob * np.log(load_prob), 0.0), axis=1
        )
        result = {
            "run_id": args.run_id,
            "experiment_id": str(config.get("experiment_id", "P2")),
            "variant": args.variant,
            "status": "complete" if all(assertions.values()) else "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config),
            "config_sha256": file_sha256(args.config),
            "code_commit": resolve_code_commit(),
            "protocol": {
                "gg_semantics": (
                    "upstream last-backward gradient geometry at epoch end; "
                    "deterministic exact-capacity Hungarian repair"
                ),
                "epochs": 1,
                "steps": total_steps,
                "scheduler_total_steps": scheduler_total_steps,
                "global_batch_size": world_size,
                "training_examples_with_padding": total_steps * world_size,
                "full_train_examples": int(manifest["original_train_count"]),
                "padding_repeats": int(manifest["padding_repeat_count"]),
                "route_pooling_scope": str(
                    config.get("route_pooling_scope", "full_supervised_sequence")
                ),
                "router_mode": str(config.get("router_mode", "residual_mask")),
                "assignment_method": str(config.get("assignment_method", "tf")),
                "allow_unequal_capacity": bool(
                    config.get("allow_unequal_capacity", False)
                ),
                "evaluation_role": str(
                    manifest.get("evaluation_role", "official_test")
                ),
                "router_lr_multiplier": router_lr_multiplier,
                "final_test_used_for_selection": False,
                "official_test_loaded": bool(
                    not args.skip_eval
                    and manifest.get("evaluation_role", "official_test")
                    == "official_test"
                ),
                "initial_bank_checkpoint": initial_bank,
            },
            "timing": {
                "scope": "SFT train loop only; excludes model/tokenizer load, adapter injection, DeepSpeed initialization, evaluation, checkpointing and artifact upload",
                "training_wall_seconds_local_rank0": training_wall_seconds_local,
                "training_wall_seconds_max_rank": float(training_wall_seconds_max.item()),
            },
            "hashes": {
                "order_manifest_file_sha256": file_sha256(config["order_manifest"]),
                "order_manifest_content_sha256": manifest["manifest_sha256"],
                "train_unique_ids_sha256": manifest["train_unique_ids_sha256"],
                "train_order_ids_sha256": manifest["train_order_ids_sha256"],
                "test_ids_sha256": manifest["test_ids_sha256"],
                "evaluation_ids_sha256": manifest["test_ids_sha256"],
                "tf_assignment_file_sha256": file_sha256(config["tf_assignment_path"]),
                "tf_assignment_sha256": canonical_json_sha256(
                    {
                        name: labels.tolist()
                        for name, labels in sorted(tf_assignments.items())
                    }
                ),
            },
            "metrics": {
                "loss": {
                    "first": float(losses[0]),
                    "last": float(losses[-1]),
                    "min": float(losses.min()),
                    "max": float(losses.max()),
                    "median": float(np.median(losses)),
                },
                "rouge": metrics,
                "trainable_parameters": trainable_count,
                "first_backward_router_gradient_norm": first_backward_router_grad,
                "first_update_router_parameter_delta_norm": first_update_router_delta,
                "cumulative_expert_gradient_norm": {
                    name: cumulative[index].tolist()
                    for index, name in enumerate(layer_names)
                },
                "expert_parameter_delta_norm": expert_parameter_delta,
                "route_load": {
                    name: route_load_np[index].tolist()
                    for index, name in enumerate(layer_names)
                },
                "route_load_entropy": {
                    name: float(entropy[index])
                    for index, name in enumerate(layer_names)
                },
                "dead_expert_count": int((route_load_np == 0).sum()),
                "mean_abs_router_residual": {
                    name: float(
                        route_abs_raw_sum[index].cpu().item()
                        / max(route_observations[index].cpu().item(), 1)
                    )
                    for index, name in enumerate(layer_names)
                },
                "router_parameter_delta_norm": {
                    name: math.sqrt(
                        sum(
                            value * value
                            for key, value in deltas.items()
                            if key.startswith(f"{name}.residual_router.")
                        )
                    )
                    for name in layer_names
                },
                "assertions": assertions,
                "rank_model_state_sha256": rank_model_state_shas,
                "rank_model_state_consensus": model_state_consensus,
            },
            "artifacts": {
                "loss_curve": str(output_dir / "loss_curve.json"),
                "route_trace": str(output_dir / "route_trace.json"),
                "predictions_dir": str(output_dir),
            },
        }
        if not args.skip_checkpoint:
            result["artifacts"]["checkpoint"] = str(
                output_dir / "trainable_checkpoint.pt"
            )
        if args.variant == "gg":
            result["gg_assignment"] = gg_assignments
            result["gg_regroup_audit"] = gg_audit
            (output_dir / "gg_assignment.json").write_text(
                json.dumps(
                    {"assignments": gg_assignments, "audit": gg_audit},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            result["artifacts"]["gg_assignment"] = str(
                output_dir / "gg_assignment.json"
            )
        (output_dir / "loss_curve.json").write_text(
            json.dumps(loss_curve, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "route_trace.json").write_text(
            json.dumps(route_trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not args.skip_checkpoint:
            save_trainable_checkpoint(
                engine.module,
                output_dir / "trainable_checkpoint.pt",
                {
                    "run_id": args.run_id,
                    "variant": args.variant,
                    "config_sha256": result["config_sha256"],
                    "order_manifest_content_sha256": manifest["manifest_sha256"],
                    "milestone_steps": total_steps,
                    "bank_contract": {
                        "initialization_method": str(
                            config.get("initialization_method", "svd")
                        ),
                        "num_experts": int(config["num_experts"]),
                        "rank": int(config["rank"]),
                        "lora_alpha": float(config["lora_alpha"]),
                        "router_mode": str(config.get("router_mode", "residual_mask")),
                        "router_bias": bool(config.get("router_bias", True)),
                        "apply_lora_dropout": bool(
                            config.get("apply_lora_dropout", True)
                        ),
                    },
                },
            )
        result["artifact_sha256"] = {
            label: file_sha256(path)
            for label, path in result["artifacts"].items()
            if Path(path).is_file()
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "run_complete", "result": result}, sort_keys=True))
    dist.barrier()
    for hook in gradient_hooks:
        hook.remove()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
