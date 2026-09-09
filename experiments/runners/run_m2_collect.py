#!/usr/bin/env python3
"""Collect frozen M2 q/Fisher profiles and matched-GG gradient geometry.

No downstream/final test manifests are accepted.  GG geometry is the mean of
per-sequence A/B gradients on the *preselected assignment prefix* only; this
is the frozen-calibration analogue of BADIT-GG's regrouping signal and is
stored separately from q/Fisher so its provenance is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import (
    collect_fisher_score,
    freeze_for_calibration,
    sample_fisher_target,
    supervised_tensors,
)
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def _canonical_slot_scalars(layer: object, values: torch.Tensor) -> np.ndarray:
    """Map expert-major slot values to immutable primitive IDs."""

    flattened = values.detach().float().reshape(-1).cpu().numpy()
    indices = layer.primitive_indices.detach().reshape(-1).cpu().numpy()
    if sorted(indices.tolist()) != list(range(flattened.size)):
        raise AssertionError("M2 collection requires a canonical primitive permutation")
    canonical = np.empty_like(flattened)
    canonical[indices] = flattened
    return canonical


def _canonical_gradient_vectors(
    layer: object, gradient_a: torch.Tensor, gradient_b: torch.Tensor
) -> np.ndarray:
    vectors = []
    for expert in range(layer.num_experts):
        for primitive in range(layer.rank):
            vectors.append(
                torch.cat(
                    [
                        gradient_a[expert, primitive].detach().float(),
                        gradient_b[expert, :, primitive].detach().float(),
                    ]
                )
            )
    physical = torch.stack(vectors).cpu().numpy()
    indices = layer.primitive_indices.detach().reshape(-1).cpu().numpy()
    if sorted(indices.tolist()) != list(range(physical.shape[0])):
        raise AssertionError("M2 GG collection requires a primitive permutation")
    canonical = np.empty_like(physical)
    canonical[indices] = physical
    return canonical


def _gate_and_optional_gg_gradients(
    model: torch.nn.Module,
    layers: list[tuple[str, object]],
    tensors: dict[str, torch.Tensor],
    *,
    collect_gg: bool,
) -> tuple[float, int, np.ndarray, list[np.ndarray] | None]:
    outputs = model(**tensors, use_cache=False)
    labels = tensors["labels"]
    target_tokens = int((labels[:, 1:] != -100).sum().item())
    if target_tokens <= 0:
        raise ValueError("sequence contains no predicted target tokens")
    targets: list[torch.Tensor] = [layer.primitive_gates for _, layer in layers]
    if collect_gg:
        for _, layer in layers:
            targets.extend((layer.lora_A, layer.lora_B))
    gradients = torch.autograd.grad(
        outputs.loss, targets, retain_graph=False, create_graph=False, allow_unused=False
    )
    gate_gradients = gradients[: len(layers)]
    q = np.stack(
        [-_canonical_slot_scalars(layer, grad) for grad, (_, layer) in zip(gate_gradients, layers, strict=True)]
    )
    if not np.isfinite(q).all():
        raise FloatingPointError("non-finite M2 gate gradient")
    if not collect_gg:
        return float(outputs.loss.detach().item()), target_tokens, q, None
    parameter_gradients = gradients[len(layers) :]
    gg = []
    for layer_index, (_, layer) in enumerate(layers):
        gradient_a = parameter_gradients[2 * layer_index]
        gradient_b = parameter_gradients[2 * layer_index + 1]
        gg.append(_canonical_gradient_vectors(layer, gradient_a, gradient_b))
    if not all(np.isfinite(value).all() for value in gg):
        raise FloatingPointError("non-finite M2 GG gradient")
    return float(outputs.loss.detach().item()), target_tokens, q, gg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if bool(config.get("official_test_loaded", False)) or bool(
        config.get("downstream_test_loaded", False)
    ):
        raise AssertionError("M2 must not load any downstream/final test")
    validate_fidelity_precision(config)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(int(config["seed"]) + rank)
    torch.cuda.manual_seed_all(int(config["seed"]) + rank)

    output_dir = Path(config["output_dir"])
    collection = output_dir / "collection"
    collection.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    all_roles = {role: load_role_instances(manifest, role) for role in ("assignment", "fidelity", "fisher")}
    selected_roles = {}
    for role, count_key in (
        ("assignment", "assignment_probes_per_task"),
        ("fisher", "fisher_probes_per_task"),
        ("fidelity", "fidelity_probes_per_task"),
    ):
        count = int(config[count_key])
        indices = task_prefix_indices(
            [{"task": row["task"]} for row in all_roles[role]], count
        )
        selected_roles[role] = [all_roles[role][index] for index in indices]
    selected_assignment_ids = {row["sample_id"] for row in selected_roles["assignment"]}
    role_rows = {role: rows[rank::world] for role, rows in selected_roles.items()}

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config["torch_dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        local_files_only=True,
        dtype=dtype,
        attn_implementation=config["attention_implementation"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injection = inject_from_config(model, config)
    initial_bank = load_initial_bank_checkpoint(model, config)
    model.eval()
    layers = freeze_for_calibration(model)
    # q uses virtual gates while GG needs the frozen A/B derivative.  They are
    # differentiated together from one exact token-normalized forward.
    for _, layer in layers:
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
    layer_names = [name for name, _ in layers]
    if layer_names != list(injection.layers):
        raise AssertionError("injected layer order drift")

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "rank": rank,
        "world": world,
        "layer_names": layer_names,
        "roles": {},
        "selected_assignment_sample_ids": sorted(selected_assignment_ids),
        "gg_gradient_reduction": "mean_per_sequence_assignment_prefix",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(rank),
        },
        "initial_bank_checkpoint": initial_bank,
        "started_at_unix": time.time(),
    }
    gg_sums: list[np.ndarray] | None = None
    gg_count = 0
    for role in ("assignment", "fidelity"):
        q_rows = []
        role_meta = []
        for row in role_rows[role]:
            tensors = supervised_tensors(
                tokenizer,
                row,
                max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]),
                device=device,
            )
            use_gg = role == "assignment" and row["sample_id"] in selected_assignment_ids
            loss, target_tokens, q, gg = _gate_and_optional_gg_gradients(
                model, layers, tensors, collect_gg=use_gg
            )
            q_rows.append(q)
            role_meta.append(
                {
                    "sample_id": row["sample_id"],
                    "task": row["task"],
                    "loss": loss,
                    "target_tokens": target_tokens,
                    "gg_selected": use_gg,
                }
            )
            if gg is not None:
                if gg_sums is None:
                    gg_sums = [np.zeros_like(value, dtype=np.float64) for value in gg]
                for layer_index, value in enumerate(gg):
                    gg_sums[layer_index] += value
                gg_count += 1
        arrays[f"q_{role}"] = np.stack(q_rows).astype(np.float32)
        metadata["roles"][role] = role_meta
        print(f"rank={rank} role={role} complete={len(role_meta)}", flush=True)

    fisher_rows = []
    fisher_meta = []
    for local_index, row in enumerate(role_rows["fisher"]):
        tensors, target_ids = sample_fisher_target(
            model,
            tokenizer,
            row,
            max_length=int(config["max_sequence_length"]),
            max_new_tokens=int(config["max_fisher_new_tokens"]),
            seed=int(config["seed"]) * 1_000_003 + rank * 10_007 + local_index,
            device=device,
        )
        loss, target_tokens, score = collect_fisher_score(model, layers, tensors)
        fisher_rows.append(score)
        fisher_meta.append(
            {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "loss": loss,
                "target_tokens": target_tokens,
                "sampled_target_ids": target_ids,
            }
        )
    arrays["fisher_scores"] = np.stack(fisher_rows).astype(np.float32)
    metadata["roles"]["fisher"] = fisher_meta
    arrays["gg_count"] = np.asarray([gg_count], dtype=np.int64)
    if gg_sums is None:
        raise AssertionError("rank collected no selected assignment gradients")
    for layer_index, value in enumerate(gg_sums):
        arrays[f"gg_sum_{layer_index}"] = value.astype(np.float32)
    metadata["finished_at_unix"] = time.time()
    metadata["max_reconstruction_error"] = max(injection.reconstruction_errors.values())
    np.savez(collection / f"rank{rank}.npz", **arrays)
    (collection / f"rank{rank}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"rank={rank} M2 collection complete", flush=True)


if __name__ == "__main__":
    main()
