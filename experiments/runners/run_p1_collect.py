#!/usr/bin/env python3
"""Collect per-sequence q profiles and model-Fisher scores for P1."""

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
    collect_gate_gradient,
    freeze_for_calibration,
    sample_fisher_target,
    supervised_tensors,
)
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(int(config["seed"]) + rank)
    torch.cuda.manual_seed_all(int(config["seed"]) + rank)

    output_dir = Path(config["output_dir"]) / "collection"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    q_roles = tuple(
        config.get(
            "q_roles",
            ("assignment", "damping_validation", "fidelity"),
        )
    )
    invalid_roles = sorted(
        set(q_roles) - {"assignment", "damping_validation", "fidelity"}
    )
    if invalid_roles:
        raise ValueError(f"invalid q roles: {invalid_roles}")
    role_rows = {
        role: load_role_instances(manifest, role)[rank::world]
        for role in (*q_roles, "fisher")
    }

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
    layer_names = [name for name, _ in layers]
    if layer_names != list(injection.layers):
        raise AssertionError("injected layer order drift")

    # A formal timing replay may restrict q_roles to assignment-only.  Keep
    # setup/model-load outside the timed calibration scope, synchronize all
    # ranks at the boundary, and retain rank-local peak memory evidence.
    torch.cuda.synchronize(device)
    torch.distributed.barrier() if torch.distributed.is_initialized() else None
    torch.cuda.reset_peak_memory_stats(device)
    calibration_started_at_unix = time.time()
    calibration_started_at_monotonic = time.perf_counter()

    arrays = {}
    metadata = {
        "rank": rank,
        "world": world,
        "layer_names": layer_names,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(rank),
        },
        "roles": {},
        "started_at_unix": time.time(),
        "calibration_started_at_unix": calibration_started_at_unix,
        "initial_bank_checkpoint": initial_bank,
        "role_wall_seconds": {},
    }
    for role in q_roles:
        role_started = time.perf_counter()
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
            loss, target_tokens, gradient = collect_gate_gradient(
                model, layers, tensors
            )
            q_rows.append(-gradient)
            role_meta.append(
                {
                    "sample_id": row["sample_id"],
                    "task": row["task"],
                    "loss": loss,
                    "target_tokens": target_tokens,
                }
            )
        arrays[f"q_{role}"] = np.stack(q_rows).astype(np.float32)
        metadata["roles"][role] = role_meta
        torch.cuda.synchronize(device)
        metadata["role_wall_seconds"][role] = time.perf_counter() - role_started
        print(f"rank={rank} role={role} complete={len(role_meta)}", flush=True)

    fisher_started = time.perf_counter()
    fisher_rows = []
    fisher_meta = []
    for local_index, row in enumerate(role_rows["fisher"]):
        fisher_tensors, target_ids = sample_fisher_target(
            model,
            tokenizer,
            row,
            max_length=int(config["max_sequence_length"]),
            max_new_tokens=int(config["max_fisher_new_tokens"]),
            seed=int(config["seed"]) * 1_000_003 + rank * 10_007 + local_index,
            device=device,
        )
        loss, target_tokens, score = collect_fisher_score(
            model, layers, fisher_tensors
        )
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
    torch.cuda.synchronize(device)
    metadata["role_wall_seconds"]["fisher"] = time.perf_counter() - fisher_started
    torch.distributed.barrier() if torch.distributed.is_initialized() else None
    torch.cuda.synchronize(device)
    metadata["calibration_finished_at_unix"] = time.time()
    metadata["calibration_wall_seconds_rank"] = (
        time.perf_counter() - calibration_started_at_monotonic
    )
    metadata["peak_memory_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    metadata["peak_memory_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved(device)
    )
    metadata["finished_at_unix"] = time.time()
    metadata["max_reconstruction_error"] = max(
        injection.reconstruction_errors.values()
    )
    np.savez(output_dir / f"rank{rank}.npz", **arrays)
    (output_dir / f"rank{rank}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"rank={rank} collection complete", flush=True)


if __name__ == "__main__":
    main()
