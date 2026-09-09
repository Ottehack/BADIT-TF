#!/usr/bin/env python3
"""Collect only the frozen Fisher and fidelity profiles needed by M4."""

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
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def canonicalize(layers: list[tuple[str, object]], values: np.ndarray) -> np.ndarray:
    output = []
    for row, (_, layer) in zip(values, layers, strict=True):
        indices = layer.primitive_indices.detach().reshape(-1).cpu().numpy()
        if sorted(indices.tolist()) != list(range(row.size)):
            raise AssertionError("M4 collection requires a primitive permutation")
        canonical = np.empty_like(row)
        canonical[indices] = row
        output.append(canonical)
    return np.stack(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"):
        raise AssertionError("M4 must not load final/downstream test")
    validate_fidelity_precision(config)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != int(config["world_size"]):
        raise RuntimeError("M4 world size mismatch")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(int(config["seed"]) + rank)
    torch.cuda.manual_seed_all(int(config["seed"]) + rank)

    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    selected: dict[str, list[dict]] = {}
    for role, count_key in (("fisher", "fisher_probes_per_task"), ("fidelity", "fidelity_probes_per_task")):
        rows = load_role_instances(manifest, role)
        indices = task_prefix_indices([{"task": row["task"]} for row in rows], int(config[count_key]))
        selected[role] = [rows[index] for index in indices]

    dtype = {"float32": torch.float32}[config["torch_dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], local_files_only=True, dtype=dtype,
        attn_implementation=config["attention_implementation"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injection = inject_from_config(model, config)
    trained = load_initial_bank_checkpoint(model, config)
    if trained is None or trained["file_sha256"] != config["initial_bank_checkpoint_sha256"]:
        raise AssertionError("M4 trained checkpoint did not load exactly")
    model.eval()
    layers = freeze_for_calibration(model)
    layer_names = [name for name, _ in layers]
    if layer_names != list(injection.layers):
        raise AssertionError("M4 collection layer order drift")

    output = Path(config["output_dir"]) / "collection"
    output.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "rank": rank, "world": world, "layer_names": layer_names, "roles": {},
        "trained_checkpoint": trained,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(rank)},
        "started_at_unix": time.time(),
    }
    fidelity_values = []
    fidelity_meta = []
    for row in selected["fidelity"][rank::world]:
        tensors = supervised_tensors(
            tokenizer, row, max_length=int(config["max_sequence_length"]),
            max_target_length=int(config["max_target_length"]), device=device,
        )
        loss, target_tokens, physical_q = collect_gate_gradient(model, layers, tensors)
        fidelity_values.append(-canonicalize(layers, physical_q))
        fidelity_meta.append({"sample_id": row["sample_id"], "task": row["task"],
                              "loss": loss, "target_tokens": target_tokens})
    arrays["q_fidelity"] = np.stack(fidelity_values).astype(np.float32)
    metadata["roles"]["fidelity"] = fidelity_meta
    print(f"rank={rank} M4 fidelity={len(fidelity_meta)}", flush=True)

    fisher_values = []
    fisher_meta = []
    for local_index, row in enumerate(selected["fisher"][rank::world]):
        tensors, target_ids = sample_fisher_target(
            model, tokenizer, row, max_length=int(config["max_sequence_length"]),
            max_new_tokens=int(config["max_fisher_new_tokens"]),
            seed=int(config["seed"]) * 1_000_003 + rank * 10_007 + local_index,
            device=device,
        )
        loss, target_tokens, physical_score = collect_fisher_score(model, layers, tensors)
        fisher_values.append(canonicalize(layers, physical_score))
        fisher_meta.append({"sample_id": row["sample_id"], "task": row["task"],
                            "loss": loss, "target_tokens": target_tokens,
                            "sampled_target_ids": target_ids})
    arrays["fisher_scores"] = np.stack(fisher_values).astype(np.float32)
    metadata["roles"]["fisher"] = fisher_meta
    metadata["finished_at_unix"] = time.time()
    np.savez(output / f"rank{rank}.npz", **arrays)
    (output / f"rank{rank}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"rank={rank} M4 collection complete", flush=True)


if __name__ == "__main__":
    main()
