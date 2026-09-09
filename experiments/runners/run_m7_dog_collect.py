#!/usr/bin/env python3
"""Collect sufficient statistics for Table VIII BADIT-GG/DOG stability.

Only the frozen calibration-assignment role is loaded.  Per-sequence A/B
gradients are reduced into full, task and deterministic batch-replicate sums;
no official or downstream test role is accessible to this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import freeze_for_calibration, supervised_tensors
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances
from run_m2_collect import _gate_and_optional_gg_gradients


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_config"]
    if digest(source_path) != config["source_config_sha256"]:
        raise AssertionError("M7-DOG source config hash drift")
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") is not False or config.get("downstream_test_loaded") is not False:
        raise AssertionError("M7-DOG must close all test roles")
    if bool(source.get("official_test_loaded", False)) or bool(source.get("downstream_test_loaded", False)):
        raise AssertionError("M7-DOG source config exposes a test role")
    validate_fidelity_precision(source)

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != int(config["world_size"]):
        raise AssertionError(f"M7-DOG requires world_size={config['world_size']}, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(int(config["seed"]) + rank)
    torch.cuda.manual_seed_all(int(config["seed"]) + rank)

    manifest_path = ROOT / source["split_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = load_role_instances(manifest, "assignment")
    selected_indices = task_prefix_indices(
        [{"task": row["task"]} for row in all_rows],
        int(source["assignment_probes_per_task"]),
    )
    selected = [all_rows[index] for index in selected_indices]
    tasks = sorted({str(row["task"]) for row in selected})
    if len(tasks) != 15:
        raise AssertionError("M7-DOG requires exactly 15 calibration tasks")

    rng = np.random.default_rng(int(config["seed"]))
    task_splits = []
    for _ in range(int(config["task_split_replicates"])):
        perm = rng.permutation(tasks).tolist()
        task_splits.append((perm[:7], perm[7:]))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[source["torch_dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(source["model_path"], local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        source["model_path"], local_files_only=True, dtype=dtype,
        attn_implementation=source["attention_implementation"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injection = inject_from_config(model, source)
    initial_bank = load_initial_bank_checkpoint(model, source)
    model.eval()
    layers = freeze_for_calibration(model)
    for _, layer in layers:
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
    layer_names = [name for name, _ in layers]
    if layer_names != list(injection.layers):
        raise AssertionError("M7-DOG layer order drift")

    # Match M7's frozen layer-specific batch subsets exactly.  The explicit
    # formula keeps TF and DOG perturbations comparable without sharing any
    # downstream/test observation.
    batch_ids: list[list[set[str]]] = []
    for layer_index in range(len(layer_names)):
        layer_batches = []
        for replicate in range(int(config["batch_replicates"])):
            brng = np.random.default_rng(int(config["seed"]) + 1_400_003 + replicate * 100_003 + layer_index * 4_099)
            ids: set[str] = set()
            for task in tasks:
                candidates = [str(row["sample_id"]) for row in selected if str(row["task"]) == task]
                ids.update(str(value) for value in brng.choice(candidates, size=int(config["batch_probes_per_task"]), replace=False).tolist())
            layer_batches.append(ids)
        batch_ids.append(layer_batches)

    full_sums = None
    task_sums: dict[str, list[np.ndarray]] = {}
    batch_sums: list[list[np.ndarray]] = []
    full_count = 0
    task_counts = {task: 0 for task in tasks}
    batch_counts = [[0 for _ in range(int(config["batch_replicates"]))] for _ in layer_names]
    local_rows = selected[rank::world]
    processed_ids = []
    for row in local_rows:
        tensors = supervised_tensors(
            tokenizer, row, max_length=int(source["max_sequence_length"]),
            max_target_length=int(source["max_target_length"]), device=device,
        )
        _, _, _, gg = _gate_and_optional_gg_gradients(model, layers, tensors, collect_gg=True)
        if gg is None:
            raise AssertionError("M7-DOG gradient collection returned no geometry")
        if full_sums is None:
            full_sums = [np.zeros_like(value, dtype=np.float64) for value in gg]
            task_sums = {task: [np.zeros_like(value, dtype=np.float64) for value in gg] for task in tasks}
            batch_sums = [[np.zeros_like(value, dtype=np.float64) for value in gg] for _ in range(int(config["batch_replicates"]))]
        task = str(row["task"])
        sample_id = str(row["sample_id"])
        for layer_index, value in enumerate(gg):
            full_sums[layer_index] += value
            task_sums[task][layer_index] += value
        full_count += 1
        task_counts[task] += 1
        for layer_index, value in enumerate(gg):
            for replicate, ids in enumerate(batch_ids[layer_index]):
                if sample_id in ids:
                    batch_sums[replicate][layer_index] += value
                    batch_counts[layer_index][replicate] += 1
        processed_ids.append(sample_id)
        print(f"rank={rank} M7-DOG {len(processed_ids)}/{len(local_rows)}", flush=True)
    if full_sums is None:
        raise AssertionError("M7-DOG rank collected no rows")

    arrays: dict[str, np.ndarray] = {}
    if len(batch_sums) != int(config["batch_replicates"]):
        raise AssertionError("M7-DOG batch accumulator count drift")
    for layer_index, value in enumerate(full_sums):
        arrays[f"full_sum_{layer_index}"] = value
        for task_index, task in enumerate(tasks):
            arrays[f"task_{task_index}_sum_{layer_index}"] = task_sums[task][layer_index]
        for replicate in range(int(config["batch_replicates"])):
            arrays[f"batch_{replicate}_sum_{layer_index}"] = batch_sums[replicate][layer_index]
    output = ROOT / config["output_dir"] / "partials"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / f"rank{rank}.npz", **arrays)
    metadata = {
        "rank": rank, "world": world, "model": config["model"],
        "layer_names": layer_names, "tasks": tasks,
        "processed_sample_ids": processed_ids, "full_count": full_count,
        "task_counts": task_counts, "batch_counts": batch_counts,
        "batch_sample_ids": [[sorted(ids) for ids in layer] for layer in batch_ids],
        "task_splits": [{"left_7": left, "right_8": right} for left, right in task_splits],
        "source_config": config["source_config"], "source_config_sha256": config["source_config_sha256"],
        "split_manifest": source["split_manifest"], "split_manifest_sha256": digest(manifest_path),
        "initial_bank_checkpoint": initial_bank,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(rank)},
    }
    (output / f"rank{rank}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
