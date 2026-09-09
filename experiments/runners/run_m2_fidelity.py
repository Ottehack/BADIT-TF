#!/usr/bin/env python3
"""Evaluate every frozen M2 assignment on all held-out fidelity layers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import (
    free_gate_vector,
    freeze_for_calibration,
    intervention_losses,
    supervised_tensors,
    tied_gate_vector,
)
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def _method_labels(candidates: dict, layer_name: str):
    yield "contiguous", -1, np.asarray(candidates["contiguous"], dtype=np.int64)
    for index, labels in enumerate(candidates["random_balanced"]):
        yield "random_balanced", index, np.asarray(labels, dtype=np.int64)
    for method in ("gg_dog", "raw_q", "tf"):
        yield method, -1, np.asarray(candidates[method][layer_name], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method-chunk-size", type=int, default=6)
    args = parser.parse_args()
    if args.method_chunk_size < 1:
        raise ValueError("method chunk size must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if bool(config.get("official_test_loaded", False)) or bool(
        config.get("downstream_test_loaded", False)
    ):
        raise AssertionError("M2 must not load a downstream/final test")
    validate_fidelity_precision(config)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    output_dir = Path(config["output_dir"])
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    all_rows = load_role_instances(manifest, "fidelity")
    selected_indices = task_prefix_indices(
        [{"task": row["task"]} for row in all_rows],
        int(config["fidelity_probes_per_task"]),
    )
    rows = [all_rows[index] for index in selected_indices]
    profiles = np.load(output_dir / "m2_profiles.npz")
    q_fidelity = profiles["q_fidelity"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    candidates = json.loads(
        (output_dir / "m2_assignment_candidates.json").read_text(encoding="utf-8")
    )
    if q_fidelity.shape[0] != len(rows):
        raise AssertionError("M2 q-fidelity rows do not match frozen manifest")
    eta = float(config["fidelity_eta"])
    if eta <= 0:
        raise ValueError("fidelity eta must be positive")

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
    if layer_names != candidates["layer_names"] or layer_names != list(injection.layers):
        raise AssertionError("M2 layer order drift")

    result_dir = output_dir / "fidelity"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"rank{rank}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for sample_index in range(rank, len(rows), world):
            row = rows[sample_index]
            tensors = supervised_tensors(
                tokenizer,
                row,
                max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]),
                device=device,
            )
            for layer_index, (layer_name, layer) in enumerate(layers):
                q = q_fidelity[sample_index, layer_index]
                damped = fisher[layer_index] + float(candidates["rho"][layer_name])
                free = free_gate_vector(q, damped, eta=eta)
                methods = list(_method_labels(candidates, layer_name))
                for start in range(0, len(methods), args.method_chunk_size):
                    chunk = methods[start : start + args.method_chunk_size]
                    gate_vectors = [free]
                    predicted = []
                    for method, assignment_index, labels in chunk:
                        tied, regret = tied_gate_vector(
                            q,
                            damped,
                            labels,
                            eta=eta,
                            num_experts=int(config["num_experts"]),
                        )
                        gate_vectors.append(tied)
                        predicted.append(
                            (method, assignment_index, regret, tied)
                        )
                    losses = intervention_losses(
                        model, layer, tensors, np.stack(gate_vectors)
                    )
                    repeat_free_loss = float(
                        intervention_losses(model, layer, tensors, free[None, :])[0]
                    )
                    free_loss = float(losses[0])
                    for offset, (method, assignment_index, regret, tied) in enumerate(
                        predicted, start=1
                    ):
                        displacement = np.abs(tied - 1.0)
                        record = {
                            "sample_id": row["sample_id"],
                            "task": row["task"],
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "eta": eta,
                            "method": method,
                            "assignment_index": assignment_index,
                            "predicted_regret": float(regret),
                            "free_loss": free_loss,
                            "repeat_free_loss": repeat_free_loss,
                            "repeat_noise_abs": abs(repeat_free_loss - free_loss),
                            "tied_loss": float(losses[offset]),
                            "observed_gap": float(losses[offset] - losses[0]),
                            "median_abs_gate_displacement": float(np.median(displacement)),
                            "p95_abs_gate_displacement": float(np.quantile(displacement, 0.95)),
                            "max_abs_gate_displacement": float(np.max(displacement)),
                            "initial_bank_checkpoint_sha256": (
                                initial_bank["file_sha256"] if initial_bank else None
                            ),
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"rank={rank} fidelity sample={sample_index} complete", flush=True)
    print(result_path)


if __name__ == "__main__":
    main()
