#!/usr/bin/env python3
"""Evaluate the frozen nonempty-only assignment on existing M2 profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import free_gate_vector, freeze_for_calibration, intervention_losses, supervised_tensors, tied_gate_vector
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"):
        raise AssertionError("Table VI fidelity must remain test-closed")
    validate_fidelity_precision(config)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != int(config["world_size"]):
        raise AssertionError("world size mismatch")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    manifest = json.loads(Path(config["split_manifest"]).read_text())
    all_rows = load_role_instances(manifest, "fidelity")
    indices = task_prefix_indices([{"task": row["task"]} for row in all_rows], int(config["fidelity_probes_per_task"]))
    rows = [all_rows[index] for index in indices]
    source = Path(config["profile_source_dir"])
    profiles = np.load(source / "m2_profiles.npz")
    q_fidelity = profiles["q_fidelity"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    candidates = json.loads((source / "m2_assignment_candidates.json").read_text())
    assignments = json.loads(Path(config["tpami_assignment_path"]).read_text())["tf_no_capacity"]
    if q_fidelity.shape[0] != len(rows):
        raise AssertionError("profile/sample count mismatch")

    dtype = {"float32": torch.float32}[config["torch_dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], local_files_only=True, dtype=dtype,
        attn_implementation=config["attention_implementation"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injection = inject_from_config(model, config)
    bank = load_initial_bank_checkpoint(model, config)
    model.eval()
    layers = freeze_for_calibration(model)
    layer_names = [name for name, _ in layers]
    if layer_names != candidates["layer_names"] or layer_names != list(injection.layers) or set(assignments) != set(layer_names):
        raise AssertionError("layer identity drift")

    output = Path(config["output_dir"]) / "fidelity"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank{rank}.jsonl"
    eta = float(config["fidelity_eta"])
    with path.open("w") as handle:
        for sample_index in range(rank, len(rows), world):
            row = rows[sample_index]
            tensors = supervised_tensors(
                tokenizer, row, max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]), device=device,
            )
            for layer_index, (layer_name, layer) in enumerate(layers):
                q = q_fidelity[sample_index, layer_index]
                damped = fisher[layer_index] + float(candidates["rho"][layer_name])
                free = free_gate_vector(q, damped, eta=eta)
                labels = np.asarray(assignments[layer_name], dtype=np.int64)
                tied, regret = tied_gate_vector(q, damped, labels, eta=eta, num_experts=int(config["num_experts"]))
                losses = intervention_losses(model, layer, tensors, np.stack([free, tied]))
                repeat = float(intervention_losses(model, layer, tensors, free[None, :])[0])
                displacement = np.abs(tied - 1.0)
                handle.write(json.dumps({
                    "sample_id": row["sample_id"], "task": row["task"],
                    "layer_index": layer_index, "layer_name": layer_name,
                    "eta": eta, "method": "tf_no_capacity", "assignment_index": -1,
                    "predicted_regret": float(regret), "free_loss": float(losses[0]),
                    "repeat_free_loss": repeat, "repeat_noise_abs": abs(repeat - float(losses[0])),
                    "tied_loss": float(losses[1]), "observed_gap": float(losses[1] - losses[0]),
                    "median_abs_gate_displacement": float(np.median(displacement)),
                    "p95_abs_gate_displacement": float(np.quantile(displacement, 0.95)),
                    "max_abs_gate_displacement": float(np.max(displacement)),
                    "initial_bank_checkpoint_sha256": bank["file_sha256"] if bank else None,
                }, sort_keys=True) + "\n")
            print(f"rank={rank} no-capacity fidelity sample={sample_index} complete", flush=True)


if __name__ == "__main__":
    main()
