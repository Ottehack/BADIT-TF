#!/usr/bin/env python3
"""Evaluate all H0 TF candidates on validation or report-only fidelity roles."""

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
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("damping_validation", "fidelity"), required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    config = yaml.safe_load(args.config.read_text())
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    output_dir = Path(config["output_dir"])
    manifest = json.loads(Path(config["split_manifest"]).read_text())
    rows = load_role_instances(manifest, args.role)
    indices = list(range(rank, len(rows), world))
    profiles = np.load(output_dir / "h0_profiles.npz")
    q_all = profiles[f"q_{args.role}"]
    candidates = json.loads(
        (output_dir / "h0_assignment_candidates.json").read_text()
    )
    trial_ids = sorted(candidates["trials"])
    if len(trial_ids) != 13:
        raise AssertionError("H0 must contain 13 trials")

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
    inject_from_config(model, config)
    initial_bank = load_initial_bank_checkpoint(model, config)
    model.eval()
    layers = freeze_for_calibration(model)
    layer_names = [name for name, _ in layers]
    if layer_names != candidates["layer_names"]:
        raise ValueError("layer order mismatch")

    result_dir = output_dir / "h0_finite" / args.role
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"rank{rank}.jsonl"
    eta = float(config["fidelity_eta"])
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_index in indices:
            row = rows[sample_index]
            tensors = supervised_tensors(
                tokenizer,
                row,
                max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]),
                device=device,
            )
            for layer_index in [int(value) for value in config["pilot_layer_indices"]]:
                layer_name = layer_names[layer_index]
                layer = layers[layer_index][1]
                q = q_all[sample_index, layer_index].astype(np.float64)
                for chunk_start in range(0, len(trial_ids), args.chunk_size):
                    chunk = trial_ids[chunk_start : chunk_start + args.chunk_size]
                    vectors = []
                    predicted = []
                    for trial_id in chunk:
                        fisher = profiles[candidates["fisher_keys"][trial_id]][layer_index]
                        rho = float(candidates["rho"][trial_id][layer_name])
                        damped = fisher + rho
                        labels = np.asarray(candidates["tf"][trial_id][layer_name])
                        free = free_gate_vector(q, damped, eta=eta)
                        tied, regret = tied_gate_vector(
                            q,
                            damped,
                            labels,
                            eta=eta,
                            num_experts=int(config["num_experts"]),
                        )
                        vectors.extend((free, tied))
                        predicted.append((trial_id, regret, free))
                    losses = intervention_losses(
                        model, layer, tensors, np.stack(vectors)
                    )
                    for trial_offset, (trial_id, regret, free) in enumerate(predicted):
                        free_loss = float(losses[2 * trial_offset])
                        tied_loss = float(losses[2 * trial_offset + 1])
                        repeat_noise = None
                        if trial_id == "H0-00":
                            repeat = float(
                                intervention_losses(
                                    model, layer, tensors, free[None, :]
                                )[0]
                            )
                            repeat_noise = abs(repeat - free_loss)
                        record = {
                            "role": args.role,
                            "sample_id": row["sample_id"],
                            "task": row["task"],
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "trial_id": trial_id,
                            "eta": eta,
                            "predicted_regret": float(regret),
                            "free_loss": free_loss,
                            "tied_loss": tied_loss,
                            "observed_gap": tied_loss - free_loss,
                            "repeat_noise_abs": repeat_noise,
                            "initial_bank_checkpoint_sha256": initial_bank["file_sha256"],
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"rank={rank} role={args.role} sample={sample_index} complete",
                flush=True,
            )
    print(output_path)


if __name__ == "__main__":
    main()
