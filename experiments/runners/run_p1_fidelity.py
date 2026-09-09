#!/usr/bin/env python3
"""Evaluate finite-loss assignment gaps for P1 on representative layers."""

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
    parser.add_argument(
        "--role",
        choices=(
            "damping_validation",
            "fidelity",
            "confirmation_fidelity_v2",
            "confirmation_fidelity_v3",
        ),
        required=True,
    )
    parser.add_argument("--selected-damping", type=Path)
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Optional output directory; defaults to <P1 output>/<role>.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="Optional diagnostic intervention scale override.",
    )
    parser.add_argument(
        "--epsilon-key",
        action="append",
        dest="epsilon_keys",
        help="Optional damping key restriction; may be repeated.",
    )
    parser.add_argument(
        "--unbatched",
        action="store_true",
        help="Evaluate each gate vector separately with batch size one.",
    )
    parser.add_argument(
        "--measure-repeat-noise",
        action="store_true",
        help="Repeat the free intervention once and record absolute loss noise.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    output_dir = Path(config["output_dir"])
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    all_rows = load_role_instances(manifest, args.role)
    indices = list(range(rank, len(all_rows), world))
    profiles = np.load(output_dir / "profiles.npz")
    q_all = profiles[f"q_{args.role}"]
    fisher = profiles["fisher"]
    candidates = json.loads(
        (output_dir / "assignment_candidates.json").read_text(encoding="utf-8")
    )
    if args.epsilon_keys:
        epsilon_keys = list(args.epsilon_keys)
    elif args.role in {
        "fidelity",
        "confirmation_fidelity_v2",
        "confirmation_fidelity_v3",
    }:
        if args.selected_damping is None:
            raise ValueError("--selected-damping is required for fidelity")
        selected = json.loads(args.selected_damping.read_text(encoding="utf-8"))
        epsilon_keys = [selected["selected_epsilon_key"]]
    else:
        epsilon_keys = [f"{float(value):.4g}" for value in config["damping_epsilons"]]
    eta = float(config["fidelity_eta"] if args.eta is None else args.eta)
    if eta <= 0:
        raise ValueError("--eta must be positive")
    unknown = sorted(set(epsilon_keys) - set(candidates["rho"]))
    if unknown:
        raise ValueError(f"unknown damping keys: {unknown}")

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
    if layer_names != candidates["layer_names"]:
        raise ValueError("layer names differ from profile collection")

    result_dir = args.result_dir or (output_dir / args.role)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"rank{rank}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_index in indices:
            row = all_rows[sample_index]
            tensors = supervised_tensors(
                tokenizer,
                row,
                max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]),
                device=device,
            )
            for layer_index in config["pilot_layer_indices"]:
                layer_name = layer_names[int(layer_index)]
                layer = layers[int(layer_index)][1]
                q = q_all[sample_index, int(layer_index)].astype(np.float64)
                for epsilon_key in epsilon_keys:
                    rho = float(candidates["rho"][epsilon_key][layer_name])
                    damped = fisher[int(layer_index)] + rho
                    method_labels = [
                        ("contiguous", -1, np.asarray(candidates["contiguous"])),
                        *[
                            ("random_balanced", random_index, np.asarray(labels))
                            for random_index, labels in enumerate(
                                candidates["random_balanced"]
                            )
                        ],
                        (
                            "tf",
                            -1,
                            np.asarray(candidates["tf"][epsilon_key][layer_name]),
                        ),
                    ]
                    free = free_gate_vector(
                        q, damped, eta=eta
                    )
                    gate_vectors = [free]
                    predicted = []
                    for method, assignment_index, labels in method_labels:
                        tied, regret = tied_gate_vector(
                            q,
                            damped,
                            labels,
                            eta=eta,
                            num_experts=int(config["num_experts"]),
                        )
                        gate_vectors.append(tied)
                        predicted.append((method, assignment_index, regret))
                    if args.unbatched:
                        losses = np.asarray(
                            [
                                float(
                                    intervention_losses(
                                        model,
                                        layer,
                                        tensors,
                                        vector[None, :],
                                    )[0]
                                )
                                for vector in gate_vectors
                            ],
                            dtype=np.float64,
                        )
                    else:
                        losses = intervention_losses(
                            model, layer, tensors, np.stack(gate_vectors)
                        )
                    free_loss = float(losses[0])
                    repeat_free_loss = None
                    repeat_noise_abs = None
                    if args.measure_repeat_noise:
                        repeat_free_loss = float(
                            intervention_losses(
                                model, layer, tensors, free[None, :]
                            )[0]
                        )
                        repeat_noise_abs = abs(repeat_free_loss - free_loss)
                    for offset, (method, assignment_index, regret) in enumerate(
                        predicted, start=1
                    ):
                        record = {
                            "role": args.role,
                            "sample_id": row["sample_id"],
                            "task": row["task"],
                            "layer_index": int(layer_index),
                            "layer_name": layer_name,
                            "epsilon_key": epsilon_key,
                            "eta": eta,
                            "method": method,
                            "assignment_index": assignment_index,
                            "predicted_regret": regret,
                            "free_loss": free_loss,
                            "repeat_free_loss": repeat_free_loss,
                            "repeat_noise_abs": repeat_noise_abs,
                            "tied_loss": float(losses[offset]),
                            "observed_gap": float(losses[offset] - losses[0]),
                            "initial_bank_checkpoint_sha256": (
                                initial_bank["file_sha256"] if initial_bank else None
                            ),
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"rank={rank} role={args.role} sample={sample_index} complete",
                flush=True,
            )
    print(output_path)


if __name__ == "__main__":
    main()
