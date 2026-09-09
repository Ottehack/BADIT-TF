#!/usr/bin/env python3
"""Evaluate the frozen four-scale trained-router deployment curve."""

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
    freeze_for_calibration,
    intervention_losses,
    supervised_route_attention_mask,
    supervised_tensors,
)
from badit_tf.core import iter_ability_layers
from badit_tf.deployment_scale import (
    canonical_primitive_delta,
    quadratic_loss_delta,
    relative_prediction_error,
    validate_deployment_etas,
)
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def set_route_masks(model: torch.nn.Module, mask: torch.Tensor) -> None:
    """Set prompt-pooling masks without importing generation evaluators."""
    for _, layer in iter_ability_layers(model):
        layer.set_attention_mask(mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"):
        raise AssertionError("M3 must not load final/downstream test")
    validate_fidelity_precision(config)
    etas = validate_deployment_etas(config["deployment_etas"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != int(config["world_size"]):
        raise RuntimeError("M3 world size mismatch")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    all_rows = load_role_instances(manifest, "fidelity")
    indices = task_prefix_indices(
        [{"task": row["task"]} for row in all_rows], int(config["fidelity_probes_per_task"])
    )
    rows = [all_rows[index] for index in indices]
    profiles = np.load(Path(config["output_dir"]) / "m3_profiles.npz")
    q_fidelity = profiles["q_fidelity"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    profile_metadata = json.loads(
        (Path(config["output_dir"]) / "m3_profile_metadata.json").read_text(encoding="utf-8")
    )
    if q_fidelity.shape[0] != len(rows):
        raise ValueError("M3 q rows do not match frozen fidelity IDs")

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
        raise AssertionError("M3 trained checkpoint did not load exactly")
    model.eval()
    routed_layers = list(iter_ability_layers(model))
    layer_names = [name for name, _ in routed_layers]
    if layer_names != list(injection.layers) or layer_names != profile_metadata["layer_names"]:
        raise AssertionError("M3 layer identity/order drift")

    output = Path(config["output_dir"]) / "deployment"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"rank{rank}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for sample_index in range(rank, len(rows), world):
            row = rows[sample_index]
            tensors = supervised_tensors(
                tokenizer, row, max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]), device=device,
            )
            for _, layer in routed_layers:
                layer.disable_calibration()
                layer.topk_enabled.fill_(True)
            set_route_masks(
                model,
                supervised_route_attention_mask(tensors, scope=str(config["route_pooling_scope"])),
            )
            with torch.no_grad():
                model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"], use_cache=False)
            route_codes = [layer._last_route["actual"][0].float().cpu().numpy() for _, layer in routed_layers]
            dense_codes = [layer._last_route["dense"][0].float().cpu().numpy() for _, layer in routed_layers]
            layers = freeze_for_calibration(model)
            for layer_index, (layer_name, layer) in enumerate(layers):
                code = route_codes[layer_index]
                dense = dense_codes[layer_index]
                active = int(np.count_nonzero(code))
                if active != int(config["top_k"]):
                    raise AssertionError(f"M3 route has {active} active experts for {layer_name}")
                canonical_direction = canonical_primitive_delta(
                    code, layer.primitive_indices.detach().cpu().numpy()
                )
                physical_direction = np.repeat(code - 1.0, int(config["rank"]))
                gate_vectors = [np.ones_like(physical_direction)] + [
                    1.0 + eta * physical_direction for eta in etas
                ]
                losses = intervention_losses(model, layer, tensors, np.stack(gate_vectors))
                repeat = float(
                    intervention_losses(model, layer, tensors, np.ones((1, physical_direction.size)))[0]
                )
                baseline = float(losses[0])
                rho = float(config["epsilon_f"]) * float(np.median(fisher[layer_index]))
                damped = fisher[layer_index] + rho
                for offset, eta in enumerate(etas, start=1):
                    delta = eta * canonical_direction
                    predicted = quadratic_loss_delta(q_fidelity[sample_index, layer_index], damped, delta)
                    observed = float(losses[offset] - losses[0])
                    record = {
                        "sample_id": row["sample_id"], "task": row["task"],
                        "layer_index": layer_index, "layer_name": layer_name,
                        "eta": eta, "predicted_delta_loss": predicted,
                        "observed_delta_loss": observed,
                        "relative_error": relative_prediction_error(predicted, observed),
                        "baseline_loss": baseline, "repeat_baseline_loss": repeat,
                        "repeat_noise_abs": abs(repeat - baseline),
                        "active_experts": active,
                        "route_code": code.tolist(), "dense_code": dense.tolist(),
                        "max_abs_scaled_displacement": float(np.max(np.abs(delta))),
                        "trained_checkpoint_sha256": trained["file_sha256"],
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"rank={rank} M3 sample={sample_index} complete", flush=True)
    print(result_path)


if __name__ == "__main__":
    main()
