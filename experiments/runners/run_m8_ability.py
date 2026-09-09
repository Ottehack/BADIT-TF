#!/usr/bin/env python3
"""Evaluate held-out expert specificity, sufficiency and composition for M8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import (
    per_sequence_nll,
    supervised_route_attention_mask,
    supervised_tensors,
)
from badit_tf.core import iter_ability_layers
from badit_tf.m2 import task_prefix_indices
from badit_tf.p2_evaluation import clear_route_overrides, set_route_masks
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_intervention_assignments(
    path: Path,
    epsilon: float,
    method: str,
    layer_names: list[str],
    *,
    random_index: int | None = None,
    rank: int = 4,
) -> dict[str, list[int]]:
    """Resolve the frozen assignment used by one matched Table IX checkpoint."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if method == "tf":
        values = payload["tf"]
        assignments = (
            values
            if all(isinstance(value, list) for value in values.values())
            else values[str(epsilon)]
        )
    elif method in {"raw_q", "tf_no_capacity", "gg_dog"}:
        assignments = payload[method]
    elif method == "contiguous":
        labels = [index // int(rank) for index in range(32)]
        assignments = {name: labels for name in layer_names}
    elif method == "random_balanced":
        if random_index is None:
            raise AssertionError("random-balanced intervention requires an index")
        candidates = payload["random_balanced"]
        if random_index < 0 or random_index >= len(candidates):
            raise AssertionError("random-balanced intervention index is invalid")
        labels = candidates[random_index]
        assignments = {name: labels for name in layer_names}
    else:
        raise AssertionError(f"unsupported intervention assignment: {method}")
    if set(assignments) != set(layer_names):
        raise AssertionError("intervention assignment layer mismatch")
    for name, labels in assignments.items():
        if len(labels) != 32 or any(int(value) not in range(8) for value in labels):
            raise AssertionError(f"invalid intervention labels for {name}")
    return assignments


def select_profile_rows(
    q_fidelity: np.ndarray,
    metadata_rows: list[dict],
    selected_rows: list[dict],
) -> np.ndarray:
    """Select frozen profile rows by immutable sample ID, preserving run order."""

    positions = {
        str(row["sample_id"]): index for index, row in enumerate(metadata_rows)
    }
    selected_ids = [str(row["sample_id"]) for row in selected_rows]
    if len(positions) != len(metadata_rows):
        raise AssertionError("M8 profile metadata contains duplicate sample IDs")
    if any(sample_id not in positions for sample_id in selected_ids):
        raise AssertionError("M8 selected fidelity sample missing from frozen profiles")
    return q_fidelity[[positions[sample_id] for sample_id in selected_ids]]


def selected_route_support(
    dense: np.ndarray, actual: np.ndarray, top_k: int
) -> np.ndarray:
    """Recover the runtime Top-K support without treating a zero weight as absent."""

    if dense.ndim != 1 or actual.shape != dense.shape:
        raise AssertionError("M8 route vector shape mismatch")
    # Match torch.argsort(descending=True, stable=True): ties retain ascending
    # expert order.  A selected residual coefficient may legitimately be
    # exactly zero, so ``actual != 0`` is not a valid support indicator.
    support = np.argsort(-dense, kind="stable")[: int(top_k)]
    mask = np.zeros(dense.shape, dtype=bool)
    mask[support] = True
    if not np.array_equal(actual[mask], dense[mask]) or np.any(actual[~mask] != 0):
        raise AssertionError("M8 route mask/dense mismatch")
    return support


@torch.no_grad()
def variant_losses(
    model, layer, tensors: dict[str, torch.Tensor], codes: np.ndarray
) -> np.ndarray:
    override = torch.as_tensor(
        codes, dtype=layer.lora_A.dtype, device=layer.lora_A.device
    )
    batch = override.shape[0]
    repeated = {key: value.repeat(batch, 1) for key, value in tensors.items()}
    layer.set_route_code_override(override)
    try:
        outputs = model(
            input_ids=repeated["input_ids"],
            attention_mask=repeated["attention_mask"],
            use_cache=False,
        )
        losses = per_sequence_nll(outputs.logits, repeated["labels"])
    finally:
        layer.set_route_code_override(None)
    return losses.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"):
        raise AssertionError("M8 must not load official/downstream test")
    profile_path = Path(config["m8_profile_source"]) / "m4_profiles.npz"
    metadata_path = Path(config["m8_profile_source"]) / "m4_profile_metadata.json"
    if (
        digest(profile_path) != config["m8_profiles_sha256"]
        or digest(metadata_path) != config["m8_profile_metadata_sha256"]
    ):
        raise AssertionError("M8 frozen profile hash mismatch")
    profiles = np.load(profile_path)
    q_fidelity = profiles["q_fidelity"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    all_rows = load_role_instances(manifest, "fidelity")
    indices = task_prefix_indices(
        [{"task": row["task"]} for row in all_rows],
        int(config["fidelity_probes_per_task"]),
    )
    rows = [all_rows[index] for index in indices]
    if q_fidelity.shape[0] != len(metadata["fidelity"]):
        raise AssertionError("M8 frozen profile/metadata row count mismatch")
    q_fidelity = select_profile_rows(q_fidelity, metadata["fidelity"], rows)

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != int(config["world_size"]):
        raise RuntimeError("M8 world size mismatch")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = {"float32": torch.float32}[config["torch_dtype"]]
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
    trained = load_initial_bank_checkpoint(model, config)
    if (
        trained is None
        or trained["file_sha256"] != config["initial_bank_checkpoint_sha256"]
    ):
        raise AssertionError("M8 trained checkpoint mismatch")
    model.eval()
    routed_layers = list(iter_ability_layers(model))
    layer_names = [name for name, _ in routed_layers]
    if layer_names != list(injection.layers) or layer_names != metadata["layer_names"]:
        raise AssertionError("M8 layer order mismatch")
    selected_layer_indices = [
        int(value) for value in config.get("layer_indices", range(len(layer_names)))
    ]
    if len(selected_layer_indices) != len(set(selected_layer_indices)) or any(
        index < 0 or index >= len(layer_names) for index in selected_layer_indices
    ):
        raise AssertionError("M8 selected layer indices are invalid")
    intervention_method = str(config.get("intervention_assignment_method", "tf"))
    assignments = load_intervention_assignments(
        Path(config["tf_assignment_path"]),
        float(config["tf_assignment_epsilon"]),
        intervention_method,
        layer_names,
        random_index=config.get("assignment_random_index"),
        rank=int(config["rank"]),
    )

    output = Path(config["output_dir"])
    shard_dir = output / "ability"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank{rank}.jsonl"
    persistent = Path(config["persistent_output_dir"]) / "ability" / shard_path.name
    persistent.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("w", encoding="utf-8") as handle:
        for sample_index in range(rank, len(rows), world):
            row = rows[sample_index]
            tensors = supervised_tensors(
                tokenizer,
                row,
                max_length=int(config["max_sequence_length"]),
                max_target_length=int(config["max_target_length"]),
                device=device,
            )
            for _, layer in routed_layers:
                layer.disable_calibration()
                layer.topk_enabled.fill_(True)
            set_route_masks(
                model,
                supervised_route_attention_mask(
                    tensors, scope=str(config["route_pooling_scope"])
                ),
            )
            clear_route_overrides(model)
            with torch.no_grad():
                model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                    use_cache=False,
                )
            route_states = [
                (
                    layer._last_route["dense"][0].float().cpu().numpy(),
                    layer._last_route["actual"][0].float().cpu().numpy(),
                )
                for _, layer in routed_layers
            ]
            for layer_index in selected_layer_indices:
                layer_name, layer = routed_layers[layer_index]
                dense, code = route_states[layer_index]
                active = selected_route_support(dense, code, int(config["top_k"]))
                target = int(active[np.argmax(code[active])])
                candidates = active[active != target]
                control = int(
                    candidates[np.argmin(np.abs(code[candidates] - code[target]))]
                )
                target_deleted = code.copy()
                target_deleted[target] = 0
                control_deleted = code.copy()
                control_deleted[control] = 0
                residual_only = np.zeros_like(code)
                target_only = np.zeros_like(code)
                target_only[target] = code[target]
                losses = variant_losses(
                    model,
                    layer,
                    tensors,
                    np.stack(
                        [
                            code,
                            target_deleted,
                            control_deleted,
                            residual_only,
                            target_only,
                        ]
                    ),
                )
                (
                    full_loss,
                    target_deleted_loss,
                    control_deleted_loss,
                    residual_loss,
                    target_only_loss,
                ) = map(float, losses)
                labels = np.asarray(assignments[layer_name], dtype=np.int64)
                rho = float(config["epsilon_f"]) * float(np.median(fisher[layer_index]))
                damped = fisher[layer_index] + rho
                coefficients = np.asarray(
                    [
                        q_fidelity[sample_index, layer_index][labels == expert].sum()
                        / damped[labels == expert].sum()
                        for expert in range(int(config["num_experts"]))
                    ]
                )
                mass = np.abs(coefficients)
                probability = mass / max(float(mass.sum()), 1e-12)
                positive = probability[probability > 0]
                effective_experts = float(np.exp(-np.sum(positive * np.log(positive))))
                top1_mass = float(probability.max(initial=0))
                multi = effective_experts >= float(
                    config["multi_expert_neff_min"]
                ) and top1_mass <= float(config["multi_expert_top1_mass_max"])
                denominator = residual_loss - full_loss
                sufficiency = (residual_loss - target_only_loss) / (
                    denominator + np.copysign(1e-8, denominator if denominator else 1.0)
                )
                record = {
                    "sample_id": row["sample_id"],
                    "task": row["task"],
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "target_expert": target,
                    "control_expert": control,
                    "active_experts": active.tolist(),
                    "route_code": code.tolist(),
                    "full_loss": full_loss,
                    "target_deleted_loss": target_deleted_loss,
                    "control_deleted_loss": control_deleted_loss,
                    "residual_only_loss": residual_loss,
                    "target_only_loss": target_only_loss,
                    "target_deletion_increase": target_deleted_loss - full_loss,
                    "control_deletion_increase": control_deleted_loss - full_loss,
                    "specificity_gap": target_deleted_loss - control_deleted_loss,
                    "sufficiency": sufficiency,
                    "composition_gain": target_only_loss - full_loss,
                    "effective_experts": effective_experts,
                    "top1_mass": top1_mass,
                    "multi_expert": bool(multi),
                    "coefficient_renormalization": False,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            shutil.copy2(shard_path, persistent)
            print(f"rank={rank} M8 sample={sample_index} complete", flush=True)


if __name__ == "__main__":
    main()
