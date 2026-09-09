#!/usr/bin/env python3
"""Evaluate frozen-router grouping/routing decomposition by primitive intervention."""

from __future__ import annotations

import argparse
import hashlib
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
from badit_tf.decomposition import (
    canonical_to_physical,
    grouping_routing_decomposition,
    labels_from_primitive_indices,
)
from badit_tf.m2 import task_prefix_indices, validate_fidelity_precision
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def set_route_masks(model: torch.nn.Module, mask: torch.Tensor) -> None:
    for _, layer in iter_ability_layers(model):
        layer.set_attention_mask(mask)


def method_labels(candidates: dict, layer_name: str, internal_labels: np.ndarray):
    yield "contiguous", -1, np.asarray(candidates["contiguous"], dtype=np.int64)
    for index, labels in enumerate(candidates["random_balanced"]):
        yield "random_balanced", index, np.asarray(labels, dtype=np.int64)
    for method in ("gg_dog", "raw_q"):
        yield method, -1, np.asarray(candidates[method][layer_name], dtype=np.int64)
    yield "tf", -1, internal_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method-chunk-size", type=int, default=4)
    args = parser.parse_args()
    if args.method_chunk_size < 1:
        raise ValueError("method chunk size must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"):
        raise AssertionError("M4 must not load final/downstream test")
    if config.get("mapping_change_scope") != "analysis_only_P":
        raise AssertionError("M4 permits only analysis mapping P to change")
    if config.get("coefficient_renormalization") is not False:
        raise AssertionError("M4 coefficient renormalization must be disabled")
    eta = float(config["decomposition_eta"])
    if eta not in {0.10, 0.0001}:
        raise AssertionError("M4 eta is outside the original/recovery frozen values")
    if eta == 0.0001:
        amendment_path = Path(config["m4_protocol_amendment"])
        if sha256(amendment_path) != config["m4_protocol_amendment_sha256"]:
            raise AssertionError("M4 local-eta protocol amendment hash drift")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment.get("status") != "ACTIVE_BEFORE_M4_R1_FORWARD":
            raise AssertionError("M4 local-eta protocol amendment is not active")
    validate_fidelity_precision(config)
    candidates_path = Path(config["m4_assignment_candidates"])
    if sha256(candidates_path) != config["m4_assignment_candidates_sha256"]:
        raise AssertionError("M4 assignment candidate hash drift")
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != int(config["world_size"]):
        raise RuntimeError("M4 world size mismatch")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    all_rows = load_role_instances(manifest, "fidelity")
    indices = task_prefix_indices([{"task": row["task"]} for row in all_rows], int(config["fidelity_probes_per_task"]))
    rows = [all_rows[index] for index in indices]
    profiles = np.load(Path(config["output_dir"]) / "m4_profiles.npz")
    q_fidelity = profiles["q_fidelity"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    profile_metadata = json.loads(
        (Path(config["output_dir"]) / "m4_profile_metadata.json").read_text(encoding="utf-8")
    )
    if q_fidelity.shape[0] != len(rows):
        raise ValueError("M4 q rows do not match frozen fidelity IDs")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], local_files_only=True, dtype=torch.float32,
        attn_implementation=config["attention_implementation"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injection = inject_from_config(model, config)
    trained = load_initial_bank_checkpoint(model, config)
    if trained is None or trained["file_sha256"] != config["initial_bank_checkpoint_sha256"]:
        raise AssertionError("M4 trained checkpoint did not load exactly")
    model.eval()
    routed_layers = list(iter_ability_layers(model))
    layer_names = [name for name, _ in routed_layers]
    if layer_names != list(injection.layers) or layer_names != profile_metadata["layer_names"] or layer_names != candidates["layer_names"]:
        raise AssertionError("M4 layer identity/order drift")
    internal_assignments = {
        layer_name: labels_from_primitive_indices(layer.primitive_indices.detach().cpu().numpy()).tolist()
        for layer_name, layer in routed_layers
    }
    if canonical_json_sha256(internal_assignments) != config["m0_tf_assignment_sha256"]:
        raise AssertionError("M4 checkpoint primitive mapping hash differs from frozen M0 evidence")

    output = Path(config["output_dir"]) / "decomposition"
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
            set_route_masks(model, supervised_route_attention_mask(tensors, scope=str(config["route_pooling_scope"])))
            with torch.no_grad():
                model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"], use_cache=False)
            route_codes = [layer._last_route["actual"][0].float().cpu().numpy() for _, layer in routed_layers]
            layers = freeze_for_calibration(model)
            for layer_index, (layer_name, layer) in enumerate(layers):
                route_code = route_codes[layer_index]
                if int(np.count_nonzero(route_code)) != int(config["top_k"]):
                    raise AssertionError(f"M4 route top-k drift for {layer_name}")
                internal_indices = layer.primitive_indices.detach().cpu().numpy()
                internal_labels = labels_from_primitive_indices(internal_indices)
                q = q_fidelity[sample_index, layer_index]
                rho = float(config["epsilon_f"]) * float(np.median(fisher[layer_index]))
                damped = fisher[layer_index] + rho
                free_canonical = q / damped
                free_physical = canonical_to_physical(free_canonical, internal_indices)
                free_gate = 1.0 + eta * free_physical
                entries = list(method_labels(candidates, layer_name, internal_labels))
                repeat_free = float(intervention_losses(model, layer, tensors, free_gate[None, :])[0])
                for start in range(0, len(entries), args.method_chunk_size):
                    chunk = entries[start:start + args.method_chunk_size]
                    decompositions = []
                    gates = [free_gate]
                    for method, assignment_index, labels in chunk:
                        decomposition = grouping_routing_decomposition(
                            q, damped, labels, route_code.astype(np.float64) - 1.0
                        )
                        routed_physical = canonical_to_physical(
                            decomposition.routed_primitive_delta, internal_indices
                        )
                        gates.append(1.0 + eta * routed_physical)
                        decompositions.append((method, assignment_index, decomposition))
                    losses = intervention_losses(model, layer, tensors, np.stack(gates))
                    free_loss = float(losses[0])
                    for offset, (method, assignment_index, decomposition) in enumerate(decompositions, start=1):
                        record = {
                            "sample_id": row["sample_id"], "task": row["task"],
                            "layer_index": layer_index, "layer_name": layer_name,
                            "eta": eta, "method": method, "assignment_index": assignment_index,
                            "grouping": decomposition.grouping, "routing": decomposition.routing,
                            "total_pred": decomposition.total, "direct_total": decomposition.direct_total,
                            "orthogonality_error": decomposition.orthogonality_error,
                            "free_loss": free_loss, "repeat_free_loss": repeat_free,
                            "repeat_noise_abs": abs(repeat_free - free_loss),
                            "routed_loss": float(losses[offset]),
                            "observed_gap": float(losses[offset] - losses[0]),
                            "route_code": route_code.tolist(),
                            "active_experts": int(np.count_nonzero(route_code)),
                            "trained_checkpoint_sha256": trained["file_sha256"],
                            "tf_mapping_matches_checkpoint": True,
                            "primitive_intervention": True,
                            "coefficient_renormalization": False,
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"rank={rank} M4 sample={sample_index} complete", flush=True)
    print(result_path)


if __name__ == "__main__":
    main()
