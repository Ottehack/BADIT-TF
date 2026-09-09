#!/usr/bin/env python3
"""Merge M2 collection shards and freeze all five assignment comparators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from badit_tf.m2 import build_assignment_payload, task_prefix_indices


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_role(
    collection: Path, role: str, expected_ids: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    key = "fisher_scores" if role == "fisher" else f"q_{role}"
    for metadata_path in sorted(collection.glob("rank*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(collection / f"{metadata_path.stem}.npz")
        for item, array in zip(metadata["roles"][role], arrays[key], strict=True):
            sample_id = str(item["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate M2 {role} ID: {sample_id}")
            rows[sample_id] = (array, item)
    missing = [sample_id for sample_id in expected_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"missing M2 {role} samples: {missing[:8]}")
    ordered = [rows[sample_id] for sample_id in expected_ids]
    return np.stack([item[0] for item in ordered]), [item[1] for item in ordered]


def merge_gg_sums(collection: Path, layer_count: int) -> tuple[np.ndarray, int]:
    sums: list[np.ndarray | None] = [None] * layer_count
    total_count = 0
    for metadata_path in sorted(collection.glob("rank*.json")):
        arrays = np.load(collection / f"{metadata_path.stem}.npz")
        count = int(arrays["gg_count"][0])
        if count < 0:
            raise ValueError("negative GG gradient count")
        total_count += count
        for layer_index in range(layer_count):
            value = arrays[f"gg_sum_{layer_index}"].astype(np.float64)
            if sums[layer_index] is None:
                sums[layer_index] = value
            else:
                sums[layer_index] += value
    if total_count <= 0 or any(value is None for value in sums):
        raise ValueError("incomplete M2 GG gradient collection")
    return np.stack([value for value in sums if value is not None]), total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if bool(config.get("official_test_loaded", False)) or bool(
        config.get("downstream_test_loaded", False)
    ):
        raise AssertionError("M2 cannot load a downstream/final test")
    amendment_path = Path(config["m2_protocol_amendment"])
    if sha256_file(amendment_path) != str(config["m2_protocol_amendment_sha256"]):
        raise AssertionError("M2 protocol amendment hash drift")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("status") != "ACTIVE_BEFORE_M2_DISPATCH":
        raise AssertionError("M2 protocol amendment is not active")
    eta_locks = list(config.get("eta_selection_locks", []))
    if not eta_locks:
        raise ValueError("M2 config must cite the P1 eta-selection lock")
    for entry in eta_locks:
        lock_path = Path(entry["path"])
        if sha256_file(lock_path) != entry["sha256"]:
            raise AssertionError(f"eta selection lock hash drift: {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if float(lock["selected_eta"]) != float(config["fidelity_eta"]):
            raise AssertionError("M2 eta does not match the cited P1 lock")

    output_dir = Path(config["output_dir"])
    collection = output_dir / "collection"
    manifest_path = Path(config["split_manifest"])
    if sha256_file(manifest_path) != str(config["split_manifest_sha256"]):
        raise AssertionError("M2 split manifest hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count_keys = {
        "assignment": "assignment_probes_per_task",
        "fisher": "fisher_probes_per_task",
        "fidelity": "fidelity_probes_per_task",
    }
    expected_ids = {}
    for role, count_key in count_keys.items():
        source_rows = manifest["roles"][role]
        indices = task_prefix_indices(source_rows, int(config[count_key]))
        expected_ids[role] = [str(source_rows[index]["sample_id"]) for index in indices]
    q_assignment, assignment_meta = merge_role(
        collection, "assignment", expected_ids["assignment"]
    )
    fisher_scores, fisher_meta = merge_role(
        collection, "fisher", expected_ids["fisher"]
    )
    q_fidelity, fidelity_meta = merge_role(
        collection, "fidelity", expected_ids["fidelity"]
    )
    first = json.loads(sorted(collection.glob("rank*.json"))[0].read_text(encoding="utf-8"))
    layer_names = list(first["layer_names"])
    gg_sums, gg_count = merge_gg_sums(collection, len(layer_names))
    expected_gg_count = int(config["assignment_probes_per_task"]) * len(
        sorted({str(row["task"]) for row in assignment_meta})
    )
    if gg_count != expected_gg_count:
        raise AssertionError(
            f"GG collection count={gg_count}, expected selected assignment count={expected_gg_count}"
        )

    payload, arrays = build_assignment_payload(
        q_assignment=q_assignment,
        assignment_metadata=assignment_meta,
        fisher_scores=fisher_scores,
        fisher_metadata=fisher_meta,
        gg_mean_vectors=gg_sums / float(gg_count),
        layer_names=layer_names,
        num_experts=int(config["num_experts"]),
        rank=int(config["rank"]),
        assignment_probes_per_task=int(config["assignment_probes_per_task"]),
        fisher_probes_per_task=int(config["fisher_probes_per_task"]),
        epsilon_f=float(config["epsilon_f"]),
        random_assignments=int(config["random_assignments"]),
        restarts=int(config["assignment_restarts"]),
        max_iterations=int(config["assignment_max_iterations"]),
        tolerance=float(config["assignment_tolerance"]),
        seed=int(config["seed"]),
    )
    payload["gg_gradient_count"] = gg_count
    payload["gg_gradient_reduction"] = first["gg_gradient_reduction"]
    payload["split_manifest_sha256"] = sha256_file(manifest_path)
    payload["eta_selection_locks"] = eta_locks
    arrays["q_assignment"] = q_assignment.astype(np.float32)
    arrays["q_fidelity"] = q_fidelity.astype(np.float32)
    np.savez(output_dir / "m2_profiles.npz", **arrays)
    (output_dir / "m2_assignment_candidates.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m2_profile_metadata.json").write_text(
        json.dumps(
            {
                "assignment": assignment_meta,
                "fisher": fisher_meta,
                "fidelity": fidelity_meta,
                "selected_assignment_indices": arrays["assignment_indices"].tolist(),
                "selected_fisher_indices": arrays["fisher_indices"].tolist(),
                "gg_gradient_count": gg_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "m2_assignment_candidates.json")


if __name__ == "__main__":
    main()
