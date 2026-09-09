#!/usr/bin/env python3
"""Freeze Table VI/IX assignment payloads from the selected H0 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from badit_tf.assignment import (
    build_natural_modulation_profiles,
    solve_nonempty_tf_assignment,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/configs/tpami_table_vi_ix/assignments"
MODELS = {
    "qwen3_4b": {
        "h0": "h0_calibration_sweep_qwen3_4b_seed24001_remote_l20z",
        "selected": "experiments/configs/h1_qwen3_4b_h0_selected_assignment.json",
        "candidate": "experiments/configs/m4/candidates/m4_qwen3_4b_assignment_candidates.json",
    },
    "llama3_3b": {
        "h0": "h0_calibration_sweep_llama3_3b_seed24001_local_a100",
        "selected": "experiments/configs/h1_llama3_3b_h0_selected_assignment.json",
        "candidate": "experiments/configs/m4/candidates/m4_llama3_3b_assignment_candidates.json",
    },
    "gemma2_2b": {
        "h0": "h0_calibration_sweep_gemma2_2b_seed24001_local_a100",
        "selected": "experiments/configs/h1_gemma2_2b_h0_selected_assignment.json",
        "candidate": "experiments/configs/m4/candidates/m4_gemma2_2b_assignment_candidates.json",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_role(
    collection: Path, role: str, expected_ids: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for metadata_path in sorted(collection.glob("rank*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(collection / f"{metadata_path.stem}.npz")
        key = "fisher_scores" if role == "fisher" else f"q_{role}"
        for item, array in zip(metadata["roles"][role], arrays[key], strict=True):
            sample_id = str(item["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate {role} sample ID: {sample_id}")
            rows[sample_id] = (array, item)
    missing = [sample_id for sample_id in expected_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"missing {role} sample IDs: {missing[:5]}")
    ordered = [rows[sample_id] for sample_id in expected_ids]
    return np.stack([row[0] for row in ordered]), [row[1] for row in ordered]


def task_prefix_indices(metadata: list[dict[str, Any]], count: int) -> np.ndarray:
    indices: list[int] = []
    for task in sorted({str(row["task"]) for row in metadata}):
        available = [
            index
            for index, row in enumerate(metadata)
            if str(row["task"]) == task
        ]
        if len(available) < count:
            raise ValueError(f"{task}: need {count}, found {len(available)}")
        indices.extend(available[:count])
    return np.asarray(indices, dtype=np.int64)


def prepare_model(slug: str, spec: dict[str, str]) -> dict[str, Any]:
    existing_path = OUTPUT / f"{slug}_assignments.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        capacities = existing["tf_no_capacity_capacities"]
        return {
            "model_slug": slug,
            "path": str(existing_path.relative_to(ROOT)),
            "sha256": digest(existing_path),
            "unequal_layers": sum(
                len(set(value)) > 1 for value in capacities.values()
            ),
            "capacity_min": min(min(value) for value in capacities.values()),
            "capacity_max": max(max(value) for value in capacities.values()),
        }
    h0 = ROOT / "experiments/raw_results" / spec["h0"]
    base_config = json.loads(
        (h0 / "h0_selection_lock.json").read_text(encoding="utf-8")
    )["selected_config"]
    manifest_path = ROOT / "experiments/configs/splits/h0_calibration_seed24001.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = h0 / "collection"
    q, q_meta = merge_role(
        collection,
        "assignment",
        [str(row["sample_id"]) for row in manifest["roles"]["assignment"]],
    )
    fisher_scores, fisher_meta = merge_role(
        collection,
        "fisher",
        [str(row["sample_id"]) for row in manifest["roles"]["fisher"]],
    )
    q_indices = task_prefix_indices(
        q_meta, int(base_config["assignment_probes_per_task"])
    )
    fisher_indices = task_prefix_indices(
        fisher_meta, int(base_config["fisher_probes_per_task"])
    )
    selected_q = q[q_indices].astype(np.float64)
    fisher = np.mean(np.square(fisher_scores[fisher_indices].astype(np.float64)), axis=0)
    selected_meta = [q_meta[index] for index in q_indices]
    counts = {
        task: sum(str(row["task"]) == task for row in selected_meta)
        for task in sorted({str(row["task"]) for row in selected_meta})
    }
    weights = np.asarray(
        [1.0 / (len(counts) * counts[str(row["task"])]) for row in selected_meta],
        dtype=np.float64,
    )
    selected_path = ROOT / spec["selected"]
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    candidates_path = ROOT / spec["candidate"]
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    layer_names = list(selected["layer_names"])
    if layer_names != list(candidates["layer_names"]) or len(layer_names) != q.shape[1]:
        raise AssertionError(f"{slug}: layer order mismatch")

    no_capacity: dict[str, list[int]] = {}
    solver_audit: dict[str, Any] = {}
    capacities: dict[str, list[int]] = {}
    for layer_index, layer_name in enumerate(layer_names):
        layer_fisher = fisher[layer_index]
        rho = float(base_config["epsilon_f"]) * float(np.median(layer_fisher))
        profiles, damped, _ = build_natural_modulation_profiles(
            selected_q[:, layer_index], layer_fisher, weights, rho
        )
        result = solve_nonempty_tf_assignment(
            profiles,
            damped,
            num_experts=8,
            restarts=int(base_config["solver_restarts"]),
            max_iterations=50,
            seed=24001 + layer_index * 4099,
        )
        no_capacity[layer_name] = result.labels.tolist()
        capacities[layer_name] = np.bincount(result.labels, minlength=8).tolist()
        solver_audit[layer_name] = result.to_jsonable()
    if not all(min(value) >= 1 and sum(value) == 32 for value in capacities.values()):
        raise AssertionError(f"{slug}: invalid nonempty capacity")

    payload = {
        "schema_version": 1,
        "experiment_id": "TPAMI-TABLE-VI-IX-ASSIGNMENT-FREEZE",
        "model_slug": slug,
        "layer_names": layer_names,
        "tf": selected["tf"],
        "raw_q": candidates["raw_q"],
        "gg_dog": candidates["gg_dog"],
        "random_balanced": candidates["random_balanced"],
        "tf_no_capacity": no_capacity,
        "tf_no_capacity_capacities": capacities,
        "tf_no_capacity_solver_audit": solver_audit,
        "paper_constraint": "all experts nonempty; expert size is not forced to rank",
        "selected_h0_config": base_config,
        "source": {
            "h0_result": str(h0.relative_to(ROOT)),
            "h0_lock_sha256": digest(h0 / "h0_selection_lock.json"),
            "selected_assignment": spec["selected"],
            "selected_assignment_sha256": digest(selected_path),
            "comparison_candidates": spec["candidate"],
            "comparison_candidates_sha256": digest(candidates_path),
            "split_manifest": str(manifest_path.relative_to(ROOT)),
            "split_manifest_sha256": digest(manifest_path),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{slug}_assignments.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "model_slug": slug,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "unequal_layers": sum(len(set(value)) > 1 for value in capacities.values()),
        "capacity_min": min(min(value) for value in capacities.values()),
        "capacity_max": max(max(value) for value in capacities.values()),
    }


def main() -> None:
    records = [prepare_model(slug, spec) for slug, spec in MODELS.items()]
    manifest = {
        "schema_version": 1,
        "event": "TPAMI_TABLE_VI_IX_ASSIGNMENTS_FROZEN",
        "records": records,
    }
    path = ROOT / "experiments/materials/tpami_table_vi_ix_assignment_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(path.relative_to(ROOT)), "records": records}))


if __name__ == "__main__":
    main()
