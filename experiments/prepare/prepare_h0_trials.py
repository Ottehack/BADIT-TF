#!/usr/bin/env python3
"""Solve all 13 preregistered H0 calibration configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from badit_tf.assignment import (
    build_natural_modulation_profiles,
    solve_balanced_tf_assignment,
)


def merge_role(
    collection: Path, role: str, expected_ids: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for metadata_path in sorted(collection.glob("rank*.json")):
        metadata = json.loads(metadata_path.read_text())
        arrays = np.load(collection / f"{metadata_path.stem}.npz")
        key = "fisher_scores" if role == "fisher" else f"q_{role}"
        for item, array in zip(metadata["roles"][role], arrays[key], strict=True):
            if item["sample_id"] in rows:
                raise ValueError(f"duplicate {role} ID {item['sample_id']}")
            rows[item["sample_id"]] = (array, item)
    missing = [sample_id for sample_id in expected_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"missing {role} IDs: {missing[:5]}")
    ordered = [rows[sample_id] for sample_id in expected_ids]
    return np.stack([row[0] for row in ordered]), [row[1] for row in ordered]


def task_prefix_indices(metadata: list[dict[str, Any]], count: int) -> np.ndarray:
    tasks = sorted({row["task"] for row in metadata})
    indices = []
    for task in tasks:
        task_indices = [index for index, row in enumerate(metadata) if row["task"] == task]
        if len(task_indices) < count:
            raise ValueError(f"{task} has {len(task_indices)} rows but needs {count}")
        indices.extend(task_indices[:count])
    return np.asarray(indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("experiments/configs/h0_calibration_grid.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    grid = json.loads(args.grid.read_text())
    manifest = json.loads(Path(config["split_manifest"]).read_text())
    output_dir = Path(config["output_dir"])
    collection = output_dir / "collection"
    role_ids = {
        role: [row["sample_id"] for row in manifest["roles"][role]]
        for role in ("assignment", "fisher", "damping_validation", "fidelity")
    }
    q_assignment, assignment_meta = merge_role(
        collection, "assignment", role_ids["assignment"]
    )
    fisher_scores, fisher_meta = merge_role(
        collection, "fisher", role_ids["fisher"]
    )
    q_validation, validation_meta = merge_role(
        collection, "damping_validation", role_ids["damping_validation"]
    )
    q_fidelity, fidelity_meta = merge_role(
        collection, "fidelity", role_ids["fidelity"]
    )
    first_meta = json.loads(sorted(collection.glob("rank*.json"))[0].read_text())
    layer_names = first_meta["layer_names"]
    num_experts = int(config["num_experts"])
    rank = int(config["rank"])

    candidates: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": "H0-A2-CALIBRATION-SWEEP",
        "layer_names": layer_names,
        "trials": {},
        "tf": {},
        "rho": {},
        "solver_audit": {},
        "fisher_keys": {},
    }
    profile_arrays: dict[str, np.ndarray] = {
        "q_damping_validation": q_validation.astype(np.float32),
        "q_fidelity": q_fidelity.astype(np.float32),
    }
    all_assertions = {}
    for trial_index, trial in enumerate(grid["trials"]):
        trial_id = trial["trial_id"]
        assignment_count = int(trial["assignment_probes_per_task"])
        fisher_count = int(trial["fisher_probes_per_task"])
        assignment_indices = task_prefix_indices(assignment_meta, assignment_count)
        fisher_indices = task_prefix_indices(fisher_meta, fisher_count)
        selected_q = q_assignment[assignment_indices].astype(np.float64)
        selected_fisher_scores = fisher_scores[fisher_indices].astype(np.float64)
        fisher = np.mean(np.square(selected_fisher_scores), axis=0)
        fisher_key = f"fisher_{trial_index:02d}"
        profile_arrays[fisher_key] = fisher.astype(np.float64)
        selected_meta = [assignment_meta[index] for index in assignment_indices]
        task_counts: dict[str, int] = {}
        for row in selected_meta:
            task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
        task_weights = np.asarray(
            [1.0 / (15 * task_counts[row["task"]]) for row in selected_meta]
        )
        candidates["trials"][trial_id] = trial
        candidates["tf"][trial_id] = {}
        candidates["rho"][trial_id] = {}
        candidates["solver_audit"][trial_id] = {}
        candidates["fisher_keys"][trial_id] = fisher_key
        for layer_index, layer_name in enumerate(layer_names):
            median = float(np.median(fisher[layer_index]))
            if not np.isfinite(median) or median <= 0:
                raise FloatingPointError(f"invalid Fisher median for {trial_id}/{layer_name}")
            rho = float(trial["epsilon_f"]) * median
            profiles, damped, _ = build_natural_modulation_profiles(
                selected_q[:, layer_index, :],
                fisher[layer_index],
                task_weights,
                rho,
            )
            result = solve_balanced_tf_assignment(
                profiles,
                damped,
                num_experts=num_experts,
                rank=rank,
                restarts=int(trial["solver_restarts"]),
                max_iterations=int(config["assignment_max_iterations"]),
                tolerance=float(config["assignment_tolerance"]),
                seed=int(config["seed"]) + trial_index * 100_003 + layer_index * 4099,
            )
            candidates["tf"][trial_id][layer_name] = result.labels.tolist()
            candidates["rho"][trial_id][layer_name] = rho
            candidates["solver_audit"][trial_id][layer_name] = {
                "objective": result.objective,
                "best_restart": result.best_restart,
                "audits": [
                    {
                        "restart": audit.restart,
                        "seed": audit.seed,
                        "iterations": audit.iterations,
                        "converged": audit.converged,
                        "objective": audit.objective,
                        "objective_history": list(audit.objective_history),
                    }
                    for audit in result.audits
                ],
            }
        all_assertions[trial_id] = {
            "assignment_count_exact": len(assignment_indices) == 15 * assignment_count,
            "fisher_count_exact": len(fisher_indices) == 15 * fisher_count,
            "q_finite": bool(np.isfinite(selected_q).all()),
            "fisher_finite_positive": bool(
                np.isfinite(fisher).all() and np.all(np.median(fisher, axis=1) > 0)
            ),
            "all_layers_solved": len(candidates["tf"][trial_id]) == len(layer_names),
        }
    if len(candidates["trials"]) != 13 or not all(
        all(values.values()) for values in all_assertions.values()
    ):
        raise AssertionError(all_assertions)
    np.savez(output_dir / "h0_profiles.npz", **profile_arrays)
    (output_dir / "h0_assignment_candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "h0_prepare_result.json").write_text(
        json.dumps(
            {
                "run_id": config["run_id"],
                "status": "complete",
                "trial_count": 13,
                "assertions": all_assertions,
                "validation_sample_ids": [row["sample_id"] for row in validation_meta],
                "fidelity_sample_ids": [row["sample_id"] for row in fidelity_meta],
                "official_test_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output_dir / "h0_assignment_candidates.json")


if __name__ == "__main__":
    main()
