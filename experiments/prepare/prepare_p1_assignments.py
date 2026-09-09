#!/usr/bin/env python3
"""Merge P1 shards and solve candidate assignments for every damping value."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from badit_tf.assignment import (
    build_natural_modulation_profiles,
    solve_balanced_tf_assignment,
)


def _merge_role(
    collection: Path,
    role: str,
    expected_ids: list[str],
) -> tuple[np.ndarray, list[dict]]:
    rows = {}
    for metadata_path in sorted(collection.glob("rank*.json")):
        rank = metadata_path.stem
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(collection / f"{rank}.npz")
        key = "fisher_scores" if role == "fisher" else f"q_{role}"
        for item, array in zip(metadata["roles"][role], arrays[key], strict=True):
            if item["sample_id"] in rows:
                raise ValueError(f"duplicate collected ID {item['sample_id']}")
            rows[item["sample_id"]] = (array, item)
    missing = [sample_id for sample_id in expected_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"missing {role} samples: {missing[:8]}")
    ordered = [rows[sample_id] for sample_id in expected_ids]
    return np.stack([item[0] for item in ordered]), [item[1] for item in ordered]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    collection = output_dir / "collection"
    manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
    q_roles = tuple(
        config.get(
            "q_roles",
            ("assignment", "damping_validation", "fidelity"),
        )
    )
    if "assignment" not in q_roles or "damping_validation" not in q_roles:
        raise ValueError("assignment and damping_validation q roles are required")
    role_ids = {
        role: [item["sample_id"] for item in manifest["roles"][role]]
        for role in (*q_roles, "fisher")
    }
    q_assignment, assignment_meta = _merge_role(
        collection, "assignment", role_ids["assignment"]
    )
    q_damping, damping_meta = _merge_role(
        collection, "damping_validation", role_ids["damping_validation"]
    )
    q_fidelity = None
    fidelity_meta = None
    if "fidelity" in q_roles:
        q_fidelity, fidelity_meta = _merge_role(
            collection, "fidelity", role_ids["fidelity"]
        )
    fisher_scores, fisher_meta = _merge_role(
        collection, "fisher", role_ids["fisher"]
    )
    fisher = np.mean(np.square(fisher_scores.astype(np.float64)), axis=0)
    if not np.all(np.isfinite(fisher)) or np.any(fisher < 0):
        raise FloatingPointError("invalid model-Fisher diagonal")

    task_counts = {}
    for item in assignment_meta:
        task_counts[item["task"]] = task_counts.get(item["task"], 0) + 1
    task_weights = np.asarray(
        [1.0 / (15 * task_counts[item["task"]]) for item in assignment_meta]
    )
    num_experts = int(config["num_experts"])
    rank = int(config["rank"])
    primitive_count = num_experts * rank
    layer_names = json.loads(
        sorted(collection.glob("rank*.json"))[0].read_text(encoding="utf-8")
    )["layer_names"]
    contiguous = np.arange(primitive_count, dtype=np.int64) // rank
    rng = np.random.default_rng(int(config["seed"]) + 991)
    random_labels = []
    for _ in range(int(config["random_assignments"])):
        labels = contiguous.copy()
        rng.shuffle(labels)
        random_labels.append(labels.tolist())

    candidates = {
        "schema_version": 1,
        "layer_names": layer_names,
        "contiguous": contiguous.tolist(),
        "random_balanced": random_labels,
        "tf": {},
        "rho": {},
        "solver_audit": {},
    }
    for epsilon in config["damping_epsilons"]:
        epsilon_key = f"{float(epsilon):.4g}"
        candidates["tf"][epsilon_key] = {}
        candidates["rho"][epsilon_key] = {}
        candidates["solver_audit"][epsilon_key] = {}
        for layer_index, layer_name in enumerate(layer_names):
            median = float(np.median(fisher[layer_index]))
            if median <= 0:
                raise ValueError(f"zero Fisher median in {layer_name}")
            rho = float(epsilon) * median
            profiles, damped, _ = build_natural_modulation_profiles(
                q_assignment[:, layer_index, :],
                fisher[layer_index],
                task_weights,
                rho,
            )
            result = solve_balanced_tf_assignment(
                profiles,
                damped,
                num_experts=num_experts,
                rank=rank,
                restarts=int(config["assignment_restarts"]),
                max_iterations=int(config["assignment_max_iterations"]),
                tolerance=float(config["assignment_tolerance"]),
                seed=int(config["seed"]) + layer_index * 4099,
            )
            candidates["tf"][epsilon_key][layer_name] = result.labels.tolist()
            candidates["rho"][epsilon_key][layer_name] = rho
            candidates["solver_audit"][epsilon_key][layer_name] = {
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

    profile_arrays = {
        "q_assignment": q_assignment.astype(np.float32),
        "q_damping_validation": q_damping.astype(np.float32),
        "fisher": fisher.astype(np.float64),
    }
    if q_fidelity is not None:
        profile_arrays["q_fidelity"] = q_fidelity.astype(np.float32)
    np.savez(
        output_dir / "profiles.npz",
        **profile_arrays,
    )
    metadata_roles = {
        "role_ids": role_ids,
        "assignment": assignment_meta,
        "damping_validation": damping_meta,
        "fisher": fisher_meta,
    }
    if fidelity_meta is not None:
        metadata_roles["fidelity"] = fidelity_meta
    (output_dir / "profile_metadata.json").write_text(
        json.dumps(
            metadata_roles,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "assignment_candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "assignment_candidates.json")


if __name__ == "__main__":
    main()
