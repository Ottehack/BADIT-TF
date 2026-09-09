#!/usr/bin/env python3
"""Freeze H0 calibration roles and the downstream tuning pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from badit_tf.splits import stable_sample_id


ROLE_SIZES = {
    "assignment": 32,
    "fisher": 32,
    "damping_validation": 8,
    "fidelity": 16,
}
EXTENSION_SIZE = 16


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        row["sample_id"]
        for rows in manifest.get("roles", {}).values()
        for row in rows
    }


def task_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"H0:{seed}:{task}".encode()).digest()[:8]
    return int.from_bytes(digest, "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("experiments/configs/splits/default_seed1.json"),
    )
    parser.add_argument(
        "--warmstart-manifest",
        type=Path,
        default=Path(
            "experiments/configs/splits/p1_r2_warmstart_seed23001_order.json"
        ),
    )
    parser.add_argument(
        "--confirmation-v2",
        type=Path,
        default=Path(
            "experiments/configs/splits/confirmation_fidelity_v2_seed1.json"
        ),
    )
    parser.add_argument(
        "--confirmation-v3",
        type=Path,
        default=Path(
            "experiments/configs/splits/confirmation_fidelity_v3_seed1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/configs/splits/h0_calibration_seed24001.json"),
    )
    parser.add_argument(
        "--tuning-output",
        type=Path,
        default=Path("experiments/configs/splits/h1_tuning_seed1.json"),
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=Path("experiments/configs/h0_calibration_grid.json"),
    )
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--selected-warmstart-steps", type=int, default=1)
    parser.add_argument("--warmstart-world-size", type=int, default=8)
    args = parser.parse_args()

    base = json.loads(args.base_manifest.read_text())
    warm = json.loads(args.warmstart_manifest.read_text())
    v2 = json.loads(args.confirmation_v2.read_text())
    v3 = json.loads(args.confirmation_v3.read_text())
    frozen_role_ids = role_ids(base)
    warm_consumed_count = (
        args.selected_warmstart_steps * args.warmstart_world_size
    )
    warm_consumed_ids = {
        row["sample_id"] for row in warm["train_order"][:warm_consumed_count]
    }
    confirmation_ids = role_ids(v2) | role_ids(v3)
    excluded = frozen_role_ids | warm_consumed_ids | confirmation_ids

    roles = {
        role: list(base["roles"][role])
        for role in ("tune_validation", *ROLE_SIZES)
    }
    tasks: dict[str, Any] = {}
    h0_ids: set[str] = set()
    extension_ids: set[str] = set()
    future_tuning_ids: set[str] = set()

    for task, task_meta in sorted(base["tasks"].items()):
        train_path = Path(task_meta["train_file"])
        instances = json.loads(train_path.read_text())["Instances"]
        metadata_by_id = {}
        for index, instance in enumerate(instances):
            sample_id = stable_sample_id(task, index, instance)
            metadata_by_id[sample_id] = {
                "sample_id": sample_id,
                "task": task,
                "source_file": str(train_path),
                "instance_index": index,
            }
        eligible = [
            sample_id
            for sample_id in task_meta["tune_train_ids"]
            if sample_id not in excluded
        ]
        rng = np.random.default_rng(task_seed(args.seed, task))
        eligible = [eligible[index] for index in rng.permutation(len(eligible))]
        required = EXTENSION_SIZE
        if len(eligible) < required:
            raise ValueError(f"{task} has only {len(eligible)} H0-eligible rows")
        task_roles = {
            role: list(task_meta["roles"][role])
            for role in ("tune_validation", *ROLE_SIZES)
        }
        chosen = eligible[:EXTENSION_SIZE]
        rows = [metadata_by_id[sample_id] for sample_id in chosen]
        for role in ("assignment", "fisher"):
            task_roles[role].extend(chosen)
            roles[role].extend(rows)
        extension_ids.update(chosen)
        h0_ids.update(
            sample_id
            for role in ROLE_SIZES
            for sample_id in task_roles[role]
        )
        tune_train_ids = [
            sample_id
            for sample_id in task_meta["tune_train_ids"]
            if sample_id not in h0_ids and sample_id not in confirmation_ids
        ]
        future_tuning_ids.update(tune_train_ids)
        tasks[task] = {
            **{key: value for key, value in task_meta.items() if key != "roles"},
            "roles": task_roles,
            "tune_train_ids": tune_train_ids,
        }

    role_sets = {
        name: {row["sample_id"] for row in rows}
        for name, rows in roles.items()
    }
    role_names = list(role_sets)
    assignment_fisher_overlap = role_sets["assignment"].intersection(
        role_sets["fisher"]
    )
    other_pairs_disjoint = all(
        not role_sets[left].intersection(role_sets[right])
        for index, left in enumerate(role_names)
        for right in role_names[index + 1 :]
        if {left, right} != {"assignment", "fisher"}
    )
    active_probe_counts = (
        (16, 16),
        (4, 16),
        (8, 16),
        (32, 16),
        (16, 4),
        (16, 8),
        (16, 32),
    )
    every_trial_disjoint = True
    for task_meta in tasks.values():
        for assignment_count, fisher_count in active_probe_counts:
            active = {
                "assignment": set(task_meta["roles"]["assignment"][:assignment_count]),
                "fisher": set(task_meta["roles"]["fisher"][:fisher_count]),
                "damping_validation": set(task_meta["roles"]["damping_validation"]),
                "fidelity": set(task_meta["roles"]["fidelity"]),
            }
            names = list(active)
            every_trial_disjoint &= all(
                not active[left].intersection(active[right])
                for index, left in enumerate(names)
                for right in names[index + 1 :]
            )
    assertions = {
        "fifteen_tasks": len(tasks) == 15,
        "role_counts_exact": all(
            len(roles[role]) == count * 15 for role, count in ROLE_SIZES.items()
        ),
        "non_probe_extension_roles_pairwise_disjoint": other_pairs_disjoint,
        "assignment_fisher_overlap_is_exact_shared_extension": (
            assignment_fisher_overlap == extension_ids
            and len(extension_ids) == 16 * 15
        ),
        "every_one_factor_trial_has_disjoint_active_roles": every_trial_disjoint,
        "extensions_disjoint_from_frozen_p1_roles": not extension_ids.intersection(
            frozen_role_ids
        ),
        "warmstart_consumed_count_exact": len(warm_consumed_ids)
        == warm_consumed_count,
        "extensions_disjoint_from_consumed_warmstart": not extension_ids.intersection(
            warm_consumed_ids
        ),
        "h0_disjoint_from_confirmation_v2_v3": not h0_ids.intersection(
            confirmation_ids
        ),
        "future_tune_train_disjoint_from_h0": not future_tuning_ids.intersection(h0_ids),
        "future_tune_train_disjoint_from_confirmation": not future_tuning_ids.intersection(confirmation_ids),
        "official_test_not_loaded": True,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)

    manifest = {
        "schema_version": 2,
        "split_name": "h0_calibration_seed24001",
        "seed": args.seed,
        "source_root": base["source_root"],
        "source_base_manifest": str(args.base_manifest),
        "source_base_manifest_sha256": file_sha(args.base_manifest),
        "excluded_manifests": {
            "warmstart": {
                "path": str(args.warmstart_manifest),
                "sha256": file_sha(args.warmstart_manifest),
                "selected_steps": args.selected_warmstart_steps,
                "world_size": args.warmstart_world_size,
                "consumed_count": warm_consumed_count,
                "consumed_ids_sha256": canonical_sha(sorted(warm_consumed_ids)),
            },
            "confirmation_v2": {"path": str(args.confirmation_v2), "sha256": file_sha(args.confirmation_v2)},
            "confirmation_v3": {"path": str(args.confirmation_v3), "sha256": file_sha(args.confirmation_v3)},
        },
        "counts": {**ROLE_SIZES, "tune_validation": len(roles["tune_validation"])},
        "reuse_contract": {
            "assignment_base_per_task": 16,
            "fisher_base_per_task": 16,
            "damping_validation_base_per_task": 8,
            "fidelity_base_per_task": 16,
            "assignment_extension_per_task": 16,
            "fisher_extension_per_task": 16,
            "shared_extension_across_one_factor_trials": True,
            "active_role_rule": "The shared extension is assignment-only in the assignment-32 trial and Fisher-only in the Fisher-32 trial; no trial uses it for both roles.",
            "reason": "H0 reuses A2/P1 roles and adds nested 32-probe prefixes without exceeding small-task capacity",
        },
        "roles": roles,
        "tasks": tasks,
        "assertions": assertions,
        "sample_id_hashes": {
            role: canonical_sha(sorted(ids)) for role, ids in role_sets.items()
        },
        "content_access_rule": "calibration roles only; no downstream or official test",
    }
    manifest["manifest_sha256"] = canonical_sha(manifest)

    tuning_manifest = {
        **manifest,
        "split_name": "h1_tuning_seed1",
        "purpose": "H1-H4 tune_train/tune_validation with H0 and confirmations excluded",
    }
    tuning_manifest.pop("manifest_sha256", None)
    tuning_manifest["manifest_sha256"] = canonical_sha(tuning_manifest)

    default = {
        "assignment_probes_per_task": 16,
        "fisher_probes_per_task": 16,
        "epsilon_f": 0.1,
        "solver_restarts": 5,
    }
    trials = [{"trial_id": "H0-00", **default, "changed_axis": "default"}]
    axes = (
        ("assignment_probes_per_task", [4, 8, 32]),
        ("fisher_probes_per_task", [4, 8, 32]),
        ("epsilon_f", [0.0001, 0.001, 0.01]),
        ("solver_restarts", [1, 3, 10]),
    )
    trial_index = 1
    for axis, values in axes:
        for value in values:
            trials.append(
                {
                    "trial_id": f"H0-{trial_index:02d}",
                    **default,
                    axis: value,
                    "changed_axis": axis,
                }
            )
            trial_index += 1
    grid = {
        "schema_version": 1,
        "experiment_id": "H0-A2-CALIBRATION-SWEEP",
        "status": "locked_before_profile",
        "trial_count": len(trials),
        "trials": trials,
        "validation_selection_rule": (
            "Require all mechanism/solver/evaluator assertions; maximize task-bootstrap "
            "95% CI lower bound of validation Spearman. Lower bounds within 0.002 tie; "
            "then maximize rank accuracy, then minimize assignment+Fisher probes, then "
            "minimize restarts. Fidelity is report-only and downstream/official test is forbidden."
        ),
        "locked_eta": 0.0001,
        "validation_role": "damping_validation",
        "report_only_role": "fidelity",
        "downstream_test_used": False,
        "calibration_manifest": str(args.output),
        "calibration_manifest_sha256": manifest["manifest_sha256"],
    }
    grid["grid_sha256"] = canonical_sha(grid)

    for path, payload in (
        (args.output, manifest),
        (args.tuning_output, tuning_manifest),
        (args.grid_output, grid),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "calibration_manifest": str(args.output),
        "calibration_manifest_sha256": manifest["manifest_sha256"],
        "tuning_manifest": str(args.tuning_output),
        "tuning_manifest_sha256": tuning_manifest["manifest_sha256"],
        "grid": str(args.grid_output),
        "grid_sha256": grid["grid_sha256"],
        "assertions": assertions,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
