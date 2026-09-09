#!/usr/bin/env python3
"""Aggregate one complete M8 ability-evidence run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_commit() -> str:
    value = (
        os.environ.get("BADIT_TF_CODE_COMMIT", "").strip()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("invalid M8 code commit")
    return value


def stats(rows: list[dict]) -> dict:
    keys = (
        "target_deletion_increase",
        "control_deletion_increase",
        "specificity_gap",
        "sufficiency",
        "composition_gain",
        "effective_experts",
        "top1_mass",
    )
    output = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    multi = [row for row in rows if row["multi_expert"]]
    output["multi_expert_fraction"] = len(multi) / len(rows)
    output["multi_expert_composition_gain"] = (
        float(np.mean([row["composition_gain"] for row in multi])) if multi else None
    )
    output["units"] = len(rows)
    output["multi_expert_units"] = len(multi)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    shards = sorted((output / "ability").glob("rank*.jsonl"))
    if len(shards) != int(config["world_size"]):
        raise AssertionError("M8 shard count mismatch")
    rows = [
        json.loads(line)
        for path in shards
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_layers = config.get("layer_indices")
    evaluated_layer_count = (
        len(selected_layers)
        if selected_layers is not None
        else int(config["expected_layer_count"])
    )
    expected = int(config["fidelity_probes_per_task"]) * 15 * evaluated_layer_count
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    assertions = {
        "all_units_retained": len(rows) == expected,
        "fifteen_tasks_present": len(by_task) == 15,
        "selected_layers_complete": {int(row["layer_index"]) for row in rows}
        == set(
            map(
                int,
                selected_layers
                if selected_layers is not None
                else range(int(config["expected_layer_count"])),
            )
        ),
        "target_control_distinct": all(
            row["target_expert"] != row["control_expert"] for row in rows
        ),
        "top4_route_recorded": all(
            len(row["active_experts"]) == int(config["top_k"]) for row in rows
        ),
        "coefficient_renormalization_disabled": all(
            not row["coefficient_renormalization"] for row in rows
        ),
        "official_test_not_loaded": not config.get("official_test_loaded"),
        "downstream_test_not_loaded": not config.get("downstream_test_loaded"),
        "all_values_finite": all(
            np.isfinite(row[key])
            for row in rows
            for key in (
                "full_loss",
                "target_deleted_loss",
                "control_deleted_loss",
                "residual_only_loss",
                "target_only_loss",
                "specificity_gap",
                "sufficiency",
                "composition_gain",
            )
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "model": config["model_name"],
        "m0_seed": config["m0_seed"],
        "status": "complete",
        "code_commit": code_commit(),
        "config_sha256": digest(args.config),
        "trained_checkpoint_sha256": config["initial_bank_checkpoint_sha256"],
        "profile_source_sha256": {
            "profiles": config["m8_profiles_sha256"],
            "metadata": config["m8_profile_metadata_sha256"],
        },
        "protocol": {
            "official_test_loaded": False,
            "coefficient_renormalization": False,
            "intervention_assignment_method": config.get(
                "intervention_assignment_method", "tf"
            ),
            "assignment_random_index": config.get("assignment_random_index"),
            "matched_checkpoint_source_run_id": config.get(
                "matched_checkpoint_source_run_id"
            ),
            "target_rule": "max active coefficient",
            "control_rule": "closest active non-target coefficient",
            "multi_expert_neff_min": config["multi_expert_neff_min"],
            "multi_expert_top1_mass_max": config["multi_expert_top1_mass_max"],
        },
        "metrics": {
            "macro": stats(rows),
            "per_task": {
                task: stats(values) for task, values in sorted(by_task.items())
            },
            "assertions": assertions,
        },
        "artifacts": {"shards": [str(path) for path in shards]},
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"]["macro"], sort_keys=True))


if __name__ == "__main__":
    main()
