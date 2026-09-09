#!/usr/bin/env python3
"""Apply the frozen P3-R8 retention/plasticity selection rule."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from badit_tf.training import canonical_json_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-lock", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-drift", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.grid_lock.read_text(encoding="utf-8"))
    rule = lock["selection_rule"]
    rows = []
    stage0_hashes = []
    for cell in lock["cells"]:
        run_id = cell["run_id"]
        run_root = args.result_root / run_id
        result_path = run_root / "result.json"
        stage0_path = run_root / "stages" / "task_0.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
        matrix = result["metrics"]["evaluation_matrix_a_t_s"]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise RuntimeError(f"{run_id}: expected a 2x2 matrix")
        stage_assertions_passed = bool(
            result["status"] == "complete"
            and all(result["metrics"]["assertions"].values())
        )
        task0_diagonal = float(matrix[0][0])
        retained_task0 = float(matrix[1][0])
        task1_diagonal = float(matrix[1][1])
        retention_fraction = (
            retained_task0 / task0_diagonal if task0_diagonal > 0 else 0.0
        )
        stage0_hash = stage0["hashes"]["checkpoint"]["model_state_sha256"]
        stage0_hashes.append(stage0_hash)
        rows.append(
            {
                **cell,
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "stage0_result_sha256": file_sha256(stage0_path),
                "stage0_model_state_sha256": stage0_hash,
                "stage_assertions_passed": stage_assertions_passed,
                "task0_diagonal": task0_diagonal,
                "task0_after_task1": retained_task0,
                "task0_retention_fraction": retention_fraction,
                "task1_diagonal": task1_diagonal,
                "final_relative_anchor_drift": float(
                    result["metrics"]["final_relative_anchor_drift"]["task_1"]
                ),
                "drift_source": "original_stage_result",
            }
        )
    controls = [
        row
        for row in rows
        if float(row["coefficient"]) == float(rule["control_coefficient"])
    ]
    if len(controls) != 1:
        raise RuntimeError("expected exactly one lambda-0 control")
    control_drift_sha256 = None
    if args.control_drift is not None:
        drift = json.loads(args.control_drift.read_text(encoding="utf-8"))
        controls[0]["final_relative_anchor_drift"] = float(
            drift["groups"]["all_trainable"]["relative_squared_drift"]
        )
        controls[0]["drift_source"] = "checkpoint_derived_amendment"
        controls[0]["drift_artifact"] = str(args.control_drift)
        control_drift_sha256 = file_sha256(args.control_drift)
    control_task1 = float(controls[0]["task1_diagonal"])
    if control_task1 <= 0:
        raise RuntimeError("lambda-0 task-1 diagonal must be positive")
    for row in rows:
        row["task1_plasticity_fraction_of_control"] = (
            float(row["task1_diagonal"]) / control_task1
        )
        row["selection_gate_passed"] = bool(
            row["stage_assertions_passed"]
            and row["task0_diagonal"] > 0
            and row["task1_diagonal"] > 0
            and row["task0_retention_fraction"]
            >= float(rule["minimum_task0_retention_fraction_after_task1"])
            and row["task1_plasticity_fraction_of_control"]
            >= float(rule["minimum_task1_plasticity_fraction_of_control"])
        )
    passing = sorted(
        (row for row in rows if row["selection_gate_passed"]),
        key=lambda row: float(row["coefficient"]),
    )
    selected = passing[0] if passing else None
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid_lock_path": str(args.grid_lock),
        "grid_lock_file_sha256": file_sha256(args.grid_lock),
        "grid_lock_content_sha256": lock["lock_content_sha256"],
        "selection_rule": rule,
        "control_drift_artifact_sha256": control_drift_sha256,
        "stage0_model_state_identical_across_grid": len(set(stage0_hashes)) == 1,
        "control_task1_diagonal": control_task1,
        "cells": rows,
        "selected_coefficient": (
            float(selected["coefficient"]) if selected is not None else None
        ),
        "selected_run_id": selected["run_id"] if selected is not None else None,
        "decision": (
            "LOCK_FOR_THIRD_TASK_CONFIRMATION"
            if selected is not None
            else "NO_COEFFICIENT_PASSED_STABILITY_PLASTICITY_GATE"
        ),
        "official_test_loaded": False,
    }
    if not payload["stage0_model_state_identical_across_grid"]:
        raise RuntimeError("task-0 model state differs across anchor grid")
    payload["selection_lock_content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite selection lock {args.output}")
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
