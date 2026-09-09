#!/usr/bin/env python3
"""Apply the frozen P3-R9 functional rehearsal selection rule."""

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
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite selection lock {args.output}")
    lock = json.loads(args.grid_lock.read_text(encoding="utf-8"))
    rule = lock["selection_rule"]
    rows = []
    stage0_hashes = []
    for cell in lock["cells"]:
        run_root = args.result_root / cell["run_id"]
        result_path = run_root / "result.json"
        stage0_path = run_root / "stages" / "task_0.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
        matrix = result["metrics"]["evaluation_matrix_a_t_s"]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise RuntimeError(f"{cell['run_id']}: expected a 2x2 matrix")
        task0_diagonal = float(matrix[0][0])
        task0_retained = float(matrix[1][0])
        task1_diagonal = float(matrix[1][1])
        stage0_hash = stage0["hashes"]["checkpoint"]["model_state_sha256"]
        stage0_hashes.append(stage0_hash)
        rows.append(
            {
                **cell,
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "stage0_result_sha256": file_sha256(stage0_path),
                "stage0_model_state_sha256": stage0_hash,
                "stage_assertions_passed": bool(
                    result["status"] == "complete"
                    and all(result["metrics"]["assertions"].values())
                ),
                "task0_diagonal": task0_diagonal,
                "task0_after_task1": task0_retained,
                "task0_retention_fraction": (
                    task0_retained / task0_diagonal
                    if task0_diagonal > 0
                    else 0.0
                ),
                "task1_diagonal": task1_diagonal,
                "task1_rehearsal_record_count": int(
                    result["metrics"]["rehearsal_record_count"]["task_1"]
                ),
                "task1_rehearsal_unique_sample_count": int(
                    result["metrics"]["rehearsal_unique_sample_count"]["task_1"]
                ),
            }
        )
    controls = [row for row in rows if float(row["coefficient"]) == 0.0]
    if len(controls) != 1:
        raise RuntimeError("expected exactly one rehearsal coefficient-0 control")
    control = controls[0]
    if control["task1_diagonal"] <= 0:
        raise RuntimeError("control task-1 diagonal must be positive")
    for row in rows:
        row["task0_improvement_over_control"] = (
            row["task0_after_task1"] - control["task0_after_task1"]
        )
        row["task1_plasticity_fraction_of_control"] = (
            row["task1_diagonal"] / control["task1_diagonal"]
        )
        is_control = float(row["coefficient"]) == 0.0
        row["selection_gate_passed"] = bool(
            not is_control
            and row["stage_assertions_passed"]
            and row["task0_diagonal"] > 0
            and row["task1_diagonal"] > 0
            and row["task0_retention_fraction"]
            >= float(rule["minimum_task0_retention_fraction_after_task1"])
            and row["task0_improvement_over_control"]
            >= float(rule["minimum_task0_absolute_improvement_over_control"])
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
        "stage0_model_state_identical_across_grid": len(set(stage0_hashes)) == 1,
        "control_task0_after_task1": control["task0_after_task1"],
        "control_task1_diagonal": control["task1_diagonal"],
        "cells": rows,
        "selected_coefficient": (
            float(selected["coefficient"]) if selected is not None else None
        ),
        "selected_run_id": selected["run_id"] if selected is not None else None,
        "decision": (
            "LOCK_FOR_THIRD_TASK_CONFIRMATION"
            if selected is not None
            else "FUNCTIONAL_REHEARSAL_SCREEN_FAILED"
        ),
        "official_test_loaded": False,
    }
    if not payload["stage0_model_state_identical_across_grid"]:
        raise RuntimeError("task-0 model state differs across rehearsal grid")
    payload["selection_lock_content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
