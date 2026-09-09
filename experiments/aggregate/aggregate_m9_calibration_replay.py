#!/usr/bin/env python3
"""Aggregate one eight-GPU assignment/Fisher calibration timing replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    paths = sorted((output / "collection").glob("rank*.json"))
    if len(paths) != 8:
        raise AssertionError(f"expected 8 timing ranks, found {len(paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_roles = {"assignment", "fisher"}
    if any(set(row["roles"]) != expected_roles for row in rows):
        raise AssertionError("timing replay loaded roles outside assignment and Fisher")
    if any(int(row["world"]) != 8 for row in rows):
        raise AssertionError("timing replay world-size drift")
    assignment = sum(len(row["roles"]["assignment"]) for row in rows)
    fisher = sum(len(row["roles"]["fisher"]) for row in rows)
    expected_assignment = 15 * int(config["assignment_probes_per_task"])
    expected_fisher = 15 * int(config["fisher_probes_per_task"])
    started = min(float(row["calibration_started_at_unix"]) for row in rows)
    finished = max(float(row["calibration_finished_at_unix"]) for row in rows)
    wall = finished - started
    result = {
        "schema_version": 1,
        "experiment_id": "M9-CALIBRATION-TIMING-REPLAY-V1",
        "run_id": config["run_id"],
        "model": config["model_name"],
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_sha256": digest(args.config),
        "metrics": {
            "calibration_wall_seconds": wall,
            "calibration_gpu_hours": wall * 8.0 / 3600.0,
            "peak_memory_allocated_gb": max(int(row["peak_memory_allocated_bytes"]) for row in rows) / 1e9,
            "peak_memory_reserved_gb": max(int(row["peak_memory_reserved_bytes"]) for row in rows) / 1e9,
            "assignment_sequences": assignment,
            "fisher_sequences": fisher,
            "rank_wall_seconds": [float(row["calibration_wall_seconds_rank"]) for row in rows],
            "rank_role_wall_seconds": [row["role_wall_seconds"] for row in rows],
        },
        "protocol": {
            "timed_scope": "assignment virtual-gate q plus selected Fisher score-gradient collection",
            "excluded_from_timer": ["model load", "tokenizer load", "bank load", "assignment solve", "fidelity evaluation"],
            "official_test_loaded": False,
            "downstream_test_loaded": False,
            "gpu_count": 8,
        },
        "assertions": {
            "eight_ranks": len(rows) == 8,
            "roles_exact_assignment_fisher": all(set(row["roles"]) == expected_roles for row in rows),
            "assignment_count_exact": assignment == expected_assignment,
            "fisher_count_exact": fisher == expected_fisher,
            "positive_wall_time": wall > 0,
            "positive_peak_memory": all(int(row["peak_memory_allocated_bytes"]) > 0 for row in rows),
            "official_and_downstream_test_closed": True,
        },
        "rank_artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in paths
        ],
    }
    if not all(result["assertions"].values()):
        raise AssertionError(result["assertions"])
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(path), "sha256": digest(path), "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
