#!/usr/bin/env python3
"""Aggregate one trained-checkpoint M3 deployment-scale run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from badit_tf.deployment_scale import validate_deployment_etas


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_code_commit() -> str:
    """Resolve provenance without importing training-only sklearn dependencies."""
    commit = os.environ.get("BADIT_TF_CODE_COMMIT", "").strip()
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise RuntimeError(f"invalid code commit: {commit!r}")
    return commit.lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    etas = validate_deployment_etas(config["deployment_etas"])
    output = Path(config["output_dir"])
    source_output = Path(config.get("source_output_dir", config["output_dir"]))
    records = []
    paths = sorted((source_output / "deployment").glob("rank*.jsonl"))
    if len(paths) != int(config["world_size"]):
        raise ValueError("M3 deployment rank shard count mismatch")
    for path in paths:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    tasks = sorted({row["task"] for row in records})
    layer_names = sorted({row["layer_name"] for row in records})
    expected_units = len(tasks) * int(config["fidelity_probes_per_task"]) * len(layer_names)
    metrics = {}
    for eta in etas:
        rows = [row for row in records if float(row["eta"]) == eta]
        if len(rows) != expected_units:
            raise AssertionError(f"M3 eta={eta} has {len(rows)} not {expected_units} units")
        correlation = spearmanr(
            [row["predicted_delta_loss"] for row in rows],
            [row["observed_delta_loss"] for row in rows],
        ).statistic
        metrics[str(eta)] = {
            "spearman": float(correlation),
            "predicted_delta_loss_mean": float(np.mean([row["predicted_delta_loss"] for row in rows])),
            "observed_delta_loss_mean": float(np.mean([row["observed_delta_loss"] for row in rows])),
            "relative_error_median": float(np.median([row["relative_error"] for row in rows])),
            "relative_error_p95": float(np.quantile([row["relative_error"] for row in rows], 0.95)),
            "max_abs_scaled_displacement": float(np.max([row["max_abs_scaled_displacement"] for row in rows])),
            "units": len(rows),
        }
    noise = [row["repeat_noise_abs"] for row in records]
    checkpoint_hashes = sorted({row["trained_checkpoint_sha256"] for row in records})
    analysis_scope = str(config.get("analysis_scope"))
    assertions = {
        "frozen_eta_grid_complete": set(metrics) == {str(eta) for eta in etas},
        "all_eta_cover_all_units": all(row["units"] == expected_units for row in metrics.values()),
        "all_routes_exact_top4": all(row["active_experts"] == int(config["top_k"]) for row in records),
        "single_exact_trained_checkpoint": checkpoint_hashes == [config["initial_bank_checkpoint_sha256"]],
        "all_metrics_finite": all(np.isfinite(row[key]) for row in metrics.values() for key in ("spearman", "predicted_delta_loss_mean", "observed_delta_loss_mean", "relative_error_median", "relative_error_p95")),
        "official_test_not_loaded": not bool(config.get("official_test_loaded")),
        "downstream_test_not_loaded": not bool(config.get("downstream_test_loaded")),
        "analysis_scope_declared": analysis_scope in {"smoke_not_table_result", "formal_table"},
        "formal_uses_full_frozen_probe_count": (
            analysis_scope != "formal_table"
            or int(config["fidelity_probes_per_task"])
            == int(config["formal_fidelity_probes_per_task"])
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"M3 terminal assertions failed: {assertions}")
    result = {
        "experiment_id": config["experiment_id"], "run_id": config["run_id"],
        "todo_id": "M3", "status": "complete", "model": config["model_name"],
        "setting": config["setting"], "seed": config["seed"],
        "git_commit": resolve_code_commit(), "config_sha256": sha256(args.config),
        "split_manifest_sha256": config["split_manifest_sha256"],
        "trained_checkpoint_sha256": checkpoint_hashes[0],
        "protocol": {
            "evaluation_role": "frozen_fidelity", "official_test_loaded": False,
            "downstream_test_loaded": False, "analysis_scope": analysis_scope,
            "eta_grid": list(etas),
        },
        "metrics": {
            "by_eta": metrics, "tasks": tasks, "layers": len(layer_names),
            "units_per_eta": expected_units,
            "repeated_forward_noise_abs": {"median": float(np.median(noise)), "p95": float(np.quantile(noise, 0.95)), "max": float(np.max(noise))},
            "assertions": assertions,
        },
        "artifacts": {"profiles": str(source_output / "m3_profiles.npz"), "records": str(source_output / "deployment")},
    }
    if source_output != output:
        result["aggregation_recovery"] = {
            "source_output_dir": str(source_output),
            "source_run_id": config["aggregation_recovery_of"],
            "model_forward_repeated": False,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
