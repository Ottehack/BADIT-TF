#!/usr/bin/env python3
"""Aggregate H0 validation/fidelity rows and lock one calibration config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for item in sorted(path.glob("rank*.jsonl"))
        for line in item.read_text().splitlines()
        if line.strip()
    ]


def correlation(rows: list[dict[str, Any]]) -> float:
    return float(
        spearmanr(
            [row["predicted_regret"] for row in rows],
            [row["observed_gap"] for row in rows],
        ).statistic
    )


def task_bootstrap(
    rows: list[dict[str, Any]], *, seed: int, replicates: int
) -> tuple[list[float], list[float]]:
    tasks = sorted({row["task"] for row in rows})
    by_task = {task: [row for row in rows if row["task"] == task] for task in tasks}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        sample_rows = [row for task in sampled for row in by_task[str(task)]]
        value = correlation(sample_rows)
        if np.isfinite(value):
            values.append(value)
    return values, [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def rank_accuracy(rows: list[dict[str, Any]]) -> tuple[float, int]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], int(row["layer_index"]))].append(row)
    correct = 0
    total = 0
    for group in groups.values():
        for left, right in combinations(group, 2):
            predicted = left["predicted_regret"] - right["predicted_regret"]
            observed = left["observed_gap"] - right["observed_gap"]
            if predicted == 0 or observed == 0:
                continue
            total += 1
            correct += int(np.sign(predicted) == np.sign(observed))
    return (float(correct / total) if total else float("nan")), total


def top_bottom(rows: list[dict[str, Any]]) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: float(row["predicted_regret"]))
    size = max(1, int(math.ceil(0.1 * len(ordered))))
    bottom = np.asarray([row["observed_gap"] for row in ordered[:size]])
    top = np.asarray([row["observed_gap"] for row in ordered[-size:]])
    return {
        "decile_size": size,
        "bottom_mean": float(bottom.mean()),
        "top_mean": float(top.mean()),
        "top_minus_bottom": float(top.mean() - bottom.mean()),
    }


def distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def adjusted_rand(left: list[int], right: list[int]) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    n = len(left_array)
    if n != len(right_array) or n < 2:
        raise ValueError("invalid ARI vectors")
    left_values = np.unique(left_array)
    right_values = np.unique(right_array)
    table = np.asarray(
        [
            [np.sum((left_array == a) & (right_array == b)) for b in right_values]
            for a in left_values
        ],
        dtype=np.float64,
    )
    choose2 = lambda x: x * (x - 1.0) / 2.0
    index = float(choose2(table).sum())
    left_sum = float(choose2(table.sum(axis=1)).sum())
    right_sum = float(choose2(table.sum(axis=0)).sum())
    total = choose2(float(n))
    expected = left_sum * right_sum / total
    maximum = 0.5 * (left_sum + right_sum)
    return 1.0 if maximum == expected else float((index - expected) / (maximum - expected))


def solver_metrics(audit: dict[str, Any]) -> dict[str, float]:
    layers = list(audit.values())
    best_iterations = []
    converged = []
    objectives = []
    spreads = []
    for layer in layers:
        audits = layer["audits"]
        best = next(item for item in audits if item["restart"] == layer["best_restart"])
        best_iterations.append(best["iterations"])
        converged.append(best["converged"])
        objectives.append(layer["objective"])
        values = np.asarray([item["objective"] for item in audits], dtype=np.float64)
        spreads.append(float(values.std() / max(abs(values.mean()), 1e-12)))
    return {
        "iterations_mean": float(np.mean(best_iterations)),
        "convergence_rate": float(np.mean(converged)),
        "best_regret_mean": float(np.mean(objectives)),
        "restart_spread_cv_mean": float(np.mean(spreads)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--grid", type=Path, default=Path("experiments/configs/h0_calibration_grid.json")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    grid = json.loads(args.grid.read_text())
    output_dir = Path(config["output_dir"])
    candidates_path = output_dir / "h0_assignment_candidates.json"
    candidates = json.loads(candidates_path.read_text())
    validation_rows = load_rows(output_dir / "h0_finite" / "damping_validation")
    fidelity_rows = load_rows(output_dir / "h0_finite" / "fidelity")
    trial_ids = sorted(candidates["trials"])
    expected_validation = 15 * 8 * len(config["pilot_layer_indices"]) * 13
    expected_fidelity = 15 * 16 * len(config["pilot_layer_indices"]) * 13
    if len(validation_rows) != expected_validation or len(fidelity_rows) != expected_fidelity:
        raise AssertionError(
            f"row count mismatch: validation={len(validation_rows)}/{expected_validation}, "
            f"fidelity={len(fidelity_rows)}/{expected_fidelity}"
        )
    noise = [
        row["repeat_noise_abs"]
        for row in validation_rows
        if row["trial_id"] == "H0-00" and row["repeat_noise_abs"] is not None
    ]
    noise_stats = distribution(noise)
    metrics = []
    bootstrap_payload = {}
    for trial_index, trial_id in enumerate(trial_ids):
        validation = [row for row in validation_rows if row["trial_id"] == trial_id]
        fidelity = [row for row in fidelity_rows if row["trial_id"] == trial_id]
        bootstrap, ci = task_bootstrap(
            validation,
            seed=int(config["seed"]) + 77_003 + trial_index,
            replicates=int(config["bootstrap_replicates"]),
        )
        bootstrap_payload[trial_id] = bootstrap
        accuracy, pairs = rank_accuracy(validation)
        fidelity_accuracy, fidelity_pairs = rank_accuracy(fidelity)
        observed_abs = np.abs([row["observed_gap"] for row in validation])
        trial = candidates["trials"][trial_id]
        metrics.append(
            {
                "trial_id": trial_id,
                "config": trial,
                "validation_spearman": correlation(validation),
                "validation_task_bootstrap_95ci": ci,
                "validation_rank_accuracy": accuracy,
                "validation_rank_accuracy_pairs": pairs,
                "validation_top_bottom": top_bottom(validation),
                "validation_absolute_observed_gap": distribution(observed_abs),
                "eligible_above_repeat_noise": float(np.median(observed_abs))
                > noise_stats["p95"],
                "fidelity_spearman_report_only": correlation(fidelity),
                "fidelity_rank_accuracy_report_only": fidelity_accuracy,
                "fidelity_rank_accuracy_pairs": fidelity_pairs,
                "solver": solver_metrics(candidates["solver_audit"][trial_id]),
            }
        )
    eligible = [item for item in metrics if item["eligible_above_repeat_noise"]]
    if not eligible:
        raise AssertionError("all H0 trials are below repeat noise")
    best_lower = max(item["validation_task_bootstrap_95ci"][0] for item in eligible)
    tied = [
        item
        for item in eligible
        if item["validation_task_bootstrap_95ci"][0] >= best_lower - 0.002
    ]
    selected = max(
        tied,
        key=lambda item: (
            item["validation_rank_accuracy"],
            -(
                int(item["config"]["assignment_probes_per_task"])
                + int(item["config"]["fisher_probes_per_task"])
            ),
            -int(item["config"]["solver_restarts"]),
        ),
    )
    selected_id = selected["trial_id"]
    selected_labels = candidates["tf"][selected_id]
    for item in metrics:
        item["stability_ari_to_selected"] = float(
            np.mean(
                [
                    adjusted_rand(
                        candidates["tf"][item["trial_id"]][layer_name],
                        selected_labels[layer_name],
                    )
                    for layer_name in candidates["layer_names"]
                ]
            )
        )
    assertions = {
        "all_13_trials_reported": len(metrics) == 13,
        "validation_rows_complete": len(validation_rows) == expected_validation,
        "fidelity_rows_complete": len(fidelity_rows) == expected_fidelity,
        "all_metrics_finite": all(
            np.isfinite(item["validation_spearman"])
            and np.isfinite(item["fidelity_spearman_report_only"])
            for item in metrics
        ),
        "selection_uses_validation_only": True,
        "fidelity_report_only": True,
        "downstream_and_official_test_not_loaded": True,
    }
    lock = {
        "schema_version": 1,
        "experiment_id": "H0-A2-CALIBRATION-SWEEP",
        "selected_trial_id": selected_id,
        "selected_config": selected["config"],
        "selection_rule": grid["validation_selection_rule"],
        "selection_metric": {
            "validation_spearman": selected["validation_spearman"],
            "task_bootstrap_95ci": selected["validation_task_bootstrap_95ci"],
            "rank_accuracy": selected["validation_rank_accuracy"],
        },
        "config_sha256": file_sha(args.config),
        "grid_sha256": grid["grid_sha256"],
        "candidates_sha256": file_sha(candidates_path),
        "lock_timestamp": datetime.now(timezone.utc).isoformat(),
        "fidelity_used_for_selection": False,
        "downstream_test_used": False,
    }
    lock_path = output_dir / "h0_selection_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    (output_dir / "h0_bootstrap_samples.json").write_text(
        json.dumps(bootstrap_payload, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "run_id": config["run_id"],
        "experiment_id": "H0-A2-CALIBRATION-SWEEP",
        "status": "complete" if all(assertions.values()) else "failed",
        "decision": "H0_CALIBRATION_LOCKED" if all(assertions.values()) else "H0_FAILED",
        "selected_trial_id": selected_id,
        "selected_config": selected["config"],
        "metrics": {
            "assertions": assertions,
            "repeat_forward_noise": noise_stats,
            "trials": metrics,
        },
        "artifacts": {
            "selection_lock": str(lock_path),
            "bootstrap_samples": str(output_dir / "h0_bootstrap_samples.json"),
            "assignment_candidates": str(candidates_path),
        },
        "protocol": {
            "validation_role": "damping_validation",
            "fidelity_role": "report_only",
            "official_test_loaded": False,
            "downstream_test_loaded": False,
            "trial_count": 13,
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
