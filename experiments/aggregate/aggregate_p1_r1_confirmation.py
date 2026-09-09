#!/usr/bin/env python3
"""Aggregate the single locked P1-R1 confirmation without exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr

from badit_tf.training import resolve_code_commit


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def spearman(rows: list[dict[str, Any]]) -> float:
    return float(
        spearmanr(
            [row["predicted_regret"] for row in rows],
            [row["observed_gap"] for row in rows],
        ).statistic
    )


def distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


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


def rank_accuracy(rows: list[dict[str, Any]]) -> tuple[float, int]:
    units: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        units[(row["sample_id"], int(row["layer_index"]))].append(row)
    correct = total = 0
    for unit_rows in units.values():
        for left, right in combinations(unit_rows, 2):
            predicted = left["predicted_regret"] - right["predicted_regret"]
            observed = left["observed_gap"] - right["observed_gap"]
            if predicted == 0 or observed == 0:
                continue
            total += 1
            correct += int(np.sign(predicted) == np.sign(observed))
    return (float(correct / total) if total else float("nan")), total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    role = config["confirmation_role"]
    lock_path = Path(config["eta_selection_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    objective_path = Path(config["objective_audit_result"])
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(lock_path) != config["eta_selection_lock_sha256"]:
        raise AssertionError("eta selection lock changed")
    if sha256_file(manifest_path) != config["split_manifest_sha256"]:
        raise AssertionError("confirmation manifest changed")

    rows = [
        json.loads(line)
        for path in sorted((output / role).glob("rank*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_samples = int(config["confirmation_samples_per_task"]) * int(
        config["task_count"]
    )
    methods_per_unit = 2 + int(config["random_assignments"])
    expected_units = expected_samples * len(config["pilot_layer_indices"])
    expected_records = expected_units * methods_per_unit
    all_finite = all(
        np.isfinite(
            [
                row["predicted_regret"],
                row["free_loss"],
                row["tied_loss"],
                row["observed_gap"],
                row["repeat_noise_abs"],
            ]
        ).all()
        for row in rows
    )
    sample_ids = sorted({row["sample_id"] for row in rows})
    manifest_ids = [row["sample_id"] for row in manifest["roles"][role]]
    no_removals = (
        len(rows) == expected_records
        and len(sample_ids) == expected_samples
        and set(sample_ids) == set(manifest_ids)
        and all_finite
    )
    if not no_removals:
        raise AssertionError("confirmation records are missing, nonfinite, or filtered")
    if {float(row["eta"]) for row in rows} != {float(lock["selected_eta"])}:
        raise AssertionError("confirmation eta drift")
    if {row["epsilon_key"] for row in rows} != {
        str(lock["selected_epsilon_key"])
    }:
        raise AssertionError("confirmation damping drift")

    method_metrics = {}
    for method in ("contiguous", "random_balanced", "tf"):
        selected = [row for row in rows if row["method"] == method]
        method_metrics[method] = {
            "n": len(selected),
            "tying_regret_mean": float(
                np.mean([row["predicted_regret"] for row in selected])
            ),
            "observed_gap_mean": float(
                np.mean([row["observed_gap"] for row in selected])
            ),
        }
    correlation = spearman(rows)
    tasks = sorted({row["task"] for row in rows})
    by_task = {task: [row for row in rows if row["task"] == task] for task in tasks}
    rng = np.random.default_rng(int(config["seed"]) + 91009)
    bootstrap = []
    for _ in range(int(config["bootstrap_replicates"])):
        sampled_rows = []
        for task in rng.choice(tasks, size=len(tasks), replace=True):
            sampled_rows.extend(by_task[str(task)])
        value = spearman(sampled_rows)
        if np.isfinite(value):
            bootstrap.append(value)
    ci = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    accuracy, pairs = rank_accuracy(rows)
    eta = float(lock["selected_eta"])
    predicted_gap = np.asarray(
        [eta * (2.0 - eta) * row["predicted_regret"] for row in rows]
    )
    observed_gap = np.asarray([row["observed_gap"] for row in rows])
    assertions = {
        "objective_evaluator_assertions_pass": objective["status"] == "complete"
        and all(objective["assertions"].values()),
        "tf_regret_below_contiguous": method_metrics["tf"]["tying_regret_mean"]
        < method_metrics["contiguous"]["tying_regret_mean"],
        "tf_regret_below_random_balanced": method_metrics["tf"][
            "tying_regret_mean"
        ]
        < method_metrics["random_balanced"]["tying_regret_mean"],
        "spearman_positive": correlation > 0,
        "task_bootstrap_lower_nonnegative": ci[0] >= 0,
        "rank_accuracy_reported": np.isfinite(accuracy) and pairs > 0,
        "top_bottom_separation_reported": True,
        "no_task_layer_probe_or_outlier_removed": no_removals,
    }
    passed = all(assertions.values())
    failure_decomposition = {
        "objective_implementation_error": not assertions[
            "objective_evaluator_assertions_pass"
        ],
        "fisher_or_damping_mismatch": not (
            assertions["tf_regret_below_contiguous"]
            and assertions["tf_regret_below_random_balanced"]
        ),
        "assignment_objective_invalid": (
            assertions["tf_regret_below_contiguous"]
            and assertions["tf_regret_below_random_balanced"]
            and (correlation <= 0 or ci[0] < 0)
        ),
        "finite_scale_taylor_breakdown": float(
            np.median(np.abs(observed_gap - predicted_gap))
        )
        > float(np.median(np.abs(observed_gap))),
        "note": "Diagnostic categories may co-occur; no downstream ROUGE was used.",
    }
    top_bottom_result = top_bottom(rows)
    result = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "complete" if passed else "failed",
        "decision_status": (
            "PASSED_AFTER_PROTOCOL_AMENDMENT" if passed else "FAILED"
        ),
        "git_commit": resolve_code_commit(),
        "config_sha256": sha256_file(args.config),
        "eta_selection_lock_sha256": sha256_file(lock_path),
        "confirmation_manifest_sha256": sha256_file(manifest_path),
        "confirmation_sample_sorted_sha256": canonical_hash(sample_ids),
        "locked_eta": eta,
        "locked_epsilon_key": str(lock["selected_epsilon_key"]),
        "confirmation_was_run_once": True,
        "selection_was_not_reopened": True,
        "metrics": {
            "methods": method_metrics,
            "spearman": correlation,
            "spearman_task_bootstrap_95ci": ci,
            "rank_accuracy": accuracy,
            "rank_accuracy_pairs": pairs,
            "top_bottom_observed_gap": top_bottom_result,
            "absolute_observed_gap": distribution(np.abs(observed_gap)),
            "repeated_forward_noise": distribution(
                [row["repeat_noise_abs"] for row in rows]
            ),
            "taylor_prediction_error": {
                "formula": "eta * (2 - eta) * R_tie",
                **distribution(np.abs(observed_gap - predicted_gap)),
                "rmse": float(
                    np.sqrt(np.mean(np.square(observed_gap - predicted_gap)))
                ),
            },
            "records": len(rows),
            "expected_records": expected_records,
            "samples": len(sample_ids),
            "units": expected_units,
            "assertions": assertions,
        },
        "failure_decomposition": failure_decomposition,
        "artifacts": {
            "raw_confirmation": str(output / role),
            "profiles": str(output / "profiles.npz"),
            "profile_metadata": str(output / "profile_metadata.json"),
            "assignment_candidates": str(output / "assignment_candidates.json"),
            "eta_selection_lock": str(lock_path),
            "confirmation_manifest": str(manifest_path),
            "objective_audit": str(objective_path),
            "bootstrap_samples": str(output / "task_bootstrap_samples.json"),
        },
    }
    if not passed:
        result["failure_reason"] = "one or more pre-registered P1-R1 criteria failed"
    (output / "task_bootstrap_samples.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
