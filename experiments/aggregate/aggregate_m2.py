#!/usr/bin/env python3
"""Aggregate one model's M2 finite-loss fidelity records without selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr

from badit_tf.training import resolve_code_commit


METHOD_ORDER = ("contiguous", "random_balanced", "gg_dog", "raw_q", "tf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(rows: list[dict[str, Any]]) -> float:
    value = spearmanr(
        [float(row["predicted_regret"]) for row in rows],
        [float(row["observed_gap"]) for row in rows],
    ).statistic
    return float(value) if np.isfinite(value) else float("nan")


def rank_accuracy(rows: list[dict[str, Any]], *, rng: np.random.Generator, pairs: int) -> tuple[float, int]:
    if len(rows) < 2:
        return float("nan"), 0
    correct = 0
    used = 0
    for _ in range(pairs):
        left_index, right_index = rng.choice(len(rows), size=2, replace=False)
        left, right = rows[int(left_index)], rows[int(right_index)]
        prediction = float(left["predicted_regret"]) - float(right["predicted_regret"])
        observed = float(left["observed_gap"]) - float(right["observed_gap"])
        if prediction == 0 or observed == 0:
            continue
        used += 1
        correct += int(np.sign(prediction) == np.sign(observed))
    return (float(correct / used) if used else float("nan")), used


def top_bottom_gap(rows: list[dict[str, Any]]) -> tuple[float, int]:
    ordered = sorted(rows, key=lambda row: float(row["predicted_regret"]))
    count = max(1, int(np.ceil(0.1 * len(ordered))))
    bottom = np.mean([float(row["observed_gap"]) for row in ordered[:count]])
    top = np.mean([float(row["observed_gap"]) for row in ordered[-count:]])
    return float(top - bottom), count


def average_random_assignments(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Average the preregistered random assignments before all statistics."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["sample_id"]), int(row["layer_index"]), str(row["method"]))].append(row)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, _, method), rows in grouped.items():
        if method == "random_balanced":
            indices = sorted(int(row["assignment_index"]) for row in rows)
            if indices != list(range(20)):
                raise AssertionError("M2 random baseline must retain exactly indices 0..19")
            representative = dict(rows[0])
            for key in (
                "predicted_regret",
                "observed_gap",
                "free_loss",
                "tied_loss",
                "repeat_noise_abs",
                "median_abs_gate_displacement",
                "p95_abs_gate_displacement",
                "max_abs_gate_displacement",
            ):
                representative[key] = float(np.mean([float(row[key]) for row in rows]))
            representative["assignment_index"] = "mean_20"
            output[method].append(representative)
        else:
            if len(rows) != 1:
                raise AssertionError(f"M2 method {method} has duplicate unit records")
            output[method].append(rows[0])
    return dict(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    records = []
    for path in sorted((output_dir / "fidelity").glob("rank*.jsonl")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not records:
        raise ValueError("no M2 fidelity records")
    by_method = average_random_assignments(records)
    if tuple(method for method in METHOD_ORDER if method in by_method) != METHOD_ORDER:
        raise AssertionError(f"M2 method set mismatch: {sorted(by_method)}")
    expected_units = len({(row["sample_id"], row["layer_index"]) for row in records})
    if expected_units <= 0:
        raise AssertionError("no M2 fidelity units")
    for method, rows in by_method.items():
        if len(rows) != expected_units:
            raise AssertionError(f"{method} retained {len(rows)} not {expected_units} units")

    tasks = sorted({str(row["task"]) for row in records})
    rng = np.random.default_rng(int(config["seed"]) + 8201)
    method_metrics: dict[str, Any] = {}
    bootstrap_samples: dict[str, list[float]] = {}
    for method in METHOD_ORDER:
        rows = by_method[method]
        bootstrap = []
        per_task = {task: [row for row in rows if row["task"] == task] for task in tasks}
        for _ in range(int(config["bootstrap_replicates"])):
            sampled = []
            for task in rng.choice(tasks, size=len(tasks), replace=True):
                sampled.extend(per_task[str(task)])
            value = spearman(sampled)
            if np.isfinite(value):
                bootstrap.append(value)
        accuracy, accuracy_pairs = rank_accuracy(
            rows,
            rng=rng,
            pairs=int(config.get("rank_accuracy_pair_samples", 10_000)),
        )
        separation, decile_count = top_bottom_gap(rows)
        bootstrap_samples[method] = [float(value) for value in bootstrap]
        method_metrics[method] = {
            "predicted_regret_mean": float(np.mean([row["predicted_regret"] for row in rows])),
            "observed_gap_mean": float(np.mean([row["observed_gap"] for row in rows])),
            "observed_gap_median_abs": float(np.median(np.abs([row["observed_gap"] for row in rows]))),
            "spearman": spearman(rows),
            "task_bootstrap_95ci": (
                [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
                if bootstrap
                else [float("nan"), float("nan")]
            ),
            "rank_accuracy": accuracy,
            "rank_accuracy_pairs": accuracy_pairs,
            "top_bottom_observed_gap_separation": separation,
            "decile_units": decile_count,
            "gate_displacement": {
                "median": float(np.median([row["median_abs_gate_displacement"] for row in rows])),
                "p95": float(np.quantile([row["p95_abs_gate_displacement"] for row in rows], 0.95)),
                "max": float(np.max([row["max_abs_gate_displacement"] for row in rows])),
            },
            "units": len(rows),
        }
    noise = [float(row["repeat_noise_abs"]) for row in records]
    result = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "model": config["model_name"],
        "status": "complete",
        "git_commit": resolve_code_commit(),
        "config_sha256": sha256_file(args.config),
        "split_manifest_sha256": config["split_manifest_sha256"],
        "eta": float(config["fidelity_eta"]),
        "metrics": {
            "methods": method_metrics,
            "fidelity_units": expected_units,
            "raw_records": len(records),
            "tasks": tasks,
            "repeated_forward_noise_abs": {
                "median": float(np.median(noise)),
                "p95": float(np.quantile(noise, 0.95)),
                "max": float(np.max(noise)),
            },
            "assertions": {
                "five_methods_present": True,
                "all_methods_cover_all_units": True,
                "random_balanced_averages_exactly_20_assignments": True,
                "official_test_not_loaded": not bool(config.get("official_test_loaded", False)),
                "downstream_test_not_loaded": not bool(config.get("downstream_test_loaded", False)),
            },
        },
        "artifacts": {
            "assignment_candidates": str(output_dir / "m2_assignment_candidates.json"),
            "profiles": str(output_dir / "m2_profiles.npz"),
            "collection": str(output_dir / "collection"),
            "fidelity": str(output_dir / "fidelity"),
            "bootstrap_samples": str(output_dir / "m2_bootstrap_samples.json"),
        },
    }
    (output_dir / "m2_bootstrap_samples.json").write_text(
        json.dumps(bootstrap_samples, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
