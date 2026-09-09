#!/usr/bin/env python3
"""Aggregate the P1 objective-fidelity go/no-go result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spearman(rows: list[dict]) -> float:
    return float(
        spearmanr(
            [row["predicted_regret"] for row in rows],
            [row["observed_gap"] for row in rows],
        ).statistic
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    selected = json.loads((output_dir / "selected_damping.json").read_text(encoding="utf-8"))
    records = []
    for path in sorted((output_dir / "fidelity").glob("rank*.jsonl")):
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not records:
        raise ValueError("no fidelity records")
    method_means = {}
    for method in ("contiguous", "random_balanced", "tf"):
        selected_rows = [row for row in records if row["method"] == method]
        method_means[method] = {
            "predicted_regret": float(
                np.mean([row["predicted_regret"] for row in selected_rows])
            ),
            "observed_gap": float(
                np.mean([row["observed_gap"] for row in selected_rows])
            ),
            "n": len(selected_rows),
        }
    correlation = _spearman(records)

    by_unit = {}
    for row in records:
        key = (row["sample_id"], row["layer_index"])
        by_unit.setdefault(key, []).append(row)
    correct = 0
    total = 0
    for unit_rows in by_unit.values():
        for left, right in combinations(unit_rows, 2):
            pred_delta = left["predicted_regret"] - right["predicted_regret"]
            obs_delta = left["observed_gap"] - right["observed_gap"]
            if pred_delta == 0 or obs_delta == 0:
                continue
            total += 1
            correct += int(np.sign(pred_delta) == np.sign(obs_delta))
    rank_accuracy = float(correct / total) if total else float("nan")

    rng = np.random.default_rng(int(config["seed"]) + 7331)
    tasks = sorted({row["task"] for row in records})
    bootstrap = []
    by_task = {task: [row for row in records if row["task"] == task] for task in tasks}
    for _ in range(int(config["bootstrap_replicates"])):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled_rows = []
        for task in sampled_tasks:
            sampled_rows.extend(by_task[str(task)])
        value = _spearman(sampled_rows)
        if np.isfinite(value):
            bootstrap.append(value)
    ci = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    assertions = {
        "tf_regret_below_contiguous": method_means["tf"]["predicted_regret"]
        < method_means["contiguous"]["predicted_regret"],
        "tf_regret_below_random": method_means["tf"]["predicted_regret"]
        < method_means["random_balanced"]["predicted_regret"],
        "spearman_positive": correlation > 0,
        "bootstrap_ci_not_negative": ci[0] >= 0,
        "rank_accuracy_above_chance": rank_accuracy > 0.5,
    }
    result = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "model": config["model_name"],
        "setting": "calibration_fidelity",
        "seed": config["seed"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": _sha256(args.config),
        "status": "complete" if all(assertions.values()) else "failed",
        "metrics": {
            "selected_damping": selected,
            "methods": method_means,
            "spearman": correlation,
            "spearman_task_bootstrap_95ci": ci,
            "rank_accuracy": rank_accuracy,
            "rank_accuracy_pairs": total,
            "fidelity_units": len(by_unit),
            "records": len(records),
            "pilot_layer_indices": config["pilot_layer_indices"],
            "assertions": assertions,
        },
        "artifacts": {
            "profiles": str(output_dir / "profiles.npz"),
            "assignment_candidates": str(output_dir / "assignment_candidates.json"),
            "selected_damping": str(output_dir / "selected_damping.json"),
            "collection": str(output_dir / "collection"),
            "damping_validation": str(output_dir / "damping_validation"),
            "fidelity": str(output_dir / "fidelity"),
        },
    }
    if result["status"] != "complete":
        result["failure_reason"] = "one or more P1 go/no-go assertions failed"
    path = output_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

