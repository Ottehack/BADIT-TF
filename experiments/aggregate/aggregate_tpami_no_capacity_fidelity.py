#!/usr/bin/env python3
"""Aggregate one Table VI nonempty-only fidelity run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config["output_dir"])
    shards = sorted((output / "fidelity").glob("rank*.jsonl"))
    if len(shards) != int(config["world_size"]): raise AssertionError("shard count mismatch")
    rows = [json.loads(line) for path in shards for line in path.read_text().splitlines() if line]
    expected = int(config["fidelity_probes_per_task"]) * 15 * int(config["expected_layer_count"])
    tasks = {row["task"] for row in rows}
    rho = float(spearmanr([r["predicted_regret"] for r in rows], [r["observed_gap"] for r in rows]).statistic)
    assertions = {
        "all_units_retained": len(rows) == expected, "fifteen_tasks_present": len(tasks) == 15,
        "all_values_finite": all(np.isfinite(r[k]) for r in rows for k in ("predicted_regret", "observed_gap", "repeat_noise_abs")),
        "official_test_not_loaded": not config.get("official_test_loaded"),
        "downstream_test_not_loaded": not config.get("downstream_test_loaded"),
    }
    if not all(assertions.values()): raise AssertionError(assertions)
    result = {
        "schema_version": 1, "experiment_id": config["experiment_id"], "run_id": config["run_id"],
        "model": config["model_name"], "status": "complete", "config_sha256": sha(args.config),
        "metrics": {"predicted_regret_mean": float(np.mean([r["predicted_regret"] for r in rows])),
                    "spearman": rho, "observed_gap_mean": float(np.mean([r["observed_gap"] for r in rows])),
                    "units": len(rows), "assertions": assertions},
        "protocol": {"method": "tf_no_capacity", "official_test_loaded": False, "coefficient_renormalization": False},
        "artifacts": {"shards": [str(path) for path in shards]},
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__": main()
