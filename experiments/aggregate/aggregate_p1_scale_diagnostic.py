#!/usr/bin/env python3
"""Aggregate the validation-only P1 intervention-locality diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eta_slug(value: float) -> str:
    return f"{value:.4g}".replace(".", "p").replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_dir = Path(source_config["output_dir"])
    output_dir = Path(config["output_dir"])
    profiles = np.load(source_dir / "profiles.npz")
    candidates = json.loads(
        (source_dir / "assignment_candidates.json").read_text(encoding="utf-8")
    )
    epsilon_key = str(config["selected_epsilon_key"])
    q = profiles["q_damping_validation"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    layer_names = candidates["layer_names"]

    metrics = []
    for eta_value in config["eta_values"]:
        eta = float(eta_value)
        eta_dir = output_dir / f"eta_{_eta_slug(eta)}"
        records = []
        for path in sorted(eta_dir.glob("rank*.jsonl")):
            records.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if not records:
            raise ValueError(f"no records for eta={eta}")
        correlation = float(
            spearmanr(
                [row["predicted_regret"] for row in records],
                [row["observed_gap"] for row in records],
            ).statistic
        )
        deltas = []
        for layer_index in source_config["pilot_layer_indices"]:
            layer_name = layer_names[int(layer_index)]
            rho = float(candidates["rho"][epsilon_key][layer_name])
            deltas.append(
                eta
                * q[:, int(layer_index), :]
                / (fisher[int(layer_index)] + rho)
            )
        absolute_delta = np.abs(np.concatenate([item.ravel() for item in deltas]))
        metrics.append(
            {
                "eta": eta,
                "records": len(records),
                "spearman": correlation,
                "gate_absolute_delta": {
                    "median": float(np.quantile(absolute_delta, 0.5)),
                    "p90": float(np.quantile(absolute_delta, 0.9)),
                    "p99": float(np.quantile(absolute_delta, 0.99)),
                    "max": float(np.max(absolute_delta)),
                    "fraction_above_one": float(np.mean(absolute_delta > 1.0)),
                },
            }
        )
    result = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "complete",
        "setting": "validation_only_intervention_locality_diagnostic",
        "seed": config["seed"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": _sha256(args.config),
        "source_config_sha256": _sha256(source_config_path),
        "source_run_id": config["source_run_id"],
        "selected_epsilon_key": epsilon_key,
        "selection_policy": config["selection_policy"],
        "metrics": {"eta_sweep": metrics},
        "artifacts": {
            "source_profiles": str(source_dir / "profiles.npz"),
            "source_assignments": str(source_dir / "assignment_candidates.json"),
            "finite_gap_records": str(output_dir),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
