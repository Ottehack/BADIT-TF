#!/usr/bin/env python3
"""Aggregate validation-only eta grid and atomically lock P1-R1 scale."""

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

from badit_tf.training import resolve_code_commit


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(values: list[str]) -> str:
    payload = ("\n".join(sorted(values)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def eta_slug(value: float) -> str:
    return f"{value:.4g}".replace(".", "p").replace("-", "m")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for item in sorted(path.glob("rank*.jsonl"))
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def spearman(rows: list[dict[str, Any]]) -> float:
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
        sample_rows = []
        for task in sampled:
            sample_rows.extend(by_task[str(task)])
        value = spearman(sample_rows)
        if np.isfinite(value):
            values.append(value)
    return values, [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def rank_accuracy(rows: list[dict[str, Any]]) -> tuple[float, int]:
    units: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        units[(row["sample_id"], int(row["layer_index"]))].append(row)
    correct = 0
    total = 0
    for unit_rows in units.values():
        for left, right in combinations(unit_rows, 2):
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
    bottom = np.asarray(
        [row["observed_gap"] for row in ordered[:size]], dtype=np.float64
    )
    top = np.asarray(
        [row["observed_gap"] for row in ordered[-size:]], dtype=np.float64
    )
    return {
        "decile_size": size,
        "bottom_mean": float(np.mean(bottom)),
        "top_mean": float(np.mean(top)),
        "top_minus_bottom": float(np.mean(top) - np.mean(bottom)),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def gate_statistics(
    *,
    eta: float,
    q: np.ndarray,
    fisher: np.ndarray,
    candidates: dict[str, Any],
    epsilon_key: str,
    layer_indices: list[int],
    num_experts: int,
) -> dict[str, Any]:
    free_offsets = []
    tied_offsets = []
    layer_names = candidates["layer_names"]
    method_labels = [
        np.asarray(candidates["contiguous"]),
        *[np.asarray(item) for item in candidates["random_balanced"]],
    ]
    for sample_index in range(q.shape[0]):
        for layer_index in layer_indices:
            layer_name = layer_names[layer_index]
            rho = float(candidates["rho"][epsilon_key][layer_name])
            damped = fisher[layer_index] + rho
            free = q[sample_index, layer_index] / damped
            free_offsets.append(eta * free)
            labels_for_layer = [
                *method_labels,
                np.asarray(candidates["tf"][epsilon_key][layer_name]),
            ]
            for labels in labels_for_layer:
                tied = np.empty_like(free)
                for expert in range(num_experts):
                    selected = labels == expert
                    tied[selected] = (
                        q[sample_index, layer_index, selected].sum()
                        / damped[selected].sum()
                    )
                tied_offsets.append(eta * tied)
    free_array = np.concatenate(free_offsets)
    tied_array = np.concatenate(tied_offsets)

    def summarize(offset: np.ndarray) -> dict[str, Any]:
        absolute = np.abs(offset)
        gates = 1.0 + offset
        return {
            "absolute_offset": distribution(absolute),
            "gate_below_zero_fraction": float(np.mean(gates < 0)),
            "offset_above_0p1_fraction": float(np.mean(absolute > 0.1)),
            "offset_above_0p5_fraction": float(np.mean(absolute > 0.5)),
            "offset_above_1p0_fraction": float(np.mean(absolute > 1.0)),
        }

    return {"free": summarize(free_array), "tied_all_methods": summarize(tied_array)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    profiles = np.load(output_dir / "profiles.npz")
    candidates = json.loads(
        (output_dir / "assignment_candidates.json").read_text(encoding="utf-8")
    )
    q = profiles["q_damping_validation"].astype(np.float64)
    fisher = profiles["fisher"].astype(np.float64)
    epsilon_key = str(config["selected_epsilon_key"])
    layer_indices = [int(item) for item in config["pilot_layer_indices"]]
    objective_audit_path = Path(config["objective_audit_result"])
    objective_audit = json.loads(objective_audit_path.read_text(encoding="utf-8"))
    if objective_audit["status"] != "complete" or not all(
        objective_audit["assertions"].values()
    ):
        raise AssertionError("objective/evaluator audit is not complete")

    eta_metrics = []
    bootstrap_payload = {}
    expected_records = (
        int(config["damping_validation_samples_per_task"])
        * int(config["task_count"])
        * len(layer_indices)
        * (2 + int(config["random_assignments"]))
    )
    for eta_value in config["eta_grid"]:
        eta = float(eta_value)
        rows = load_rows(output_dir / "eta_grid" / f"eta_{eta_slug(eta)}")
        if len(rows) != expected_records:
            raise AssertionError(
                f"eta={eta} has {len(rows)} records, expected {expected_records}"
            )
        if {float(row["eta"]) for row in rows} != {eta}:
            raise AssertionError(f"eta label drift for {eta}")
        sample_ids = sorted({row["sample_id"] for row in rows})
        if len(sample_ids) != int(config["damping_validation_samples_per_task"]) * int(
            config["task_count"]
        ):
            raise AssertionError("validation sample count drift")
        correlation = spearman(rows)
        bootstrap, ci = task_bootstrap(
            rows,
            seed=int(config["seed"]) + 17011 + int(round(eta * 1e7)),
            replicates=int(config["bootstrap_replicates"]),
        )
        bootstrap_payload[f"{eta:.4g}"] = bootstrap
        accuracy, pairs = rank_accuracy(rows)
        observed = np.asarray([row["observed_gap"] for row in rows])
        predicted_gap = np.asarray(
            [
                eta * (2.0 - eta) * float(row["predicted_regret"])
                for row in rows
            ]
        )
        taylor_error = observed - predicted_gap
        repeat_noise = np.asarray(
            [float(row["repeat_noise_abs"]) for row in rows]
        )
        noise_p95 = float(np.quantile(repeat_noise, 0.95))
        median_abs_gap = float(np.median(np.abs(observed)))
        eligible = median_abs_gap > noise_p95
        eta_metrics.append(
            {
                "eta": eta,
                "records": len(rows),
                "validation_sample_sorted_sha256": canonical_hash(sample_ids),
                "spearman": correlation,
                "task_bootstrap_95ci": ci,
                "rank_accuracy": accuracy,
                "rank_accuracy_pairs": pairs,
                "top_bottom_observed_gap": top_bottom(rows),
                "absolute_observed_gap": distribution(np.abs(observed)),
                "taylor_prediction_error": {
                    "prediction_formula": "eta * (2 - eta) * R_tie",
                    **distribution(np.abs(taylor_error)),
                    "rmse": float(np.sqrt(np.mean(np.square(taylor_error)))),
                    "signed_mean": float(np.mean(taylor_error)),
                },
                "repeated_forward_noise": distribution(repeat_noise),
                "eligible_above_noise": eligible,
                "gate_displacement": gate_statistics(
                    eta=eta,
                    q=q,
                    fisher=fisher,
                    candidates=candidates,
                    epsilon_key=epsilon_key,
                    layer_indices=layer_indices,
                    num_experts=int(config["num_experts"]),
                ),
            }
        )

    eligible = [item for item in eta_metrics if item["eligible_above_noise"]]
    if not eligible:
        raise AssertionError("all eta candidates are below repeated-forward noise")
    best_lower = max(item["task_bootstrap_95ci"][0] for item in eligible)
    tied = [
        item
        for item in eligible
        if abs(item["task_bootstrap_95ci"][0] - best_lower) <= 1e-12
    ]
    selected = max(tied, key=lambda item: item["eta"])
    validation_hashes = {
        item["validation_sample_sorted_sha256"] for item in eta_metrics
    }
    if len(validation_hashes) != 1:
        raise AssertionError("eta candidates used different validation samples")
    validation_hash = validation_hashes.pop()

    fisher_norm = {
        str(index): {
            "layer_name": candidates["layer_names"][index],
            "l1": float(np.linalg.norm(fisher[index], ord=1)),
            "l2": float(np.linalg.norm(fisher[index])),
            "median": float(np.median(fisher[index])),
            "max": float(np.max(fisher[index])),
        }
        for index in layer_indices
    }
    result = {
        "run_id": config["run_id"],
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "git_commit": resolve_code_commit(),
        "config_sha256": sha256_file(args.config),
        "base_config_sha256": sha256_file(Path(config["base_config"])),
        "objective_audit_result_sha256": sha256_file(objective_audit_path),
        "selected_epsilon_key": epsilon_key,
        "fisher_norm_by_pilot_layer": fisher_norm,
        "eta_metrics": eta_metrics,
        "selection_rule": (
            "Exclude eta when median absolute observed gap is not above p95 "
            "repeated-forward noise; maximize task-bootstrap Spearman lower "
            "95% bound; exact ties choose larger eta."
        ),
        "selected_eta": selected["eta"],
        "selected_metric": selected,
        "validation_sample_sorted_sha256": validation_hash,
        "confirmation_fidelity_loaded": False,
        "original_fidelity_loaded": False,
        "superni_final_test_loaded": False,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "eta_bootstrap_samples.json").write_text(
        json.dumps(bootstrap_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "run_id": config["run_id"],
        "selected_eta": selected["eta"],
        "selected_epsilon_key": epsilon_key,
        "selection_rule": result["selection_rule"],
        "validation_sample_sorted_sha256": validation_hash,
        "config_path": str(args.config),
        "config_sha256": result["config_sha256"],
        "base_config_sha256": result["base_config_sha256"],
        "profiles_sha256": sha256_file(output_dir / "profiles.npz"),
        "assignments_sha256": sha256_file(
            output_dir / "assignment_candidates.json"
        ),
        "objective_audit_result": str(objective_audit_path),
        "objective_audit_result_sha256": result[
            "objective_audit_result_sha256"
        ],
        "code_commit": result["git_commit"],
        "confirmation_fidelity_loaded_before_lock": False,
        "original_fidelity_used_for_selection": False,
        "superni_final_test_used_for_selection": False,
    }
    lock_path = output_dir / "eta_selection_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
