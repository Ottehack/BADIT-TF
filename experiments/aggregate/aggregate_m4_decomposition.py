#!/usr/bin/env python3
"""Aggregate one immutable M4 run without filtering any probe/layer unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

METHOD_ORDER = ("contiguous", "random_balanced", "gg_dog", "raw_q", "tf")


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


def average_random(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(str(row["sample_id"]), int(row["layer_index"]), str(row["method"]))].append(row)
    output: dict[str, list[dict]] = defaultdict(list)
    averaged = ("grouping", "routing", "total_pred", "direct_total", "orthogonality_error",
                "free_loss", "repeat_free_loss", "repeat_noise_abs", "routed_loss", "observed_gap")
    for (_, _, method), rows in grouped.items():
        if method == "random_balanced":
            if sorted(int(row["assignment_index"]) for row in rows) != list(range(20)):
                raise AssertionError("M4 random baseline must retain exactly indices 0..19")
            representative = dict(rows[0])
            for key in averaged:
                representative[key] = float(np.mean([float(row[key]) for row in rows]))
            representative["assignment_index"] = "mean_20"
            output[method].append(representative)
        else:
            if len(rows) != 1:
                raise AssertionError(f"M4 method {method} has duplicate unit records")
            output[method].append(rows[0])
    return dict(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    records = []
    shard_paths = sorted((output / "decomposition").glob("rank*.jsonl"))
    if len(shard_paths) != int(config["world_size"]):
        raise AssertionError("M4 decomposition shard count mismatch")
    for path in shard_paths:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not records:
        raise ValueError("no M4 decomposition records")
    by_method = average_random(records)
    if tuple(method for method in METHOD_ORDER if method in by_method) != METHOD_ORDER:
        raise AssertionError(f"M4 method set mismatch: {sorted(by_method)}")
    expected_units = len({(row["sample_id"], row["layer_index"]) for row in records})
    metrics = {}
    for method in METHOD_ORDER:
        rows = by_method[method]
        if len(rows) != expected_units:
            raise AssertionError(f"M4 {method} retained {len(rows)} not {expected_units} units")
        pred = np.asarray([row["total_pred"] for row in rows], dtype=np.float64)
        observed = np.asarray([row["observed_gap"] for row in rows], dtype=np.float64)
        correlation = spearmanr(pred, observed).statistic
        metrics[method] = {
            "grouping_mean": float(np.mean([row["grouping"] for row in rows])),
            "routing_mean": float(np.mean([row["routing"] for row in rows])),
            "total_pred_mean": float(np.mean(pred)),
            "observed_gap_mean": float(np.mean(observed)),
            "spearman_total_pred_vs_observed": float(correlation),
            "orthogonality_error_max": float(np.max([row["orthogonality_error"] for row in rows])),
            "units": len(rows),
        }
    noise = [float(row["repeat_noise_abs"]) for row in records]
    result = {
        "schema_version": 1, "experiment_id": config["experiment_id"],
        "run_id": config["run_id"], "model": config["model_name"],
        "m0_seed": int(config["m0_seed"]), "status": "complete",
        "git_commit": resolve_code_commit(), "config_sha256": sha256(args.config),
        "split_manifest_sha256": config["split_manifest_sha256"],
        "trained_checkpoint_sha256": config["initial_bank_checkpoint_sha256"],
        "assignment_candidates_sha256": config["m4_assignment_candidates_sha256"],
        "eta": float(config["decomposition_eta"]),
        "metrics": {
            "methods": metrics, "fidelity_units": expected_units, "raw_records": len(records),
            "repeated_forward_noise_abs": {"median": float(np.median(noise)),
                                           "p95": float(np.quantile(noise, 0.95)),
                                           "max": float(np.max(noise))},
            "assertions": {
                "five_methods_present": True,
                "all_methods_cover_all_units": True,
                "random_balanced_averages_exactly_20_assignments": True,
                "grouping_plus_routing_equals_direct_total": max(
                    float(row["orthogonality_error"]) for row in records
                ) < 1e-7,
                "tf_mapping_matches_checkpoint": all(bool(row["tf_mapping_matches_checkpoint"]) for row in records),
                "primitive_intervention": all(bool(row["primitive_intervention"]) for row in records),
                "coefficient_renormalization_disabled": all(not bool(row["coefficient_renormalization"]) for row in records),
                "official_test_not_loaded": not bool(config.get("official_test_loaded")),
                "downstream_test_not_loaded": not bool(config.get("downstream_test_loaded")),
            },
        },
        "artifacts": {"profiles": str(output / "m4_profiles.npz"),
                      "profile_metadata": str(output / "m4_profile_metadata.json"),
                      "collection": str(output / "collection"),
                      "decomposition": str(output / "decomposition")},
    }
    if not all(result["metrics"]["assertions"].values()):
        raise AssertionError("M4 terminal assertions failed")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
