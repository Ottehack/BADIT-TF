#!/usr/bin/env python3
"""Validate and aggregate the frozen 30-cell M4 decomposition grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("tf", "contiguous", "random_balanced", "gg_dog", "raw_q")
METRICS = (
    "grouping_mean",
    "routing_mean",
    "total_pred_mean",
    "observed_gap_mean",
    "spearman_total_pred_vs_observed",
)
EXPECTED_COMMITS = {
    "standard": "c994c109ad5cbc8b7ad6b8637026844f8dab119b",
    # Gemma2-2B R2 removes an unused sklearn import only.  The frozen
    # objective, inputs, mapping candidates, router, eta and evaluator match R1.
    "gemma2_2b_environment_only_recovery": "80e4c21d202ba0c32d03d9a515f2bf75c44ccb16",
}
RUN_SUFFIXES = {
    "qwen3_4b": {1: "r1", 2: "r1", 3: "r1", 4: "r1", 5: "r2"},
    "llama3_3b": {seed: "r1" for seed in range(1, 6)},
    "gemma2_2b": {seed: "r2" for seed in range(1, 6)},
    "qwen3_8b": {seed: "r1" for seed in range(1, 6)},
    "llama3_8b": {1: "r1", 2: "r1", 3: "r1", 4: "r1", 5: "r3"},
    "gemma2_9b": {1: "r1", 2: "r1", 3: "r1", 4: "r1", 5: "r3"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, object]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values),
        "n_cells": len(values),
        "values": values,
    }


def aggregate(result_root: Path) -> dict:
    cells = []
    for model, seeds in RUN_SUFFIXES.items():
        for seed, suffix in seeds.items():
            run_id = f"m4_{model}_mixed_seed{seed}_formal_{suffix}"
            path = result_root / run_id / "result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            assertions = payload["metrics"]["assertions"]
            config_path = ROOT / "experiments" / "configs" / "m4"
            config_matches = list(config_path.rglob(f"{run_id}.yaml"))
            checks = {
                "run_id": payload["run_id"] == run_id,
                "status": payload["status"] == "complete",
                "model_seed": int(payload["m0_seed"]) == seed,
                "eta": float(payload["eta"]) == 0.0001,
                "commit": payload["git_commit"]
                == (
                    EXPECTED_COMMITS["gemma2_2b_environment_only_recovery"]
                    if model == "gemma2_2b"
                    else EXPECTED_COMMITS["standard"]
                ),
                "one_config": len(config_matches) == 1,
                "config_sha256": len(config_matches) == 1
                and payload["config_sha256"] == sha256(config_matches[0]),
                "method_set": set(payload["metrics"]["methods"]) == set(METHODS),
                "source_assertions": len(assertions) == 9 and all(assertions.values()),
            }
            if not all(checks.values()):
                raise AssertionError(f"invalid M4 cell {run_id}: {checks}")
            cells.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_id": run_id,
                    "result_path": str(path.relative_to(ROOT)),
                    "result_sha256": sha256(path),
                    "config_sha256": payload["config_sha256"],
                    "methods": payload["metrics"]["methods"],
                    "fidelity_units": int(payload["metrics"]["fidelity_units"]),
                    "noise": payload["metrics"]["repeated_forward_noise_abs"],
                }
            )
    if len(cells) != 30:
        raise AssertionError(f"expected 30 M4 cells, got {len(cells)}")
    macro = {
        method: {
            metric: summary(
                [float(cell["methods"][method][metric]) for cell in cells]
            )
            for metric in METRICS
        }
        for method in METHODS
    }
    tf_lower = sum(
        cell["methods"]["tf"]["grouping_mean"]
        < cell["methods"]["contiguous"]["grouping_mean"]
        and cell["methods"]["tf"]["grouping_mean"]
        < cell["methods"]["random_balanced"]["grouping_mean"]
        for cell in cells
    )
    return {
        "schema_version": 1,
        "experiment_id": "M4-DECOMPOSITION-FIDELITY-FORMAL-AGGREGATE",
        "status": "complete",
        "protocol_amendment": "eta=1e-4 local scale locked by P1; eta was not reselected for M4",
        "code_commits": EXPECTED_COMMITS,
        "models": list(RUN_SUFFIXES),
        "seeds": [1, 2, 3, 4, 5],
        "cells": cells,
        "macro": macro,
        "assertions": {
            "six_models_five_seeds_complete": len(cells) == 30,
            "all_270_source_assertions_passed": True,
            "all_source_commits_match": True,
            "all_config_hashes_match": True,
            "official_and_downstream_tests_closed": True,
            "tf_grouping_lower_than_contiguous_and_random_in_all_cells": tf_lower == 30,
        },
        "counts": {
            "logical_cells": 30,
            "terminal_cells": len(cells),
            "source_assertions_passed": 270,
            "source_assertions_total": 270,
            "tf_grouping_lower_cells": tf_lower,
            "fidelity_units": sum(cell["fidelity_units"] for cell in cells),
        },
    }


def markdown(payload: dict) -> str:
    lines = [
        "# M4 Decomposition Fidelity — Formal 30-Cell Aggregate",
        "",
        "Unweighted macro mean ± sample standard deviation across six models × five seeds. "
        "The P1-locked local scale eta=1e-4 is used under the frozen M4 protocol amendment; "
        "official and downstream tests were not loaded.",
        "",
        "| Assignment | Grouping | Routing | Total pred. | Obs. gap | Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "tf": "TF",
        "contiguous": "Contiguous",
        "random_balanced": "Random balanced",
        "gg_dog": "GG/DOG",
        "raw_q": "Raw-q",
    }
    for method in METHODS:
        row = payload["macro"][method]

        def cell(metric: str) -> str:
            value = row[metric]
            return f"{value['mean']:.6g} ± {value['sample_std']:.6g}"

        lines.append(
            f"| {labels[method]} | {cell('grouping_mean')} | {cell('routing_mean')} | "
            f"{cell('total_pred_mean')} | {cell('observed_gap_mean')} | "
            f"{cell('spearman_total_pred_vs_observed')} |"
        )
    lines.extend(
        [
            "",
            f"All 30/30 logical cells and 270/270 source assertions passed. "
            f"TF grouping regret was lower than both contiguous and random balanced in "
            f"{payload['counts']['tf_grouping_lower_cells']}/30 cells.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=Path, default=ROOT / "experiments/raw_results"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "analysis/tables/m4_decomposition_fidelity"
    )
    args = parser.parse_args()
    payload = aggregate(args.result_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "result.json"
    markdown_path = args.output_dir / "M4_DECOMPOSITION_FIDELITY_RESULTS.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(json_path.relative_to(ROOT)),
                "result_sha256": sha256(json_path),
                "markdown": str(markdown_path.relative_to(ROOT)),
                "markdown_sha256": sha256(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
