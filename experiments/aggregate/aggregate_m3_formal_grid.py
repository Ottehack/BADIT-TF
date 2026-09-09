#!/usr/bin/env python3
"""Validate and aggregate the six-model formal M3 deployment-scale table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "experiments/materials/m3_six_model_formal_grid_manifest.json"
RECONCILIATION = ROOT / "experiments/materials/m3_formal_terminal_reconciliation.json"
ETAS = [0.1, 0.25, 0.5, 1.0]
METRICS = [
    "predicted_delta_loss_mean", "observed_delta_loss_mean", "spearman",
    "relative_error_median", "relative_error_p95", "max_abs_scaled_displacement",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def aggregate(
    result_root: Path, grid_path: Path = GRID,
    reconciliation_path: Path | None = None,
) -> dict:
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if grid["eta_grid"] != ETAS or len(grid["trials"]) != 6 or grid["unresolved"]:
        raise AssertionError("M3 formal grid is not the frozen complete six-model grid")
    if reconciliation_path is None:
        terminal = {
            trial["run_id"]: {
                "terminal_run_id": trial["run_id"],
                "terminal_config_sha256": trial["config_sha256"],
            }
            for trial in grid["trials"]
        }
        reconciliation_sha256 = None
    else:
        reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        terminal = {row["base_run_id"]: row for row in reconciliation["terminal_results"]}
        if set(terminal) != {trial["run_id"] for trial in grid["trials"]}:
            raise AssertionError("M3 terminal reconciliation does not cover the frozen grid exactly")
        reconciliation_sha256 = sha256(reconciliation_path)
    model_rows = []
    seen_models = set()
    for trial in grid["trials"]:
        terminal_row = terminal[trial["run_id"]]
        path = result_root / terminal_row["terminal_run_id"] / "result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assertions = payload["metrics"]["assertions"]
        checks = {
            "run_id": payload["run_id"] == terminal_row["terminal_run_id"],
            "model": payload["model"] == trial["model"],
            "status": payload["status"] == "complete",
            "scope": payload["protocol"]["analysis_scope"] == "formal_table",
            "eta_grid": payload["protocol"]["eta_grid"] == ETAS,
            "official_test_closed": payload["protocol"]["official_test_loaded"] is False,
            "downstream_test_closed": payload["protocol"]["downstream_test_loaded"] is False,
            "config_sha256": payload["config_sha256"] == terminal_row["terminal_config_sha256"],
            "checkpoint_sha256": payload["trained_checkpoint_sha256"] == trial["trained_checkpoint_sha256"],
            "split_sha256": payload["split_manifest_sha256"] == "c2244e81cfa42c723d64e351326e632ea2a6fa8540a79d02884137eb2c9b3af6",
            "result_sha256": (
                "result_sha256" not in terminal_row
                or sha256(path) == terminal_row["result_sha256"]
            ),
            "all_terminal_assertions": bool(assertions) and all(assertions.values()),
            "all_eta_present": set(payload["metrics"]["by_eta"]) == {str(eta) for eta in ETAS},
        }
        if not all(checks.values()):
            raise AssertionError(f"invalid M3 formal result {trial['run_id']}: {checks}")
        if payload["model"] in seen_models:
            raise AssertionError(f"duplicate M3 model: {payload['model']}")
        seen_models.add(payload["model"])
        model_rows.append({
            "model": payload["model"], "base_run_id": trial["run_id"],
            "terminal_run_id": payload["run_id"],
            "result_path": display_path(path), "result_sha256": sha256(path),
            "by_eta": payload["metrics"]["by_eta"],
            "repeated_forward_noise_abs": payload["metrics"]["repeated_forward_noise_abs"],
            "units_per_eta": payload["metrics"]["units_per_eta"],
        })
    macro = {}
    for eta in ETAS:
        key = str(eta)
        metric_rows = {}
        for metric in METRICS:
            values = [float(row["by_eta"][key][metric]) for row in model_rows]
            metric_rows[metric] = {
                "mean": mean(values), "sample_std": stdev(values), "n_models": len(values),
                "per_model": {row["model"]: value for row, value in zip(model_rows, values)},
            }
        metric_rows["units"] = sum(int(row["by_eta"][key]["units"]) for row in model_rows)
        macro[key] = metric_rows
    return {
        "schema_version": 1, "experiment_id": "M3-DEPLOYMENT-SCALE-FORMAL-AGGREGATE",
        "status": "complete", "models": [row["model"] for row in model_rows],
        "eta_grid": ETAS, "model_results": model_rows, "macro_by_eta": macro,
        "terminal_reconciliation_sha256": reconciliation_sha256,
        "assertions": {
            "six_models_complete": len(model_rows) == 6,
            "one_result_per_model": len(seen_models) == 6,
            "official_test_closed": True, "downstream_test_closed": True,
            "fixed_eta_grid_complete": True, "all_source_assertions_pass": True,
        },
    }


def markdown(payload: dict) -> str:
    lines = [
        "# M3 Deployment-Scale Fidelity — Six-Model Formal Aggregate", "",
        "Macro-average is the unweighted mean across the six frozen model results. "
        "The `±` term is the sample standard deviation across models; official and "
        "downstream tests were not loaded.", "",
        "| eta | Pred. ΔL | Obs. ΔL | Spearman | Median relative error | p95 relative error | n models | units |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for eta in payload["eta_grid"]:
        row = payload["macro_by_eta"][str(eta)]
        def cell(metric: str) -> str:
            value = row[metric]
            return f"{value['mean']:.6g} ± {value['sample_std']:.6g}"
        lines.append(
            f"| {eta:.2f} | {cell('predicted_delta_loss_mean')} | "
            f"{cell('observed_delta_loss_mean')} | {cell('spearman')} | "
            f"{cell('relative_error_median')} | {cell('relative_error_p95')} | 6 | {row['units']} |"
        )
    lines.extend(["", "## Per-model values", ""])
    for model in payload["model_results"]:
        lines.extend([f"### {model['model']}", "", f"Result: `{model['result_path']}`", "",
                      "| eta | Pred. ΔL | Obs. ΔL | Spearman | Median rel. error | p95 rel. error | units |",
                      "|---:|---:|---:|---:|---:|---:|---:|"])
        for eta in payload["eta_grid"]:
            row = model["by_eta"][str(eta)]
            lines.append(
                f"| {eta:.2f} | {row['predicted_delta_loss_mean']:.6g} | "
                f"{row['observed_delta_loss_mean']:.6g} | {row['spearman']:.6g} | "
                f"{row['relative_error_median']:.6g} | {row['relative_error_p95']:.6g} | {row['units']} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=ROOT / "experiments/raw_results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/tables/m3_deployment_scale")
    parser.add_argument("--reconciliation", type=Path, default=RECONCILIATION)
    args = parser.parse_args()
    payload = aggregate(args.result_root, reconciliation_path=args.reconciliation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "result.json"
    md_path = args.output_dir / "M3_DEPLOYMENT_SCALE_RESULTS.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "json_sha256": sha256(json_path),
                      "markdown": str(md_path), "markdown_sha256": sha256(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
