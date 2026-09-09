#!/usr/bin/env python3
"""Validate and aggregate the frozen 15-cell M8 ability-evidence grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "experiments/configs/m8/ability_v1"
RELEASE_MANIFEST = ROOT / "experiments/materials/m8_ability_v1_release_manifest.json"
DEFAULT_OUTPUT = ROOT / "analysis/tables/m8_ability_evidence"
METRICS = (
    "target_deletion_increase",
    "control_deletion_increase",
    "specificity_gap",
    "sufficiency",
    "composition_gain",
    "multi_expert_composition_gain",
    "effective_experts",
    "top1_mass",
    "multi_expert_fraction",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, Any]:
    if not values or not all(np.isfinite(values)):
        raise AssertionError("M8 aggregate received empty or non-finite values")
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "positive_count": sum(value > 0 for value in values),
        "values": values,
    }


def validate_cell(cell: dict[str, Any], expected_commit: str) -> dict[str, Any]:
    config_path = ROOT / cell["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_path = ROOT / config["output_dir"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = {
        "run_id": result.get("run_id") == cell["run_id"] == config.get("run_id"),
        "status": result.get("status") == "complete",
        "formal_scope": cell.get("analysis_scope") == "formal_table",
        "code_commit": result.get("code_commit") == expected_commit,
        "config_sha256": digest(config_path) == cell["config_sha256"] == result.get("config_sha256"),
        "checkpoint_sha256": result.get("trained_checkpoint_sha256") == cell["trained_checkpoint_sha256"],
        "profile_sha256": result.get("profile_source_sha256") == {
            "profiles": cell["m4_profiles_sha256"],
            "metadata": cell["m4_profile_metadata_sha256"],
        },
        "assignment_sha256": config.get("tf_assignment_sha256") == cell["tf_assignment_sha256"],
        "assertions": all(result.get("metrics", {}).get("assertions", {}).values()),
        "official_test_closed": result.get("protocol", {}).get("official_test_loaded") is False,
        "downstream_test_closed": config.get("downstream_test_loaded") is False,
        "renormalization_closed": result.get("protocol", {}).get("coefficient_renormalization") is False,
    }
    if not all(checks.values()):
        raise AssertionError(f"{cell['run_id']}: {checks}")
    macro = result["metrics"]["macro"]
    return {
        "run_id": cell["run_id"],
        "model": cell["model"],
        "seed": int(cell["seed"]),
        "units": int(macro["units"]),
        "multi_expert_units": int(macro["multi_expert_units"]),
        "metrics": {metric: float(macro[metric]) for metric in METRICS},
        "config": cell["config"],
        "config_sha256": cell["config_sha256"],
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": digest(result_path),
        "checks": checks,
    }


def aggregate(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)
    if set(by_model) != {"Qwen3-4B", "Llama3-3B", "Gemma2-2B"}:
        raise AssertionError("M8 formal grid must contain exactly three small models")
    per_model: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        rows.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != [1, 2, 3, 4, 5]:
            raise AssertionError(f"{model}: M8 seeds are not exactly 1..5")
        per_model[model] = {
            "seeds": [row["seed"] for row in rows],
            "units": sum(row["units"] for row in rows),
            "metrics": {
                metric: summarize([row["metrics"][metric] for row in rows])
                for metric in METRICS
            },
        }
    macro = {
        metric: summarize(
            [per_model[model]["metrics"][metric]["mean"] for model in sorted(per_model)]
        )
        for metric in METRICS
    }
    return per_model, macro


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M8 Ability Evidence Results",
        "",
        "Formal held-out fidelity probes from the frozen M0 checkpoints. Values are mean ± sample standard deviation across five fixed seeds per model; Macro is the equal-weight mean across the three model means. No coefficient renormalization was used, and official/downstream tests remained closed.",
        "",
        "| Model | Target deletion | Control deletion | Specificity gap | Sufficiency | Composition gain | Positive specificity seeds | Positive composition seeds | Units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [*sorted(payload["per_model"]), "Macro"]
    for label in labels:
        metrics = payload["macro"] if label == "Macro" else payload["per_model"][label]["metrics"]
        cell = lambda key: f"{metrics[key]['mean']:.6f} ± {metrics[key]['sample_std']:.6f}"
        if label == "Macro":
            specificity = f"{metrics['specificity_gap']['positive_count']}/3 models"
            composition = f"{metrics['composition_gain']['positive_count']}/3 models"
            units = payload["total_units"]
        else:
            specificity = f"{metrics['specificity_gap']['positive_count']}/5"
            composition = f"{metrics['composition_gain']['positive_count']}/5"
            units = payload["per_model"][label]["units"]
        lines.append(
            f"| {label} | {cell('target_deletion_increase')} | {cell('control_deletion_increase')} | "
            f"{cell('specificity_gap')} | {cell('sufficiency')} | {cell('composition_gain')} | "
            f"{specificity} | {composition} | {units} |"
        )
    lines.extend([
        "",
        "## Per-seed values",
        "",
        "| Model | Seed | Specificity gap | Sufficiency | Composition gain | Multi-expert fraction | Result SHA256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for record in payload["records"]:
        metrics = record["metrics"]
        lines.append(
            f"| {record['model']} | {record['seed']} | {metrics['specificity_gap']:.9f} | "
            f"{metrics['sufficiency']:.9f} | {metrics['composition_gain']:.9f} | "
            f"{metrics['multi_expert_fraction']:.6f} | `{record['result_sha256']}` |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Gemma2-2B and Llama3-3B have positive mean specificity in all five seeds. Qwen3-4B has two negative-specificity seeds, which are retained; therefore M8 supports a positive macro tendency, not universal per-seed specificity. Composition gain is positive in all 15 formal cells. Sufficiency is reported exactly as preregistered and is not clipped when its denominator is small.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    formal = [cell for cell in manifest["cells"] if cell["analysis_scope"] == "formal_table"]
    if len(formal) != 15 or manifest["formal_cells"] != 15:
        raise AssertionError("M8 release manifest formal-cell count drifted")
    records = [validate_cell(cell, manifest["code_commit"]) for cell in formal]
    records.sort(key=lambda row: (row["model"], row["seed"]))
    per_model, macro = aggregate(records)
    payload = {
        "schema_version": 1,
        "event": "M8_ABILITY_EVIDENCE_FORMAL_AGGREGATE",
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_release_manifest": str(RELEASE_MANIFEST.relative_to(ROOT)),
        "source_release_manifest_sha256": digest(RELEASE_MANIFEST),
        "formal_cells": len(records),
        "total_units": sum(record["units"] for record in records),
        "total_multi_expert_units": sum(record["multi_expert_units"] for record in records),
        "official_test_loaded": False,
        "downstream_test_loaded": False,
        "coefficient_renormalization": False,
        "records": records,
        "per_model": per_model,
        "macro": macro,
        "assertions": {
            "fifteen_formal_cells": len(records) == 15,
            "three_models_five_seeds": all(value["seeds"] == [1, 2, 3, 4, 5] for value in per_model.values()),
            "all_source_assertions_passed": all(all(record["checks"].values()) for record in records),
            "all_units_retained": sum(record["units"] for record in records) == 108000,
            "official_test_not_loaded": True,
            "downstream_test_not_loaded": True,
        },
    }
    if not all(payload["assertions"].values()):
        raise AssertionError(payload["assertions"])
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    markdown_path = output / "M8_ABILITY_EVIDENCE_RESULTS.md"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": digest(result_path),
        "markdown": str(markdown_path.relative_to(ROOT)),
        "markdown_sha256": digest(markdown_path),
        "formal_cells": len(records),
        "total_units": payload["total_units"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
