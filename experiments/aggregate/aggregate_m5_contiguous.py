#!/usr/bin/env python3
"""Aggregate the frozen 18-cell M5 Contiguous-SVD control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/raw_results"
OFFICIAL = ROOT / "analysis/tables/m0_m1_official_aggregate.json"
OUTPUT = ROOT / "analysis/tables/m5_tf_ablation_contiguous"
COMMIT = "4ab4c7ee734e70a5141eb78329931a778a5f5805"
MODELS = {
    "Gemma2-2B": "gemma2_2b",
    "Llama3-3B": "llama3_3b",
    "Qwen3-4B": "qwen3_4b",
}
METRICS = {
    "mixed": ("macro_rouge_l",),
    "sequential": ("continual_score", "forget_rate", "forward", "backward"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(values: list[float]) -> dict[str, object]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values),
        "n": len(values),
        "values": values,
    }


def values(payload: dict, setting: str) -> dict[str, float]:
    if setting == "mixed":
        return {"macro_rouge_l": float(payload["metrics"]["rouge"]["macro"]["rougeL"])}
    return {key: float(payload["metrics"]["continual"][key]) for key in METRICS[setting]}


def main() -> None:
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    cells = {}
    assertions_total = 0
    for model, slug in MODELS.items():
        for setting in METRICS:
            tf_cell = official["cells"][f"{model}.{setting}.tf"]
            tf_by_seed = {int(row["seed"]): row for row in tf_cell["per_seed"]}
            per_seed = []
            for seed in (1, 2, 3):
                run_id = f"m5_{slug}_{setting}_contiguous_seed{seed}_formal_r1"
                result_path = RESULTS / run_id / "result.json"
                config_path = (
                    ROOT / "experiments/configs/m5/formal_contiguous_r1" / f"{run_id}.yaml"
                )
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                source_assertions = payload["metrics"]["assertions"]
                checks = {
                    "run_id": payload["run_id"] == run_id,
                    "status": payload["status"] == "complete",
                    "commit": payload["code_commit"] == COMMIT,
                    "config_sha256": payload["config_sha256"] == digest(config_path),
                    "assignment": payload["protocol"]["assignment_method"] == "contiguous",
                    "official_test_not_selection": payload["protocol"].get(
                        "official_test_used_for_selection",
                        payload["protocol"].get("final_test_used_for_selection"),
                    )
                    is False,
                    "source_assertions": bool(source_assertions)
                    and all(source_assertions.values()),
                }
                if not all(checks.values()):
                    raise AssertionError(f"invalid M5 result {run_id}: {checks}")
                assertions_total += len(source_assertions)
                contiguous = values(payload, setting)
                tf = {key: float(tf_by_seed[seed]["values"][key]) for key in METRICS[setting]}
                per_seed.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "result_path": str(result_path.relative_to(ROOT)),
                        "result_sha256": digest(result_path),
                        "source_tf_run_id": tf_by_seed[seed]["terminal_registry_run_id"],
                        "contiguous": contiguous,
                        "tf": tf,
                        "tf_minus_contiguous": {
                            key: tf[key] - contiguous[key] for key in METRICS[setting]
                        },
                    }
                )
            cells[f"{model}.{setting}"] = {
                "model": model,
                "setting": setting,
                "per_seed": per_seed,
                "contiguous": {
                    key: stat([row["contiguous"][key] for row in per_seed])
                    for key in METRICS[setting]
                },
                "tf": {
                    key: stat([row["tf"][key] for row in per_seed])
                    for key in METRICS[setting]
                },
                "tf_minus_contiguous": {
                    key: stat([row["tf_minus_contiguous"][key] for row in per_seed])
                    for key in METRICS[setting]
                },
            }

    macro = {}
    for setting in METRICS:
        selected = [cells[f"{model}.{setting}"] for model in MODELS]
        macro[setting] = {}
        for arm in ("tf", "contiguous", "tf_minus_contiguous"):
            macro[setting][arm] = {
                key: stat([float(cell[arm][key]["mean"]) for cell in selected])
                for key in METRICS[setting]
            }

    payload = {
        "schema_version": 1,
        "experiment_id": "M5-TF-ABLATION-CONTIGUOUS-FORMAL-AGGREGATE",
        "status": "complete",
        "scope": "Contiguous-SVD row only; four other ablation rows remain unrun",
        "code_commit": COMMIT,
        "source_official_aggregate": str(OFFICIAL.relative_to(ROOT)),
        "source_official_aggregate_sha256": digest(OFFICIAL),
        "cells": cells,
        "macro": macro,
        "counts": {
            "logical_cells": 18,
            "terminal_cells": 18,
            "source_assertions_passed": assertions_total,
            "source_assertions_total": assertions_total,
        },
        "assertions": {
            "three_models_two_settings_three_seeds_complete": len(cells) == 6,
            "all_source_assertions_passed": True,
            "all_config_hashes_match": True,
            "all_source_commits_match": True,
            "official_test_not_used_for_selection": True,
            "original_failed_gate_rows_preserved": True,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# M5 TF Ablation — Contiguous-SVD Formal Row",
        "",
        "Mean ± sample standard deviation over seeds 1–3 within each model. Macro values first average seeds within each model, then average the three model means.",
        "",
        "| Model | Setting | Metric | TF | Contiguous | TF − Contiguous |",
        "|---|---|---|---:|---:|---:|",
    ]
    for cell in cells.values():
        for metric in METRICS[cell["setting"]]:
            def fmt(arm: str) -> str:
                row = cell[arm][metric]
                return f"{row['mean']:.4f} ± {row['sample_std']:.4f}"
            lines.append(
                f"| {cell['model']} | {cell['setting']} | {metric} | "
                f"{fmt('tf')} | {fmt('contiguous')} | {fmt('tf_minus_contiguous')} |"
            )
    lines += ["", "## Three-model macro", "", "| Setting | Metric | TF | Contiguous | TF − Contiguous |", "|---|---|---:|---:|---:|"]
    for setting in METRICS:
        for metric in METRICS[setting]:
            row = macro[setting]
            lines.append(
                f"| {setting} | {metric} | {row['tf'][metric]['mean']:.4f} | "
                f"{row['contiguous'][metric]['mean']:.4f} | "
                f"{row['tf_minus_contiguous'][metric]['mean']:.4f} |"
            )
    lines += ["", "This is one completed M5 ablation row, not the full six-row ablation table."]
    markdown_path = OUTPUT / "M5_CONTIGUOUS_RESULTS.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path.relative_to(ROOT)), "result_sha256": digest(result_path), "markdown": str(markdown_path.relative_to(ROOT)), "markdown_sha256": digest(markdown_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
