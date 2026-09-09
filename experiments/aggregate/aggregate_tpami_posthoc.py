#!/usr/bin/env python3
"""Recover TPAMI table cells that require analysis only, not new training.

Every value in this artifact is regenerated from immutable local raw results.
Protocol-ambiguous or genuinely unmeasured cells remain outside this script.
"""

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
MODEL_ORDER = [
    "Qwen3-8B",
    "Qwen3-4B",
    "Llama3-8B",
    "Llama3-3B",
    "Gemma2-9B",
    "Gemma2-2B",
]
OFFICIAL = ROOT / "analysis/tables/m0_m1_official_aggregate.json"
OFFICIAL_RESULTS = ROOT / "experiments/raw_results/m0_m1_official_result_records"
M6_MANIFEST = ROOT / "experiments/materials/m6_discovery_transfer_release_manifest.json"
M7_RESULT = ROOT / "experiments/raw_results/m7_partition_stability_small_models_formal_r1/result.json"
H0_QWEN = ROOT / "experiments/raw_results/h0_calibration_sweep_qwen3_4b_seed24001_remote_l20z"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(values: list[float]) -> dict[str, Any]:
    if not values or not all(np.isfinite(values)):
        raise AssertionError("summary received empty or non-finite values")
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": [float(value) for value in values],
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def terminal_result(official: dict[str, Any], model: str, setting: str, variant: str, seed: int) -> tuple[Path, dict[str, Any]]:
    rows = official["cells"][f"{model}.{setting}.{variant}"]["per_seed"]
    match = next(row for row in rows if int(row["seed"]) == seed)
    path = OFFICIAL_RESULTS / match["terminal_registry_run_id"] / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, read_json(path)


def paired_bootstrap(official: dict[str, Any], replicates: int, seed: int) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Tasks are resampled first; paired seeds are resampled within each task."""

    rng = np.random.default_rng(seed)
    matrices: dict[str, np.ndarray] = {}
    used: list[Path] = []
    for model in MODEL_ORDER:
        by_seed: list[list[float]] = []
        task_order: list[str] | None = None
        for paired_seed in range(1, 6):
            tf_path, tf = terminal_result(official, model, "mixed", "tf", paired_seed)
            gg_path, gg = terminal_result(official, model, "mixed", "gg", paired_seed)
            used.extend([tf_path, gg_path])
            tf_tasks = tf["metrics"]["rouge"]["per_task"]
            gg_tasks = gg["metrics"]["rouge"]["per_task"]
            if set(tf_tasks) != set(gg_tasks) or len(tf_tasks) != 15:
                raise AssertionError(f"{model} seed {paired_seed}: paired task set mismatch")
            if task_order is None:
                task_order = sorted(tf_tasks)
            by_seed.append([float(tf_tasks[task]["rougeL"] - gg_tasks[task]["rougeL"]) for task in task_order])
        matrices[model] = np.asarray(by_seed, dtype=np.float64).T  # [tasks, seeds]

    samples = {model: np.empty(replicates, dtype=np.float64) for model in MODEL_ORDER}
    macro = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        model_values = []
        for model in MODEL_ORDER:
            matrix = matrices[model]
            task_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            task_means = []
            for task_index in task_indices:
                seed_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
                task_means.append(float(matrix[task_index, seed_indices].mean()))
            value = float(np.mean(task_means))
            samples[model][index] = value
            model_values.append(value)
        macro[index] = float(np.mean(model_values))

    result = {
        model: {
            "point_estimate": float(matrices[model].mean()),
            "hierarchical_bootstrap_95ci": percentile_interval(samples[model]),
            "tasks": int(matrices[model].shape[0]),
            "paired_seeds": int(matrices[model].shape[1]),
        }
        for model in MODEL_ORDER
    }
    result["Macro average"] = {
        "point_estimate": float(np.mean([matrices[model].mean() for model in MODEL_ORDER])),
        "hierarchical_bootstrap_95ci": percentile_interval(macro),
        "models": len(MODEL_ORDER),
        "tasks_per_model": 15,
        "paired_seeds": 5,
    }
    sample_payload = {model: samples[model].tolist() for model in MODEL_ORDER} | {"Macro average": macro.tolist()}
    return result, sample_payload, used


def task_indices(result: dict[str, Any]) -> dict[str, int]:
    order = list(result["protocol"]["task_order"])
    if len(order) != 15 or len(set(order)) != 15:
        raise AssertionError("sequential task order must contain 15 unique tasks")
    return {task: index for index, task in enumerate(order)}


def sequential_subset(result: dict[str, Any], tasks: list[str]) -> tuple[float, float]:
    matrix = np.asarray(result["metrics"]["evaluation_matrix_a_t_s"], dtype=np.float64)
    if matrix.shape != (15, 15):
        raise AssertionError("sequential matrix must be 15x15")
    indices = task_indices(result)
    selected = [indices[task] for task in tasks]
    final_score = float(matrix[-1, selected].mean())
    prior = [index for index in selected if index < matrix.shape[0] - 1]
    forget = float(np.mean([np.max(matrix[:-1, index]) - matrix[-1, index] for index in prior])) if prior else 0.0
    return final_score, forget


def mixed_subset(result: dict[str, Any], tasks: list[str]) -> float:
    per_task = result["metrics"]["rouge"]["per_task"]
    return float(statistics.mean(float(per_task[task]["rougeL"]) for task in tasks))


def discovery_transfer(official: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    manifest = read_json(M6_MANIFEST)
    four = sorted(
        [record for record in manifest["records"] if record["scope"] == "four_category"],
        key=lambda row: (row["setting"], int(row["fold"])),
    )
    category_by_fold = {
        int(record["fold"]): (list(record["discovery_tasks"]), list(record["heldout_tasks"]))
        for record in four
        if record["setting"] == "mixed"
    }
    if len(category_by_fold) != 5:
        raise AssertionError("M6 must define five rotating category folds")
    used = [M6_MANIFEST]
    rows: dict[str, Any] = {}

    for label, variant in (("all15", "tf"), ("gg", "gg")):
        values = defaultdict(list)
        for fold in range(1, 6):
            discovery, heldout = category_by_fold[fold]
            mixed_path, mixed = terminal_result(official, "Qwen3-4B", "mixed", variant, fold)
            sequential_path, sequential = terminal_result(official, "Qwen3-4B", "sequential", variant, fold)
            used.extend([mixed_path, sequential_path])
            values["seen_mt"].append(mixed_subset(mixed, discovery))
            values["unseen_mt"].append(mixed_subset(mixed, heldout))
            unseen_st, unseen_forget = sequential_subset(sequential, heldout)
            values["unseen_st"].append(unseen_st)
            values["unseen_forget"].append(unseen_forget)
            values["overall_mt"].append(float(mixed["metrics"]["rouge"]["macro"]["rougeL"]))
        rows[label] = {metric: summary(metric_values) for metric, metric_values in values.items()}

    for scope in ("four_category", "random12"):
        values = defaultdict(list)
        records = sorted(
            [record for record in manifest["records"] if record["scope"] == scope],
            key=lambda row: (row["setting"], int(row["fold"])),
        )
        for record in [row for row in records if row["setting"] == "mixed"]:
            path = ROOT / "experiments/raw_results" / record["run_id"] / "result.json"
            if not path.is_file():
                continue
            result = read_json(path); used.append(path)
            values["seen_mt"].append(mixed_subset(result, record["discovery_tasks"]))
            values["unseen_mt"].append(mixed_subset(result, record["heldout_tasks"]))
            values["overall_mt"].append(float(result["metrics"]["rouge"]["macro"]["rougeL"]))
        for record in [row for row in records if row["setting"] == "sequential"]:
            path = ROOT / "experiments/raw_results" / record["run_id"] / "result.json"
            if not path.is_file():
                continue
            result = read_json(path); used.append(path)
            unseen_st, unseen_forget = sequential_subset(result, record["heldout_tasks"])
            values["unseen_st"].append(unseen_st)
            values["unseen_forget"].append(unseen_forget)
        rows[scope] = {
            metric: summary(metric_values) for metric, metric_values in values.items()
        } | {
            "mixed_complete": len(values["overall_mt"]) == 5,
            "sequential_complete": len(values["unseen_st"]) == 5,
        }
    return rows, used


def m8_topk_scores() -> tuple[dict[str, Any], list[Path]]:
    cells = []
    used: list[Path] = []
    for path in sorted((ROOT / "experiments/raw_results").glob("m8_*_mixed_seed*_formal_r1/ability/rank*.jsonl")):
        used.append(path)
        run_id = path.parents[1].name
        parts = run_id.split("_")
        model_key = "_".join(parts[1:3])
        model = {"qwen3_4b": "Qwen3-4B", "llama3_3b": "Llama3-3B", "gemma2_2b": "Gemma2-2B"}[model_key]
        seed = int(next(part[4:] for part in parts if part.startswith("seed")))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["multi_expert"]:
                    cells.append((model, seed, -float(row["target_only_loss"]), -float(row["full_loss"])))
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for model, seed, top1, top4 in cells:
        grouped[(model, seed)].append((top1, top4))
    if len(grouped) != 15:
        raise AssertionError(f"expected 15 formal M8 cells, got {len(grouped)}")
    records = []
    for (model, seed), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        records.append({
            "model": model,
            "seed": seed,
            "multi_expert_units": len(values),
            "top1_score": float(array[:, 0].mean()),
            "top4_score": float(array[:, 1].mean()),
            "composition_gain": float((array[:, 1] - array[:, 0]).mean()),
        })
    per_model = {}
    for model in ("Qwen3-4B", "Llama3-3B", "Gemma2-2B"):
        selected = [row for row in records if row["model"] == model]
        per_model[model] = {
            metric: summary([row[metric] for row in selected])
            for metric in ("top1_score", "top4_score", "composition_gain")
        }
    macro = {
        metric: summary([per_model[model][metric]["mean"] for model in sorted(per_model)])
        for metric in ("top1_score", "top4_score", "composition_gain")
    }
    return {
        "score_definition": "S = -token-normalized NLL; therefore Top-4 - Top-1 equals target_only_loss - full_loss",
        "subset": "pre-registered multi-expert subset only",
        "records": records,
        "per_model": per_model,
        "macro": macro,
    }, used


def solver_row(path: Path, selected: str) -> dict[str, Any]:
    payload = read_json(path)
    audit = payload["solver_audit"]
    layers = audit[selected]
    values = []
    for layer in layers.values():
        restarts = layer["audits"]
        objectives = np.asarray([float(row["objective"]) for row in restarts], dtype=np.float64)
        mean_objective = float(objectives.mean())
        best = next(row for row in restarts if int(row["restart"]) == int(layer["best_restart"]))
        values.append({
            "iterations": float(best["iterations"]),
            "converged": float(bool(best["converged"])),
            "best_regret": float(layer["objective"]),
            "restart_spread_cv": float(objectives.std(ddof=0) / max(abs(mean_objective), 1e-12)),
        })
    return {
        "iterations_mean": float(statistics.mean(row["iterations"] for row in values)),
        "convergence_rate": float(statistics.mean(row["converged"] for row in values)),
        "best_regret_mean": float(statistics.mean(row["best_regret"] for row in values)),
        "restart_spread_cv_mean": float(statistics.mean(row["restart_spread_cv"] for row in values)),
        "layers": len(values),
        "selected_solver_key": selected,
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
    }


def solver_table() -> tuple[dict[str, Any], list[Path]]:
    specs = {
        "Qwen3-8B": (ROOT / "experiments/raw_results/h4_qwen3_8b_calibration_seed24001_r5/h0_assignment_candidates.json", "H0-00"),
        "Qwen3-4B": (H0_QWEN / "h0_assignment_candidates.json", "H0-00"),
        "Llama3-8B": (ROOT / "experiments/raw_results/m2_llama3_8b_fidelity_fp32_r2/m2_assignment_candidates.json", "tf"),
        "Llama3-3B": (ROOT / "experiments/raw_results/h0_calibration_sweep_llama3_3b_seed24001_local_a100/h0_assignment_candidates.json", "H0-11"),
        "Gemma2-9B": (ROOT / "experiments/raw_results/m2_gemma2_9b_fidelity_fp32_r2/m2_assignment_candidates.json", "tf"),
        "Gemma2-2B": (ROOT / "experiments/raw_results/h0_calibration_sweep_gemma2_2b_seed24001_local_a100/h0_assignment_candidates.json", "H0-05"),
    }
    for path, _ in specs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {model: solver_row(*specs[model]) for model in MODEL_ORDER}, [path for path, _ in specs.values()]


def h0_sensitivity() -> tuple[list[dict[str, Any]], list[Path]]:
    result_path = H0_QWEN / "result.json"
    result = read_json(result_path)
    timing_path = ROOT / "experiments/raw_results/table_xxi_qwen3_4b_calibration_timing_replay_v1/result.json"
    timing_payload = read_json(timing_path)
    if timing_payload.get("status") != "complete" or not all(timing_payload.get("assertions", {}).values()):
        raise AssertionError("Table XXI timing replay is incomplete")
    timing = {row["trial_id"]: row for row in timing_payload["rows"]}
    trials = {row["trial_id"]: row for row in result["metrics"]["trials"]}
    fidelity_rows: dict[str, list[float]] = defaultdict(list)
    jsonls = sorted((H0_QWEN / "h0_finite/fidelity").glob("rank*.jsonl"))
    for path in jsonls:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                fidelity_rows[row["trial_id"]].append(float(row["predicted_regret"]))
    axes = [
        ("assignment_probes_per_task", [4, 8, 16, 32]),
        ("fisher_probes_per_task", [4, 8, 16, 32]),
        ("epsilon_f", [1e-4, 1e-3, 1e-2, 1e-1]),
        ("solver_restarts", [1, 3, 5, 10]),
    ]
    default = result["selected_config"]
    rows = []
    for axis, values in axes:
        for value in values:
            if float(default[axis]) == float(value):
                trial = trials[result["selected_trial_id"]]
            else:
                matches = [row for row in trials.values() if row["config"]["changed_axis"] == axis and float(row["config"][axis]) == float(value)]
                if len(matches) != 1:
                    raise AssertionError(f"{axis}={value}: expected one H0 trial, found {len(matches)}")
                trial = matches[0]
            regrets = fidelity_rows[trial["trial_id"]]
            if not regrets:
                raise AssertionError(f"{trial['trial_id']}: missing fidelity records")
            rows.append({
                "factor": axis,
                "value": value,
                "trial_id": trial["trial_id"],
                "heldout_regret": float(statistics.mean(regrets)),
                "heldout_units": len(regrets),
                "stability_ari": float(trial["stability_ari_to_selected"]),
                "fidelity_spearman": float(trial["fidelity_spearman_report_only"]),
                "calibration_time_seconds": float(timing[trial["trial_id"]]["calibration_time_seconds"]),
                "calibration_time_scope": "cached profile construction plus exact-capacity TF solve over all injected layers; median of five exact replays",
            })
    return rows, [result_path, timing_path, *jsonls]


def parameter_ratio_audit(official: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    keys = (
        "model_name", "target_modules", "num_experts", "rank", "lora_alpha",
        "router_mode", "router_bias", "top_k", "route_pooling_scope",
    )
    rows = {}
    used: list[Path] = []
    for model in MODEL_ORDER:
        checks = []
        observed_counts = []
        for setting in ("mixed", "sequential"):
            for seed in range(1, 6):
                tf_path, tf = terminal_result(official, model, setting, "tf", seed)
                gg_path, gg = terminal_result(official, model, setting, "gg", seed)
                used.extend([tf_path, gg_path])
                tf_cfg_path, gg_cfg_path = ROOT / tf["config_path"], ROOT / gg["config_path"]
                tf_cfg, gg_cfg = yaml.safe_load(tf_cfg_path.read_text()), yaml.safe_load(gg_cfg_path.read_text())
                used.extend([tf_cfg_path, gg_cfg_path])
                checks.append(all(tf_cfg.get(key) == gg_cfg.get(key) for key in keys))
                for payload in (tf, gg):
                    value = payload.get("metrics", {}).get("trainable_parameters")
                    if value is not None:
                        observed_counts.append(int(value))
        if not all(checks) or not observed_counts:
            raise AssertionError(f"{model}: TF/GG trainable-parameter contract mismatch")
        rows[model] = {
            "tf_over_gg": 1.0,
            "paired_config_cells": len(checks),
            "all_parameterization_keys_equal": all(checks),
            "observed_trainable_parameter_counts": sorted(set(observed_counts)),
            "derivation": "exact static equality of all trainable-parameter-defining config fields; corroborated by available runtime counts",
        }
    return rows, used


def assignment_timing_replay() -> tuple[dict[str, Any], list[Path]]:
    path = ROOT / "experiments/raw_results/m9_assignment_cpu_timing_replay_v1/result.json"
    payload = read_json(path)
    if payload.get("status") != "complete" or not all(payload.get("assertions", {}).values()):
        raise AssertionError("M9 assignment timing replay is not complete and valid")
    rows = {str(row["model"]): row for row in payload["rows"]}
    if list(rows) != MODEL_ORDER or any(int(row["repeats"]) != 5 for row in rows.values()):
        raise AssertionError("M9 assignment timing replay coverage mismatch")
    return rows, [path]


def calibration_timing_replays() -> tuple[dict[str, Any], list[Path]]:
    slugs = {
        "Qwen3-8B": "qwen3_8b", "Qwen3-4B": "qwen3_4b",
        "Llama3-8B": "llama3_8b", "Llama3-3B": "llama3_3b",
        "Gemma2-9B": "gemma2_9b", "Gemma2-2B": "gemma2_2b",
    }
    rows, used = {}, []
    for model in MODEL_ORDER:
        path = ROOT / f"experiments/raw_results/m9_calibration_timing_{slugs[model]}_v1/result.json"
        payload = read_json(path)
        if payload.get("status") != "complete" or payload.get("model") != model or not all(payload.get("assertions", {}).values()):
            raise AssertionError(f"invalid M9 calibration timing result: {model}")
        rows[model] = payload["metrics"]
        used.append(path)
    return rows, used


def throughput_replays() -> tuple[dict[str, Any], list[Path]]:
    slugs = {
        "Qwen3-8B": "qwen3_8b", "Qwen3-4B": "qwen3_4b",
        "Llama3-8B": "llama3_8b", "Llama3-3B": "llama3_3b",
        "Gemma2-9B": "gemma2_9b", "Gemma2-2B": "gemma2_2b",
    }
    rows, used = {}, []
    for model in MODEL_ORDER:
        path = ROOT / f"experiments/raw_results/m9_throughput_{slugs[model]}_v1/result.json"
        payload = read_json(path)
        if payload.get("status") != "complete" or payload.get("model") != model or not all(payload.get("assertions", {}).values()):
            raise AssertionError(f"invalid M9 throughput result: {model}")
        metrics = payload["metrics"]
        rows[model] = {
            "tf_tokens_per_second": float(metrics["tf"]["tokens_per_second"]),
            "gg_tokens_per_second": float(metrics["gg"]["tokens_per_second"]),
            "tf_over_gg": float(metrics["tf_over_gg"]),
            "aggregate_generated_tokens_per_method": int(metrics["tf"]["aggregate_generated_tokens"]),
            "protocol": payload["protocol"],
        }
        used.append(path)
    return rows, used


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TPAMI Post-hoc Recoveries",
        "",
        "Only analysis-only recoveries are included. No new SFT or protocol-ambiguous comparison is introduced.",
        "",
        "## Table II hierarchical bootstrap CI",
        "",
        "| Model | MT delta | 95% CI |",
        "|---|---:|---:|",
    ]
    for model in [*MODEL_ORDER, "Macro average"]:
        row = payload["table_ii"][model]
        lines.append(f"| {model} | {row['point_estimate']:.4f} | [{row['hierarchical_bootstrap_95ci'][0]:.4f}, {row['hierarchical_bootstrap_95ci'][1]:.4f}] |")
    lines.extend(["", "## Table XX solver", "", "| Model | Iter. | Conv. | Best regret | Restart spread |", "|---|---:|---:|---:|---:|"])
    for model in MODEL_ORDER:
        row = payload["table_xx"][model]
        lines.append(f"| {model} | {row['iterations_mean']:.4f} | {row['convergence_rate']:.4f} | {row['best_regret_mean']:.4f} | {row['restart_spread_cv_mean']:.6f} |")
    lines.extend([
        "",
        "## Remaining boundaries",
        "",
        "- Table XII fixed-load TF/GG throughput is complete for all six models.",
        "- Table IX comparison methods remain excluded pending the already identified checkpoint/assignment semantics decision.",
        "- M6 sequential rows appear automatically after all ten result.json files are returned locally.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/tables/tpami_posthoc")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=24001)
    args = parser.parse_args()
    official = read_json(OFFICIAL)
    table_ii, bootstrap_samples, used_ii = paired_bootstrap(official, args.bootstrap_replicates, args.bootstrap_seed)
    table_vii, used_vii = discovery_transfer(official)
    table_ix_tf, used_ix = m8_topk_scores()
    table_xx, used_xx = solver_table()
    table_xxi, used_xxi = h0_sensitivity()
    table_xii_params, used_xii = parameter_ratio_audit(official)
    table_xii_assignment, used_xii_assignment = assignment_timing_replay()
    table_xii_calibration, used_xii_calibration = calibration_timing_replays()
    table_xii_throughput, used_xii_throughput = throughput_replays()
    assertions = {
        "table_ii_all_models_and_macro": set(table_ii) == {*MODEL_ORDER, "Macro average"},
        "table_ii_10000_bootstrap_samples": args.bootstrap_replicates == 10_000 and all(len(values) == 10_000 for values in bootstrap_samples.values()),
        "table_vii_all15_and_gg_complete": all(all(metric in table_vii[row] for metric in ("seen_mt", "unseen_mt", "unseen_st", "unseen_forget", "overall_mt")) for row in ("all15", "gg")),
        "table_ix_tf_15_cells": len(table_ix_tf["records"]) == 15,
        "table_xx_six_models": list(table_xx) == MODEL_ORDER,
        "table_xxi_16_rows": len(table_xxi) == 16,
        "table_xxi_16_timing_cells": len(table_xxi) == 16 and all(row["calibration_time_seconds"] > 0 for row in table_xxi),
        "table_xii_six_exact_parameter_ratios": len(table_xii_params) == 6 and all(row["tf_over_gg"] == 1.0 for row in table_xii_params.values()),
        "table_xii_six_exact_assignment_replays": len(table_xii_assignment) == 6 and all(row["assignment_solve_seconds"] > 0 for row in table_xii_assignment.values()),
        "table_xii_six_calibration_memory_replays": len(table_xii_calibration) == 6 and all(row["calibration_gpu_hours"] > 0 and row["peak_memory_allocated_gb"] > 0 for row in table_xii_calibration.values()),
        "table_xii_six_throughput_replays": len(table_xii_throughput) == 6 and all(row["tf_over_gg"] > 0 for row in table_xii_throughput.values()),
        "no_new_training": True,
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    used = sorted(set([OFFICIAL, *used_ii, *used_vii, *used_ix, *used_xx, *used_xxi, *used_xii, *used_xii_assignment, *used_xii_calibration, *used_xii_throughput]))
    payload = {
        "schema_version": 1,
        "status": "complete_analysis_only_recovery",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "new_sft": False,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "missing_values_not_imputed": True,
        },
        "table_ii": table_ii,
        "table_vii": table_vii,
        "table_ix_tf_topk": table_ix_tf,
        "table_xii_trainable_params": table_xii_params,
        "table_xii_assignment_timing": table_xii_assignment,
        "table_xii_calibration_timing": table_xii_calibration,
        "table_xii_throughput": table_xii_throughput,
        "table_xx": table_xx,
        "table_xxi": table_xxi,
        "assertions": assertions,
        "sources": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in used],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    samples_path = args.output_dir / "table_ii_bootstrap_samples.json"
    markdown_path = args.output_dir / "TPAMI_POSTHOC_RESULTS.md"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps({
        "replicates": args.bootstrap_replicates,
        "seed": args.bootstrap_seed,
        "samples": bootstrap_samples,
    }, separators=(",", ":")) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    manifest = {
        "result": str(result_path.relative_to(ROOT)), "result_sha256": sha256(result_path),
        "bootstrap_samples": str(samples_path.relative_to(ROOT)), "bootstrap_samples_sha256": sha256(samples_path),
        "markdown": str(markdown_path.relative_to(ROOT)), "markdown_sha256": sha256(markdown_path),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
