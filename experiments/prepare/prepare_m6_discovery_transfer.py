#!/usr/bin/env python3
"""Freeze M6 discovery-only assignments and the 20-cell Qwen3-4B SFT grid."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from run_m7_partition_stability import merge_collection, profiles, select_prefix, solve


ROOT = Path(__file__).resolve().parents[2]
H0_RUN = "h0_calibration_sweep_qwen3_4b_seed24001_remote_l20z"
H0_DIR = ROOT / "experiments/raw_results" / H0_RUN
H0_CONFIG = ROOT / "experiments/configs/h0_profile_fp32.yaml"
CANDIDATES = H0_DIR / "h0_assignment_candidates.json"
BASE_ASSIGNMENT = ROOT / "experiments/configs/h1_qwen3_4b_h0_selected_assignment.json"
CONFIG_DIR = ROOT / "experiments/configs/m6/discovery_transfer_v1"
ASSIGNMENT_DIR = ROOT / "experiments/configs/m6/assignments_v1"
SPLIT_MANIFEST = ROOT / "experiments/materials/m6_discovery_transfer_splits_v1.json"
GRID_MANIFEST = ROOT / "experiments/materials/m6_discovery_transfer_grid_v1.json"
RANDOM_FOLD_SEED = 20260812

CATEGORIES = {
    "qa": [
        "task002_quoref_answer_generation",
        "task073_commonsenseqa_answer_generation",
        "task591_sciq_answer_generation",
    ],
    "summarization": [
        "task1290_xsum_summarization",
        "task1572_samsum_summary",
        "task511_reddit_tifu_long_text_summarization",
    ],
    "information_extraction": [
        "task1510_evalution_relation_extraction",
        "task181_outcome_extraction",
        "task748_glucose_reverse_cause_event_detection",
    ],
    "generation": [
        "task1590_diplomacy_text_generation",
        "task1729_personachat_generate_next",
        "task639_multi_woz_user_utterance_generation",
    ],
    "classification": [
        "task1687_sentiment140_classification",
        "task363_sst2_polarity_classification",
        "task875_emotion_classification",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def immutable_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"refusing to overwrite frozen M6 artifact: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def freeze_folds(tasks: list[str]) -> list[dict[str, Any]]:
    category_folds = [
        {
            "scope": "four_category",
            "fold": index,
            "heldout_label": category,
            "heldout_tasks": sorted(CATEGORIES[category]),
        }
        for index, category in enumerate(CATEGORIES, start=1)
    ]
    permutation = list(tasks)
    random.Random(RANDOM_FOLD_SEED).shuffle(permutation)
    random_folds = [
        {
            "scope": "random12",
            "fold": index + 1,
            "heldout_label": f"random_fold_{index + 1}",
            "heldout_tasks": sorted(permutation[index * 3 : (index + 1) * 3]),
        }
        for index in range(5)
    ]
    folds = category_folds + random_folds
    for row in folds:
        row["discovery_tasks"] = sorted(set(tasks) - set(row["heldout_tasks"]))
        row["heldout_tasks_sha256"] = canonical_sha(row["heldout_tasks"])
        row["discovery_tasks_sha256"] = canonical_sha(row["discovery_tasks"])
        if len(row["heldout_tasks"]) != 3 or len(row["discovery_tasks"]) != 12:
            raise AssertionError("M6 fold must be exactly 12 discovery / 3 held-out tasks")
    for scope in ("four_category", "random12"):
        heldout = [task for row in folds if row["scope"] == scope for task in row["heldout_tasks"]]
        if sorted(heldout) != tasks:
            raise AssertionError(f"{scope} folds must hold out every task exactly once")
    return folds


def build_assignment(
    fold: dict[str, Any],
    q: np.ndarray,
    q_meta: list[dict[str, Any]],
    fisher_scores: np.ndarray,
    fisher_meta: list[dict[str, Any]],
    candidates: dict[str, Any],
    trial_id: str,
) -> tuple[Path, dict[str, Any]]:
    q_indices = [i for i, row in enumerate(q_meta) if str(row["task"]) in fold["discovery_tasks"]]
    f_indices = [i for i, row in enumerate(fisher_meta) if str(row["task"]) in fold["discovery_tasks"]]
    q_selected = q[q_indices]
    q_meta_selected = [q_meta[i] for i in q_indices]
    f_selected = fisher_scores[f_indices]
    f_meta_selected = [fisher_meta[i] for i in f_indices]
    if {str(row["task"]) for row in q_meta_selected + f_meta_selected} != set(fold["discovery_tasks"]):
        raise AssertionError("M6 discovery profile tasks drifted")
    if any(str(row["task"]) in fold["heldout_tasks"] for row in q_meta_selected + f_meta_selected):
        raise AssertionError("M6 held-out task leaked into discovery")
    fisher = np.square(f_selected.astype(np.float64)).mean(axis=0)
    trial = candidates["trials"][trial_id]
    labels: dict[str, list[int]] = {}
    for layer, layer_name in enumerate(candidates["layer_names"]):
        rho = float(candidates["rho"][trial_id][layer_name])
        natural, damped = profiles(q_selected, q_meta_selected, fisher, rho, layer)
        solver_seed = 6_000_000 + (0 if fold["scope"] == "four_category" else 1_000_000) + fold["fold"] * 10_000 + layer
        layer_labels = solve(natural, damped, int(trial["solver_restarts"]), solver_seed)
        counts = np.bincount(layer_labels, minlength=8)
        if counts.tolist() != [4] * 8:
            raise AssertionError(f"M6 capacity violation: {fold['scope']} fold {fold['fold']} layer {layer}")
        labels[layer_name] = layer_labels.astype(int).tolist()
    name = f"m6_qwen3_4b_{fold['scope']}_fold{fold['fold']}_assignment.json"
    path = ASSIGNMENT_DIR / name
    payload = {
        "schema_version": 1,
        "experiment_id": "M6-DISCOVERY-TRANSFER",
        "assignment_method": "tf",
        "epsilon_f": 0.1,
        "selected_trial_id": trial_id,
        "layer_names": candidates["layer_names"],
        "discovery_scope": fold["scope"],
        "fold": fold["fold"],
        "heldout_label": fold["heldout_label"],
        "heldout_tasks": fold["heldout_tasks"],
        "discovery_tasks": fold["discovery_tasks"],
        "heldout_tasks_sha256": fold["heldout_tasks_sha256"],
        "discovery_tasks_sha256": fold["discovery_tasks_sha256"],
        "source_h0_run_id": H0_RUN,
        "source_h0_config": str(H0_CONFIG.relative_to(ROOT)),
        "source_h0_config_sha256": digest(H0_CONFIG),
        "source_h0_candidates_sha256": digest(CANDIDATES),
        "assignment_sample_ids_sha256": canonical_sha(sorted(str(row["sample_id"]) for row in q_meta_selected)),
        "fisher_sample_ids_sha256": canonical_sha(sorted(str(row["sample_id"]) for row in f_meta_selected)),
        "tf": {"0.1": labels},
        "assertions": {
            "twelve_discovery_tasks": len(fold["discovery_tasks"]) == 12,
            "three_heldout_tasks": len(fold["heldout_tasks"]) == 3,
            "heldout_discovery_disjoint": not set(fold["heldout_tasks"]) & set(fold["discovery_tasks"]),
            "heldout_not_read_by_assignment": True,
            "exact_equal_capacity": True,
            "official_test_not_read_by_assignment": True,
        },
    }
    immutable_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path, payload


def build_config(fold: dict[str, Any], assignment_path: Path, setting: str) -> tuple[Path, dict[str, Any]]:
    seed = int(fold["fold"])
    source_path = ROOT / f"experiments/configs/m0_m1_final/m0_qwen3_4b_{setting}_tf_seed{seed}.yaml"
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    run_id = f"m6_qwen3_4b_{setting}_{fold['scope']}_fold{fold['fold']}_seed{seed}"
    config.update({
        "run_id": run_id,
        "experiment_id": "M6-DISCOVERY-TRANSFER",
        "todo_id": "M6",
        "m0_m1_final": False,
        "m6_discovery_transfer": True,
        "assignment_method": "tf",
        "tf_assignment_path": str(assignment_path.relative_to(ROOT)),
        "m6_assignment_sha256": digest(assignment_path),
        "tf_assignment_epsilon": 0.1,
        "discovery_scope": fold["scope"],
        "discovery_fold": int(fold["fold"]),
        "discovery_heldout_label": fold["heldout_label"],
        "discovery_tasks": fold["discovery_tasks"],
        "heldout_tasks": fold["heldout_tasks"],
        "discovery_tasks_sha256": fold["discovery_tasks_sha256"],
        "heldout_tasks_sha256": fold["heldout_tasks_sha256"],
        "source_m0_config": str(source_path.relative_to(ROOT)),
        "source_m0_config_sha256": digest(source_path),
        "selection_metric_source": "none_post_selection_control",
        "official_test_used_for_selection": False,
        "official_test_loaded": True,
    })
    config_path = CONFIG_DIR / f"{run_id}.yaml"
    immutable_write(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return config_path, config


def main() -> None:
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    base = json.loads(BASE_ASSIGNMENT.read_text(encoding="utf-8"))
    trial_id = str(base["selected_trial_id"])
    trial = candidates["trials"][trial_id]
    q_all, q_meta_all = merge_collection(H0_DIR / "collection", "assignment")
    f_all, f_meta_all = merge_collection(H0_DIR / "collection", "fisher")
    q, q_meta = select_prefix(q_all, q_meta_all, int(trial["assignment_probes_per_task"]))
    fisher_scores, fisher_meta = select_prefix(f_all, f_meta_all, int(trial["fisher_probes_per_task"]))
    tasks = sorted({str(row["task"]) for row in q_meta})
    if sorted(task for values in CATEGORIES.values() for task in values) != tasks:
        raise AssertionError("M6 category map does not exactly cover the frozen 15 tasks")
    folds = freeze_folds(tasks)
    split_payload = {
        "schema_version": 1,
        "event": "M6_DISCOVERY_TRANSFER_SPLITS_FROZEN",
        "random_fold_seed": RANDOM_FOLD_SEED,
        "category_map": CATEGORIES,
        "all_tasks": tasks,
        "all_tasks_sha256": canonical_sha(tasks),
        "folds": folds,
        "rule": "Five category folds and one seeded random five-fold partition; each fold uses 12 discovery tasks and holds out three tasks. SFT and final evaluation still use all 15 tasks.",
        "official_test_used_for_split_construction": False,
    }
    immutable_write(SPLIT_MANIFEST, json.dumps(split_payload, indent=2, sort_keys=True) + "\n")
    cells = []
    for fold in folds:
        assignment_path, assignment = build_assignment(
            fold, q, q_meta, fisher_scores, fisher_meta, candidates, trial_id
        )
        for setting in ("mixed", "sequential"):
            config_path, config = build_config(fold, assignment_path, setting)
            cells.append({
                "run_id": config["run_id"],
                "scope": fold["scope"],
                "fold": fold["fold"],
                "seed": config["seed"],
                "setting": setting,
                "heldout_label": fold["heldout_label"],
                "heldout_tasks": fold["heldout_tasks"],
                "discovery_tasks": fold["discovery_tasks"],
                "assignment": str(assignment_path.relative_to(ROOT)),
                "assignment_sha256": digest(assignment_path),
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": digest(config_path),
                "source_m0_config": config["source_m0_config"],
                "source_m0_config_sha256": config["source_m0_config_sha256"],
            })
    if len(cells) != 20:
        raise AssertionError("M6 grid must contain exactly 20 new SFT cells")
    grid = {
        "schema_version": 1,
        "event": "M6_DISCOVERY_TRANSFER_GRID_FROZEN",
        "status": "prepared_not_released",
        "source_split_manifest": str(SPLIT_MANIFEST.relative_to(ROOT)),
        "source_split_manifest_sha256": digest(SPLIT_MANIFEST),
        "source_h0_candidates_sha256": digest(CANDIDATES),
        "base_all15_assignment": str(BASE_ASSIGNMENT.relative_to(ROOT)),
        "base_all15_assignment_sha256": digest(BASE_ASSIGNMENT),
        "new_sft_cells": 20,
        "reuse": {
            "all15": "Reuse the five Qwen3-4B M0 TF final seeds per setting.",
            "gg": "Reuse the five Qwen3-4B M1 matched-GG final seeds per setting.",
        },
        "official_test_used_for_selection": False,
        "cells": cells,
    }
    immutable_write(GRID_MANIFEST, json.dumps(grid, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "splits": str(SPLIT_MANIFEST.relative_to(ROOT)),
        "splits_sha256": digest(SPLIT_MANIFEST),
        "grid": str(GRID_MANIFEST.relative_to(ROOT)),
        "grid_sha256": digest(GRID_MANIFEST),
        "assignments": len(folds),
        "cells": len(cells),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
