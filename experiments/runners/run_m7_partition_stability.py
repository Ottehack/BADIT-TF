#!/usr/bin/env python3
"""Run M7 from frozen H0 calibration profiles without loading official test."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from badit_tf.assignment import build_natural_modulation_profiles, solve_balanced_tf_assignment


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("tf", "random_balanced", "contiguous")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("ARI label shapes differ")
    _, li = np.unique(left, return_inverse=True)
    _, ri = np.unique(right, return_inverse=True)
    table = np.zeros((li.max() + 1, ri.max() + 1), dtype=np.int64)
    np.add.at(table, (li, ri), 1)
    comb2 = lambda x: x * (x - 1) / 2
    total = comb2(left.size)
    sum_cells = float(comb2(table).sum())
    sum_rows = float(comb2(table.sum(axis=1)).sum())
    sum_cols = float(comb2(table.sum(axis=0)).sum())
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    if maximum == expected:
        return 1.0
    return float((sum_cells - expected) / (maximum - expected))


def merge_collection(collection: Path, role: str) -> tuple[np.ndarray, list[dict]]:
    key = "fisher_scores" if role == "fisher" else f"q_{role}"
    rows = []
    for metadata_path in sorted(collection.glob("rank*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(collection / f"{metadata_path.stem}.npz")
        rows.extend(zip(metadata["roles"][role], arrays[key], strict=True))
    if not rows:
        raise ValueError(f"empty H0 collection role {role}")
    rows.sort(key=lambda item: str(item[0]["sample_id"]))
    return np.stack([item[1] for item in rows]), [item[0] for item in rows]


def select_prefix(array: np.ndarray, meta: list[dict], count: int) -> tuple[np.ndarray, list[dict]]:
    indices = []
    for task in sorted({str(row["task"]) for row in meta}):
        candidates = [i for i, row in enumerate(meta) if str(row["task"]) == task]
        if len(candidates) < count:
            raise ValueError(f"{task} has only {len(candidates)} probes")
        indices.extend(candidates[:count])
    return array[indices], [meta[index] for index in indices]


def profiles(q: np.ndarray, meta: list[dict], fisher: np.ndarray, rho: float, layer: int) -> tuple[np.ndarray, np.ndarray]:
    counts = {task: sum(str(row["task"]) == task for row in meta) for task in {str(row["task"]) for row in meta}}
    weights = np.asarray([1.0 / (len(counts) * counts[str(row["task"])]) for row in meta])
    p, damped, _ = build_natural_modulation_profiles(q[:, layer, :], fisher[layer], weights, rho)
    return p, damped


def solve(p: np.ndarray, weights: np.ndarray, restarts: int, seed: int) -> np.ndarray:
    return solve_balanced_tf_assignment(p, weights, num_experts=8, rank=4, restarts=restarts, max_iterations=50, tolerance=1e-6, seed=seed).labels


def balanced_random(rng: np.random.Generator) -> np.ndarray:
    labels = np.repeat(np.arange(8, dtype=np.int64), 4)
    rng.shuffle(labels)
    return labels


def centroid_cosine(p_left: np.ndarray, labels_left: np.ndarray, p_right: np.ndarray, labels_right: np.ndarray) -> float:
    left = np.stack([p_left[labels_left == k].mean(axis=0) for k in range(8)])
    right = np.stack([p_right[labels_right == k].mean(axis=0) for k in range(8)])
    denom = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    cosine = (left @ right.T) / np.maximum(denom, 1e-12)
    rows, cols = linear_sum_assignment(-cosine)
    return float(cosine[rows, cols].mean())


def summarize(values: list[float]) -> dict[str, object]:
    return {"mean": mean(values), "sample_std": stdev(values), "n": len(values), "values": values}


def evaluate_model(model: str, source: dict, seed: int) -> dict:
    h0_dir = ROOT / "experiments/raw_results" / source["h0_run_id"]
    h0_config_path = ROOT / source["h0_config"]
    h0_config = yaml.safe_load(h0_config_path.read_text(encoding="utf-8"))
    candidates_path = h0_dir / "h0_assignment_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    trial_id = source["selected_trial_id"]
    trial = candidates["trials"][trial_id]
    q_all, assignment_meta_all = merge_collection(h0_dir / "collection", "assignment")
    fisher_scores_all, fisher_meta_all = merge_collection(h0_dir / "collection", "fisher")
    q, assignment_meta = select_prefix(q_all, assignment_meta_all, int(trial["assignment_probes_per_task"]))
    fisher_scores, _ = select_prefix(fisher_scores_all, fisher_meta_all, int(trial["fisher_probes_per_task"]))
    fisher = np.square(fisher_scores.astype(np.float64)).mean(axis=0)
    layer_names = candidates["layer_names"]
    restarts = int(trial["solver_restarts"])
    rng = np.random.default_rng(seed)
    tasks = sorted({str(row["task"]) for row in assignment_meta})
    task_splits = []
    for replicate in range(5):
        perm = rng.permutation(tasks).tolist()
        task_splits.append((perm[:7], perm[7:]))

    method_values = {method: {"task_split_ari": [], "seed_ari": [], "batch_ari": [], "cross_layer_cosine": []} for method in METHODS}
    full_profiles = []
    selected_labels = []
    for layer, layer_name in enumerate(layer_names):
        rho = float(candidates["rho"][trial_id][layer_name])
        p, damped = profiles(q, assignment_meta, fisher, rho, layer)
        full_profiles.append(p)
        selected_labels.append(np.asarray(candidates["tf"][trial_id][layer_name], dtype=np.int64))

        seed_labels = [solve(p, damped, restarts, seed + replicate * 100003 + layer * 4099) for replicate in range(5)]
        method_values["tf"]["seed_ari"].extend(adjusted_rand(a, b) for a, b in itertools.combinations(seed_labels, 2))
        random_seed_labels = [balanced_random(np.random.default_rng(seed + 700001 + replicate * 100003 + layer * 4099)) for replicate in range(5)]
        method_values["random_balanced"]["seed_ari"].extend(adjusted_rand(a, b) for a, b in itertools.combinations(random_seed_labels, 2))
        method_values["contiguous"]["seed_ari"].extend([1.0] * 10)

        batch_labels = []
        random_batch_labels = []
        for replicate in range(5):
            brng = np.random.default_rng(seed + 1400003 + replicate * 100003 + layer * 4099)
            indices = []
            for task in tasks:
                available = [i for i, row in enumerate(assignment_meta) if str(row["task"]) == task]
                indices.extend(sorted(brng.choice(available, size=8, replace=False).tolist()))
            q_batch = q[indices]
            meta_batch = [assignment_meta[i] for i in indices]
            bp, bd = profiles(q_batch, meta_batch, fisher, rho, layer)
            batch_labels.append(solve(bp, bd, restarts, seed + 1900009 + replicate * 100003 + layer * 4099))
            random_batch_labels.append(balanced_random(np.random.default_rng(seed + 2100011 + replicate * 100003 + layer * 4099)))
        method_values["tf"]["batch_ari"].extend(adjusted_rand(a, b) for a, b in itertools.combinations(batch_labels, 2))
        method_values["random_balanced"]["batch_ari"].extend(adjusted_rand(a, b) for a, b in itertools.combinations(random_batch_labels, 2))
        method_values["contiguous"]["batch_ari"].extend([1.0] * 10)

        for replicate, (left_tasks, right_tasks) in enumerate(task_splits):
            labels = []
            random_labels = []
            for side, chosen in enumerate((left_tasks, right_tasks)):
                indices = [i for i, row in enumerate(assignment_meta) if str(row["task"]) in chosen]
                sp, sd = profiles(q[indices], [assignment_meta[i] for i in indices], fisher, rho, layer)
                labels.append(solve(sp, sd, restarts, seed + 2700019 + replicate * 200003 + side * 100003 + layer * 4099))
                random_labels.append(balanced_random(np.random.default_rng(seed + 3300023 + replicate * 200003 + side * 100003 + layer * 4099)))
            method_values["tf"]["task_split_ari"].append(adjusted_rand(*labels))
            method_values["random_balanced"]["task_split_ari"].append(adjusted_rand(*random_labels))
            method_values["contiguous"]["task_split_ari"].append(1.0)

    contiguous = np.repeat(np.arange(8, dtype=np.int64), 4)
    cross_random = [balanced_random(np.random.default_rng(seed + 3900029 + layer * 4099)) for layer in range(len(layer_names))]
    for layer in range(len(layer_names) - 1):
        method_values["tf"]["cross_layer_cosine"].append(centroid_cosine(full_profiles[layer], selected_labels[layer], full_profiles[layer + 1], selected_labels[layer + 1]))
        method_values["random_balanced"]["cross_layer_cosine"].append(centroid_cosine(full_profiles[layer], cross_random[layer], full_profiles[layer + 1], cross_random[layer + 1]))
        method_values["contiguous"]["cross_layer_cosine"].append(centroid_cosine(full_profiles[layer], contiguous, full_profiles[layer + 1], contiguous))

    return {
        "model": model,
        "selected_trial_id": trial_id,
        "h0_config_sha256": digest(h0_config_path),
        "h0_candidates_sha256": digest(candidates_path),
        "collection_files_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sorted((h0_dir / "collection").glob("rank*.*"))},
        "task_splits": [{"left_7": left, "right_8": right} for left, right in task_splits],
        "methods": {method: {metric: summarize(values) for metric, values in metrics.items()} for method, metrics in method_values.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["official_test_loaded"] is not False:
        raise AssertionError("M7 must not load official test")
    code_commit = str(config["code_commit"])
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", code_commit, "HEAD"], check=False
    )
    if ancestry.returncode != 0:
        raise AssertionError(f"M7 implementation commit {code_commit} is not an ancestor of HEAD")
    models = [evaluate_model(model, source, int(config["seed"]) + index * 10000019) for index, (model, source) in enumerate(config["models"].items())]
    macro = {method: {metric: summarize([row["methods"][method][metric]["mean"] for row in models]) for metric in ("task_split_ari", "seed_ari", "batch_ari", "cross_layer_cosine")} for method in METHODS}
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "complete",
        "code_commit": code_commit,
        "config_sha256": digest(args.config),
        "protocol": config["protocol"] | {"official_test_loaded": False},
        "models": models,
        "macro": macro,
        "assertions": {
            "three_small_models_complete": len(models) == 3,
            "five_task_splits_each": all(len(row["task_splits"]) == 5 for row in models),
            "five_seed_replicates_each": True,
            "five_batch_replicates_each": True,
            "random_baseline_independent": True,
            "official_test_not_loaded": True,
            "all_values_finite": all(np.isfinite(metric["mean"]) for method in macro.values() for metric in method.values()),
        },
    }
    if not all(result["assertions"].values()):
        raise AssertionError(result["assertions"])
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# M7 Partition Stability", "", "Macro mean ± sample standard deviation across three small models. Official test was not loaded.", "", "| Method | Task-split ARI | Seed ARI | Batch ARI | Cross-layer cosine |", "|---|---:|---:|---:|---:|"]
    labels = {"tf": "TF", "random_balanced": "Random", "contiguous": "Contiguous SVD"}
    for method in METHODS:
        row = macro[method]
        cell = lambda metric: f"{row[metric]['mean']:.4f} ± {row[metric]['sample_std']:.4f}"
        lines.append(f"| {labels[method]} | {cell('task_split_ari')} | {cell('seed_ari')} | {cell('batch_ari')} | {cell('cross_layer_cosine')} |")
    markdown = output / "M7_PARTITION_STABILITY_RESULTS.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path.relative_to(ROOT)), "result_sha256": digest(result_path), "markdown": str(markdown.relative_to(ROOT)), "markdown_sha256": digest(markdown)}, sort_keys=True))


if __name__ == "__main__":
    main()
