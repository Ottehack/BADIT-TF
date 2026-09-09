#!/usr/bin/env python3
"""Aggregate distributed M7-DOG sufficient statistics into Table VIII metrics."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from badit_tf.m2 import canonical_gg_assignment
from run_m7_partition_stability import adjusted_rand, centroid_cosine


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, object]:
    return {"mean": mean(values), "sample_std": stdev(values), "n": len(values), "values": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    partials = ROOT / config["output_dir"] / "partials"
    metadata_paths = sorted(partials.glob("rank*.json"))
    if len(metadata_paths) != int(config["world_size"]):
        raise AssertionError(f"expected {config['world_size']} M7-DOG ranks, got {len(metadata_paths)}")
    metas = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    first = metas[0]
    for meta in metas[1:]:
        for key in ("world", "model", "layer_names", "tasks", "batch_sample_ids", "task_splits", "source_config_sha256", "split_manifest_sha256"):
            if meta[key] != first[key]:
                raise AssertionError(f"M7-DOG rank metadata drift: {key}")
    ids = [sample_id for meta in metas for sample_id in meta["processed_sample_ids"]]
    expected = int(config["assignment_probes_per_task"]) * len(first["tasks"])
    if len(ids) != expected or len(set(ids)) != expected:
        raise AssertionError("M7-DOG sample coverage mismatch")
    expected_batch = int(config["batch_probes_per_task"]) * len(first["tasks"])
    if any(
        sum(int(meta["batch_counts"][layer][replicate]) for meta in metas) != expected_batch
        for layer in range(len(first["layer_names"]))
        for replicate in range(int(config["batch_replicates"]))
    ):
        raise AssertionError("M7-DOG batch coverage mismatch")

    arrays = [np.load(partials / f"rank{rank}.npz") for rank in range(int(config["world_size"]))]
    metrics = {key: [] for key in ("task_split_ari", "seed_ari", "batch_ari", "cross_layer_cosine")}
    per_layer = []
    previous_profile = previous_labels = None
    for layer_index, layer_name in enumerate(first["layer_names"]):
        full_count = sum(int(meta["full_count"]) for meta in metas)
        full = sum(array[f"full_sum_{layer_index}"] for array in arrays) / float(full_count)
        canonical_labels, canonical_audit = canonical_gg_assignment(full, num_experts=8, rank=4, seed=layer_index)

        seed_labels = [canonical_gg_assignment(full, num_experts=8, rank=4, seed=int(config["seed"]) + replicate * 100003 + layer_index * 4099)[0] for replicate in range(int(config["seed_replicates"]))]
        seed_values = [adjusted_rand(left, right) for left, right in itertools.combinations(seed_labels, 2)]
        metrics["seed_ari"].extend(seed_values)

        batch_labels = []
        for replicate in range(int(config["batch_replicates"])):
            count = sum(int(meta["batch_counts"][layer_index][replicate]) for meta in metas)
            profile = sum(array[f"batch_{replicate}_sum_{layer_index}"] for array in arrays) / float(count)
            batch_labels.append(canonical_gg_assignment(
                profile, num_experts=8, rank=4,
                seed=int(config["seed"]) + 1_900_009 + replicate * 100_003 + layer_index * 4_099,
            )[0])
        batch_values = [adjusted_rand(left, right) for left, right in itertools.combinations(batch_labels, 2)]
        metrics["batch_ari"].extend(batch_values)

        task_values = []
        tasks = list(first["tasks"])
        task_index = {task: index for index, task in enumerate(tasks)}
        for replicate, split in enumerate(first["task_splits"]):
            labels = []
            for side, selected_tasks in enumerate((split["left_7"], split["right_8"])):
                count = sum(sum(int(meta["task_counts"][task]) for task in selected_tasks) for meta in metas)
                profile = sum(
                    sum(array[f"task_{task_index[task]}_sum_{layer_index}"] for task in selected_tasks)
                    for array in arrays
                ) / float(count)
                labels.append(canonical_gg_assignment(
                    profile, num_experts=8, rank=4,
                    seed=int(config["seed"]) + 2_700_019 + replicate * 200_003 + side * 100_003 + layer_index * 4_099,
                )[0])
            task_values.append(adjusted_rand(*labels))
        metrics["task_split_ari"].extend(task_values)

        cross = None
        if previous_profile is not None:
            cross = centroid_cosine(previous_profile, previous_labels, full, canonical_labels)
            metrics["cross_layer_cosine"].append(cross)
        per_layer.append({
            "layer": layer_name, "canonical_labels": canonical_labels.tolist(),
            "canonical_solver_audit": canonical_audit,
            "task_split_ari": task_values, "seed_ari": seed_values,
            "batch_ari": batch_values, "cross_layer_cosine_from_previous": cross,
        })
        previous_profile, previous_labels = full, canonical_labels
    for array in arrays:
        array.close()
    result = {
        "schema_version": 1, "experiment_id": config["experiment_id"],
        "run_id": config["run_id"], "status": "complete", "model": config["model"],
        "code_commit": config["code_commit"], "config_sha256": digest(args.config),
        "protocol": config["protocol"] | {"official_test_loaded": False, "downstream_test_loaded": False},
        "metrics": {key: summary(values) for key, values in metrics.items()},
        "per_layer": per_layer,
        "source_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}
            for path in metadata_paths
        ],
        "assertions": {
            "eight_ranks": len(metas) == 8, "all_assignment_samples_exact_once": len(ids) == len(set(ids)) == expected,
            "five_task_splits": len(first["task_splits"]) == 5, "five_seed_replicates": int(config["seed_replicates"]) == 5,
            "five_batch_replicates": int(config["batch_replicates"]) == 5,
            "all_values_finite": all(np.isfinite(value) for values in metrics.values() for value in values),
            "official_test_not_loaded": True,
        },
    }
    if not all(result["assertions"].values()):
        raise AssertionError(result["assertions"])
    output = ROOT / config["output_dir"]
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row = result["metrics"]
    lines = [
        f"# M7 DOG Stability — {config['model']}", "",
        "Frozen calibration-assignment data only; official/downstream test closed.", "",
        "| Task-split ARI | Seed ARI | Batch ARI | Cross-layer cosine |", "|---:|---:|---:|---:|",
        "| " + " | ".join(f"{row[key]['mean']:.4f} ± {row[key]['sample_std']:.4f}" for key in ("task_split_ari", "seed_ari", "batch_ari", "cross_layer_cosine")) + " |",
    ]
    markdown = output / "M7_DOG_STABILITY_RESULTS.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path.relative_to(ROOT)), "result_sha256": digest(result_path), "markdown_sha256": digest(markdown)}, sort_keys=True))


if __name__ == "__main__":
    main()
