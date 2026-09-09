#!/usr/bin/env python3
"""Freeze tune-train/tune-validation data for sequential H1 trials."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from badit_tf.training import canonical_json_sha256, file_sha256


def records(path: Path, split: str) -> list[dict]:
    payload = json.loads((path / f"{split}.json").read_text(encoding="utf-8"))
    return [
        {
            "sample_id": f"{path.name}::{instance['id']}",
            "task": path.name,
            "source_file": str(path / f"{split}.json"),
            "instance_index": index,
        }
        for index, instance in enumerate(payload["Instances"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    split_path = Path(config["tuning_split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    world_size = int(config["world_size"])
    epochs = int(config["epochs_per_task"])
    selected = list(config["fixed_task_order"][: int(config["pilot_task_count"])])
    if selected != list(config["expected_first_tasks"]):
        raise RuntimeError("frozen first-task order mismatch")

    tasks = []
    total_padding = 0
    all_tune_train: set[str] = set()
    all_validation: set[str] = set()
    all_official_test: set[str] = set()
    for task_index, task_name in enumerate(selected):
        task_split = split["tasks"][task_name]
        train_records = records(data_root / task_name, "train")
        official_test = records(data_root / task_name, "test")
        by_id = {record["sample_id"]: record for record in train_records}
        tune_train_ids = list(task_split["tune_train_ids"])
        validation_ids = list(task_split["roles"]["tune_validation"])
        unknown = sorted((set(tune_train_ids) | set(validation_ids)) - set(by_id))
        if unknown:
            raise RuntimeError(f"unknown frozen IDs for {task_name}: {unknown[:3]}")
        if set(tune_train_ids) & set(validation_ids):
            raise RuntimeError(f"tune train/validation overlap for {task_name}")
        tune_train = [dict(by_id[sample_id]) for sample_id in tune_train_ids]
        validation = [dict(by_id[sample_id]) for sample_id in validation_ids]
        official_test_ids = {record["sample_id"] for record in official_test}
        if (set(tune_train_ids) | set(validation_ids)) & official_test_ids:
            raise RuntimeError(f"train/official-test overlap for {task_name}")
        all_tune_train.update(tune_train_ids)
        all_validation.update(validation_ids)
        all_official_test.update(official_test_ids)

        epoch_orders = []
        for epoch in range(epochs):
            order = [dict(record) for record in tune_train]
            random.Random(seed * 100_000 + task_index * 1_000 + epoch).shuffle(order)
            padded = math.ceil(len(order) / world_size) * world_size
            for index in range(padded - len(order)):
                duplicate = dict(order[index])
                duplicate["padding_repeat_of"] = duplicate["sample_id"]
                order.append(duplicate)
                total_padding += 1
            epoch_orders.append(order)
        tasks.append(
            {
                "task_index": task_index,
                "task": task_name,
                "train_count": len(tune_train),
                "evaluation_count": len(validation),
                "tune_train_ids_sha256": canonical_json_sha256(
                    sorted(tune_train_ids)
                ),
                "tune_validation_ids_sha256": canonical_json_sha256(
                    sorted(validation_ids)
                ),
                "official_test_ids_sha256": canonical_json_sha256(
                    sorted(official_test_ids)
                ),
                "epoch_order_ids_sha256": [
                    canonical_json_sha256([record["sample_id"] for record in order])
                    for order in epoch_orders
                ],
                "epoch_orders": epoch_orders,
                "evaluation_records": validation,
            }
        )
    if all_tune_train & all_validation:
        raise RuntimeError("global tune train/validation overlap")
    if (all_tune_train | all_validation) & all_official_test:
        raise RuntimeError("global tuning/official-test overlap")
    payload = {
        "schema_version": 1,
        "purpose": "P3-R3/H1 sequential validation-only tuning",
        "seed": seed,
        "world_size": world_size,
        "epochs_per_task": epochs,
        "evaluation_role": "tune_validation",
        "official_test_loaded_by_training_runner": False,
        "tuning_split_manifest": str(split_path),
        "tuning_split_file_sha256": file_sha256(split_path),
        "tuning_split_content_sha256": split["manifest_sha256"],
        "task_order_source": str(config["task_order_source"]),
        "selected_tasks": selected,
        "selected_task_order_sha256": canonical_json_sha256(selected),
        "task_order_sha256": canonical_json_sha256(config["fixed_task_order"]),
        "total_padding_repeats": total_padding,
        "global_hashes": {
            "tune_train_ids_sha256": canonical_json_sha256(sorted(all_tune_train)),
            "tune_validation_ids_sha256": canonical_json_sha256(
                sorted(all_validation)
            ),
            "official_test_ids_sha256": canonical_json_sha256(
                sorted(all_official_test)
            ),
        },
        "assertions": {
            "tune_train_validation_disjoint": not bool(
                all_tune_train & all_validation
            ),
            "tuning_official_test_disjoint": not bool(
                (all_tune_train | all_validation) & all_official_test
            ),
        },
        "tasks": tasks,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    output = Path(config["sequential_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed manifest {output}")
    else:
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "path": str(output),
                "manifest_sha256": payload["manifest_sha256"],
                "selected_tasks": selected,
                "train_counts": [task["train_count"] for task in tasks],
                "validation_counts": [
                    task["evaluation_count"] for task in tasks
                ],
                "assertions": payload["assertions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
