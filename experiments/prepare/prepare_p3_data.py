#!/usr/bin/env python3
"""Freeze P3 task order 1 and every per-epoch distributed sample order."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import yaml

from badit_tf.training import canonical_json_sha256


def records(path: Path, split: str) -> list[dict]:
    payload = json.loads((path / f"{split}.json").read_text(encoding="utf-8"))
    output = []
    for index, instance in enumerate(payload["Instances"]):
        output.append(
            {
                "sample_id": f"{path.name}::{instance['id']}",
                "task": path.name,
                "source_file": str(path / f"{split}.json"),
                "instance_index": index,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    seed = int(config["seed"])
    world_size = int(config["world_size"])
    epochs = int(config["epochs_per_task"])

    available_tasks = [
        name
        for name in os.listdir(data_root)
        if (data_root / name).is_dir()
    ]
    if config.get("fixed_task_order"):
        # The paper's Table XIV is authoritative for sequential evaluation.
        # The upstream runner shuffled ``os.listdir()``, whose enumeration
        # order is filesystem-dependent and therefore cannot define a frozen
        # experimental order.
        all_tasks = list(config["fixed_task_order"])
        missing = sorted(set(all_tasks) - set(available_tasks))
        extra = sorted(set(available_tasks) - set(all_tasks))
        if missing or extra or len(all_tasks) != len(set(all_tasks)):
            raise RuntimeError(
                "fixed task order/data-root mismatch: "
                f"missing={missing}, extra={extra}, duplicate_count="
                f"{len(all_tasks) - len(set(all_tasks))}"
            )
        task_order_source = str(config.get("task_order_source", "fixed_config"))
    else:
        # Legacy P3 behavior is retained for immutable reproduction of the
        # original mechanism-only pilot.
        all_tasks = list(available_tasks)
        random.Random(seed).shuffle(all_tasks)
        task_order_source = "legacy_os_listdir_shuffle"
    selected = all_tasks[: int(config["pilot_task_count"])]
    expected = list(config["expected_first_tasks"])
    if selected != expected:
        raise RuntimeError(
            f"task order 1 drift: observed={selected}, expected={expected}"
        )

    tasks = []
    total_padding = 0
    for task_index, task in enumerate(selected):
        train = records(data_root / task, "train")
        test = records(data_root / task, "test")
        epoch_orders = []
        for epoch in range(epochs):
            order = [dict(item) for item in train]
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
                "task": task,
                "train_count": len(train),
                "test_count": len(test),
                "train_ids_sha256": canonical_json_sha256(
                    sorted(item["sample_id"] for item in train)
                ),
                "test_ids_sha256": canonical_json_sha256(
                    sorted(item["sample_id"] for item in test)
                ),
                "epoch_order_ids_sha256": [
                    canonical_json_sha256([item["sample_id"] for item in order])
                    for order in epoch_orders
                ],
                "epoch_orders": epoch_orders,
                "test_records": test,
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "P3 Qwen3-4B task-order-1 first-three sequential recovery pilot",
        "seed": seed,
        "world_size": world_size,
        "epochs_per_task": epochs,
        "task_order_1": all_tasks,
        "task_order_source": task_order_source,
        "selected_tasks": selected,
        "task_order_sha256": canonical_json_sha256(all_tasks),
        "selected_task_order_sha256": canonical_json_sha256(selected),
        "total_padding_repeats": total_padding,
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
                "epochs_per_task": epochs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
