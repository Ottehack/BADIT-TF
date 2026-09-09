#!/usr/bin/env python3
"""Freeze the exact mixed-SFT order and final-test IDs used by P2."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from badit_tf.training import canonical_json_sha256


def records(data_root: Path, split: str) -> list[dict]:
    output = []
    for path in sorted(data_root.glob(f"*/{split}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = path.parent.name
        for index, instance in enumerate(payload["Instances"]):
            output.append(
                {
                    "sample_id": f"{task}::{instance['id']}",
                    "task": task,
                    "source_file": str(path),
                    "instance_index": index,
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    world_size = int(config["world_size"])
    train = records(Path(config["data_root"]), "train")
    test = records(Path(config["data_root"]), "test")
    random.Random(seed).shuffle(train)
    original_train_count = len(train)
    padded_count = math.ceil(original_train_count / world_size) * world_size
    padding = []
    for index in range(padded_count - original_train_count):
        duplicate = dict(train[index])
        duplicate["padding_repeat_of"] = duplicate["sample_id"]
        padding.append(duplicate)
    train.extend(padding)
    payload = {
        "schema_version": 1,
        "purpose": "P2 matched TF/GG mixed one-epoch order",
        "seed": seed,
        "world_size": world_size,
        "original_train_count": original_train_count,
        "padded_train_count": len(train),
        "padding_repeat_count": len(padding),
        "test_count": len(test),
        "train_order": train,
        "test_records": test,
        "train_unique_ids_sha256": canonical_json_sha256(
            sorted({item["sample_id"] for item in train})
        ),
        "train_order_ids_sha256": canonical_json_sha256(
            [item["sample_id"] for item in train]
        ),
        "test_ids_sha256": canonical_json_sha256(
            sorted(item["sample_id"] for item in test)
        ),
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    output = Path(config["order_manifest"])
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
                "original_train_count": original_train_count,
                "padded_train_count": len(train),
                "test_count": len(test),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
