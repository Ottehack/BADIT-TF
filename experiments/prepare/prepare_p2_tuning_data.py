#!/usr/bin/env python3
"""Freeze a 15-task mixed tune-train/tune-validation order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import yaml


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_path = Path(config["tuning_split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    world_size = int(config["world_size"])

    train_records: list[dict] = []
    for task_name, task in sorted(split["tasks"].items()):
        source_file = Path(task["train_file"])
        source = json.loads(source_file.read_text(encoding="utf-8"))
        by_id = {
            f"{task_name}::{instance['id']}": index
            for index, instance in enumerate(source["Instances"])
        }
        for sample_id in task["tune_train_ids"]:
            train_records.append(
                {
                    "sample_id": sample_id,
                    "task": task_name,
                    "source_file": str(source_file),
                    "instance_index": by_id[sample_id],
                }
            )

    validation_records = list(split["roles"]["tune_validation"])
    train_ids = {row["sample_id"] for row in train_records}
    validation_ids = {row["sample_id"] for row in validation_records}
    if train_ids & validation_ids:
        raise AssertionError("mixed tune_train/tune_validation overlap")

    official_manifest_path = Path(
        config.get(
            "official_test_manifest",
            "experiments/configs/splits/p2_mixed_seed1_order.json",
        )
    )
    official_manifest = json.loads(
        official_manifest_path.read_text(encoding="utf-8")
    )
    official_test_ids = {
        row["sample_id"] for row in official_manifest["test_records"]
    }
    if train_ids & official_test_ids or validation_ids & official_test_ids:
        raise AssertionError("mixed tuning records overlap official test")

    random.Random(seed).shuffle(train_records)
    original_train_count = len(train_records)
    padded_count = math.ceil(original_train_count / world_size) * world_size
    padding = []
    for index in range(padded_count - original_train_count):
        duplicate = dict(train_records[index])
        duplicate["padding_repeat_of"] = duplicate["sample_id"]
        padding.append(duplicate)
    train_records.extend(padding)

    payload = {
        "schema_version": 1,
        "purpose": "mixed validation-only TF assignment comparison",
        "evaluation_role": "tune_validation",
        "seed": seed,
        "world_size": world_size,
        "source_split_path": str(split_path),
        "source_split_file_sha256": file_sha256(split_path),
        "official_test_manifest_path": str(official_manifest_path),
        "official_test_ids_sha256": official_manifest["test_ids_sha256"],
        "original_train_count": original_train_count,
        "padded_train_count": len(train_records),
        "padding_repeat_count": len(padding),
        "test_count": len(validation_records),
        "train_order": train_records,
        "test_records": validation_records,
        "train_unique_ids_sha256": canonical_json_sha256(sorted(train_ids)),
        "train_order_ids_sha256": canonical_json_sha256(
            [row["sample_id"] for row in train_records]
        ),
        "test_ids_sha256": canonical_json_sha256(sorted(validation_ids)),
        "assertions": {
            "tune_train_validation_disjoint": True,
            "tuning_official_test_disjoint": True,
            "official_test_loaded": False,
        },
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
                "padded_train_count": len(train_records),
                "validation_count": len(validation_records),
                "assertions": payload["assertions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
