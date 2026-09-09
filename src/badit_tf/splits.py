"""Deterministic, disjoint split manifests for BADIT-TF."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROLE_ORDER = (
    "tune_validation",
    "assignment",
    "fisher",
    "damping_validation",
    "fidelity",
)


@dataclass(frozen=True)
class SplitCounts:
    assignment: int = 16
    fisher: int = 16
    damping_validation: int = 8
    fidelity: int = 16
    tune_validation_fraction: float = 0.10
    tune_validation_cap: int = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("Instances"), list):
        raise ValueError(f"{path} is not a SuperNI task JSON")
    return payload, payload["Instances"]


def stable_sample_id(task: str, index: int, instance: dict[str, Any]) -> str:
    upstream_id = instance.get("id")
    if upstream_id:
        return f"{task}::{upstream_id}"
    canonical = json.dumps(
        instance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{task}::row-{index}::{hashlib.sha256(canonical).hexdigest()[:16]}"


def _task_seed(seed: int, task: str) -> int:
    value = hashlib.sha256(f"{seed}:{task}".encode("utf-8")).digest()[:8]
    return int.from_bytes(value, "big", signed=False)


def build_split_manifest(
    data_root: Path,
    *,
    seed: int = 1,
    counts: SplitCounts = SplitCounts(),
) -> dict[str, Any]:
    task_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    if len(task_dirs) != 15:
        raise ValueError(f"expected 15 task directories, found {len(task_dirs)}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "source_root": str(data_root),
        "counts": {
            "assignment": counts.assignment,
            "fisher": counts.fisher,
            "damping_validation": counts.damping_validation,
            "fidelity": counts.fidelity,
            "tune_validation_fraction": counts.tune_validation_fraction,
            "tune_validation_cap": counts.tune_validation_cap,
        },
        "roles": {role: [] for role in ROLE_ORDER},
        "tasks": {},
    }
    all_ids: set[str] = set()

    for task_dir in task_dirs:
        train_path = task_dir / "train.json"
        _, instances = _load_task(train_path)
        task = task_dir.name
        ids = [
            stable_sample_id(task, index, instance)
            for index, instance in enumerate(instances)
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{task} contains duplicate stable sample IDs")
        rng = np.random.default_rng(_task_seed(seed, task))
        order = rng.permutation(len(instances)).tolist()
        tune_count = max(
            1,
            min(
                int(counts.tune_validation_cap),
                int(math.floor(len(instances) * counts.tune_validation_fraction)),
            ),
        )
        required = (
            tune_count
            + counts.assignment
            + counts.fisher
            + counts.damping_validation
            + counts.fidelity
        )
        if len(instances) < required:
            raise ValueError(
                f"{task} has {len(instances)} train instances but needs {required}"
            )

        cursor = 0
        role_sizes = {
            "tune_validation": tune_count,
            "assignment": counts.assignment,
            "fisher": counts.fisher,
            "damping_validation": counts.damping_validation,
            "fidelity": counts.fidelity,
        }
        task_roles: dict[str, list[str]] = {}
        for role in ROLE_ORDER:
            selected_indices = order[cursor : cursor + role_sizes[role]]
            cursor += role_sizes[role]
            selected = [
                {
                    "sample_id": ids[index],
                    "task": task,
                    "source_file": str(train_path),
                    "instance_index": int(index),
                }
                for index in selected_indices
            ]
            selected_ids = [item["sample_id"] for item in selected]
            overlap = all_ids.intersection(selected_ids)
            if overlap:
                raise AssertionError(f"cross-role duplicate IDs: {sorted(overlap)[:5]}")
            all_ids.update(selected_ids)
            manifest["roles"][role].extend(selected)
            task_roles[role] = selected_ids

        remaining = order[cursor:]
        manifest["tasks"][task] = {
            "train_file": str(train_path),
            "train_sha256": sha256_file(train_path),
            "train_instances": len(instances),
            "dev_instances": len(_load_task(task_dir / "dev.json")[1]),
            "test_instances": len(_load_task(task_dir / "test.json")[1]),
            "roles": task_roles,
            "tune_train_ids": [ids[index] for index in remaining],
        }

    role_sets = {
        role: {item["sample_id"] for item in rows}
        for role, rows in manifest["roles"].items()
    }
    for index, left in enumerate(ROLE_ORDER):
        for right in ROLE_ORDER[index + 1 :]:
            if role_sets[left].intersection(role_sets[right]):
                raise AssertionError(f"split leakage between {left} and {right}")
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def write_split_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_role_instances(
    manifest: dict[str, Any], role: str
) -> list[dict[str, Any]]:
    if role not in manifest.get("roles", {}):
        raise KeyError(f"unknown role {role}")
    cache: dict[str, list[dict[str, Any]]] = {}
    resolved = []
    for record in manifest["roles"][role]:
        path = record["source_file"]
        if path not in cache:
            cache[path] = _load_task(Path(path))[1]
        instance = cache[path][record["instance_index"]]
        actual_id = stable_sample_id(
            record["task"], record["instance_index"], instance
        )
        if actual_id != record["sample_id"]:
            raise ValueError(f"sample ID drift for {record['sample_id']}")
        resolved.append({**record, "instance": instance})
    return resolved

