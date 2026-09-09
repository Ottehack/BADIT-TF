#!/usr/bin/env python3
"""Merge trained-checkpoint FP32 q/Fisher shards for M3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from badit_tf.m2 import task_prefix_indices


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_role(collection: Path, role: str, expected_ids: list[str]):
    key = "fisher_scores" if role == "fisher" else f"q_{role}"
    rows = {}
    layer_names = None
    for metadata_path in sorted(collection.glob("rank*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if layer_names is None:
            layer_names = list(metadata["layer_names"])
        elif layer_names != list(metadata["layer_names"]):
            raise ValueError("M3 collection layer order drift across ranks")
        arrays = np.load(collection / f"{metadata_path.stem}.npz")
        for item, array in zip(metadata["roles"][role], arrays[key], strict=True):
            sample_id = str(item["sample_id"])
            if sample_id in rows:
                raise ValueError(f"duplicate M3 {role} ID: {sample_id}")
            rows[sample_id] = (array, item)
    if len(list(collection.glob("rank*.json"))) != 8 or len(list(collection.glob("rank*.npz"))) != 8:
        raise ValueError("M3 requires exactly eight JSON and eight NPZ collection shards")
    missing = [sample_id for sample_id in expected_ids if sample_id not in rows]
    if missing:
        raise ValueError(f"missing M3 {role} samples: {missing[:8]}")
    ordered = [rows[sample_id] for sample_id in expected_ids]
    return np.stack([row[0] for row in ordered]), [row[1] for row in ordered], layer_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"):
        raise AssertionError("M3 profile preparation cannot load final/downstream test")
    manifest_path = Path(config["split_manifest"])
    if sha256(manifest_path) != str(config["split_manifest_sha256"]):
        raise AssertionError("M3 split manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {}
    for role, count_key in (("fisher", "fisher_probes_per_task"), ("fidelity", "fidelity_probes_per_task")):
        rows = manifest["roles"][role]
        indices = task_prefix_indices(rows, int(config[count_key]))
        expected[role] = [str(rows[index]["sample_id"]) for index in indices]
    output = Path(config["output_dir"])
    fisher_scores, fisher_meta, fisher_layers = merge_role(output / "collection", "fisher", expected["fisher"])
    q_fidelity, fidelity_meta, fidelity_layers = merge_role(output / "collection", "fidelity", expected["fidelity"])
    if fisher_layers != fidelity_layers:
        raise ValueError("M3 Fisher/fidelity layer order mismatch")
    fisher = np.mean(np.square(fisher_scores.astype(np.float64)), axis=0)
    if not np.all(np.isfinite(fisher)) or np.any(fisher < 0):
        raise FloatingPointError("M3 Fisher is invalid")
    np.savez(
        output / "m3_profiles.npz",
        q_fidelity=q_fidelity.astype(np.float32),
        fisher=fisher.astype(np.float64),
    )
    metadata = {
        "schema_version": 1,
        "layer_names": fisher_layers,
        "fisher": fisher_meta,
        "fidelity": fidelity_meta,
        "split_manifest_sha256": sha256(manifest_path),
        "trained_checkpoint": config["initial_bank_checkpoint"],
        "trained_checkpoint_sha256": config["initial_bank_checkpoint_sha256"],
    }
    (output / "m3_profile_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "m3_profiles.npz")


if __name__ == "__main__":
    main()
