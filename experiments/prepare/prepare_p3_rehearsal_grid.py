#!/usr/bin/env python3
"""Expand and lock the preregistered P3-R9 functional rehearsal grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from badit_tf.training import canonical_json_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.grid.read_text(encoding="utf-8"))
    base = dict(spec["base_config"])
    outputs = []
    for cell in spec["cells"]:
        payload = {
            **base,
            "experiment_id": cell["experiment_id"],
            "rehearsal_coefficient": float(cell["coefficient"]),
        }
        output = Path(cell["config_path"])
        rendered = yaml.safe_dump(payload, sort_keys=False)
        if output.exists():
            if output.read_text(encoding="utf-8") != rendered:
                raise RuntimeError(f"refusing to overwrite changed config {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
        outputs.append(
            {
                "run_id": cell["run_id"],
                "coefficient": float(cell["coefficient"]),
                "worker": cell["worker"],
                "config_path": str(output),
                "config_sha256": file_sha256(output),
            }
        )
    lock = {
        "schema_version": 1,
        "grid_path": str(args.grid),
        "grid_file_sha256": file_sha256(args.grid),
        "grid_content_sha256": canonical_json_sha256(spec),
        "selection_rule": spec["selection_rule"],
        "cells": outputs,
        "official_test_loaded": False,
    }
    lock["lock_content_sha256"] = canonical_json_sha256(lock)
    lock_path = Path(spec["lock_path"])
    rendered_lock = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if lock_path.exists():
        if lock_path.read_text(encoding="utf-8") != rendered_lock:
            raise RuntimeError(f"refusing to overwrite changed lock {lock_path}")
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(rendered_lock, encoding="utf-8")
    print(rendered_lock, end="")


if __name__ == "__main__":
    main()
