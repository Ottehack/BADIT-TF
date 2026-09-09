#!/usr/bin/env python3
"""Freeze the M5 Contiguous-SVD formal submatrix from canonical M0 configs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "experiments/configs/m5/formal_contiguous_v1"
GRID = ROOT / "experiments/materials/m5_contiguous_formal_grid.json"
MODELS = {
    "Qwen3-4B": "qwen3_4b",
    "Llama3-3B": "llama3_3b",
    "Gemma2-2B": "gemma2_2b",
}
IDENTITY_KEYS = {
    "run_id", "experiment_id", "todo_id", "assignment_method", "m0_m1_final",
    "m0_m1_pair_id", "m0_m1_arm", "m5_formal", "m5_ablation_row",
    "m5_source_m0_config_path", "m5_source_m0_config_sha256",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if GRID.exists() or CONFIG_ROOT.exists():
        raise FileExistsError("M5 contiguous formal grid already exists")
    CONFIG_ROOT.mkdir(parents=True)
    trials = []
    for model, slug in MODELS.items():
        for setting in ("mixed", "sequential"):
            for seed in (1, 2, 3):
                source_path = ROOT / "experiments/configs/m0_m1_final" / f"m0_{slug}_{setting}_tf_seed{seed}.yaml"
                source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
                if source["model_name"] != model or source["setting"] != setting or int(source["seed"]) != seed:
                    raise AssertionError(f"source identity mismatch: {source_path}")
                if source["assignment_method"] != "tf" or source["official_test_loaded"] is not True:
                    raise AssertionError(f"source protocol mismatch: {source_path}")
                run_id = f"m5_{slug}_{setting}_contiguous_seed{seed}_formal"
                config = deepcopy(source)
                config.update({
                    "run_id": run_id,
                    "experiment_id": "M5-TF-ABLATION-CONTIGUOUS-SVD-FORMAL",
                    "todo_id": "M5",
                    "assignment_method": "contiguous",
                    "m0_m1_final": False,
                    "m5_formal": True,
                    "m5_ablation_row": "contiguous_svd",
                    "m5_source_m0_config_path": str(source_path.relative_to(ROOT)),
                    "m5_source_m0_config_sha256": sha256(source_path),
                })
                config.pop("m0_m1_pair_id", None)
                config.pop("m0_m1_arm", None)
                source_scientific = {k: v for k, v in source.items() if k not in IDENTITY_KEYS}
                target_scientific = {k: v for k, v in config.items() if k not in IDENTITY_KEYS}
                if source_scientific != target_scientific:
                    raise AssertionError(f"scientific drift beyond assignment: {run_id}")
                path = CONFIG_ROOT / f"{run_id}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                trials.append({
                    "run_id": run_id, "model": model, "model_slug": slug,
                    "setting": setting, "seed": seed,
                    "source_config": str(source_path.relative_to(ROOT)),
                    "source_config_sha256": sha256(source_path),
                    "config": str(path.relative_to(ROOT)),
                    "config_sha256": sha256(path),
                    "only_scientific_change": "assignment_method=tf_to_contiguous",
                })
    payload = {
        "schema_version": 1,
        "event": "M5_CONTIGUOUS_SVD_FORMAL_SUBMATRIX_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CONFIGS_FROZEN_RELEASE_NOT_BUILT",
        "models": list(MODELS), "settings": ["mixed", "sequential"],
        "seeds": [1, 2, 3], "logical_cells": len(trials),
        "official_test_used_for_selection": False,
        "scientific_change_from_m0": "assignment_method only: TF to canonical contiguous SVD",
        "trials": trials,
    }
    GRID.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"configs": len(trials), "grid": str(GRID.relative_to(ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
