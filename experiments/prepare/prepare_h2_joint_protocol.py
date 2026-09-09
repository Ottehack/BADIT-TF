#!/usr/bin/env python3
"""Freeze the validation-only H2 Sobol search after all H1 locks exist.

This deliberately refuses to emit runnable configs until every model has a
complete H1 selection lock.  The sampler itself is deterministic and records
the unscaled Sobol points, so an interrupted dispatcher cannot silently change
the H2 search space.  H2 only tunes the five SFT optimisation dimensions; H0
continues to supply the frozen assignment/Fisher/damping/restart choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


GRID = Path("experiments/configs/h1_one_factor_grid_v2.json")
OUTPUT = Path("experiments/configs/h2_joint_sobol_grid.json")
H2_SOURCE_AUTHORIZATION = Path("experiments/materials/h2_small_model_source_authorization.json")
MODELS = {
    "Qwen3-4B": "qwen3_4b",
    "Llama3-3B": "llama3_3b",
    "Gemma2-2B": "gemma2_2b",
}
LOCKS = {
    "Qwen3-4B": Path("experiments/raw_results/h1_qwen3_4b_one_factor_aggregate/selection_lock.json"),
    "Llama3-3B": Path("experiments/raw_results/h1_llama3_3b_one_factor_aggregate/selection_lock.json"),
    "Gemma2-2B": Path("experiments/raw_results/h1_gemma2_2b_one_factor_aggregate/selection_lock.json"),
}
WARMUP = (0.0, 0.01, 0.03, 0.05)
DROPOUT = (0.0, 0.03, 0.05, 0.10)
WEIGHT_DECAY = (0.0, 0.01, 0.10)
H2_AUTHORIZATION_SCOPES = {"formal_H2", "formal_H1_H2", "all_subsequent_experiments"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def categorical(point: float, values: tuple[float, ...]) -> float:
    return values[min(int(point * len(values)), len(values) - 1)]


def log_uniform(point: float, low: float, high: float) -> float:
    return math.exp(math.log(low) + point * (math.log(high) - math.log(low)))


def sobol_trials(seed: int, count: int) -> list[dict[str, Any]]:
    """Return exactly ``count`` H2 points with a persisted pre-scale point."""
    require(count > 0, "H2 trial count must be positive")
    engine = torch.quasirandom.SobolEngine(dimension=5, scramble=True, seed=seed)
    points = engine.draw(count).tolist()
    rows = []
    for index, point in enumerate(points):
        rows.append(
            {
                "trial_id": f"H2-{index:02d}",
                "sobol_point": [float(value) for value in point],
                "learning_rate": log_uniform(point[0], 5e-5, 8e-4),
                "router_lr_multiplier": log_uniform(point[1], 0.25, 4.0),
                "warmup_ratio": categorical(point[2], WARMUP),
                "lora_dropout": categorical(point[3], DROPOUT),
                "weight_decay": categorical(point[4], WEIGHT_DECAY),
            }
        )
    require(len({json.dumps(row["sobol_point"]) for row in rows}) == count, "duplicate Sobol point")
    return rows


def load_h1_locks(root: Path) -> dict[str, dict[str, Any]]:
    loaded = {}
    for model, relative in LOCKS.items():
        path = root / relative
        require(path.is_file(), f"H1 lock missing for {model}: {relative}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("status") == "complete_locked", f"H1 is not locked for {model}")
        require(payload.get("completed_rows") == 34, f"H1 row count incomplete for {model}")
        require(payload.get("selection_metric_source") == "tune_validation", f"H1 non-validation selection for {model}")
        require(payload.get("test_used_for_selection") is False, f"H1 test leakage for {model}")
        require(set(payload.get("selected", {})) == {"mixed", "sequential"}, f"H1 settings missing for {model}")
        loaded[model] = {"path": str(relative), "sha256": sha256(path), "selected": payload["selected"]}
    return loaded


def load_h2_material_authorization(root: Path, grid: dict[str, Any]) -> dict[str, Any]:
    """Load an explicit H2 authorization bound to the frozen H1 source.

    The ModelScope Llama/Gemma artefacts were explicitly admitted for H1 only.
    A completed H1 lock says nothing about permission to use their weights for
    a new H2 optimisation search.  The independent authorization must name the
    exact H1 source-manifest digest, so a different checkpoint cannot silently
    inherit the H1 hyperparameter selection.
    """
    path = root / H2_SOURCE_AUTHORIZATION
    require(path.is_file(), f"H2 material authorization missing: {H2_SOURCE_AUTHORIZATION}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == 1, "unexpected H2 material authorization schema")
    require(payload.get("status") == "authorized", "H2 material authorization is not authorized")
    require(isinstance(payload.get("authorized_by"), str) and payload["authorized_by"].strip(), "H2 authorization issuer missing")
    profiles = grid.get("model_protocols", {})
    authorizations = payload.get("models")
    require(isinstance(authorizations, dict), "H2 material authorization models missing")
    for model in ("Llama3-3B", "Gemma2-2B"):
        profile = profiles.get(model)
        require(isinstance(profile, dict), f"missing frozen H1 material profile for {model}")
        authorization = authorizations.get(model)
        require(isinstance(authorization, dict), f"H2 authorization missing model {model}")
        scope = authorization.get("authorization_scope")
        require(
            scope in H2_AUTHORIZATION_SCOPES,
            f"H2 material authorization missing for {model}: {scope!r}; "
            "the frozen source is not approved beyond H1",
        )
        require(
            authorization.get("source_manifest_sha256") == profile.get("source_manifest_sha256"),
            f"H2 authorization source manifest mismatch for {model}",
        )
        require(
            authorization.get("source_manifest") == profile.get("source_manifest"),
            f"H2 authorization source manifest path mismatch for {model}",
        )
    return {
        "path": str(H2_SOURCE_AUTHORIZATION),
        "sha256": sha256(path),
        "authorized_by": payload["authorized_by"],
        "model_scopes": {model: authorizations[model]["authorization_scope"] for model in ("Llama3-3B", "Gemma2-2B")},
    }


def write_frozen(path: Path, payload: dict[str, Any], *, rewrite: bool) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text and not rewrite:
        raise RuntimeError(f"refusing to overwrite frozen H2 grid: {path}")
    path.write_text(text, encoding="utf-8")


def build(root: Path, *, seed: int, count: int, rewrite: bool) -> dict[str, Any]:
    locks = load_h1_locks(root)
    grid_path = root / GRID
    require(grid_path.is_file(), f"H1 grid missing: {GRID}")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    require(grid.get("expected_trial_rows") == 102, "unexpected H1 grid contract")
    authorization = load_h2_material_authorization(root, grid)
    h2_points = sobol_trials(seed, count)
    rows: list[dict[str, Any]] = []
    for model, slug in MODELS.items():
        for setting in ("mixed", "sequential"):
            source_rows = [row for row in grid["trial_rows"] if row["model"] == model and row["setting"] == setting and row["trial_id"] == "H1-00"]
            require(len(source_rows) == 1, f"missing H1 default template for {model}/{setting}")
            source = source_rows[0]
            config_path = root / source["config_path"]
            require(sha256(config_path) == source["config_sha256"], f"H1 template drift: {config_path}")
            template = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            for point in h2_points:
                config = deepcopy(template)
                run_id = f"h2_{slug}_{setting}_{point['trial_id'].lower().replace('-', '_')}_seed1"
                config.update(
                    {
                        "experiment_id": "H2-JOINT-SOBOL",
                        "todo_id": "H2",
                        "run_id": run_id,
                        "trial_id": point["trial_id"],
                        "h2_sampler": "Sobol",
                        "h2_sampler_seed": seed,
                        "h2_sobol_point": point["sobol_point"],
                        "parent_h1_selection_lock": locks[model]["path"],
                        "parent_h1_selection_lock_sha256": locks[model]["sha256"],
                        "selection_metric_source": "tune_validation",
                        "official_test_used_for_selection": False,
                        **{key: point[key] for key in ("learning_rate", "router_lr_multiplier", "warmup_ratio", "lora_dropout", "weight_decay")},
                    }
                )
                if model != "Qwen3-4B":
                    config.update(
                        {
                            "h2_source_authorization": authorization["path"],
                            "h2_source_authorization_sha256": authorization["sha256"],
                            "h2_source_authorized_by": authorization["authorized_by"],
                            "model_source_authorization_scope": authorization["model_scopes"][model],
                        }
                    )
                relative = Path("experiments/configs/h2") / f"{run_id}.yaml"
                config_text = yaml.safe_dump(config, sort_keys=False)
                absolute = root / relative
                absolute.parent.mkdir(parents=True, exist_ok=True)
                if absolute.exists() and absolute.read_text(encoding="utf-8") != config_text and not rewrite:
                    raise RuntimeError(f"refusing to overwrite frozen H2 config: {relative}")
                absolute.write_text(config_text, encoding="utf-8")
                rows.append({"model": model, "setting": setting, "run_id": run_id, "config_path": str(relative), "config_sha256": sha256(absolute), **point})
    require(len(rows) == 6 * count, f"H2 row count {len(rows)}")
    payload = {
        "schema_version": 1,
        "experiment_id": "H2-JOINT-SOBOL",
        "status": "ready_for_registration_not_submitted",
        "selection_metric_source": "tune_validation",
        "official_test_used_for_selection": False,
        "sampler": {"name": "torch.quasirandom.SobolEngine", "seed": seed, "dimensions": 5, "points_per_model_setting": count},
        "parent_h1_grid_path": str(GRID),
        "parent_h1_grid_sha256": sha256(grid_path),
        "parent_h1_locks": locks,
        "h2_source_authorization": authorization,
        "fixed_h0_parameters_reused": True,
        "trial_rows": rows,
        "expected_trial_rows": 6 * count,
        "submission_gate": "register every row then run only after this manifest SHA256 is frozen",
    }
    write_frozen(root / OUTPUT, payload, rewrite=rewrite)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()
    result = build(args.root.resolve(), seed=args.seed, count=args.count, rewrite=args.rewrite)
    print(json.dumps({"rows": result["expected_trial_rows"], "output": str(OUTPUT), "sampler": result["sampler"]}, sort_keys=True))


if __name__ == "__main__":
    main()
