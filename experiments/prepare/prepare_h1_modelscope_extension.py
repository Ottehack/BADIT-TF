#!/usr/bin/env python3
"""Freeze the owner-authorized Llama/Gemma extension of the H1 grid.

The original H1 grid is immutable because the completed Qwen selection lock
records its SHA256.  This script therefore writes a versioned composite grid:
all 34 Qwen rows are copied unchanged and only the 68 formerly blocked
ModelScope rows are released with model-specific H0 assignments and banks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PARENT_GRID = Path("experiments/configs/h1_one_factor_grid.json")
PARENT_GRID_SHA256 = "e257d47a86fe2b18a72675d25a7f24cccd44e1898a9587abb9476344e338c057"
OUTPUT_GRID = Path("experiments/configs/h1_one_factor_grid_v2.json")

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "Llama3-3B": {
        "slug": "llama3_3b",
        "model_path": "models/Llama3-3B-modelscope-335130",
        "attention_implementation": "sdpa",
        "assignment_path": "experiments/configs/h1_llama3_3b_h0_selected_assignment.json",
        "bank_path": "experiments/raw_results/h1_llama3_3b_warmstart_step001_local_a100_v3/trainable_checkpoint.pt",
        "bank_sha256": "1933e9846f607dc27e9a6f88d920e5e043f572b39107d69247c2908957018fea",
        "h0_dir": "experiments/raw_results/h0_calibration_sweep_llama3_3b_seed24001_local_a100",
        "selected_trial_id": "H0-11",
        "selected_config": {
            "assignment_probes_per_task": 16,
            "changed_axis": "solver_restarts",
            "epsilon_f": 0.1,
            "fisher_probes_per_task": 16,
            "solver_restarts": 3,
            "trial_id": "H0-11",
        },
        "source": "ModelScope LLM-Research/Llama-3.2-3B-Instruct ID 335130",
        "source_manifest": "experiments/materials/llama3_2_3b_modelscope_335130_manifest.json",
        "source_manifest_sha256": "043f45a5d39274ff30dae744cf7d3b5427d69dbc78f2988c67636ea1309b3eda",
    },
    "Gemma2-2B": {
        "slug": "gemma2_2b",
        "model_path": "models/Gemma2-2B-modelscope-311846",
        "attention_implementation": "eager",
        "assignment_path": "experiments/configs/h1_gemma2_2b_h0_selected_assignment.json",
        "bank_path": "experiments/raw_results/h1_gemma2_2b_warmstart_step001_eager_local_a100_v2/trainable_checkpoint.pt",
        "bank_sha256": "9dfbaa7a25be6f8b16df3585e286704741bfaa70d8f0f7dc2d461bb5372c8c88",
        "h0_dir": "experiments/raw_results/h0_calibration_sweep_gemma2_2b_seed24001_local_a100",
        "selected_trial_id": "H0-05",
        "selected_config": {
            "assignment_probes_per_task": 16,
            "changed_axis": "fisher_probes_per_task",
            "epsilon_f": 0.1,
            "fisher_probes_per_task": 8,
            "solver_restarts": 5,
            "trial_id": "H0-05",
        },
        "source": "ModelScope LLM-Research/gemma-2-2b-it ID 311846",
        "source_manifest": "experiments/materials/gemma2_2b_modelscope_311846_manifest.json",
        "source_manifest_sha256": "16f512e65f214e231ebff6e7e1743292c1a3fdb140b4873d31842cde3086b989",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def dump_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def dump_yaml(payload: object) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def write_frozen(path: Path, text: str, *, rewrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text and not rewrite:
        raise RuntimeError(f"refusing to overwrite frozen artifact {path}")
    path.write_text(text, encoding="utf-8")


def validate_assignment_payload(payload: dict[str, Any]) -> None:
    layers = payload["layer_names"]
    assignments = payload["tf"]["0.1"]
    require(set(assignments) == set(layers), "assignment layers do not match layer_names")
    require(len(layers) > 0 and len(layers) == len(set(layers)), "invalid layer_names")
    for layer_name in layers:
        labels = assignments[layer_name]
        require(len(labels) == 32, f"{layer_name}: expected 32 primitives")
        require(set(labels) == set(range(8)), f"{layer_name}: missing expert label")
        counts = {label: labels.count(label) for label in range(8)}
        require(set(counts.values()) == {4}, f"{layer_name}: unbalanced assignment {counts}")


def prepare_assignment(root: Path, profile: dict[str, Any], *, rewrite: bool) -> dict[str, Any]:
    h0_dir = root / profile["h0_dir"]
    lock_path = h0_dir / "h0_selection_lock.json"
    candidates_path = h0_dir / "h0_assignment_candidates.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    require(lock["selected_trial_id"] == profile["selected_trial_id"], "H0 trial mismatch")
    require(lock["selected_config"] == profile["selected_config"], "H0 config mismatch")
    require(lock["downstream_test_used"] is False, "H0 downstream test closure failed")
    require(lock["fidelity_used_for_selection"] is False, "H0 fidelity influenced selection")
    require(lock["candidates_sha256"] == sha256_file(candidates_path), "H0 candidates SHA mismatch")
    selected_trial = profile["selected_trial_id"]
    require(selected_trial in candidates["tf"], "selected H0 assignment missing")

    assignment = {
        "schema_version": 1,
        "layer_names": candidates["layer_names"],
        "tf": {"0.1": candidates["tf"][selected_trial]},
        "source_h0_lock_sha256": sha256_file(lock_path),
        "source_h0_candidates_sha256": sha256_file(candidates_path),
        "selected_trial_id": selected_trial,
    }
    validate_assignment_payload(assignment)
    output_path = root / profile["assignment_path"]
    write_frozen(output_path, dump_json(assignment), rewrite=rewrite)
    return {
        "assignment_path": profile["assignment_path"],
        "assignment_sha256": sha256_file(output_path),
        "h0_lock_path": str(Path(profile["h0_dir"]) / "h0_selection_lock.json"),
        "h0_lock_sha256": sha256_file(lock_path),
        "h0_candidates_path": str(Path(profile["h0_dir"]) / "h0_assignment_candidates.json"),
        "h0_candidates_sha256": sha256_file(candidates_path),
        "selected_trial_id": selected_trial,
        "selected_config": profile["selected_config"],
        "layer_count": len(assignment["layer_names"]),
    }


def qwen_templates(root: Path, parent: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    templates: dict[tuple[str, str], dict[str, Any]] = {}
    rows = [row for row in parent["trial_rows"] if row["model"] == "Qwen3-4B"]
    require(len(rows) == 34, f"expected 34 Qwen template rows, got {len(rows)}")
    for row in rows:
        config_path = root / row["config_path"]
        require(sha256_file(config_path) == row["config_sha256"], f"Qwen config drift: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        key = (row["setting"], row["trial_id"])
        require(key not in templates, f"duplicate Qwen template {key}")
        templates[key] = config
    return templates


def build_model_config(
    template: dict[str, Any],
    model: str,
    setting: str,
    trial_id: str,
    profile: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    config = deepcopy(template)
    run_id = f"h1_{profile['slug']}_{setting}_{trial_id.lower()}_seed1"
    selected = profile["selected_config"]
    config.update(
        {
            "model_name": model,
            "model_path": profile["model_path"],
            "tf_assignment_path": profile["assignment_path"],
            "tf_assignment_epsilon": float(selected["epsilon_f"]),
            "initial_bank_checkpoint": profile["bank_path"],
            "initial_bank_checkpoint_sha256": profile["bank_sha256"],
            "attention_implementation": profile["attention_implementation"],
            "h0_selection_lock": str(Path(profile["h0_dir"]) / "h0_selection_lock.json"),
            "h0_selected_assignment_probes_per_task": int(selected["assignment_probes_per_task"]),
            "h0_selected_fisher_probes_per_task": int(selected["fisher_probes_per_task"]),
            "h0_selected_epsilon_f": float(selected["epsilon_f"]),
            "h0_selected_solver_restarts": int(selected["solver_restarts"]),
            "run_id": run_id,
            "model_source": profile["source"],
            "model_source_type": "ModelScope_USER_UPLOAD",
            "model_source_manifest": profile["source_manifest"],
            "model_source_manifest_sha256": profile["source_manifest_sha256"],
            "model_source_authorization_scope": "formal_H1_only",
            "model_source_protocol_amendment_date": "2026-08-03",
        }
    )
    return run_id, config


def validate_local_profile(root: Path, profile: dict[str, Any]) -> None:
    require((root / profile["model_path"]).is_dir(), f"model missing: {profile['model_path']}")
    bank_path = root / profile["bank_path"]
    require(bank_path.is_file(), f"bank missing: {bank_path}")
    require(sha256_file(bank_path) == profile["bank_sha256"], f"bank SHA mismatch: {bank_path}")
    manifest_path = root / profile["source_manifest"]
    require(manifest_path.is_file(), f"source manifest missing: {manifest_path}")
    require(
        sha256_file(manifest_path) == profile["source_manifest_sha256"],
        f"source manifest SHA mismatch: {manifest_path}",
    )


def build_grid(root: Path, *, rewrite: bool) -> dict[str, Any]:
    parent_path = root / PARENT_GRID
    require(sha256_file(parent_path) == PARENT_GRID_SHA256, "immutable parent H1 grid drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    require(parent["expected_trial_rows"] == 102, "parent grid row contract changed")
    templates = qwen_templates(root, parent)

    protocols: dict[str, Any] = {}
    for model, profile in MODEL_PROFILES.items():
        validate_local_profile(root, profile)
        protocols[model] = prepare_assignment(root, profile, rewrite=rewrite)
        protocols[model].update(
            {
                "model_path": profile["model_path"],
                "attention_implementation": profile["attention_implementation"],
                "bank_path": profile["bank_path"],
                "bank_sha256": profile["bank_sha256"],
                "source": profile["source"],
                "source_type": "ModelScope_USER_UPLOAD",
                "source_manifest": profile["source_manifest"],
                "source_manifest_sha256": profile["source_manifest_sha256"],
                "authorization_scope": "formal_H1_only",
            }
        )

    trial_rows: list[dict[str, Any]] = []
    for parent_row in parent["trial_rows"]:
        model = parent_row["model"]
        if model == "Qwen3-4B":
            trial_rows.append(deepcopy(parent_row))
            continue
        require(model in MODEL_PROFILES, f"unexpected parent model {model}")
        profile = MODEL_PROFILES[model]
        setting = parent_row["setting"]
        trial_id = parent_row["trial_id"]
        template = templates[(setting, trial_id)]
        run_id, config = build_model_config(template, model, setting, trial_id, profile)
        config_path = Path("experiments/configs/h1") / f"{run_id}.yaml"
        write_frozen(root / config_path, dump_yaml(config), rewrite=rewrite)

        row = {
            key: value
            for key, value in parent_row.items()
            if key not in {"required_checkpoint", "status"}
        }
        row.update(
            {
                "status": "ready",
                "run_id": run_id,
                "config_path": str(config_path),
                "config_sha256": sha256_file(root / config_path),
                "h0_selected_trial_id": profile["selected_trial_id"],
                "h0_selection_lock_sha256": protocols[model]["h0_lock_sha256"],
                "initial_bank_checkpoint_sha256": profile["bank_sha256"],
                "attention_implementation": profile["attention_implementation"],
                "model_source_type": "ModelScope_USER_UPLOAD",
                "model_source_manifest_sha256": profile["source_manifest_sha256"],
                "model_source_authorization_scope": "formal_H1_only",
            }
        )
        trial_rows.append(row)

    manifest = {
        "schema_version": 2,
        "experiment_id": "H1-ONE-FACTOR",
        "protocol_amendment": "owner-authorized ModelScope sources for formal H1 only",
        "protocol_amendment_date": "2026-08-03",
        "parent_grid_path": str(PARENT_GRID),
        "parent_grid_sha256": PARENT_GRID_SHA256,
        "selection_metric_source": "tune_validation",
        "official_test_used_for_selection": False,
        "unique_configs_per_model_setting": 17,
        "expected_trial_rows": 102,
        "formal_rows_added": 68,
        "model_protocols": protocols,
        "trial_rows": trial_rows,
    }
    write_frozen(root / OUTPUT_GRID, dump_json(manifest), rewrite=rewrite)
    audit_grid(root, manifest)
    return manifest


def audit_grid(root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    parent_path = root / PARENT_GRID
    require(sha256_file(parent_path) == PARENT_GRID_SHA256, "parent grid SHA mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if manifest is None:
        manifest = json.loads((root / OUTPUT_GRID).read_text(encoding="utf-8"))
    require(manifest["parent_grid_sha256"] == PARENT_GRID_SHA256, "parent reference mismatch")
    rows = manifest["trial_rows"]
    require(len(rows) == 102, f"expected 102 rows, got {len(rows)}")
    identities = {(row["model"], row["setting"], row["trial_id"]) for row in rows}
    require(len(identities) == 102, "duplicate H1 trial identity")
    require(all(row["status"] == "ready" for row in rows), "non-ready H1 row remains")

    qwen_parent = [row for row in parent["trial_rows"] if row["model"] == "Qwen3-4B"]
    qwen_v2 = [row for row in rows if row["model"] == "Qwen3-4B"]
    require(qwen_v2 == qwen_parent, "Qwen rows changed in versioned extension")

    by_model: dict[str, int] = {}
    for row in rows:
        by_model[row["model"]] = by_model.get(row["model"], 0) + 1
        config_path = root / row["config_path"]
        require(config_path.is_file(), f"missing H1 config {config_path}")
        require(sha256_file(config_path) == row["config_sha256"], f"config SHA mismatch {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(config["selection_metric_source"] == "tune_validation", "invalid selection source")
        require(config["official_test_used_for_selection"] is False, "test selection is enabled")
        require(config["trial_id"] == row["trial_id"], "trial ID mismatch")
        require(config["setting"] == row["setting"], "setting mismatch")
        require(config["run_id"] == row["run_id"], "run ID mismatch")
        if row["model"] in MODEL_PROFILES:
            profile = MODEL_PROFILES[row["model"]]
            require(config["model_path"] == profile["model_path"], "model path mismatch")
            require(config["attention_implementation"] == profile["attention_implementation"], "attention mismatch")
            require(config["initial_bank_checkpoint_sha256"] == profile["bank_sha256"], "bank SHA mismatch")
            require(config["model_source_authorization_scope"] == "formal_H1_only", "scope mismatch")
    require(by_model == {"Qwen3-4B": 34, "Llama3-3B": 34, "Gemma2-2B": 34}, f"row counts {by_model}")
    return {"rows": len(rows), "ready": len(rows), "by_model": by_model}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rewrite-generated", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.check_only:
        result = audit_grid(root)
    else:
        manifest = build_grid(root, rewrite=bool(args.rewrite_generated))
        result = audit_grid(root, manifest)
        result["grid_path"] = str(OUTPUT_GRID)
        result["grid_sha256"] = sha256_file(root / OUTPUT_GRID)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
