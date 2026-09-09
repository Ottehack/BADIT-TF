#!/usr/bin/env python3
"""Mechanically freeze one Llama/Gemma H4 TF/GG matrix from the H3 lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "experiments/raw_results/h3_confirmation_aggregate/selection_lock.json"
LOCK_SHA = "e2f67778b1a3a6202d6ceb5fc7051da47c1a4e6c7a440e93e553d308a3871162"
PROFILES = {
    "llama3_8b": ("Llama3-8B", "Llama3-3B", "models/Llama3-8B-modelscope"),
    "gemma2_9b": ("Gemma2-9B", "Gemma2-2B", "models/Gemma2-9B-modelscope-google"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", choices=PROFILES)
    parser.add_argument("--assignment-path", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--bank-path", required=True)
    parser.add_argument("--bank-sha256", required=True)
    args = parser.parse_args()
    assert sha(LOCK) == LOCK_SHA
    model_name, source_model, model_path = PROFILES[args.slug]
    lock = json.loads(LOCK.read_text())
    out = ROOT / "experiments/configs/h4"; out.mkdir(parents=True, exist_ok=True)
    trials = []
    for setting in ("mixed", "sequential"):
        candidates = lock["h4_candidates"][f"{source_model}/{setting}"]
        assert len(candidates) == 3
        for row in candidates:
            source = ROOT / row["candidate"]["config_path"]
            source_cfg = yaml.safe_load(source.read_text())
            trial_id = str(row["candidate"]["trial_id"])
            for variant in ("tf", "gg"):
                run_id = f"h4_{args.slug}_{setting}_{trial_id.lower().replace('-', '_')}_{variant}_seed1"
                config = dict(source_cfg)
                config.update({
                    "experiment_id": "H4-LARGE-MODEL-TRANSFER", "todo_id": "H4", "model_name": model_name, "model_path": model_path,
                    "tf_assignment_path": args.assignment_path, "tf_assignment_epsilon": 0.1, "assignment_method": variant,
                    "initial_bank_checkpoint": args.bank_path, "initial_bank_checkpoint_sha256": args.bank_sha256,
                    "seed": 1, "run_id": run_id, "h4_source_model": source_model, "h4_source_trial_id": trial_id,
                    "h4_source_config_path": str(source.relative_to(ROOT)), "h4_source_config_sha256": sha(source),
                    "h4_h3_selection_lock": str(LOCK.relative_to(ROOT)), "h4_h3_selection_lock_sha256": LOCK_SHA,
                    "h4_model_native_assignment_sha256": args.assignment_sha256,
                    "official_test_used_for_selection": False, "h4_validation_only": True,
                })
                if setting == "sequential":
                    config["public_epoch_regroup"] = variant == "gg"; config["public_gradient_sync"] = "mean"; config["public_capacity_repair"] = "exact_capacity"; config["require_public_regroup_capacity_integrity"] = variant == "gg"
                path = out / f"{run_id}.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False))
                trials.append({"run_id": run_id, "setting": setting, "variant": variant, "source_trial": trial_id, "config_path": str(path.relative_to(ROOT)), "config_sha256": sha(path)})
    assert len(trials) == 12 and len({x['run_id'] for x in trials}) == 12
    manifest = {"experiment_id": "H4", "model": model_name, "selection_lock_sha256": LOCK_SHA, "assignment_path": args.assignment_path, "assignment_sha256": args.assignment_sha256, "bank_sha256": args.bank_sha256, "test_closed": True, "trials": trials}
    path = ROOT / f"experiments/materials/h4_{args.slug}_dispatch_manifest.json"; path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trials": len(trials), "manifest": str(path.relative_to(ROOT)), "manifest_sha256": sha(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
