#!/usr/bin/env python3
"""Freeze one model-native H4 calibration config after a warm-start is complete."""
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
    "llama3_8b": ("Llama3-8B", "models/Llama3-8B-modelscope"),
    "gemma2_9b": ("Gemma2-9B", "models/Gemma2-9B-modelscope-google"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", choices=PROFILES)
    parser.add_argument("--bank-path", required=True)
    parser.add_argument("--bank-sha256", required=True)
    args = parser.parse_args()
    assert sha(LOCK) == LOCK_SHA
    model_name, model_path = PROFILES[args.slug]
    run_id = f"h4_{args.slug}_calibration_seed24001"
    config = {
        "run_id": run_id, "experiment_id": "H4-LARGE-MODEL-NATIVE-CALIBRATION", "todo_id": "H4",
        "model_name": model_name, "model_path": model_path,
        "split_manifest": "experiments/configs/splits/h0_calibration_seed24001.json",
        "output_dir": f"experiments/raw_results/{run_id}",
        "initial_bank_checkpoint": args.bank_path, "initial_bank_checkpoint_sha256": args.bank_sha256,
        "seed": 24001, "q_roles": ["assignment", "damping_validation", "fidelity"],
        "max_sequence_length": 1024, "max_target_length": 50, "max_fisher_new_tokens": 50,
        "torch_dtype": "bfloat16", "attention_implementation": "flash_attention_2", "target_modules": ["gate_proj"],
        "num_experts": 8, "rank": 4, "lora_alpha": 32, "lora_dropout": 0.05, "apply_lora_dropout": False,
        "top_k": 4, "dense_steps": 1, "router_mode": "residual_mask", "router_bias": True,
        "initialization_method": "kaiming_zero", "svd_method": "exact", "residual_implementation": "paired_subtraction",
        "svd_oversample": 0, "svd_niter": 0, "assignment_restarts": 5, "assignment_max_iterations": 50,
        "assignment_tolerance": 0.000001, "fidelity_eta": 0.0001,
        "h4_parent_h3_lock": str(LOCK.relative_to(ROOT)), "h4_parent_h3_lock_sha256": LOCK_SHA,
        "h4_h0_selected_hyperparameters": {"trial_id": "H0-00", "assignment_probes_per_task": 16, "fisher_probes_per_task": 16, "epsilon_f": 0.1, "solver_restarts": 5},
        "official_test_loaded": False, "downstream_test_loaded": False,
    }
    path = ROOT / "experiments/configs/h4" / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({"run_id": run_id, "config": str(path.relative_to(ROOT)), "config_sha256": sha(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
