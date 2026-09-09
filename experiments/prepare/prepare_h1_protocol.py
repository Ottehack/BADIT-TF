#!/usr/bin/env python3
"""Freeze H1 one-factor trials after the immutable H0 selection lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ALLOW_REWRITE = False


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path: Path, payload: object, *, yaml_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        yaml.safe_dump(payload, sort_keys=False)
        if yaml_output
        else json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    if path.exists() and path.read_text() != text and not ALLOW_REWRITE:
        raise RuntimeError(f"refusing to overwrite frozen artifact {path}")
    path.write_text(text)


def trials() -> list[dict]:
    default = {
        "learning_rate": 2e-4,
        "router_lr_multiplier": 1.0,
        "warmup_ratio": 0.03,
        "lora_dropout": 0.05,
        "weight_decay": 0.0,
    }
    replacements = [
        ("learning_rate", value) for value in (5e-5, 1e-4, 4e-4, 8e-4)
    ] + [
        ("router_lr_multiplier", value) for value in (0.25, 0.5, 2.0, 4.0)
    ] + [
        ("warmup_ratio", value) for value in (0.0, 0.01, 0.05)
    ] + [
        ("lora_dropout", value) for value in (0.0, 0.03, 0.10)
    ] + [
        ("weight_decay", value) for value in (0.01, 0.10)
    ]
    result = [{"trial_id": "H1-00", "changed_axis": "default", **default}]
    for index, (axis, value) in enumerate(replacements, start=1):
        item = dict(default)
        item[axis] = value
        result.append(
            {"trial_id": f"H1-{index:02d}", "changed_axis": axis, **item}
        )
    if len(result) != 17 or len({json.dumps(x, sort_keys=True) for x in result}) != 17:
        raise AssertionError("H1 must contain exactly 17 unique configs")
    return result


def base_config(setting: str) -> dict:
    base = {
        "experiment_id": "H1-ONE-FACTOR",
        "todo_id": "H1",
        "model_name": "Qwen3-4B",
        "model_path": "models/Qwen3-4B",
        "data_root": "data/SuperNI",
        "tuning_split_manifest": "experiments/configs/splits/h1_tuning_seed1.json",
        "tf_assignment_path": "experiments/configs/h1_qwen3_4b_h0_selected_assignment.json",
        "tf_assignment_epsilon": 0.1,
        "assignment_method": "tf",
        "initial_bank_checkpoint": "local/artifacts/checkpoints/initial_bank.pt",
        "initial_bank_checkpoint_sha256": "014e6e4a2ae4391c64e357c2daaa484b4d4f0dd9dc5ff7377960136cb0c4da56",
        "output_root": "experiments/raw_results",
        "seed": 1,
        "world_size": 8,
        "max_sequence_length": 1024,
        "max_target_length": 50,
        "max_new_tokens": 50,
        "torch_dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "target_modules": ["gate_proj"],
        "num_experts": 8,
        "rank": 4,
        "lora_alpha": 32,
        "apply_lora_dropout": True,
        "allow_runtime_dropout_override": True,
        "top_k": 4,
        "dense_steps": 1,
        "router_mode": "residual_mask",
        "router_bias": True,
        "initialization_method": "kaiming_zero",
        "svd_method": "exact",
        "residual_implementation": "paired_subtraction",
        "svd_oversample": 0,
        "svd_niter": 0,
        "route_pooling_scope": "prompt_only",
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1e-8,
        "gradient_clipping": 1.0,
        "loss_explosion_factor": 10.0,
        "log_every_steps": 50,
        "selection_metric_source": "tune_validation",
        "official_test_used_for_selection": False,
        "h0_selection_lock": "experiments/raw_results/h0_calibration_sweep_qwen3_4b_seed24001_remote_l20z/h0_selection_lock.json",
        "h0_selected_assignment_probes_per_task": 16,
        "h0_selected_fisher_probes_per_task": 16,
        "h0_selected_epsilon_f": 0.1,
        "h0_selected_solver_restarts": 5,
    }
    if setting == "mixed":
        base.update(
            {
                "setting": "mixed",
                "epochs": 1,
                "order_manifest": "experiments/configs/splits/h1_r2_mixed_seed1_order.json",
                "official_test_manifest": "experiments/configs/splits/p2_mixed_seed1_order.json",
            }
        )
    elif setting == "sequential":
        order = [
            "task591_sciq_answer_generation",
            "task073_commonsenseqa_answer_generation",
            "task1687_sentiment140_classification",
            "task875_emotion_classification",
            "task1572_samsum_summary",
            "task639_multi_woz_user_utterance_generation",
            "task1510_evalution_relation_extraction",
            "task1590_diplomacy_text_generation",
            "task511_reddit_tifu_long_text_summarization",
            "task181_outcome_extraction",
            "task363_sst2_polarity_classification",
            "task1729_personachat_generate_next",
            "task002_quoref_answer_generation",
            "task1290_xsum_summarization",
            "task748_glucose_reverse_cause_event_detection",
        ]
        base.update(
            {
                "setting": "sequential",
                "sequential_manifest": "experiments/configs/splits/h1_r2_sequential_order1_seed1.json",
                "pilot_task_count": 15,
                "task_order_source": "TPAMI.pdf Table XIV seed 1",
                "fixed_task_order": order,
                "expected_first_tasks": order,
                "epochs_per_task": 10,
                "scheduler_scope": "per_task",
                "optimizer_scope": "per_task",
                "primary_task_metric": "rougeL",
                "resume_loss_atol": 0.0,
            }
        )
    else:
        raise ValueError(setting)
    return base


def main() -> None:
    global ALLOW_REWRITE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h0-dir",
        type=Path,
        default=Path("experiments/raw_results/h0_calibration_sweep_qwen3_4b_seed24001_remote_l20z"),
    )
    parser.add_argument(
        "--rewrite-generated",
        action="store_true",
        help="Rewrite generated H1 configs before any formal trial begins.",
    )
    args = parser.parse_args()
    ALLOW_REWRITE = bool(args.rewrite_generated)
    lock_path = args.h0_dir / "h0_selection_lock.json"
    candidates_path = args.h0_dir / "h0_assignment_candidates.json"
    lock = json.loads(lock_path.read_text())
    candidates = json.loads(candidates_path.read_text())
    if lock["selected_trial_id"] != "H0-00" or lock["selected_config"] != {
        "assignment_probes_per_task": 16,
        "changed_axis": "default",
        "epsilon_f": 0.1,
        "fisher_probes_per_task": 16,
        "solver_restarts": 5,
        "trial_id": "H0-00",
    }:
        raise AssertionError("unexpected H0 selection lock")
    assignment = {
        "schema_version": 1,
        "layer_names": candidates["layer_names"],
        "tf": {"0.1": candidates["tf"]["H0-00"]},
        "source_h0_lock_sha256": sha(lock_path),
        "source_h0_candidates_sha256": sha(candidates_path),
        "selected_trial_id": "H0-00",
    }
    assignment_path = Path("experiments/configs/h1_qwen3_4b_h0_selected_assignment.json")
    write_frozen(assignment_path, assignment)

    grid = trials()
    trial_rows = []
    for model in ("Qwen3-4B", "Llama3-3B", "Gemma2-2B"):
        for setting in ("mixed", "sequential"):
            for trial in grid:
                row = {"model": model, "setting": setting, **trial}
                if model == "Qwen3-4B":
                    config = base_config(setting)
                    config.update(trial)
                    run_id = f"h1_{model.lower().replace('-', '_')}_{setting}_{trial['trial_id'].lower()}_seed1"
                    config["run_id"] = run_id
                    config_path = Path("experiments/configs/h1") / f"{run_id}.yaml"
                    write_frozen(config_path, config, yaml_output=True)
                    row.update(
                        {
                            "status": "ready",
                            "run_id": run_id,
                            "config_path": str(config_path),
                            "config_sha256": sha(config_path),
                        }
                    )
                else:
                    row.update(
                        {
                            "status": "blocked_model_checkpoint_access",
                            "required_checkpoint": (
                                "meta-llama/Llama-3.2-3B-Instruct"
                                if model == "Llama3-3B"
                                else "google/gemma-2-2b-it"
                            ),
                        }
                    )
                trial_rows.append(row)
    manifest = {
        "schema_version": 1,
        "experiment_id": "H1-ONE-FACTOR",
        "selection_metric_source": "tune_validation",
        "official_test_used_for_selection": False,
        "h0_selection_lock_sha256": sha(lock_path),
        "qwen_assignment_sha256": sha(assignment_path),
        "unique_configs_per_model_setting": 17,
        "expected_trial_rows": 102,
        "trial_rows": trial_rows,
    }
    write_frozen(Path("experiments/configs/h1_one_factor_grid.json"), manifest)
    print(json.dumps({"rows": len(trial_rows), "ready": sum(x["status"] == "ready" for x in trial_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
