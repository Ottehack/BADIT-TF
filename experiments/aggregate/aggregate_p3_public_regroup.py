#!/usr/bin/env python3
"""Aggregate the one-epoch public BADIT distributed-regroup integrity probe."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from badit_tf.training import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_root"]) / args.run_id
    stage_path = output_dir / "stages" / "task_0.json"
    if not stage_path.exists():
        result = {
            "run_id": args.run_id,
            "experiment_id": config["experiment_id"],
            "status": "failed",
            "decision": "STAGE_DID_NOT_PRODUCE_STRUCTURED_RESULT",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config),
            "config_sha256": file_sha256(args.config),
            "metrics": {"assertions": {"stage_result_exists": False}},
            "artifacts": {"expected_stage_result": str(stage_path)},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True))
        return
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage_assertions = stage["metrics"]["assertions"]
    events = stage["metrics"]["public_regroup_events"]
    assertions = {
        "stage_complete": stage["status"] == "complete"
        and all(stage_assertions.values()),
        "initial_model_cross_rank_consensus": bool(
            stage_assertions["initial_model_cross_rank_consensus"]
        ),
        "exactly_one_epoch_end_regroup": len(events) == 1
        and events[0]["epoch"] == 1
        and events[0]["applied"],
        "all_36_layers_regrouped": len(events) == 1
        and len(events[0]["layers"]) == 36,
        "public_regroup_cross_rank_consensus": len(events) == 1
        and bool(events[0]["model_state_cross_rank_consensus"]),
        "public_regroup_capacity_integrity": bool(
            not config.get("require_public_regroup_capacity_integrity", False)
            or (
                len(events) == 1
                and events[0]["duplicate_copies_total"] == 0
                and events[0]["omitted_source_slots_total"] == 0
            )
        ),
        "official_test_not_loaded": stage["protocol"]["evaluation_role"]
        == "tune_validation"
        and not stage["protocol"]["official_test_used_for_selection"],
    }
    passed = all(assertions.values())
    result = {
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "status": "complete" if passed else "failed",
        "decision": (
            "DISTRIBUTED_REGROUP_INTEGRITY_PASSED_RELEASE_FULL_CONTROL"
            if passed
            else "PUBLIC_REGROUP_INTEGRITY_FAILED_BLOCK_FULL_CONTROL"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": stage["code_commit"],
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "protocol": {
            "upstream_commit": config["public_upstream_commit"],
            "upstream_config": config["public_lora_config"],
            "world_size": int(config["world_size"]),
            "task": stage["task"],
            "executed_steps": stage["protocol"]["effective_steps_this_task"],
            "audit_scope": config["audit_scope"],
            "known_prefix_deviation": config["known_prefix_deviation"],
            "public_gradient_sync": str(
                config.get("public_gradient_sync", "none")
            ),
            "public_capacity_repair": str(
                config.get("public_capacity_repair", "public_greedy")
            ),
            "official_test_used_for_selection": False,
        },
        "metrics": {
            "assertions": assertions,
            "stage_assertions": stage_assertions,
            "initial_rank_model_state_sha256": stage["metrics"][
                "initial_rank_model_state_sha256"
            ],
            "public_regroup_events": events,
        },
        "artifacts": {
            "stage_result": str(stage_path),
            "rank_regroup_audits": [
                str(output_dir / f"public_regroup_rank_{rank}.json")
                for rank in range(int(config["world_size"]))
            ],
            "checkpoint_dir": str(output_dir / "checkpoints" / "task_0"),
        },
        "artifact_sha256": {"stage_result": file_sha256(stage_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
