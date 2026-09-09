#!/usr/bin/env python3
"""Apply the preregistered P2 gate to the matched TF/GG pair."""

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
    parser.add_argument("--tf-result", type=Path, required=True)
    parser.add_argument("--gg-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tf = json.loads(args.tf_result.read_text(encoding="utf-8"))
    gg = json.loads(args.gg_result.read_text(encoding="utf-8"))
    tf_rouge = float(tf["metrics"]["rouge"]["macro"]["rougeL"])
    gg_rouge = float(gg["metrics"]["rouge"]["macro"]["rougeL"])
    difference = tf_rouge - gg_rouge
    tf_assertions = tf["metrics"]["assertions"]
    gg_assertions = gg["metrics"]["assertions"]
    assertions = {
        "tf_component_complete": tf["status"] == "complete"
        and all(tf_assertions.values()),
        "gg_component_complete": gg["status"] == "complete"
        and all(gg_assertions.values()),
        "matched_trainable_parameter_count": (
            tf["metrics"]["trainable_parameters"]
            == gg["metrics"]["trainable_parameters"]
        ),
        "matched_train_order": (
            tf["hashes"]["train_order_ids_sha256"]
            == gg["hashes"]["train_order_ids_sha256"]
        ),
        "matched_test_ids": (
            tf["hashes"]["test_ids_sha256"] == gg["hashes"]["test_ids_sha256"]
        ),
        "matched_steps_and_batch": (
            tf["protocol"]["steps"] == gg["protocol"]["steps"]
            and tf["protocol"]["global_batch_size"]
            == gg["protocol"]["global_batch_size"]
        ),
        "tf_not_worse_than_gg_by_more_than_one_point": (
            difference >= -float(config["performance_margin_points"])
        ),
        "no_test_record_removed": (
            tf_assertions["test_records_complete"]
            and gg_assertions["test_records_complete"]
        ),
    }
    result = {
        "run_id": args.run_id,
        "experiment_id": "P2",
        "status": "complete" if all(assertions.values()) else "failed",
        "decision": "GO_TO_P3" if all(assertions.values()) else "NO_GO_P3_BLOCKED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "components": {
            "tf": str(args.tf_result),
            "tf_sha256": file_sha256(args.tf_result),
            "gg": str(args.gg_result),
            "gg_sha256": file_sha256(args.gg_result),
        },
        "metrics": {
            "tf_macro_rougeL": tf_rouge,
            "gg_macro_rougeL": gg_rouge,
            "tf_minus_gg_macro_rougeL": difference,
            "margin_threshold_points": -float(
                config["performance_margin_points"]
            ),
            "tf_loss": tf["metrics"]["loss"],
            "gg_loss": gg["metrics"]["loss"],
            "tf_dead_expert_count": tf["metrics"]["dead_expert_count"],
            "gg_dead_expert_count": gg["metrics"]["dead_expert_count"],
            "assertions": assertions,
        },
        "artifacts": {
            "tf_result": str(args.tf_result),
            "gg_result": str(args.gg_result),
        },
    }
    if result["status"] != "complete":
        result["failure_reason"] = (
            "one or more preregistered P2 matched-pair criteria failed"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
