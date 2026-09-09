#!/usr/bin/env python3
"""Aggregate restart-delimited P3 stages into one decision artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from badit_tf.sequential import continual_metrics
from badit_tf.training import file_sha256, resolve_code_commit


def validate_stage_code_commits(
    stages: list[dict[str, object]], config: dict[str, object]
) -> tuple[bool, dict[str, object]]:
    """Require one commit unless a frozen recovery lineage is explicitly bound."""

    observed = [str(stage["code_commit"]) for stage in stages]
    details: dict[str, object] = {"observed_by_task": observed}
    if len(set(observed)) == 1:
        details["mode"] = "single_commit"
        return True, details
    ranges = config.get("recovery_stage_commit_ranges")
    if not isinstance(ranges, list):
        details["mode"] = "mixed_commit_unbound"
        return False, details
    expected: list[str | None] = [None] * len(stages)
    valid_schema = True
    for item in ranges:
        if not isinstance(item, dict):
            valid_schema = False
            continue
        try:
            start, end, commit = int(item["start_task"]), int(item["end_task"]), str(item["code_commit"])
        except (KeyError, TypeError, ValueError):
            valid_schema = False
            continue
        if start < 0 or end < start or end >= len(stages):
            valid_schema = False
            continue
        for index in range(start, end + 1):
            if expected[index] is not None:
                valid_schema = False
            expected[index] = commit
    exact = valid_schema and all(commit is not None for commit in expected) and observed == expected
    details.update(
        {
            "mode": "explicit_recovery_lineage",
            "expected_by_task": expected,
            "ranges": ranges,
            "exact": exact,
        }
    )
    return exact, details


def build_continual_sanity_gates(
    evaluation_matrix: list[list[float]], config: dict[str, object]
) -> tuple[dict[str, object], str | None]:
    """Evaluate preregistered continual-quality gates without hiding scores."""

    if not evaluation_matrix or "continual_sanity_min_retention_fraction" not in config:
        return {}, None
    minimum = float(config["continual_sanity_min_retention_fraction"])
    diagonal = [
        float(evaluation_matrix[index][index])
        for index in range(len(evaluation_matrix))
    ]
    final_prior = [
        float(evaluation_matrix[-1][index])
        for index in range(len(evaluation_matrix) - 1)
    ]
    retention_fractions = [
        final / diagonal[index] if diagonal[index] > 0 else 0.0
        for index, final in enumerate(final_prior)
    ]
    mean_retention = sum(retention_fractions) / len(retention_fractions)
    gates: dict[str, object] = {
        "minimum_mean_retention_fraction": minimum,
        "diagonal_scores": diagonal,
        "final_prior_scores": final_prior,
        "retention_fractions": retention_fractions,
        "mean_final_retention_fraction": mean_retention,
        "all_diagonal_scores_positive": all(value > 0 for value in diagonal),
        "all_final_prior_scores_positive": all(value > 0 for value in final_prior),
        "mean_retention_fraction_passed": mean_retention >= minimum,
    }
    required_boolean_gates = [
        "all_diagonal_scores_positive",
        "all_final_prior_scores_positive",
        "mean_retention_fraction_passed",
    ]
    if "continual_sanity_min_final_diagonal_score" in config:
        minimum_final_diagonal = float(
            config["continual_sanity_min_final_diagonal_score"]
        )
        gates["minimum_final_diagonal_score"] = minimum_final_diagonal
        gates["final_diagonal_score"] = diagonal[-1]
        gates["final_diagonal_score_passed"] = diagonal[-1] >= minimum_final_diagonal
        required_boolean_gates.append("final_diagonal_score_passed")
    quality_passed = all(bool(gates[key]) for key in required_boolean_gates)
    return gates, (
        "CONTINUAL_SANITY_PASSED"
        if quality_passed
        else "CONTINUAL_SANITY_QUALITY_FAILED"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output-name", default="result.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_root"]) / args.run_id
    task_count = int(config.get("pilot_task_count", 3))
    if task_count < 2:
        raise ValueError("P3 aggregation requires at least two tasks")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("output-name must be a basename")
    stage_paths = [
        Path(str(config.get("aggregation_source_output_dir", output_dir)))
        / "stages"
        / f"task_{index}.json"
        for index in range(task_count)
    ]
    source_dir = Path(str(config.get("aggregation_source_output_dir", output_dir)))
    stages = [json.loads(path.read_text(encoding="utf-8")) for path in stage_paths]
    stage_code_commit_valid, stage_code_commit_details = validate_stage_code_commits(
        stages, config
    )
    stage_assertions = {
        f"task_{index}_complete": stage["status"] == "complete"
        and all(stage["metrics"]["assertions"].values())
        for index, stage in enumerate(stages)
    }
    resume_keys = [
        key
        for stage in stages[1:]
        for key, value in stage["metrics"]["assertions"].items()
        if key.startswith("load_")
        or key.startswith("metadata_")
        or key.startswith("resume_")
        or key
        in {
            "fixed_plus_one_forward_exact",
            "router_parameters_exact",
            "topk_state_exact",
            "optimizer_updates_exact",
            "assignment_exact",
            "global_switch_step_consistent",
            "all_ranks_resume_first_step_exact",
        }
    ]
    assertions = {
        **stage_assertions,
        "restart_boundaries_audited": len(stages) == task_count
        and all(
            stage["protocol"]["restart_boundary_before_task"] for stage in stages[1:]
        ),
        "stage_code_commit_consistent": stage_code_commit_valid,
        "resume_assertions_present": bool(resume_keys),
        "all_resume_assertions_passed": all(
            value
            for stage in stages[1:]
            for key, value in stage["metrics"]["assertions"].items()
            if key.startswith("load_")
            or key.startswith("metadata_")
            or key.startswith("resume_")
            or key
            in {
                "fixed_plus_one_forward_exact",
                "router_parameters_exact",
                "topk_state_exact",
                "optimizer_updates_exact",
                "assignment_exact",
                "global_switch_step_consistent",
                "all_ranks_resume_first_step_exact",
            }
        ),
    }
    evaluation_matrix = []
    baseline = []
    cl_metrics = {}
    if not args.skip_eval:
        baseline_payload = json.loads(
            (source_dir / "evaluation_baseline.json").read_text(encoding="utf-8")
        )
        baseline = baseline_payload["scores"]
        evaluation_matrix = [
            json.loads(
                (source_dir / f"evaluation_after_task_{index}.json").read_text(
                    encoding="utf-8"
                )
            )["scores"]
            for index in range(task_count)
        ]
        cl_metrics = continual_metrics(evaluation_matrix, baseline)
        assertions["forgetting_matrix_complete"] = len(
            evaluation_matrix
        ) == task_count and all(len(row) == task_count for row in evaluation_matrix)
        evaluation_payloads = [
            json.loads((source_dir / label).read_text(encoding="utf-8"))
            for label in [
                "evaluation_baseline.json",
                *[f"evaluation_after_task_{index}.json" for index in range(task_count)],
            ]
        ]
        assertions["all_evaluations_retained"] = all(
            payload["records"] == payload["expected_records"]
            and payload["unique_sample_ids"] == payload["expected_records"]
            for payload in evaluation_payloads
        )
    quality_gates, quality_decision = build_continual_sanity_gates(
        evaluation_matrix, config
    )
    result = {
        "run_id": args.run_id,
        "experiment_id": str(config.get("experiment_id", "P3")),
        "status": "complete" if all(assertions.values()) else "failed",
        "decision": quality_decision
        or (
            f"{str(config.get('experiment_id', 'P3')).replace('-', '_')}_MECHANISM_PASSED"
            if all(assertions.values())
            else f"{str(config.get('experiment_id', 'P3')).replace('-', '_')}_FAILED_STOP"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": stages[0]["code_commit"],
        "aggregation_code_commit": resolve_code_commit(),
        "config_path": str(args.config),
        "config_sha256": file_sha256(args.config),
        "protocol": {
            "model": config["model_name"],
            "task_order": list(config["expected_first_tasks"]),
            "task_order_source": str(config.get("task_order_source", "legacy")),
            "epochs_per_task": int(config["epochs_per_task"]),
            "scheduler_scope": str(config.get("scheduler_scope", "global")),
            "optimizer_scope": str(config.get("optimizer_scope", "global")),
            "route_pooling_scope": str(
                config.get("route_pooling_scope", "full_supervised_sequence")
            ),
            "assignment_method": str(config.get("assignment_method", "tf")),
            "router_mode": str(config.get("router_mode", "residual_mask")),
            "router_bias": bool(config.get("router_bias", True)),
            "svd_method": str(config.get("svd_method", "randomized")),
            "residual_implementation": str(
                config.get("residual_implementation", "paired_subtraction")
            ),
            "public_epoch_regroup": bool(config.get("public_epoch_regroup", False)),
            "public_gradient_sync": str(config.get("public_gradient_sync", "none")),
            "public_capacity_repair": str(
                config.get("public_capacity_repair", "public_greedy")
            ),
            "primary_task_metric": str(config.get("primary_task_metric", "task_aware")),
            "evaluation_role": stages[0]["protocol"].get(
                "evaluation_role", "official_test"
            ),
            "router_lr_multiplier": float(config.get("router_lr_multiplier", 1.0)),
            "rehearsal_coefficient": float(config.get("rehearsal_coefficient", 0.0)),
            "rehearsal_source_role": "prior tune_train manifest epoch orders",
            "formal_budget": not args.skip_eval,
            "restart_processes": task_count,
            "official_test_used_for_selection": False,
        },
        "metrics": {
            "baseline": baseline,
            "evaluation_matrix_a_t_s": evaluation_matrix,
            "continual": cl_metrics,
            "continual_sanity_gates": quality_gates,
            "public_regroup_events": {
                f"task_{index}": stage["metrics"].get("public_regroup_events", [])
                for index, stage in enumerate(stages)
            },
            "final_relative_anchor_drift": {
                f"task_{index}": stage["metrics"].get(
                    "final_relative_anchor_drift", 0.0
                )
                for index, stage in enumerate(stages)
            },
            "rehearsal_record_count": {
                f"task_{index}": int(stage["metrics"].get("rehearsal_record_count", 0))
                for index, stage in enumerate(stages)
            },
            "rehearsal_unique_sample_count": {
                f"task_{index}": int(
                    stage["metrics"].get("rehearsal_unique_sample_count", 0)
                )
                for index, stage in enumerate(stages)
            },
            "assertions": assertions,
            "stage_code_commit_provenance": stage_code_commit_details,
            "stage_resume_assertions": {
                f"task_{index}": stage["metrics"]["assertions"]
                for index, stage in enumerate(stages)
            },
        },
        "artifacts": {
            "stage_results": [str(path) for path in stage_paths],
            "source_checkpoint_dirs": [
                str(source_dir / "checkpoints" / f"task_{index}")
                for index in range(task_count)
            ],
            "source_evaluation_dir": str(source_dir),
            "recovery_output_dir": str(output_dir),
        },
        "artifact_sha256": {
            f"stage_{index}": file_sha256(path)
            for index, path in enumerate(stage_paths)
        },
    }
    (output_dir / args.output_name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
