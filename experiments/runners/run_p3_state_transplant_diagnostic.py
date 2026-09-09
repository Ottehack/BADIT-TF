#!/usr/bin/env python3
"""No-training router/expert state transplants for frozen P3-R2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.core import inject_ability_layers, iter_ability_layers
from badit_tf.sequential import load_trainable_and_ability_state
from badit_tf.training import canonical_json_sha256, file_sha256
from run_p2_train import load_record, score_predictions, seed_everything
from run_p3_route_diagnostic import generate_condition, source_predictions


CONDITIONS = (
    "early_all",
    "final_all",
    "final_experts_early_router",
    "early_experts_final_router",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def is_router_key(name: str) -> bool:
    return ".residual_router." in name


def hybrid_state(
    *,
    expert_payload: dict[str, Any],
    router_payload: dict[str, Any],
    buffer_payload: dict[str, Any],
) -> dict[str, Any]:
    expert_state = expert_payload["model_state"]["trainable_state"]
    router_state = router_payload["model_state"]["trainable_state"]
    if expert_state.keys() != router_state.keys():
        raise AssertionError("checkpoint trainable-state keys differ")
    trainable_state = {
        name: (router_state[name] if is_router_key(name) else expert_state[name])
        for name in expert_state
    }
    return {
        "trainable_state": trainable_state,
        "ability_buffers": buffer_payload["model_state"]["ability_buffers"],
    }


def condition_state(
    condition: str,
    *,
    early: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    if condition == "early_all":
        return hybrid_state(
            expert_payload=early, router_payload=early, buffer_payload=early
        )
    if condition == "final_all":
        return hybrid_state(
            expert_payload=final, router_payload=final, buffer_payload=final
        )
    if condition == "final_experts_early_router":
        return hybrid_state(
            expert_payload=final, router_payload=early, buffer_payload=final
        )
    if condition == "early_experts_final_router":
        return hybrid_state(
            expert_payload=early, router_payload=final, buffer_payload=final
        )
    raise ValueError(f"unknown condition={condition}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_config_path = Path(config["source_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size != int(config["world_size"]):
        raise RuntimeError("world-size mismatch")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(int(source_config["seed"]) + rank)

    output_dir = Path(config["output_root"]) / args.run_id
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    manifest_path = Path(source_config["sequential_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    tokenizer = AutoTokenizer.from_pretrained(
        source_config["model_path"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        source_config["model_path"],
        torch_dtype=getattr(torch, str(source_config["torch_dtype"])),
        attn_implementation=source_config["attention_implementation"],
        local_files_only=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    assignment_payload = json.loads(
        Path(source_config["tf_assignment_path"]).read_text(encoding="utf-8")
    )
    assignments = {
        name: torch.tensor(labels, dtype=torch.long)
        for name, labels in assignment_payload["tf"][
            str(source_config["tf_assignment_epsilon"])
        ].items()
    }
    inject_ability_layers(
        model,
        target_suffixes=tuple(source_config["target_modules"]),
        assignments=assignments,
        num_experts=int(source_config["num_experts"]),
        rank=int(source_config["rank"]),
        lora_alpha=float(source_config["lora_alpha"]),
        lora_dropout=float(source_config["lora_dropout"]),
        top_k=int(source_config["top_k"]),
        dense_steps=int(source_config["dense_steps"]),
        svd_seed=int(source_config["seed"]),
        svd_oversample=int(source_config["svd_oversample"]),
        svd_niter=int(source_config["svd_niter"]),
    )
    model.eval()
    model.config.use_cache = True

    source_dir = Path(config["source_run_dir"])
    checkpoint_payloads = {
        stage: torch.load(
            source_dir / "checkpoints" / f"task_{stage}" / "common.pt",
            map_location="cpu",
            weights_only=False,
        )
        for stage in range(3)
    }
    record_key = (
        "evaluation_records"
        if all("evaluation_records" in tasks[index] for index in (0, 1))
        else "test_records"
    )
    expected_by_task = {
        index: len(tasks[index][record_key]) for index in (0, 1)
    }
    cache: dict[str, dict] = {}
    rows_path = output_dir / f"predictions_rank{rank}.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for task_index in (0, 1):
            early = checkpoint_payloads[task_index]
            final = checkpoint_payloads[2]
            task = tasks[task_index]
            for condition in CONDITIONS:
                load_trainable_and_ability_state(
                    model,
                    condition_state(condition, early=early, final=final),
                )
                for _, layer in iter_ability_layers(model):
                    layer.disable_calibration()
                    layer.topk_enabled.fill_(True)
                for record_meta in task[record_key][rank::world_size]:
                    record = load_record(record_meta, cache)
                    prediction, route_stats = generate_condition(
                        model,
                        tokenizer,
                        record,
                        condition="trained_top4",
                        max_prompt_length=int(source_config["max_sequence_length"])
                        - int(source_config["max_new_tokens"]),
                        max_new_tokens=int(source_config["max_new_tokens"]),
                        device=device,
                    )
                    output = record["instance"]["output"]
                    references = (
                        [str(item) for item in output]
                        if isinstance(output, list)
                        else [str(output)]
                    )
                    handle.write(
                        json.dumps(
                            {
                                "task_index": task_index,
                                "task": task["task"],
                                "condition": condition,
                                "sample_id": record["sample_id"],
                                "prediction": prediction,
                                "references": references,
                                "route_stats": route_stats,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
    dist.barrier()

    if rank == 0:
        rows = []
        for path in sorted(output_dir.glob("predictions_rank*.jsonl")):
            rows.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["task_index"], row["condition"])].append(row)

        slices = []
        source_exact = True
        for (task_index, condition), values in sorted(grouped.items()):
            metrics = score_predictions(values)
            task_name = tasks[task_index]["task"]
            reproduced = None
            if condition in {"early_all", "final_all"}:
                stage = task_index if condition == "early_all" else 2
                source = source_predictions(source_dir, stage)
                reproduced = all(
                    source.get(row["sample_id"]) == row["prediction"]
                    for row in values
                )
                source_exact = source_exact and bool(reproduced)
            slices.append(
                {
                    "task_index": task_index,
                    "task": task_name,
                    "condition": condition,
                    "records": len(values),
                    "rougeL": float(metrics["per_task"][task_name]["rougeL"]),
                    "exact_match": float(
                        metrics["per_task"][task_name]["exact_match"]
                    ),
                    "source_prediction_exact": reproduced,
                }
            )

        by_slice = {
            (row["task_index"], row["condition"]): row for row in slices
        }
        margin = float(config["causal_margin_rougeL_points"])
        comparisons = []
        for task_index in (0, 1):
            score = {
                condition: by_slice[(task_index, condition)]["rougeL"]
                for condition in CONDITIONS
            }
            router_rescue = score["final_experts_early_router"] - score["final_all"]
            expert_rescue = score["early_experts_final_router"] - score["final_all"]
            rollback_rescue = score["early_all"] - score["final_all"]
            router_supported = router_rescue >= margin
            expert_supported = expert_rescue >= margin
            if router_supported and expert_supported:
                diagnosis = "BOTH_COMPONENT_RESCUES_SUPPORTED"
            elif router_supported:
                diagnosis = "ROUTER_DRIFT_SUPPORTED"
            elif expert_supported:
                diagnosis = "EXPERT_OVERWRITING_SUPPORTED"
            elif rollback_rescue >= margin:
                diagnosis = "COUPLED_ROUTER_EXPERT_INTERACTION_SUPPORTED"
            else:
                diagnosis = "NO_FIVE_POINT_RECOVERABLE_SIGNAL"
            comparisons.append(
                {
                    "task_index": task_index,
                    "task": tasks[task_index]["task"],
                    "rougeL": score,
                    "router_rescue": router_rescue,
                    "expert_rescue": expert_rescue,
                    "full_rollback_rescue": rollback_rescue,
                    "router_drift_supported": router_supported,
                    "expert_overwriting_supported": expert_supported,
                    "diagnosis": diagnosis,
                }
            )

        assertions = {
            "source_controls_exact": source_exact,
            "all_predictions_retained": len(rows)
            == sum(expected_by_task.values()) * len(CONDITIONS),
            "all_slices_complete": len(slices) == 8
            and all(
                row["records"] == expected_by_task[row["task_index"]]
                for row in slices
            ),
            "official_test_not_loaded": manifest.get("evaluation_role")
            == "tune_validation"
            and manifest.get("official_test_loaded_by_training_runner") is False,
        }
        result = {
            "run_id": args.run_id,
            "experiment_id": str(config["experiment_id"]),
            "status": "complete" if all(assertions.values()) else "failed",
            "decision": (
                "STATE_TRANSPLANT_DIAGNOSTIC_COMPLETE"
                if all(assertions.values())
                else "STATE_TRANSPLANT_DIAGNOSTIC_FAILED"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": os.popen("git rev-parse HEAD").read().strip(),
            "config_path": str(args.config),
            "config_sha256": file_sha256(args.config),
            "protocol": {
                "training_rerun": False,
                "conditions": list(CONDITIONS),
                "tasks": [tasks[index]["task"] for index in (0, 1)],
                "causal_margin_rougeL_points": margin,
                "selection_split": "tune_validation",
                "official_test_used_for_selection": False,
            },
            "hashes": {
                "source_result": file_sha256(config["source_result"]),
                "source_config": file_sha256(source_config_path),
                "manifest_file": file_sha256(manifest_path),
                "manifest_content": manifest["manifest_sha256"],
                "assignment": canonical_json_sha256(
                    {
                        name: labels.tolist()
                        for name, labels in sorted(assignments.items())
                    }
                ),
                "checkpoints": {
                    f"task_{stage}": file_sha256(
                        source_dir / "checkpoints" / f"task_{stage}" / "common.pt"
                    )
                    for stage in range(3)
                },
            },
            "metrics": {
                "assertions": assertions,
                "comparisons": comparisons,
                "slices": slices,
            },
            "artifacts": {"predictions_dir": str(output_dir)},
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
