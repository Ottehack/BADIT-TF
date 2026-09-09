#!/usr/bin/env python3
"""No-training route counterfactuals for the frozen P3-R2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import prompt_for_record
from badit_tf.core import inject_ability_layers, iter_ability_layers
from badit_tf.sequential import load_trainable_and_ability_state
from badit_tf.training import canonical_json_sha256, file_sha256
from run_p2_train import load_record, score_predictions, seed_everything, set_route_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def clear_route_overrides(model: torch.nn.Module) -> None:
    for _, layer in iter_ability_layers(model):
        layer.set_route_code_override(None)


@torch.no_grad()
def generate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    condition: str,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, dict[str, float]]:
    encoded = tokenizer(
        prompt_for_record(record),
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    clear_route_overrides(model)
    for _, layer in iter_ability_layers(model):
        layer.topk_enabled.fill_(condition == "trained_top4")
        if condition == "all_one":
            layer.set_route_code_override(
                torch.ones(
                    1,
                    layer.num_experts,
                    device=device,
                    dtype=layer.lora_A.dtype,
                )
            )
    set_route_masks(model, attention_mask)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    prompt_routes = []
    prompt_raw = []
    for _, layer in iter_ability_layers(model):
        prompt_routes.append(layer._last_route["actual"].float().flatten())
        prompt_raw.append(layer._last_route["raw"].float().flatten())
        if condition != "all_one":
            layer.set_route_code_override(layer._last_route["actual"])
    actual = torch.cat(prompt_routes)
    raw = torch.cat(prompt_raw)
    route_stats = {
        "actual_abs_mean": float(actual.abs().mean().item()),
        "actual_abs_max": float(actual.abs().max().item()),
        "active_fraction": float((actual != 0).float().mean().item()),
        "raw_abs_mean": float(raw.abs().mean().item()),
        "raw_abs_max": float(raw.abs().max().item()),
    }
    generated: list[int] = []
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past = outputs.past_key_values
    for _ in range(max_new_tokens):
        token = int(next_token.item())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(1, 1, dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    clear_route_overrides(model)
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), route_stats


def source_predictions(source_dir: Path, stage: int) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(source_dir.glob(f"predictions_after_task_{stage}_rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                rows[row["sample_id"]] = row["prediction"]
    return rows


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
    cache: dict[str, dict] = {}
    rows_path = output_dir / f"predictions_rank{rank}.jsonl"
    evaluation_plan = [(0, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    conditions = ["trained_top4", "dense_trained", "all_one"]
    with rows_path.open("w", encoding="utf-8") as handle:
        for stage in sorted({item[0] for item in evaluation_plan}):
            checkpoint = Path(config["source_run_dir"]) / "checkpoints" / f"task_{stage}" / "common.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            load_trainable_and_ability_state(model, payload["model_state"])
            for _, layer in iter_ability_layers(model):
                layer.disable_calibration()
            for planned_stage, task_index in evaluation_plan:
                if planned_stage != stage:
                    continue
                task = tasks[task_index]
                for condition in conditions:
                    for record_meta in task["test_records"][rank::world_size]:
                        record = load_record(record_meta, cache)
                        prediction, route_stats = generate_condition(
                            model,
                            tokenizer,
                            record,
                            condition=condition,
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
                                    "stage": stage,
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
        grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["stage"], row["task_index"], row["condition"])].append(row)
        slices = []
        exact_reproduction = True
        source_dir = Path(config["source_run_dir"])
        for (stage, task_index, condition), values in sorted(grouped.items()):
            metrics = score_predictions(values)
            task_name = tasks[task_index]["task"]
            route_stats = {
                key: float(np.mean([row["route_stats"][key] for row in values]))
                for key in values[0]["route_stats"]
            }
            reproduced = None
            if condition == "trained_top4":
                source = source_predictions(source_dir, stage)
                reproduced = all(
                    source.get(row["sample_id"]) == row["prediction"] for row in values
                )
                exact_reproduction = exact_reproduction and bool(reproduced)
            slices.append(
                {
                    "stage": stage,
                    "task_index": task_index,
                    "task": task_name,
                    "condition": condition,
                    "records": len(values),
                    "rougeL": float(metrics["per_task"][task_name]["rougeL"]),
                    "exact_match": float(
                        metrics["per_task"][task_name]["exact_match"]
                    ),
                    "source_prediction_exact": reproduced,
                    "route_stats": route_stats,
                }
            )
        by_slice = {
            (row["stage"], row["task_index"], row["condition"]): row
            for row in slices
        }
        margin = float(config["causal_margin_rougeL_points"])
        comparisons = []
        for stage, task_index in evaluation_plan:
            top4 = by_slice[(stage, task_index, "trained_top4")]["rougeL"]
            dense = by_slice[(stage, task_index, "dense_trained")]["rougeL"]
            all_one = by_slice[(stage, task_index, "all_one")]["rougeL"]
            comparisons.append(
                {
                    "stage": stage,
                    "task_index": task_index,
                    "task": tasks[task_index]["task"],
                    "trained_top4": top4,
                    "dense_trained": dense,
                    "all_one": all_one,
                    "dense_minus_top4": dense - top4,
                    "all_one_minus_top4": all_one - top4,
                    "topk_failure_supported": dense - top4 >= margin,
                    "router_code_failure_supported": all_one - top4 >= margin,
                }
            )
        topk_count = sum(row["topk_failure_supported"] for row in comparisons)
        router_count = sum(
            row["router_code_failure_supported"] for row in comparisons
        )
        if topk_count or router_count:
            decision = "ROUTER_REALIZATION_FAILURE_SUPPORTED"
        else:
            decision = "EXPERT_OR_ASSIGNMENT_FAILURE_SUPPORTED"
        result = {
            "run_id": args.run_id,
            "experiment_id": "P3-R2-D1",
            "status": "complete" if exact_reproduction else "failed",
            "decision": decision if exact_reproduction else "SOURCE_REPRODUCTION_FAILED",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": os.popen("git rev-parse HEAD").read().strip(),
            "config_path": str(args.config),
            "config_sha256": file_sha256(args.config),
            "protocol": {
                "training_rerun": False,
                "conditions": conditions,
                "evaluation_plan": evaluation_plan,
                "causal_margin_rougeL_points": margin,
                "official_test_used_for_hyperparameter_selection": False,
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
            },
            "metrics": {
                "assertions": {
                    "source_top4_predictions_exact": exact_reproduction,
                    "all_1500_predictions_retained": len(rows) == 1500,
                    "all_slices_complete": len(slices) == 15
                    and all(row["records"] == 100 for row in slices),
                },
                "comparisons": comparisons,
                "slices": slices,
                "topk_failure_slice_count": topk_count,
                "router_code_failure_slice_count": router_count,
            },
            "artifacts": {"predictions_dir": str(output_dir)},
        }
        if not all(result["metrics"]["assertions"].values()):
            result["status"] = "failed"
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
