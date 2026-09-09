#!/usr/bin/env python3
"""One restart-delimited task stage of the eight-GPU P3 sequential pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import supervised_route_attention_mask, supervised_tensors
from badit_tf.core import advance_router_step, iter_ability_layers
from badit_tf.runtime import (
    apply_layer_assignments,
    inject_from_config,
    load_initial_bank_checkpoint,
)
from badit_tf.sequential import (
    capture_rng_state,
    capture_trainable_anchor,
    deterministic_rehearsal_rank_records,
    effective_task_step_budgets,
    load_model_and_rng_checkpoint,
    load_rank_checkpoint,
    optimizer_updates_match_training_steps,
    save_rank_checkpoint,
    scheduler_initial_state,
    relative_squared_trainable_drift,
    state_sha256,
    topk_schedule_matches_protocol,
    trainable_and_ability_state,
)
from badit_tf.training import (
    balanced_gradient_geometry_assignment,
    canonical_json_sha256,
    file_sha256,
    nonzero_cosine_warmup_multiplier,
    public_badit_gradient_regroup,
    resolve_assignment_labels,
    resolve_code_commit,
    residual_router_optimizer_groups,
)
from run_p2_train import (
    clear_route_overrides,
    greedy_generate,
    load_record,
    score_predictions,
    seed_everything,
    set_route_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--steps-per-task", type=int)
    parser.add_argument(
        "--task-step-budgets",
        help="Comma-separated per-task smoke budgets; each must not exceed the natural budget.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def permitted_resume_config_sha256s(config: dict[str, Any], current_sha256: str) -> set[str]:
    """Return only explicitly frozen config identities eligible for resume.

    H3 recovery registrations use ``source_config_sha256`` while older H2
    lineage retries use ``recovery_preserves_h2_config_sha256``.  Both are
    provenance declarations, not interchangeable defaults: accepting either
    is safe only when it is present in the frozen recovery config itself.
    """

    permitted = {current_sha256}
    for key in ("source_config_sha256", "recovery_preserves_h2_config_sha256"):
        value = str(config.get(key, "")).strip()
        if value:
            permitted.add(value)
    return permitted


def task_rank_records(task: dict[str, Any], rank: int, world_size: int) -> list[dict]:
    ordered = [record for epoch in task["epoch_orders"] for record in epoch]
    return ordered[rank::world_size]


def task_evaluation_records(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Read explicit tuning records while preserving legacy P3 manifests."""

    if "evaluation_records" in task:
        return task["evaluation_records"]
    return task["test_records"]


def task_evaluation_count(task: dict[str, Any]) -> int:
    if "evaluation_count" in task:
        return int(task["evaluation_count"])
    return int(task["test_count"])


def read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON without splitting Unicode record content.

    ``str.splitlines`` also treats U+2028/U+2029 as line boundaries.  Those
    characters are legal inside a JSON string and can be emitted by a model,
    so JSONL must be split only on physical file newlines.
    """

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline=None) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line:
                rows.append(json.loads(line))
    return rows


def build_cosine_scheduler(
    optimizer: Any, *, total_steps: int, warmup_ratio: float
) -> LambdaLR:
    warmup_steps = max(1, int(math.ceil(total_steps * float(warmup_ratio))))
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: nonzero_cosine_warmup_multiplier(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )


def build_engine(config: dict[str, Any], device: torch.device, rank: int, total_steps: int):
    assignments_payload = json.loads(
        Path(config["tf_assignment_path"]).read_text(encoding="utf-8")
    )
    # ``gg`` denotes the sequential gradient-regrouping variant.  It must
    # validate the same frozen TF assignment shape, but it must *not* apply
    # that TF grouping before its first gradient collection: the canonical
    # warm-start bank is the paired starting state for TF and GG.
    assignment_method = str(config.get("assignment_method", "tf"))
    is_gg = assignment_method == "gg"
    resolved_labels = resolve_assignment_labels(
        assignments_payload,
        method="tf" if is_gg else assignment_method,
        epsilon=float(config["tf_assignment_epsilon"]),
        num_experts=int(config["num_experts"]),
        rank=int(config["rank"]),
        random_index=int(config.get("assignment_random_index", 0)),
        allow_unequal_capacity=bool(config.get("allow_unequal_capacity", False)),
    )
    assignments = {
        name: torch.tensor(labels, dtype=torch.long)
        for name, labels in resolved_labels.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=getattr(torch, str(config["torch_dtype"])),
        attn_implementation=config["attention_implementation"],
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inject_from_config(
        model,
        config,
        assignments=None if config.get("initial_bank_checkpoint") else assignments,
    )
    initial_bank = load_initial_bank_checkpoint(model, config)
    if initial_bank is not None and not is_gg:
        apply_layer_assignments(model, assignments)
    elif initial_bank is not None:
        from badit_tf.runtime import assert_canonical_layer_assignments

        assert_canonical_layer_assignments(model)
    for _, layer in iter_ability_layers(model):
        layer.disable_calibration()
    seed_everything(
        int(config["seed"])
        + (rank if bool(config.get("rank_offset_seed", True)) else 0)
    )
    router_lr_multiplier = float(config.get("router_lr_multiplier", 1.0))
    optimizer = AdamW(
        residual_router_optimizer_groups(
            model,
            learning_rate=float(config["learning_rate"]),
            router_lr_multiplier=router_lr_multiplier,
        ),
        betas=tuple(config["adam_betas"]),
        eps=float(config["adam_epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(config["warmup_ratio"]),
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": int(config["world_size"]),
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
        },
        "gradient_clipping": float(config["gradient_clipping"]),
        "steps_per_print": 1_000_000,
        "wall_clock_breakdown": False,
    }
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
        config=ds_config,
    )
    return engine, optimizer, scheduler, tokenizer, assignments, initial_bank


def batch_for_record(
    record_meta: dict[str, Any],
    *,
    cache: dict[str, dict],
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    record = load_record(record_meta, cache)
    batch = supervised_tensors(
        tokenizer,
        record,
        max_length=int(config["max_sequence_length"]),
        max_target_length=int(config["max_target_length"]),
        device=device,
    )
    return record, batch


@torch.no_grad()
def fixed_forward_signature(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    route_pooling_scope: str,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    clear_route_overrides(model)
    set_route_masks(
        model,
        supervised_route_attention_mask(batch, scope=route_pooling_scope),
    )
    outputs = model(**batch, use_cache=False)
    routes = {
        name: {
            key: value.detach().cpu()
            for key, value in layer._last_route.items()
        }
        for name, layer in iter_ability_layers(model)
    }
    selected_logits = outputs.logits[:, -1, :256].detach().cpu()
    result = {
        "loss": float(outputs.loss.detach().float().item()),
        "output_sha256": state_sha256(
            {"loss": outputs.loss.detach().cpu(), "selected_logits": selected_logits}
        ),
        "route_sha256": state_sha256(routes),
    }
    if was_training:
        model.train()
    return result


def train_step(
    engine: Any,
    batch: dict[str, torch.Tensor],
    scheduler: Any,
    world_size: int,
    *,
    route_pooling_scope: str,
    anchor_state: dict[str, torch.Tensor] | None = None,
    anchor_coefficient: float = 0.0,
    rehearsal_batch: dict[str, torch.Tensor] | None = None,
    rehearsal_coefficient: float = 0.0,
) -> dict[str, float]:
    engine.train()
    clear_route_overrides(engine.module)
    set_route_masks(
        engine.module,
        supervised_route_attention_mask(batch, scope=route_pooling_scope),
    )
    outputs = engine(**batch, use_cache=False)
    nll = outputs.loss
    if not torch.isfinite(nll):
        raise FloatingPointError(f"non-finite training NLL: {nll}")
    if anchor_coefficient < 0:
        raise ValueError("anchor_coefficient must be non-negative")
    if rehearsal_coefficient < 0:
        raise ValueError("rehearsal_coefficient must be non-negative")
    if anchor_state is None:
        anchor_penalty = torch.zeros((), dtype=torch.float32, device=nll.device)
    else:
        anchor_penalty = relative_squared_trainable_drift(
            engine.module, anchor_state
        )
    primary_objective = nll + 0.5 * float(anchor_coefficient) * anchor_penalty
    if not torch.isfinite(primary_objective):
        raise FloatingPointError(
            f"non-finite primary training objective: {primary_objective}"
        )
    reduced_nll = nll.detach().float().clone()
    dist.all_reduce(reduced_nll, op=dist.ReduceOp.SUM)
    reduced_nll.div_(world_size)
    engine.backward(primary_objective)
    rehearsal_nll = torch.zeros((), dtype=torch.float32, device=nll.device)
    if rehearsal_batch is not None and rehearsal_coefficient > 0:
        clear_route_overrides(engine.module)
        set_route_masks(
            engine.module,
            supervised_route_attention_mask(
                rehearsal_batch, scope=route_pooling_scope
            ),
        )
        rehearsal_outputs = engine(**rehearsal_batch, use_cache=False)
        rehearsal_nll = rehearsal_outputs.loss
        if not torch.isfinite(rehearsal_nll):
            raise FloatingPointError(
                f"non-finite rehearsal NLL: {rehearsal_nll}"
            )
        engine.backward(float(rehearsal_coefficient) * rehearsal_nll)
    reduced_rehearsal_nll = rehearsal_nll.detach().float().clone()
    dist.all_reduce(reduced_rehearsal_nll, op=dist.ReduceOp.SUM)
    reduced_rehearsal_nll.div_(world_size)
    global_objective = (
        reduced_nll
        + 0.5 * float(anchor_coefficient) * anchor_penalty.detach().float()
        + float(rehearsal_coefficient) * reduced_rehearsal_nll
    )
    engine.step()
    advance_router_step(engine.module)
    return {
        "local_nll": float(nll.detach().float().item()),
        "global_nll": float(reduced_nll.item()),
        "local_rehearsal_nll": float(rehearsal_nll.detach().float().item()),
        "global_rehearsal_nll": float(reduced_rehearsal_nll.item()),
        "relative_anchor_penalty": float(anchor_penalty.detach().float().item()),
        "local_objective": float(
            primary_objective.detach().float().item()
            + float(rehearsal_coefficient)
            * rehearsal_nll.detach().float().item()
        ),
        "global_objective": float(global_objective.item()),
    }


def routing_state(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: {
            "router_sha256": state_sha256(
                {
                    "weight": layer.residual_router.weight,
                    "bias": layer.residual_router.bias,
                }
            ),
            "assignment_sha256": state_sha256(
                layer.assignment_matrix()
                if layer.allow_unequal_capacity
                else layer.primitive_indices
            ),
            "topk_enabled": bool(layer.topk_enabled.item()),
            "optimizer_updates": int(layer.optimizer_updates.item()),
        }
        for name, layer in iter_ability_layers(model)
    }


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    cache: dict[str, dict],
    device: torch.device,
    output_dir: Path,
    label: str,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    model.eval()
    model.config.use_cache = True
    records = [record for task in tasks for record in task_evaluation_records(task)]
    path = output_dir / f"predictions_{label}_rank{rank}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record_meta in records[rank::world_size]:
            record = load_record(record_meta, cache)
            prediction = greedy_generate(
                model,
                tokenizer,
                record,
                max_prompt_length=int(config["max_sequence_length"])
                - int(config["max_new_tokens"]),
                max_new_tokens=int(config["max_new_tokens"]),
                device=device,
            )
            output = record["instance"]["output"]
            references = [str(item) for item in output] if isinstance(output, list) else [str(output)]
            handle.write(
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "task": record["task"],
                        "prediction": prediction,
                        "references": references,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    model.config.use_cache = False
    dist.barrier()
    if rank != 0:
        return None
    rows = []
    for prediction_file in sorted(output_dir.glob(f"predictions_{label}_rank*.jsonl")):
        rows.extend(read_prediction_jsonl(prediction_file))
    metrics = score_predictions(rows)
    scores = []
    metric_names = []
    task_aware_scores = []
    task_aware_metric_names = []
    primary_task_metric = str(config.get("primary_task_metric", "task_aware"))
    if primary_task_metric not in {"task_aware", "rougeL"}:
        raise ValueError(f"unsupported primary_task_metric={primary_task_metric}")
    for task in tasks:
        name = task["task"]
        task_aware_metric = (
            "exact_match" if name.endswith("_classification") else "rougeL"
        )
        task_aware_metric_names.append(task_aware_metric)
        task_aware_scores.append(
            float(metrics["per_task"][name][task_aware_metric])
        )
        metric = "rougeL" if primary_task_metric == "rougeL" else task_aware_metric
        metric_names.append(metric)
        scores.append(float(metrics["per_task"][name][metric]))
    result = {
        "label": label,
        "records": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "expected_records": sum(task_evaluation_count(task) for task in tasks),
        "task_metrics": metric_names,
        "scores": scores,
        "primary_task_metric": primary_task_metric,
        "task_aware_metrics": task_aware_metric_names,
        "task_aware_scores": task_aware_scores,
        "rouge_bundle": metrics,
    }
    (output_dir / f"evaluation_{label}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def post_step_state(model: torch.nn.Module, optimizer: Any, scheduler: Any) -> dict[str, str]:
    return {
        "model_state_sha256": state_sha256(trainable_and_ability_state(model)),
        "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
        "scheduler_state_sha256": state_sha256(scheduler.state_dict()),
        "rng_state_sha256": state_sha256(capture_rng_state()),
    }


def distributed_hash_consensus(
    value: Any, *, device: torch.device, world_size: int
) -> tuple[bool, list[str]]:
    digest = bytes.fromhex(state_sha256(value))
    local = torch.tensor(list(digest), dtype=torch.uint8, device=device)
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    hashes = [bytes(item.cpu().tolist()).hex() for item in gathered]
    return len(set(hashes)) == 1, hashes


def apply_public_epoch_regroup(
    model: torch.nn.Module,
    *,
    current_epoch: int,
    max_epochs: int,
    device: torch.device,
    world_size: int,
    gradient_sync: str,
    capacity_repair: str,
) -> dict[str, Any]:
    if gradient_sync not in {"none", "mean"}:
        raise ValueError(f"unsupported public gradient sync={gradient_sync}")
    if capacity_repair not in {"public_greedy", "exact_capacity"}:
        raise ValueError(f"unsupported public capacity repair={capacity_repair}")
    event: dict[str, Any] = {
        "epoch": int(current_epoch),
        "max_epochs": int(max_epochs),
        "applied": current_epoch <= max_epochs // 2,
        "gradient_sync": gradient_sync,
        "capacity_repair": capacity_repair,
        "layers": {},
    }
    if event["applied"]:
        for name, layer in iter_ability_layers(model):
            gradient_vectors = layer.cached_public_gradient_vectors()
            if gradient_sync == "mean":
                dist.all_reduce(gradient_vectors, op=dist.ReduceOp.SUM)
                gradient_vectors.div_(world_size)
            if capacity_repair == "public_greedy":
                clusters, audit = public_badit_gradient_regroup(
                    gradient_vectors,
                    num_experts=layer.num_experts,
                    rank=layer.rank,
                    rng=np.random,
                )
            else:
                labels, assignment_audit = balanced_gradient_geometry_assignment(
                    gradient_vectors,
                    num_experts=layer.num_experts,
                    rank=layer.rank,
                    seed=0,
                )
                clusters = [
                    np.flatnonzero(labels == expert).astype(int).tolist()
                    for expert in range(layer.num_experts)
                ]
                flat = [index for cluster in clusters for index in cluster]
                audit = {
                    **assignment_audit,
                    "unique_source_slots": len(set(flat)),
                    "duplicate_copies": len(flat) - len(set(flat)),
                    "omitted_source_slots": layer.num_experts * layer.rank
                    - len(set(flat)),
                    "clusters": clusters,
                }
            layer.apply_public_slot_regroup(clusters)
            event["layers"][name] = audit
    consensus, rank_hashes = distributed_hash_consensus(
        trainable_and_ability_state(model),
        device=device,
        world_size=world_size,
    )
    event["model_state_cross_rank_consensus"] = consensus
    event["rank_model_state_sha256"] = rank_hashes
    event["duplicate_copies_total"] = sum(
        int(audit["duplicate_copies"]) for audit in event["layers"].values()
    )
    event["omitted_source_slots_total"] = sum(
        int(audit["omitted_source_slots"]) for audit in event["layers"].values()
    )
    return event


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size != int(config["world_size"]):
        raise RuntimeError(f"world size {world_size} != {config['world_size']}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if bool(config.get("deterministic_training", False)):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(int(config["seed"]))

    output_dir = Path(config["output_root"]) / args.run_id
    stage_success = torch.ones(1, dtype=torch.int32, device=device)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
    dist.barrier()
    manifest = json.loads(Path(config["sequential_manifest"]).read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if not 0 <= args.task_index < len(tasks):
        raise ValueError("invalid task index")
    natural_task_step_budgets = effective_task_step_budgets(tasks, world_size)
    if args.task_step_budgets:
        task_step_budgets = [
            int(value) for value in args.task_step_budgets.split(",")
        ]
        if len(task_step_budgets) != len(tasks):
            raise ValueError("task-step-budgets length must equal task count")
        if any(
            budget <= 0 or budget > natural
            for budget, natural in zip(
                task_step_budgets, natural_task_step_budgets, strict=True
            )
        ):
            raise ValueError(
                f"invalid task-step-budgets={task_step_budgets}; "
                f"natural={natural_task_step_budgets}"
            )
    else:
        task_step_budgets = effective_task_step_budgets(
            tasks, world_size, cap=args.steps_per_task
        )
    steps_this_task = task_step_budgets[args.task_index]
    total_steps = sum(task_step_budgets)
    scheduler_scope = str(config.get("scheduler_scope", "global"))
    if scheduler_scope not in {"global", "per_task"}:
        raise ValueError(f"unsupported scheduler_scope={scheduler_scope}")
    optimizer_scope = str(config.get("optimizer_scope", "global"))
    if optimizer_scope not in {"global", "per_task"}:
        raise ValueError(f"unsupported optimizer_scope={optimizer_scope}")
    route_pooling_scope = str(
        config.get("route_pooling_scope", "full_supervised_sequence")
    )
    if route_pooling_scope not in {"full_supervised_sequence", "prompt_only"}:
        raise ValueError(f"unsupported route_pooling_scope={route_pooling_scope}")
    stage_anchor_coefficient = float(
        config.get("stage_anchor_coefficient", 0.0)
    )
    if stage_anchor_coefficient < 0:
        raise ValueError("stage_anchor_coefficient must be non-negative")
    if stage_anchor_coefficient > 0 and optimizer_scope != "per_task":
        raise ValueError(
            "stage anchoring requires optimizer_scope=per_task so no "
            "unanchored resume-reference update occurs"
        )
    rehearsal_coefficient = float(config.get("rehearsal_coefficient", 0.0))
    if rehearsal_coefficient < 0:
        raise ValueError("rehearsal_coefficient must be non-negative")
    if rehearsal_coefficient > 0 and optimizer_scope != "per_task":
        raise ValueError(
            "functional rehearsal requires optimizer_scope=per_task so the "
            "restart reference does not consume an unregistered replay sample"
        )
    scheduler_steps = steps_this_task if scheduler_scope == "per_task" else total_steps
    engine, optimizer, scheduler, tokenizer, assignments, initial_bank = build_engine(
        config, device, rank, scheduler_steps
    )
    initialization_method = str(config.get("initialization_method", "svd"))
    apply_lora_dropout = bool(config.get("apply_lora_dropout", True))
    initialization_audit = {
        name: {
            "initialization_method": layer.initialization_method,
            "apply_lora_dropout": layer.apply_lora_dropout,
            "lora_A_nonzero": int(torch.count_nonzero(layer.lora_A).item()),
            "lora_B_nonzero": int(torch.count_nonzero(layer.lora_B).item()),
            "initial_lora_A_nonzero": int(
                torch.count_nonzero(layer.initial_lora_A).item()
            ),
            "initial_lora_B_nonzero": int(
                torch.count_nonzero(layer.initial_lora_B).item()
            ),
            "initial_weight_relative_error": float(
                layer.initial_weight_relative_error.item()
            ),
        }
        for name, layer in iter_ability_layers(engine.module)
    }
    initial_routing_state = routing_state(engine.module)
    initial_model_consensus, initial_rank_model_hashes = distributed_hash_consensus(
        trainable_and_ability_state(engine.module),
        device=device,
        world_size=world_size,
    )
    fresh_optimizer_state_sha256 = state_sha256(optimizer.state_dict())
    fresh_scheduler_state = scheduler_initial_state(scheduler)
    assignment_sha256 = canonical_json_sha256(
        {name: labels.tolist() for name, labels in sorted(assignments.items())}
    )
    config_sha256 = file_sha256(args.config)
    permitted_resume_config_sha256 = permitted_resume_config_sha256s(config, config_sha256)
    cache: dict[str, dict] = {}
    task = tasks[args.task_index]
    records = task_rank_records(task, rank, world_size)[:steps_this_task]
    if len(records) != steps_this_task:
        raise RuntimeError("task rank order shorter than configured budget")
    rehearsal_records = (
        deterministic_rehearsal_rank_records(
            tasks,
            current_task_index=args.task_index,
            rank=rank,
            world_size=world_size,
            seed=int(config["seed"]),
        )
        if args.task_index > 0 and rehearsal_coefficient > 0
        else []
    )
    if args.task_index > 0 and rehearsal_coefficient > 0 and not rehearsal_records:
        raise RuntimeError("rehearsal was enabled but no prior-task records exist")
    rehearsal_used = rehearsal_records[:steps_this_task]
    if rehearsal_records and len(rehearsal_used) < steps_this_task:
        rehearsal_used = [
            rehearsal_records[index % len(rehearsal_records)]
            for index in range(steps_this_task)
        ]
    validation_ids = {
        record["sample_id"]
        for prior_task in tasks
        for record in task_evaluation_records(prior_task)
    }
    rehearsal_ids = {record["sample_id"] for record in rehearsal_used}
    rehearsal_validation_overlap = rehearsal_ids & validation_ids
    rehearsal_task_counts: dict[str, int] = {}
    for record in rehearsal_used:
        rehearsal_task_counts[record["task"]] = (
            rehearsal_task_counts.get(record["task"], 0) + 1
        )

    resume_assertions: dict[str, bool] = {}
    resume_observed: dict[str, Any] = {}
    expected_global_step = sum(task_step_budgets[: args.task_index])
    if args.task_index == 0:
        if args.resume_from is not None:
            raise RuntimeError("task 0 must not resume")
    else:
        if args.resume_from is None:
            raise RuntimeError("later tasks require --resume-from")
        if optimizer_scope == "per_task":
            metadata, load_assertions, loaded_hashes = load_model_and_rng_checkpoint(
                args.resume_from,
                rank=rank,
                model=engine.module,
                map_location="cpu",
            )
            load_assertions["optimizer_reinitialized_for_task"] = (
                state_sha256(optimizer.state_dict())
                == fresh_optimizer_state_sha256
            )
            load_assertions["scheduler_reinitialized_for_task"] = (
                scheduler.state_dict() == fresh_scheduler_state
            )
        else:
            metadata, load_assertions, loaded_hashes = load_rank_checkpoint(
                args.resume_from,
                rank=rank,
                model=engine.module,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location="cpu",
            )
        resume_assertions.update({f"load_{key}": value for key, value in load_assertions.items()})
        resume_assertions["scheduler_restarted_for_task"] = bool(
            scheduler_scope != "per_task"
            or scheduler.state_dict() == fresh_scheduler_state
        )
        resume_assertions.update(
            {
                "metadata_completed_task_consistent": metadata["completed_task_index"] == args.task_index - 1,
                "metadata_next_task_consistent": metadata["next_task_index"] == args.task_index,
                "metadata_global_step_consistent": metadata["global_step"] == expected_global_step,
                # Recovery configs have a new run ID/provenance and therefore a
                # new file SHA.  They may resume an old checkpoint only when
                # the old SHA is explicitly bound in the frozen config.
                "metadata_config_consistent": metadata["config_sha256"]
                in permitted_resume_config_sha256,
                "metadata_config_recovery_source_exact": metadata["config_sha256"]
                in permitted_resume_config_sha256,
                "metadata_manifest_consistent": metadata["manifest_sha256"] == manifest["manifest_sha256"],
                "metadata_assignment_consistent": metadata["assignment_sha256"] == assignment_sha256,
                "metadata_initial_bank_consistent": metadata.get(
                    "initial_bank_checkpoint_sha256"
                )
                == (
                    initial_bank["file_sha256"] if initial_bank is not None else None
                ),
                "metadata_budget_consistent": metadata["task_step_budgets"] == task_step_budgets,
                "metadata_scheduler_scope_consistent": metadata["scheduler_scope"] == scheduler_scope,
                "metadata_optimizer_scope_consistent": metadata.get(
                    "optimizer_scope", "global"
                )
                == optimizer_scope,
                "metadata_route_pooling_scope_consistent": metadata.get(
                    "route_pooling_scope", "full_supervised_sequence"
                )
                == route_pooling_scope,
                "metadata_stage_anchor_coefficient_consistent": float(
                    metadata.get("stage_anchor_coefficient", 0.0)
                )
                == stage_anchor_coefficient,
                "metadata_rehearsal_coefficient_consistent": float(
                    metadata.get("rehearsal_coefficient", 0.0)
                )
                == rehearsal_coefficient,
                "global_switch_step_consistent": metadata["global_switch_step"] == int(config["dense_steps"]),
            }
        )
        _, probe_batch = batch_for_record(
            records[0], cache=cache, tokenizer=tokenizer, config=config, device=device
        )
        observed_forward = fixed_forward_signature(
            engine.module,
            probe_batch,
            route_pooling_scope=route_pooling_scope,
        )
        resume_assertions["fixed_plus_one_forward_exact"] = observed_forward == metadata["fixed_next_forward"]
        observed_routing = routing_state(engine.module)
        resume_assertions["router_parameters_exact"] = all(
            observed_routing[name]["router_sha256"] == expected["router_sha256"]
            for name, expected in metadata["routing_state"].items()
        )
        resume_assertions["topk_state_exact"] = all(
            observed_routing[name]["topk_enabled"] == expected["topk_enabled"]
            for name, expected in metadata["routing_state"].items()
        )
        resume_assertions["optimizer_updates_exact"] = all(
            observed_routing[name]["optimizer_updates"] == expected["optimizer_updates"]
            for name, expected in metadata["routing_state"].items()
        )
        resume_assertions["assignment_exact"] = all(
            observed_routing[name]["assignment_sha256"] == expected["assignment_sha256"]
            for name, expected in metadata["routing_state"].items()
        )
        if optimizer_scope == "global":
            reference = torch.load(
                args.resume_from / f"next_step_reference_rank_{rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            step_metrics = train_step(
                engine,
                probe_batch,
                scheduler,
                world_size,
                route_pooling_scope=route_pooling_scope,
            )
            observed_post = post_step_state(engine.module, optimizer, scheduler)
            local_checks = {
                "resume_first_local_loss_exact": step_metrics["local_nll"]
                == reference["local_loss"],
                "resume_first_global_loss_exact": step_metrics["global_nll"]
                == reference["global_loss"],
                **{
                    f"resume_first_{key}_exact": observed_post[key]
                    == reference["post_step_state"][key]
                    for key in observed_post
                },
            }
            check_tensor = torch.tensor(
                [all(local_checks.values())], dtype=torch.int32, device=device
            )
            dist.all_reduce(check_tensor, op=dist.ReduceOp.MIN)
            resume_assertions.update(local_checks)
            resume_assertions["all_ranks_resume_first_step_exact"] = bool(
                check_tensor.item()
            )
            resume_observed = {
                "load_hashes": loaded_hashes,
                "fixed_forward": observed_forward,
                "first_step_local_loss": step_metrics["local_nll"],
                "first_step_global_loss": step_metrics["global_nll"],
                "post_step_state": observed_post,
            }
            start_step = 1
        else:
            resume_observed = {
                "load_hashes": loaded_hashes,
                "fixed_forward": observed_forward,
                "optimizer_state_sha256_at_task_start": state_sha256(
                    optimizer.state_dict()
                ),
                "optimizer_reset_intentional": True,
            }
            start_step = 0
    if args.task_index == 0:
        start_step = 0

    anchor_state = (
        capture_trainable_anchor(engine.module)
        if args.task_index > 0 and stage_anchor_coefficient > 0
        else None
    )

    if not args.skip_eval and args.task_index == 0:
        baseline = evaluate(
            engine.module,
            tokenizer,
            tasks,
            config=config,
            cache=cache,
            device=device,
            output_dir=output_dir,
            label="baseline",
            rank=rank,
            world_size=world_size,
        )
        if rank == 0 and baseline is None:
            raise AssertionError("rank 0 baseline missing")

    losses = []
    public_regroup_events: list[dict[str, Any]] = []
    public_epoch_regroup = bool(config.get("public_epoch_regroup", False))
    public_gradient_sync = str(config.get("public_gradient_sync", "none"))
    public_capacity_repair = str(
        config.get("public_capacity_repair", "public_greedy")
    )
    epoch_rank_steps = [
        len(epoch) // world_size for epoch in task["epoch_orders"]
    ]
    if any(len(epoch) % world_size for epoch in task["epoch_orders"]):
        raise RuntimeError("epoch order is not divisible by world size")
    if len(set(epoch_rank_steps)) != 1:
        raise RuntimeError(f"unequal rank-local epoch steps: {epoch_rank_steps}")
    steps_per_epoch = epoch_rank_steps[0]
    started = time.monotonic()
    for local_step in range(start_step, steps_this_task):
        _, batch = batch_for_record(
            records[local_step], cache=cache, tokenizer=tokenizer, config=config, device=device
        )
        rehearsal_batch = None
        if rehearsal_used:
            _, rehearsal_batch = batch_for_record(
                rehearsal_used[local_step],
                cache=cache,
                tokenizer=tokenizer,
                config=config,
                device=device,
            )
        step_metrics = train_step(
            engine,
            batch,
            scheduler,
            world_size,
            route_pooling_scope=route_pooling_scope,
            anchor_state=anchor_state,
            anchor_coefficient=(
                stage_anchor_coefficient if args.task_index > 0 else 0.0
            ),
            rehearsal_batch=rehearsal_batch,
            rehearsal_coefficient=(
                rehearsal_coefficient if args.task_index > 0 else 0.0
            ),
        )
        completed_steps = local_step + 1
        if public_epoch_regroup and completed_steps % steps_per_epoch == 0:
            event = apply_public_epoch_regroup(
                engine.module,
                current_epoch=completed_steps // steps_per_epoch,
                max_epochs=int(config["epochs_per_task"]),
                device=device,
                world_size=world_size,
                gradient_sync=public_gradient_sync,
                capacity_repair=public_capacity_repair,
            )
            public_regroup_events.append(event)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "public_badit_epoch_regroup",
                            "task_index": args.task_index,
                            **event,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if rank == 0:
            losses.append(
                {
                    "task_step": local_step,
                    "global_step": expected_global_step + local_step,
                    "loss": step_metrics["global_nll"],
                    "objective": step_metrics["global_objective"],
                    "relative_anchor_penalty": step_metrics[
                        "relative_anchor_penalty"
                    ],
                    "rehearsal_nll": step_metrics["global_rehearsal_nll"],
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "router_learning_rate": float(scheduler.get_last_lr()[1]),
                }
            )
            if local_step % int(config["log_every_steps"]) == 0 or local_step + 1 == steps_this_task:
                print(
                    json.dumps(
                        {
                            "event": "p3_train_step",
                            "task_index": args.task_index,
                            "task": task["task"],
                            "task_step": local_step,
                            "steps_this_task": steps_this_task,
                            "loss": step_metrics["global_nll"],
                            "objective": step_metrics["global_objective"],
                            "relative_anchor_penalty": step_metrics[
                                "relative_anchor_penalty"
                            ],
                            "rehearsal_nll": step_metrics[
                                "global_rehearsal_nll"
                            ],
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    final_relative_anchor_drift = float(
        relative_squared_trainable_drift(engine.module, anchor_state)
        .detach()
        .float()
        .item()
        if anchor_state is not None
        else 0.0
    )

    evaluation = None
    if not args.skip_eval:
        evaluation = evaluate(
            engine.module,
            tokenizer,
            tasks,
            config=config,
            cache=cache,
            device=device,
            output_dir=output_dir,
            label=f"after_task_{args.task_index}",
            rank=rank,
            world_size=world_size,
        )

    next_task_index = args.task_index + 1
    probe_record = (
        task_rank_records(tasks[next_task_index], rank, world_size)[0]
        if next_task_index < len(tasks)
        else records[-1]
    )
    _, probe_batch = batch_for_record(
        probe_record, cache=cache, tokenizer=tokenizer, config=config, device=device
    )
    fixed_next_forward = fixed_forward_signature(
        engine.module,
        probe_batch,
        route_pooling_scope=route_pooling_scope,
    )
    route_state = routing_state(engine.module)
    public_regroup_path = output_dir / f"public_regroup_rank_{rank}.json"
    public_regroup_path.write_text(
        json.dumps(public_regroup_events, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    if (
        scheduler_scope == "per_task"
        and optimizer_scope == "global"
        and next_task_index < len(tasks)
    ):
        # Upstream BADIT invokes Trainer.train once per task.  Transformers
        # discards the scheduler it created at the end of each call, so the
        # Legacy local protocol: restart only the cosine/warmup curve while
        # keeping optimizer moments.  The upstream-faithful per-task optimizer
        # branch instead creates both fresh in the next process.
        scheduler = build_cosine_scheduler(
            scheduler.optimizer,
            total_steps=task_step_budgets[next_task_index],
            warmup_ratio=float(config["warmup_ratio"]),
        )
        engine.lr_scheduler = scheduler
        fresh_scheduler_state = scheduler_initial_state(scheduler)
    scheduler_boundary_reset = bool(
        optimizer_scope == "per_task"
        or scheduler_scope != "per_task"
        or next_task_index >= len(tasks)
        or scheduler.state_dict() == fresh_scheduler_state
    )
    checkpoint_dir = output_dir / "checkpoints" / f"task_{args.task_index}"
    metadata = {
        "run_id": args.run_id,
        "completed_task_index": args.task_index,
        "next_task_index": next_task_index,
        "global_step": expected_global_step + steps_this_task,
        "steps_this_task": steps_this_task,
        "task_step_budgets": task_step_budgets,
        "total_steps": total_steps,
        "scheduler_scope": scheduler_scope,
        "optimizer_scope": optimizer_scope,
        "route_pooling_scope": route_pooling_scope,
        "stage_anchor_coefficient": stage_anchor_coefficient,
        "rehearsal_coefficient": rehearsal_coefficient,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "assignment_sha256": assignment_sha256,
        "initial_bank_checkpoint_sha256": (
            initial_bank["file_sha256"] if initial_bank is not None else None
        ),
        "global_switch_step": int(config["dense_steps"]),
        "routing_state": route_state,
        "fixed_next_forward": fixed_next_forward,
    }
    checkpoint_hashes = save_rank_checkpoint(
        checkpoint_dir,
        rank=rank,
        model=engine.module,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )
    dist.barrier()

    reference_written = optimizer_scope == "global"
    if next_task_index < len(tasks) and optimizer_scope == "global":
        step_metrics = train_step(
            engine,
            probe_batch,
            scheduler,
            world_size,
            route_pooling_scope=route_pooling_scope,
        )
        reference = {
            "source_checkpoint": str(checkpoint_dir),
            "next_task_index": next_task_index,
            "sample_id": probe_record["sample_id"],
            "local_loss": step_metrics["local_nll"],
            "global_loss": step_metrics["global_nll"],
            "post_step_state": post_step_state(engine.module, optimizer, scheduler),
        }
        torch.save(reference, checkpoint_dir / f"next_step_reference_rank_{rank}.pt")
    dist.barrier()

    if rank == 0:
        loss_values = np.asarray([item["loss"] for item in losses], dtype=np.float64)
        objective_values = np.asarray(
            [item["objective"] for item in losses], dtype=np.float64
        )
        assertions = {
            "world_size_8": world_size == int(config["world_size"]),
            "task_order_exact": manifest["selected_tasks"] == list(config["expected_first_tasks"]),
            "assignment_hash_present": bool(assignment_sha256),
            "initialization_method_exact": all(
                item["initialization_method"] == initialization_method
                for item in initialization_audit.values()
            ),
            "effective_dropout_behavior_exact": all(
                item["apply_lora_dropout"] == apply_lora_dropout
                for item in initialization_audit.values()
            ),
            "kaiming_zero_fresh_contract": bool(
                initialization_method != "kaiming_zero"
                or initial_bank is not None
                or all(
                    item["lora_A_nonzero"] > 0
                    and item["lora_B_nonzero"] == 0
                    and item["initial_lora_A_nonzero"] == 0
                    and item["initial_lora_B_nonzero"] == 0
                    and item["initial_weight_relative_error"] == 0.0
                    for item in initialization_audit.values()
                )
            ),
            "initial_bank_checkpoint_exact": bool(
                initial_bank is None
                or initial_bank["file_sha256"]
                == config.get("initial_bank_checkpoint_sha256")
            ),
            "all_layers_topk_enabled": all(item["topk_enabled"] for item in route_state.values()),
            # Retain the legacy key for old aggregators.  Its old literal
            # ``dense_steps == 1`` test incorrectly rejected the uploaded
            # BADIT contract, which enables top-k from the first forward.
            "global_switch_step_one": topk_schedule_matches_protocol(
                dense_steps=int(config["dense_steps"]),
                expected_dense_steps=int(config.get("expected_dense_steps", 1)),
            ),
            "global_switch_step_matches_protocol": topk_schedule_matches_protocol(
                dense_steps=int(config["dense_steps"]),
                expected_dense_steps=int(config.get("expected_dense_steps", 1)),
            ),
            "all_layer_optimizer_updates_match_global_step": (
                optimizer_updates_match_training_steps(
                    route_state,
                    initial_routing_state,
                    metadata["global_step"],
                )
            ),
            "losses_finite": bool(loss_values.size == 0 or np.isfinite(loss_values).all()),
            "objectives_finite": bool(
                objective_values.size == 0 or np.isfinite(objective_values).all()
            ),
            "stage_anchor_scope_exact": bool(anchor_state is not None)
            == bool(args.task_index > 0 and stage_anchor_coefficient > 0),
            "stage_anchor_drift_finite": bool(
                np.isfinite(final_relative_anchor_drift)
            ),
            "rehearsal_scope_exact": bool(rehearsal_used)
            == bool(args.task_index > 0 and rehearsal_coefficient > 0),
            "rehearsal_validation_disjoint": not bool(
                rehearsal_validation_overlap
            ),
            "rehearsal_prior_tasks_only": all(
                record["task"]
                in {item["task"] for item in tasks[: args.task_index]}
                for record in rehearsal_used
            ),
            "rehearsal_count_exact": len(rehearsal_used)
            == (
                steps_this_task
                if args.task_index > 0 and rehearsal_coefficient > 0
                else 0
            ),
            "evaluation_complete": bool(
                args.skip_eval
                or (
                    evaluation is not None
                    and evaluation["records"] == evaluation["expected_records"]
                    and evaluation["unique_sample_ids"] == evaluation["expected_records"]
                )
            ),
            "checkpoint_common_written": (checkpoint_dir / "common.pt").exists(),
            "all_rank_checkpoints_written": all(
                (checkpoint_dir / f"rank_{index}.pt").exists() for index in range(world_size)
            ),
            "next_step_reference_written": bool(
                optimizer_scope == "per_task"
                or
                next_task_index >= len(tasks)
                or all(
                    (checkpoint_dir / f"next_step_reference_rank_{index}.pt").exists()
                    for index in range(world_size)
                )
            ),
            "scheduler_boundary_reset": scheduler_boundary_reset,
            "initial_model_cross_rank_consensus": initial_model_consensus,
            "public_regroup_event_count_expected": bool(
                not public_epoch_regroup
                or len(public_regroup_events)
                == min(
                    int(config["epochs_per_task"]),
                    steps_this_task // steps_per_epoch,
                )
            ),
            "public_regroup_cross_rank_consensus": bool(
                not public_epoch_regroup
                or all(
                    event["model_state_cross_rank_consensus"]
                    for event in public_regroup_events
                )
            ),
            "public_regroup_capacity_integrity": bool(
                not bool(config.get("require_public_regroup_capacity_integrity", False))
                or all(
                    event["duplicate_copies_total"] == 0
                    and event["omitted_source_slots_total"] == 0
                    for event in public_regroup_events
                    if event["applied"]
                )
            ),
            **resume_assertions,
        }
        result = {
            "run_id": args.run_id,
            "experiment_id": str(config.get("experiment_id", "P3")),
            "stage_task_index": args.task_index,
            "task": task["task"],
            "status": "complete" if all(assertions.values()) else "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": resolve_code_commit(),
            "config_path": str(args.config),
            "config_sha256": config_sha256,
            "protocol": {
                "epochs_per_task": int(config["epochs_per_task"]),
                "effective_steps_this_task": steps_this_task,
                "task_step_budgets": task_step_budgets,
                "scheduler_scope": scheduler_scope,
                "optimizer_scope": optimizer_scope,
                "route_pooling_scope": route_pooling_scope,
                "assignment_method": str(config.get("assignment_method", "tf")),
                "allow_unequal_capacity": bool(
                    config.get("allow_unequal_capacity", False)
                ),
                "router_mode": str(config.get("router_mode", "residual_mask")),
                "router_bias": bool(config.get("router_bias", True)),
                "initialization_method": initialization_method,
                "apply_lora_dropout": apply_lora_dropout,
                "svd_method": str(config.get("svd_method", "randomized")),
                "residual_implementation": str(
                    config.get("residual_implementation", "paired_subtraction")
                ),
                "public_epoch_regroup": public_epoch_regroup,
                "public_gradient_sync": public_gradient_sync,
                "public_capacity_repair": public_capacity_repair,
                "rank_offset_seed": bool(config.get("rank_offset_seed", True)),
                "router_lr_multiplier": float(
                    config.get("router_lr_multiplier", 1.0)
                ),
                "stage_anchor_coefficient": stage_anchor_coefficient,
                "stage_anchor_penalty": (
                    "0.5 * coefficient * sum(||theta-theta_previous_stage||^2) "
                    "/ sum(||theta_previous_stage||^2)"
                ),
                "stage_anchor_applied": bool(anchor_state is not None),
                "rehearsal_coefficient": rehearsal_coefficient,
                "rehearsal_objective": (
                    "current token-normalized NLL + coefficient * one "
                    "balanced prior-task tune-train sequence NLL per step"
                ),
                "rehearsal_applied": bool(rehearsal_used),
                "rehearsal_source_role": "prior tune_train manifest epoch orders",
                "primary_task_metric": str(
                    config.get("primary_task_metric", "task_aware")
                ),
                "evaluation_role": str(
                    manifest.get("evaluation_role", "official_test")
                ),
                "formal_budget": args.steps_per_task is None
                and args.task_step_budgets is None,
                "global_batch_size": world_size,
                "restart_boundary_before_task": args.task_index > 0,
                "official_test_used_for_selection": False,
                "initial_bank_checkpoint": initial_bank,
            },
            "hashes": {
                "manifest_file_sha256": file_sha256(config["sequential_manifest"]),
                "manifest_content_sha256": manifest["manifest_sha256"],
                "selected_task_order_sha256": manifest["selected_task_order_sha256"],
                "assignment_sha256": assignment_sha256,
                "checkpoint": checkpoint_hashes,
            },
            "metrics": {
                "loss_curve": losses,
                "final_relative_anchor_drift": final_relative_anchor_drift,
                "rehearsal_record_count": len(rehearsal_used),
                "rehearsal_unique_sample_count": len(rehearsal_ids),
                "rehearsal_sample_ids_sha256": canonical_json_sha256(
                    sorted(rehearsal_ids)
                ),
                "rehearsal_task_counts": rehearsal_task_counts,
                "evaluation": evaluation,
                "routing_state": route_state,
                "fixed_next_forward": fixed_next_forward,
                "resume_observed": resume_observed,
                "initial_rank_model_state_sha256": initial_rank_model_hashes,
                "fresh_initialization_audit": initialization_audit,
                "public_regroup_events": public_regroup_events,
                "assertions": assertions,
            },
            "artifacts": {
                "checkpoint_dir": str(checkpoint_dir),
                "stage_result": str(output_dir / "stages" / f"task_{args.task_index}.json"),
                "evaluation": str(output_dir / f"evaluation_after_task_{args.task_index}.json"),
            },
        }
        stages = output_dir / "stages"
        stages.mkdir(exist_ok=True)
        stage_path = stages / f"task_{args.task_index}.json"
        stage_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "p3_stage_complete", "result": result}, sort_keys=True))
        stage_success.fill_(int(result["status"] == "complete"))
    dist.broadcast(stage_success, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if not bool(stage_success.item()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
