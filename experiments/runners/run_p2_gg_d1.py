#!/usr/bin/env python3
"""Frozen counterfactual audit of the P2 matched-GG epoch-end collapse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.core import inject_ability_layers, iter_ability_layers
from badit_tf.gg_diagnostic import (
    assignment_fragmentation,
    classify_collapse,
    labels_from_primitive_indices,
    prediction_degeneracy,
)
from badit_tf.p2_evaluation import (
    clear_route_overrides,
    load_record,
    score_predictions,
    set_route_masks,
)
from badit_tf.training import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--local-rank", type=int, default=-1)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    wanted = ("lora_A", "lora_B", "initial_lora_A", "initial_lora_B", "primitive_indices")
    for name, tensor in sorted(
        list(model.named_parameters()) + list(model.named_buffers())
    ):
        if not any(marker in name for marker in wanted):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint_with_initial_bank(
    model: torch.nn.Module, checkpoint_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainable = payload["trainable_state"]
    routing = payload["routing_buffers"]
    post_labels = {}
    for name, layer in iter_ability_layers(model):
        key = f"{name}.primitive_indices"
        labels = labels_from_primitive_indices(routing[key])
        post_labels[name] = labels
        # Fresh initial banks are contiguous.  Regroup them before loading the
        # trained current factors so the fixed subtraction path matches the
        # source model's post-regroup state exactly.
        layer.apply_assignment(labels.to(layer.primitive_indices.device))

    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    missing_parameters = sorted(set(trainable).difference(named_parameters))
    missing_buffers = sorted(set(routing).difference(named_buffers))
    if missing_parameters or missing_buffers:
        raise KeyError(
            f"checkpoint mismatch: parameters={missing_parameters}, buffers={missing_buffers}"
        )
    with torch.no_grad():
        for name, value in trainable.items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
        for name, value in routing.items():
            named_buffers[name].copy_(value.to(named_buffers[name].device))
    return post_labels, payload["metadata"]


def apply_grouping(
    model: torch.nn.Module,
    grouping: str,
    post_labels: dict[str, torch.Tensor],
) -> None:
    for name, layer in iter_ability_layers(model):
        if grouping == "post_regroup":
            labels = post_labels[name]
        elif grouping == "canonical":
            labels = torch.arange(layer.total_primitives) // layer.rank
        else:
            raise ValueError(grouping)
        layer.apply_assignment(labels.to(layer.primitive_indices.device))


@torch.no_grad()
def all_one_fp32_grouping_probe(
    model: torch.nn.Module,
    post_labels: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Check the true all-one invariant before autoregressive amplification."""

    def layer_delta(layer: Any, x: torch.Tensor) -> torch.Tensor:
        current_hidden = torch.einsum(
            "bi,kri->bkr", x, layer.lora_A.detach().float()
        )
        initial_hidden = torch.einsum(
            "bi,kri->bkr", x, layer.initial_lora_A.detach().float()
        )
        current = torch.einsum(
            "bkr,kor->bo", current_hidden, layer.lora_B.detach().float()
        )
        initial = torch.einsum(
            "bkr,kor->bo", initial_hidden, layer.initial_lora_B.detach().float()
        )
        return (current - initial) * layer.scaling

    apply_grouping(model, "post_regroup", post_labels)
    post_outputs = {}
    inputs = {}
    for index, (name, layer) in enumerate(iter_ability_layers(model)):
        generator = torch.Generator(device="cpu").manual_seed(91000 + index)
        x = torch.randn(2, layer.in_features, generator=generator, dtype=torch.float32)
        x = x.to(layer.lora_A.device)
        inputs[name] = x
        post_outputs[name] = layer_delta(layer, x)
    apply_grouping(model, "canonical", post_labels)
    per_layer = {}
    for name, layer in iter_ability_layers(model):
        canonical = layer_delta(layer, inputs[name])
        reference = post_outputs[name]
        difference = canonical - reference
        per_layer[name] = {
            "max_abs": float(difference.abs().max().item()),
            "relative_l2": float(
                torch.linalg.vector_norm(difference).item()
                / max(torch.linalg.vector_norm(reference).item(), 1e-30)
            ),
            "close": bool(torch.allclose(canonical, reference, atol=atol, rtol=rtol)),
        }
    apply_grouping(model, "post_regroup", post_labels)
    return {
        "atol": float(atol),
        "rtol": float(rtol),
        "all_layers_close": all(row["close"] for row in per_layer.values()),
        "max_abs": max(row["max_abs"] for row in per_layer.values()),
        "max_relative_l2": max(row["relative_l2"] for row in per_layer.values()),
        "per_layer": per_layer,
    }


@torch.no_grad()
def greedy_generate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    route_mode: str,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int]:
    encoded = tokenizer(
        record["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    clear_route_overrides(model)
    for _, layer in iter_ability_layers(model):
        layer.topk_enabled.fill_(route_mode == "trained_top4")
        if route_mode == "all_one":
            layer.set_route_code_override(
                torch.ones(1, layer.num_experts, device=device, dtype=input_ids.dtype)
            )
    set_route_masks(model, attention_mask)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    for _, layer in iter_ability_layers(model):
        layer.set_route_code_override(layer._last_route["actual"])
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
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)


def reference_predictions(source_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(source_dir.glob("predictions_rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                rows[row["sample_id"]] = row
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
        raise RuntimeError(f"world size {world_size} != {config['world_size']}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_everything(int(source_config["seed"]))

    output_dir = Path(config["output_root"]) / args.run_id
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    manifest_path = Path(source_config["order_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_records = manifest["test_records"]
    if args.test_limit is not None:
        test_records = test_records[: args.test_limit]
    selected_ids = [row["sample_id"] for row in test_records]
    checkpoint_path = Path(config["source_checkpoint"])
    source_result_path = Path(config["source_result"])
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(
        source_config["model_path"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = getattr(torch, str(source_config["torch_dtype"]))
    model = AutoModelForCausalLM.from_pretrained(
        source_config["model_path"],
        torch_dtype=dtype,
        attn_implementation=source_config["attention_implementation"],
        local_files_only=True,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inject_ability_layers(
        model,
        target_suffixes=tuple(source_config["target_modules"]),
        assignments={},
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
    post_labels, checkpoint_metadata = load_checkpoint_with_initial_bank(
        model, checkpoint_path
    )
    model.eval()
    model.config.use_cache = True
    post_state_sha256 = tensor_state_sha256(model)
    fp32_probe = all_one_fp32_grouping_probe(
        model,
        post_labels,
        atol=float(config["all_one_fp32_probe_atol"]),
        rtol=float(config["all_one_fp32_probe_rtol"]),
    )
    probe_pass = torch.tensor(
        int(fp32_probe["all_layers_close"]), device=device, dtype=torch.int32
    )
    dist.all_reduce(probe_pass, op=dist.ReduceOp.MIN)

    conditions = config["conditions"]
    cache: dict[str, dict] = {}
    for condition in conditions:
        name = condition["name"]
        apply_grouping(model, condition["grouping"], post_labels)
        output_path = output_dir / f"{name}_rank{rank}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record_meta in test_records[rank::world_size]:
                record = load_record(record_meta, cache)
                # Keep the prompt constructor frozen to the source P2 evaluator.
                from badit_tf.calibration import prompt_for_record

                record["prompt"] = prompt_for_record(record)
                prediction, generated_tokens = greedy_generate_condition(
                    model,
                    tokenizer,
                    record,
                    route_mode=condition["routing"],
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
                            "sample_id": record["sample_id"],
                            "task": record["task"],
                            "prediction": prediction,
                            "references": references,
                            "generated_tokens": generated_tokens,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        dist.barrier()
        if rank == 0:
            print(json.dumps({"event": "condition_complete", "condition": name}), flush=True)

    apply_grouping(model, "post_regroup", post_labels)
    local_roundtrip = int(tensor_state_sha256(model) == post_state_sha256)
    roundtrip = torch.tensor(local_roundtrip, device=device, dtype=torch.int32)
    dist.all_reduce(roundtrip, op=dist.ReduceOp.MIN)
    dist.barrier()

    if rank == 0:
        condition_rows = {}
        for condition in conditions:
            name = condition["name"]
            rows = []
            for path in sorted(output_dir.glob(f"{name}_rank*.jsonl")):
                rows.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            rows_by_id = {row["sample_id"]: row for row in rows}
            condition_rows[name] = [rows_by_id[sample_id] for sample_id in selected_ids]

        source_predictions = reference_predictions(Path(config["source_predictions_dir"]))
        reproduced = all(
            condition_rows["post_regroup_top4"][index]["prediction"]
            == source_predictions[sample_id]["prediction"]
            for index, sample_id in enumerate(selected_ids)
        )
        scores = {name: score_predictions(rows) for name, rows in condition_rows.items()}
        degeneracy = {
            name: prediction_degeneracy(rows) for name, rows in condition_rows.items()
        }
        rouge_l = {name: value["macro"]["rougeL"] for name, value in scores.items()}
        classification = classify_collapse(
            rouge_l,
            causal_margin_points=float(config["causal_margin_points"]),
        )
        source_hashes_exact = {
            "source_config": file_sha256(source_config_path)
            == config["expected_sha256"]["source_config"],
            "source_checkpoint": file_sha256(checkpoint_path)
            == config["expected_sha256"]["source_checkpoint"],
            "source_result": file_sha256(source_result_path)
            == config["expected_sha256"]["source_result"],
            "source_assignment": file_sha256(config["source_assignment"])
            == config["expected_sha256"]["source_assignment"],
            "order_manifest_file": file_sha256(manifest_path)
            == config["expected_sha256"]["order_manifest_file"],
            "order_manifest_content": manifest["manifest_sha256"]
            == config["expected_sha256"]["order_manifest_content"],
            "test_ids": manifest["test_ids_sha256"]
            == config["expected_sha256"]["test_ids"],
        }
        all_one_exact = all(
            left["prediction"] == right["prediction"]
            for left, right in zip(
                condition_rows["post_regroup_all_one"],
                condition_rows["canonical_all_one"],
            )
        )
        expected_records = len(selected_ids)
        all_one_agreement = sum(
            left["prediction"] == right["prediction"]
            for left, right in zip(
                condition_rows["post_regroup_all_one"],
                condition_rows["canonical_all_one"],
            )
        ) / max(expected_records, 1)
        formal = args.test_limit is None
        source_metric_exact = (
            not formal
            or abs(
                rouge_l["post_regroup_top4"]
                - float(source_result["metrics"]["rouge"]["macro"]["rougeL"])
            )
            <= float(config["source_metric_tolerance"])
        )
        assertions = {
            "world_size_8": world_size == int(config["world_size"]),
            "all_source_hashes_exact": all(source_hashes_exact.values()),
            "checkpoint_metadata_exact": (
                checkpoint_metadata["run_id"] == config["source_run_id"]
                and checkpoint_metadata["variant"] == "gg"
                and checkpoint_metadata["config_sha256"]
                == config["expected_sha256"]["source_config"]
                and checkpoint_metadata["order_manifest_content_sha256"]
                == config["expected_sha256"]["order_manifest_content"]
            ),
            "all_conditions_complete": all(
                len(rows) == expected_records
                and len({row["sample_id"] for row in rows}) == expected_records
                for rows in condition_rows.values()
            ),
            "post_regroup_source_predictions_exact": reproduced,
            "post_regroup_source_metric_exact": source_metric_exact,
            "all_one_grouping_fp32_probe_close_all_ranks": bool(probe_pass.item()),
            "post_grouping_roundtrip_exact_all_ranks": bool(roundtrip.item()),
            "no_test_record_removed": all(
                len(rows) == expected_records for rows in condition_rows.values()
            ),
        }
        assignments = source_result["gg_assignment"]
        fragmentation = assignment_fragmentation(
            assignments,
            num_experts=int(source_config["num_experts"]),
            rank=int(source_config["rank"]),
        )
        zero_counts = [
            int(item["zero_gradient_primitives"])
            for item in source_result["gg_regroup_audit"].values()
        ]
        status = "complete" if all(assertions.values()) else "failed"
        result = {
            "run_id": args.run_id,
            "experiment_id": "P2-GG-D1",
            "status": status,
            "decision": (
                ("SMOKE_PASSED" if not formal else classification["primary"])
                if status == "complete"
                else "DIAGNOSTIC_INVALID"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": os.popen("git rev-parse HEAD").read().strip(),
            "config_path": str(args.config),
            "config_sha256": file_sha256(args.config),
            "protocol": {
                "source_run_id": config["source_run_id"],
                "training_rerun": False,
                "formal": formal,
                "test_limit": args.test_limit,
                "test_records": expected_records,
                "conditions": conditions,
                "causal_margin_points": float(config["causal_margin_points"]),
                "test_used_for_parameter_selection": False,
            },
            "hashes": {
                **{key: file_sha256(path) for key, path in {
                    "source_config": source_config_path,
                    "source_checkpoint": checkpoint_path,
                    "source_result": source_result_path,
                    "source_assignment": Path(config["source_assignment"]),
                    "order_manifest_file": manifest_path,
                }.items()},
                "order_manifest_content": manifest["manifest_sha256"],
                "test_ids": manifest["test_ids_sha256"],
                "loaded_post_grouping_state": post_state_sha256,
            },
            "metrics": {
                "assertions": assertions,
                "source_hash_assertions": source_hashes_exact,
                "scores": scores,
                "macro_rougeL": rouge_l,
                "degeneracy": degeneracy,
                "causal_classification": classification,
                "all_one_grouping_numerics": {
                    "fp32_layer_probe": fp32_probe,
                    "autoregressive_prediction_exact": all_one_exact,
                    "autoregressive_prediction_agreement_fraction": all_one_agreement,
                    "macro_rougeL_difference": float(
                        rouge_l["canonical_all_one"]
                        - rouge_l["post_regroup_all_one"]
                    ),
                    "interpretation": (
                        "FP32 local linear probe is the invariant; BF16 greedy text "
                        "agreement is report-only because summation-order noise can "
                        "amplify autoregressively."
                    ),
                },
                "assignment_fragmentation": fragmentation,
                "last_backward_zero_gradient_primitives": {
                    "total": int(sum(zero_counts)),
                    "fraction": float(sum(zero_counts) / (len(zero_counts) * 32)),
                    "min_per_layer": int(min(zero_counts)),
                    "median_per_layer": float(torch.tensor(zero_counts).median().item()),
                    "max_per_layer": int(max(zero_counts)),
                },
            },
            "artifacts": {
                "predictions_dir": str(output_dir),
                "source_result": str(source_result_path),
                "source_checkpoint": str(checkpoint_path),
            },
        }
        if status == "failed":
            result["failure_reason"] = "one or more frozen diagnostic integrity assertions failed"
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "diagnostic_complete", "result": result}, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
