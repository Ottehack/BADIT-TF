#!/usr/bin/env python3
"""Run the Qwen3-4B BADIT-TF initialization pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.core import inject_ability_layers
from badit_tf.splits import load_role_instances


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _task_definition(source_file: str) -> str:
    with Path(source_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    definition = payload.get("Definition", "")
    if isinstance(definition, list):
        definition = definition[0] if definition else ""
    return str(definition).strip()


def _prompt(record: dict[str, Any]) -> str:
    definition = _task_definition(record["source_file"])
    instance = record["instance"]
    return (
        f"{definition}\n\nNow complete the following example -\n"
        f"Input: {instance['input']}\nOutput: "
    )


def _select_samples(manifest: dict[str, Any], per_task: int) -> list[dict[str, Any]]:
    rows = load_role_instances(manifest, "assignment")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    selected = []
    for task in sorted(by_task):
        if len(by_task[task]) < per_task:
            raise ValueError(f"{task} has fewer than {per_task} assignment samples")
        selected.extend(by_task[task][:per_task])
    if len(selected) != 15 * per_task:
        raise AssertionError("P0 must use the same number of samples for all 15 tasks")
    return selected


@torch.no_grad()
def _last_token_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    max_length: int,
) -> torch.Tensor:
    rows = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=False)
        rows.append(output.logits[:, -1, :].float().cpu())
    return torch.cat(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    started_at = time.time()
    result: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "model": config["model_name"],
        "setting": "initialization",
        "seed": config["seed"],
        "git_commit": _git_commit(),
        "config_sha256": _sha256(args.config),
        "status": "running",
        "metrics": {},
        "artifacts": {},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the P0 model pilot")
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        manifest_path = Path(config["split_manifest"])
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        selected = _select_samples(manifest, int(config["samples_per_task"]))
        sample_ids = [row["sample_id"] for row in selected]
        prompts = [_prompt(row) for row in selected]
        with (output_dir / "sample_ids.json").open("w", encoding="utf-8") as handle:
            json.dump(sample_ids, handle, indent=2)
            handle.write("\n")

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config["torch_dtype"]]
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation=config["attention_implementation"],
        ).to("cuda")
        model.eval()
        before = _last_token_logits(
            model, tokenizer, prompts, int(config["max_sequence_length"])
        )

        injection = inject_ability_layers(
            model,
            target_suffixes=tuple(config["target_modules"]),
            num_experts=int(config["num_experts"]),
            rank=int(config["rank"]),
            lora_alpha=float(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            top_k=int(config["top_k"]),
            dense_steps=int(config["dense_steps"]),
            svd_seed=int(config["seed"]),
            svd_oversample=int(config["svd_oversample"]),
            svd_niter=int(config["svd_niter"]),
        )
        model.eval()
        after = _last_token_logits(
            model, tokenizer, prompts, int(config["max_sequence_length"])
        )

        layer_checks = {}
        all_finite = True
        for name, layer in injection.layers.items():
            probe = torch.zeros(
                2, 3, layer.in_features, device=model.device, dtype=dtype
            )
            raw, dense, actual = layer.route(probe)
            matrix = layer.assignment_matrix()
            checks = {
                "weight_relative_error": injection.reconstruction_errors[name],
                "raw_route_max_abs": float(raw.abs().max().item()),
                "dense_code_max_abs_from_one": float(
                    (dense - 1).abs().max().item()
                ),
                "actual_code_max_abs_from_one": float(
                    (actual - 1).abs().max().item()
                ),
                "topk_enabled": bool(layer.topk_enabled.item()),
                "assignment_row_sums_ok": bool(torch.all(matrix.sum(1) == 1)),
                "assignment_column_sums_ok": bool(
                    torch.all(matrix.sum(0) == int(config["rank"]))
                ),
            }
            all_finite = all_finite and all(
                torch.isfinite(parameter).all().item()
                for parameter in (
                    layer.base_layer.weight,
                    layer.lora_A,
                    layer.lora_B,
                    layer.residual_router.weight,
                    layer.residual_router.bias,
                )
            )
            layer_checks[name] = checks

        forward_error = float(
            (
                torch.linalg.vector_norm(after - before)
                / torch.linalg.vector_norm(before).clamp_min(1e-30)
            ).item()
        )
        max_weight_error = max(injection.reconstruction_errors.values())
        assertions = {
            "layer_count_positive": len(injection.layers) > 0,
            "weight_reconstruction": max_weight_error
            < float(config["weight_reconstruction_tolerance"]),
            "forward_reconstruction": forward_error
            < float(config["forward_reconstruction_tolerance"]),
            "raw_route_zero": all(
                row["raw_route_max_abs"] == 0 for row in layer_checks.values()
            ),
            "dense_code_one": all(
                row["dense_code_max_abs_from_one"] == 0
                for row in layer_checks.values()
            ),
            "actual_code_one": all(
                row["actual_code_max_abs_from_one"] == 0
                for row in layer_checks.values()
            ),
            "topk_disabled": all(
                not row["topk_enabled"] for row in layer_checks.values()
            ),
            "equal_capacity": all(
                row["assignment_row_sums_ok"]
                and row["assignment_column_sums_ok"]
                for row in layer_checks.values()
            ),
            "no_nan_inf": all_finite and torch.isfinite(after).all().item(),
        }
        result["metrics"] = {
            "num_layers": len(injection.layers),
            "num_samples": len(selected),
            "max_weight_reconstruction_relative_error": max_weight_error,
            "forward_last_token_relative_error": forward_error,
            "assertions": assertions,
            "layer_checks": layer_checks,
        }
        result["artifacts"] = {
            "resolved_config": str(output_dir / "resolved_config.json"),
            "sample_ids": str(output_dir / "sample_ids.json"),
        }
        result["status"] = (
            "complete" if all(assertions.values()) else "failed"
        )
        if result["status"] != "complete":
            result["failure_reason"] = "one or more P0 assertions failed"
    except Exception as error:
        result["status"] = "failed"
        result["failure_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["started_at_unix"] = started_at
        result["finished_at_unix"] = time.time()
        result["wall_seconds"] = result["finished_at_unix"] - started_at
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

