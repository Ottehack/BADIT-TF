#!/usr/bin/env python3
"""Matched TF/GG fixed-load inference-throughput replay on eight GPUs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import prompt_for_record
from badit_tf.core import iter_ability_layers
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def clear_overrides(model):
    for _, layer in iter_ability_layers(model): layer.set_route_code_override(None)


@torch.no_grad()
def generate_exact(model, tokenizer, record, max_prompt_length, new_tokens, device):
    encoded = tokenizer(prompt_for_record(record), return_tensors="pt", truncation=True, max_length=max_prompt_length)
    input_ids = encoded["input_ids"].to(device); attention_mask = encoded["attention_mask"].to(device)
    clear_overrides(model)
    for _, layer in iter_ability_layers(model): layer.set_attention_mask(attention_mask)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    for _, layer in iter_ability_layers(model): layer.set_route_code_override(layer._last_route["actual"])
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True); past = outputs.past_key_values
    for _ in range(new_tokens):
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=device)], dim=1)
        outputs = model(input_ids=next_token, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        past = outputs.past_key_values; next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    clear_overrides(model)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("official_test_loaded") or config.get("downstream_test_loaded"): raise AssertionError("throughput replay cannot load test")
    rank = int(os.environ.get("LOCAL_RANK", "0")); world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if world != 8: raise AssertionError("M9 throughput requires eight ranks")
    torch.cuda.set_device(rank); device = torch.device("cuda", rank)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[config["torch_dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True, dtype=dtype, attn_implementation=config["attention_implementation"]).to(device)
    for p in model.parameters(): p.requires_grad_(False)
    injection = inject_from_config(model, config); model.eval()
    manifest = json.loads(Path(config["throughput_prompt_manifest"]).read_text(encoding="utf-8"))
    rows = load_role_instances(manifest, "assignment")
    if len(rows) != 15 or len({row["task"] for row in rows}) != 15: raise AssertionError("throughput prompt set must be one per task")
    local_rows = rows[rank::world]
    variants = ("tf", "gg") if rank % 2 == 0 else ("gg", "tf")
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for variant in variants:
        checkpoint = config[f"{variant}_checkpoint"]
        checkpoint_config = dict(config, initial_bank_checkpoint=checkpoint, initial_bank_checkpoint_sha256=config[f"{variant}_checkpoint_sha256"])
        loaded = load_initial_bank_checkpoint(model, checkpoint_config)
        if loaded is None or loaded["file_sha256"] != config[f"{variant}_checkpoint_sha256"]: raise AssertionError(f"{variant} checkpoint mismatch")
        for record in local_rows[:1]: generate_exact(model, tokenizer, record, int(config["max_prompt_length"]), int(config["max_new_tokens"]), device)
        torch.cuda.synchronize(device); started = time.perf_counter()
        for _ in range(int(config["repeats"])):
            for record in local_rows: generate_exact(model, tokenizer, record, int(config["max_prompt_length"]), int(config["max_new_tokens"]), device)
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
        tokens = len(local_rows) * int(config["repeats"]) * int(config["max_new_tokens"])
        metrics[variant] = {"elapsed_seconds": elapsed, "generated_tokens": tokens, "local_prompts": len(local_rows), "checkpoint_sha256": loaded["file_sha256"]}
    payload = {
        "rank": rank, "world": world, "model": config["model_name"], "run_id": config["run_id"],
        "variant_order": list(variants), "metrics": metrics, "layer_names": list(injection.layers),
        "protocol": {"exact_new_tokens": int(config["max_new_tokens"]), "repeats": int(config["repeats"]), "warmup_prompts_per_variant": min(1, len(local_rows)), "test_loaded": False},
    }
    path = output / f"rank{rank}.json"; path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rank": rank, "metrics": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
