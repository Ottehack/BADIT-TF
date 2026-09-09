#!/usr/bin/env python3
"""Run a no-data backbone/injection compatibility smoke for a frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.core import iter_ability_layers
from badit_tf.runtime import inject_from_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    # Sandbox-external GPU jobs must leave their evidence readable to the
    # regular audit environment; do not inherit a restrictive launcher umask.
    os.umask(0o022)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["smoke_scope"] != "no_data_no_training":
        raise AssertionError("compatibility smoke must not load data or train")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this checkpoint compatibility smoke")
    device = torch.device(config["device"])
    model_path = Path(config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    injection = inject_from_config(model, config)
    layers = dict(iter_ability_layers(model))
    expected = [f"model.layers.{index}.mlp.gate_proj" for index in range(config["expected_layer_count"])]
    if list(layers) != expected:
        raise AssertionError("injected layer order does not match the frozen architecture")
    tokens = tokenizer("Backbone compatibility smoke.", return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**tokens).logits
    if not torch.isfinite(logits).all():
        raise AssertionError("non-finite logits in no-data forward")
    result = {
        "run_id": config["run_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint_manifest": config["checkpoint_manifest"],
        "checkpoint_manifest_sha256": sha256(Path(config["checkpoint_manifest"])),
        "model_path": str(model_path),
        "architecture": model.config.architectures,
        "model_type": model.config.model_type,
        "expected_layer_count": config["expected_layer_count"],
        "injected_layer_count": len(layers),
        "injected_layer_order_matches": True,
        "initialization_method": config["initialization_method"],
        "num_experts": config["num_experts"],
        "rank": config["rank"],
        "experiment_data_loaded": False,
        "no_training_steps": 0,
        "forward_logits_finite": True,
        "all_assertions_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
