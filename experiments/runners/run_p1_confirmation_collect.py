#!/usr/bin/env python3
"""Collect confirmation q profiles after validating the eta-lock chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from badit_tf.calibration import (
    collect_gate_gradient,
    freeze_for_calibration,
    supervised_tensors,
)
from badit_tf.runtime import inject_from_config, load_initial_bank_checkpoint
from badit_tf.splits import load_role_instances


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    lock_path = Path(config["eta_selection_lock"])
    manifest_path = Path(config["split_manifest"])
    if sha256_file(lock_path) != config["eta_selection_lock_sha256"]:
        raise AssertionError("eta lock changed before confirmation")
    if sha256_file(manifest_path) != config["split_manifest_sha256"]:
        raise AssertionError("confirmation manifest changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if float(config["fidelity_eta"]) != float(lock["selected_eta"]):
        raise AssertionError("confirmation eta differs from lock")
    if manifest["eta_selection_lock_sha256"] != config["eta_selection_lock_sha256"]:
        raise AssertionError("confirmation split belongs to a different lock")

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    role = config["confirmation_role"]
    all_rows = load_role_instances(manifest, role)
    rows = all_rows[rank::world]
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
        dtype=dtype,
        attn_implementation=config["attention_implementation"],
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injection = inject_from_config(model, config)
    initial_bank = load_initial_bank_checkpoint(model, config)
    model.eval()
    layers = freeze_for_calibration(model)
    layer_names = [name for name, _ in layers]
    if layer_names != list(injection.layers):
        raise AssertionError("injected layer order drift")

    q_rows = []
    metadata = []
    started = time.time()
    for row in rows:
        tensors = supervised_tensors(
            tokenizer,
            row,
            max_length=int(config["max_sequence_length"]),
            max_target_length=int(config["max_target_length"]),
            device=device,
        )
        loss, target_tokens, gradient = collect_gate_gradient(model, layers, tensors)
        q_rows.append(-gradient)
        metadata.append(
            {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "loss": loss,
                "target_tokens": target_tokens,
            }
        )
    output = Path(config["output_dir"]) / "collection"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / f"rank{rank}.npz",
        **{f"q_{role}": np.stack(q_rows).astype(np.float32)},
    )
    (output / f"rank{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "world": world,
                "role": role,
                "rows": metadata,
                "layer_names": layer_names,
                "started_at_unix": started,
                "finished_at_unix": time.time(),
                "eta_lock_sha256": config["eta_selection_lock_sha256"],
                "manifest_sha256": config["split_manifest_sha256"],
                "initial_bank_checkpoint": initial_bank,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"rank={rank} confirmation q complete={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
