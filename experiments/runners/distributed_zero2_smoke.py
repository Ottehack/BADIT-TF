#!/usr/bin/env python3
"""Two-process NCCL + DeepSpeed ZeRO-2 forward/backward/update smoke."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="env4_two_process_zero2_smoke")
    args = parser.parse_args()
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(20260731)
    model = torch.nn.Linear(8, 4, bias=True).cuda()
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {
        "train_batch_size": world,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": False,
            "contiguous_gradients": True,
        },
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "steps_per_print": 1000000,
        "wall_clock_breakdown": False,
    }
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
        dist_init_required=False,
    )
    x = torch.arange(8, dtype=torch.float32, device=engine.device).reshape(1, 8)
    x = x + float(rank)
    target = torch.full((1, 4), float(rank), device=engine.device)
    loss = torch.nn.functional.mse_loss(engine(x), target)
    engine.backward(loss)
    engine.step()
    updated = torch.cat(
        [parameter.detach().flatten() for parameter in engine.module.parameters()]
    )
    changed = torch.linalg.vector_norm(updated - initial).item()
    reduced = torch.tensor(float(rank + 1), device=engine.device)
    dist.all_reduce(reduced)
    row = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "device": str(engine.device),
        "gpu": torch.cuda.get_device_name(local_rank),
        "loss": float(loss),
        "parameter_delta_l2": changed,
        "all_reduce_sum": float(reduced),
        "zero_stage": engine.zero_optimization_stage(),
        "deepspeed": deepspeed.__version__,
        "torch": torch.__version__,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"rank{rank}.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dist.barrier()
    if rank == 0:
        rows = [
            json.loads((args.output_dir / f"rank{index}.json").read_text())
            for index in range(world)
        ]
        assertions = {
            "world_size_two": world == 2,
            "distinct_local_ranks": {item["local_rank"] for item in rows} == {0, 1},
            "nccl_all_reduce_exact": all(
                item["all_reduce_sum"] == 3.0 for item in rows
            ),
            "zero_stage_two": all(item["zero_stage"] == 2 for item in rows),
            "optimizer_updated_each_rank": all(
                item["parameter_delta_l2"] > 0 for item in rows
            ),
            "finite_loss_each_rank": all(
                torch.isfinite(torch.tensor(item["loss"])).item() for item in rows
            ),
            "a100_each_rank": all(
                "A100-SXM4-80GB" in item["gpu"] for item in rows
            ),
        }
        result = {
        "run_id": args.run_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if all(assertions.values()) else "failed",
            "backend": dist.get_backend(),
            "rows": rows,
            "assertions": assertions,
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
