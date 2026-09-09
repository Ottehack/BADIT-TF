"""Checkpoint and metric helpers for sequential BADIT-TF training."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


def deterministic_rehearsal_rank_records(
    tasks: list[dict[str, Any]],
    *,
    current_task_index: int,
    rank: int,
    world_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a balanced deterministic rank-local order over prior tasks.

    Each prior task is shuffled independently after distributed striding, then
    the task pools are interleaved round-robin.  This prevents a prefix from
    silently containing only the first prior task when the current task uses
    fewer steps than the cumulative rehearsal pool.
    """

    if not 0 <= current_task_index <= len(tasks):
        raise ValueError("current_task_index is outside the task list")
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world_size")
    pools: list[list[dict[str, Any]]] = []
    for task_index, task in enumerate(tasks[:current_task_index]):
        flattened = [record for epoch in task["epoch_orders"] for record in epoch]
        if len(flattened) % world_size:
            raise ValueError("rehearsal task order is not divisible by world_size")
        pool = [dict(record) for record in flattened[rank::world_size]]
        random.Random(
            int(seed) * 1_000_000
            + int(current_task_index) * 10_000
            + int(task_index) * 100
            + int(rank)
        ).shuffle(pool)
        pools.append(pool)
    if not pools:
        return []
    order: list[dict[str, Any]] = []
    for index in range(max(len(pool) for pool in pools)):
        for pool in pools:
            if index < len(pool):
                order.append(pool[index])
    return order


def effective_task_step_budgets(
    tasks: list[dict[str, Any]], world_size: int, cap: int | None = None
) -> list[int]:
    """Return the rank-local optimizer-step budget for every task.

    Sequential SuperNI tasks do not all contain 1,000 examples.  Deriving one
    shared budget from task 0 silently over-trains, truncates, or crashes on
    later tasks.  Each epoch order is padded to a multiple of ``world_size``,
    so its flattened length must be divided independently per task.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    budgets = []
    for task in tasks:
        records = sum(len(epoch) for epoch in task["epoch_orders"])
        if records % world_size:
            raise ValueError(
                f"task {task.get('task', '<unknown>')} order is not divisible "
                f"by world_size={world_size}"
            )
        natural = records // world_size
        budgets.append(min(natural, int(cap)) if cap is not None else natural)
    return budgets


def optimizer_updates_match_training_steps(
    current_routing: dict[str, dict[str, Any]],
    initial_routing: dict[str, dict[str, Any]],
    training_steps: int,
) -> bool:
    """Check layer update counters relative to the loaded bank baseline.

    A canonical warm-start bank can already contain optimizer updates.  The
    sequential protocol must therefore audit the *delta* contributed by the
    current P3 run instead of incorrectly requiring every absolute counter to
    start at zero.
    """

    if training_steps < 0 or set(current_routing) != set(initial_routing):
        return False
    return all(
        int(current_routing[name]["optimizer_updates"])
        == int(initial_routing[name]["optimizer_updates"]) + int(training_steps)
        for name in current_routing
    )


def scheduler_initial_state(scheduler: Any) -> dict[str, Any]:
    """Capture a reusable task-boundary scheduler state."""

    return deepcopy(scheduler.state_dict())


def restore_scheduler_initial_state(
    scheduler: Any, state: dict[str, Any]
) -> None:
    """Restart a task-local scheduler and synchronize optimizer learning rates."""

    scheduler.load_state_dict(deepcopy(state))
    last_lrs = list(scheduler.get_last_lr())
    if len(last_lrs) != len(scheduler.optimizer.param_groups):
        raise RuntimeError("scheduler/optimizer parameter-group mismatch")
    for group, learning_rate in zip(
        scheduler.optimizer.param_groups, last_lrs, strict=True
    ):
        group["lr"] = float(learning_rate)


ABILITY_BUFFER_MARKERS = (
    "initial_lora_A",
    "initial_lora_B",
    "primitive_gates",
    "primitive_indices",
    "routing_assignment_labels",
    "singular_values",
    "topk_enabled",
    "calibration_enabled",
    "optimizer_updates",
    "initial_weight_relative_error",
)


def trainable_and_ability_state(model: torch.nn.Module) -> dict[str, Any]:
    """Return the compact model state needed for exact sequential recovery."""

    trainable = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    buffers = {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.named_buffers()
        if any(name.endswith(marker) for marker in ABILITY_BUFFER_MARKERS)
    }
    return {"trainable_state": trainable, "ability_buffers": buffers}


def capture_trainable_anchor(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Freeze the current trainable state on-device for a stage anchor."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def relative_squared_trainable_drift(
    model: torch.nn.Module,
    anchor: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Differentiable squared drift normalized by the anchor's squared norm."""

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(anchor):
        raise RuntimeError(
            "stage-anchor key mismatch: "
            f"missing={sorted(set(current) - set(anchor))}, "
            f"extra={sorted(set(anchor) - set(current))}"
        )
    numerator = None
    denominator = None
    for name, parameter in current.items():
        reference = anchor[name].to(
            device=parameter.device, dtype=parameter.dtype
        )
        squared_drift = (parameter - reference).float().square().sum()
        squared_reference = reference.float().square().sum()
        numerator = (
            squared_drift if numerator is None else numerator + squared_drift
        )
        denominator = (
            squared_reference
            if denominator is None
            else denominator + squared_reference
        )
    if numerator is None or denominator is None:
        raise RuntimeError("cannot anchor an empty trainable parameter set")
    return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)


def load_trainable_and_ability_state(
    model: torch.nn.Module, state: dict[str, Any]
) -> None:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    expected_parameters = {
        name for name, parameter in parameters.items() if parameter.requires_grad
    }
    observed_parameters = set(state["trainable_state"])
    if observed_parameters != expected_parameters:
        raise RuntimeError(
            "trainable checkpoint key mismatch: "
            f"missing={sorted(expected_parameters - observed_parameters)}, "
            f"extra={sorted(observed_parameters - expected_parameters)}"
        )
    with torch.no_grad():
        for name, value in state["trainable_state"].items():
            parameters[name].copy_(value.to(parameters[name].device))
        for name, value in state["ability_buffers"].items():
            if name not in buffers:
                raise RuntimeError(f"checkpoint buffer is absent from model: {name}")
            buffers[name].copy_(value.to(buffers[name].device))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])


def _hash_update(hasher: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(json.dumps(list(tensor.shape)).encode())
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray\0")
        hasher.update(str(array.dtype).encode())
        hasher.update(json.dumps(list(array.shape)).encode())
        hasher.update(array.tobytes())
    elif isinstance(value, dict):
        hasher.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_update(hasher, key)
            _hash_update(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _hash_update(hasher, item)
    else:
        hasher.update(type(value).__name__.encode() + b"\0")
        hasher.update(repr(value).encode())
        hasher.update(b"\0")


def state_sha256(value: Any) -> str:
    hasher = hashlib.sha256()
    _hash_update(hasher, value)
    return hasher.hexdigest()


def save_rank_checkpoint(
    checkpoint_dir: Path,
    *,
    rank: int,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Save replicated model state once and rank-local ZeRO/RNG state."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_state = trainable_and_ability_state(model)
    rank_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
    }
    hashes = {
        "model_state_sha256": state_sha256(model_state),
        "optimizer_state_sha256": state_sha256(rank_state["optimizer"]),
        "scheduler_state_sha256": state_sha256(rank_state["scheduler"]),
        "rng_state_sha256": state_sha256(rank_state["rng"]),
    }
    if rank == 0:
        torch.save(
            {"model_state": model_state, "metadata": metadata, "hashes": hashes},
            checkpoint_dir / "common.pt",
        )
    torch.save(
        {"rank": rank, "state": rank_state, "hashes": hashes},
        checkpoint_dir / f"rank_{rank}.pt",
    )
    return hashes


def load_rank_checkpoint(
    checkpoint_dir: Path,
    *,
    rank: int,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, bool], dict[str, str]]:
    common = torch.load(
        checkpoint_dir / "common.pt", map_location=map_location, weights_only=False
    )
    rank_paths = sorted(
        checkpoint_dir.glob("rank_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not rank_paths:
        raise RuntimeError(f"no rank checkpoints found in {checkpoint_dir}")
    all_rank_payloads = [
        torch.load(path, map_location=map_location, weights_only=False)
        for path in rank_paths
    ]
    local = all_rank_payloads[rank]
    load_trainable_and_ability_state(model, common["model_state"])
    # ZeRO-2's legacy loader expects the full list of data-parallel partition
    # state dictionaries and selects its own rank internally.  Passing only
    # the local dictionary is interpreted as a mapping indexed by rank and
    # fails with KeyError.
    optimizer.load_state_dict(
        [payload["state"]["optimizer"] for payload in all_rank_payloads],
        load_from_fp32_weights=True,
    )
    scheduler.load_state_dict(local["state"]["scheduler"])
    restore_rng_state(local["state"]["rng"])
    observed = {
        "model_state_sha256": state_sha256(trainable_and_ability_state(model)),
        "optimizer_state_sha256": state_sha256(optimizer.state_dict()),
        "scheduler_state_sha256": state_sha256(scheduler.state_dict()),
        "rng_state_sha256": state_sha256(capture_rng_state()),
    }
    expected = dict(local["hashes"])
    expected["model_state_sha256"] = common["hashes"]["model_state_sha256"]
    assertions = {key: observed[key] == expected[key] for key in observed}
    return common["metadata"], assertions, observed


def load_model_and_rng_checkpoint(
    checkpoint_dir: Path,
    *,
    rank: int,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, bool], dict[str, str]]:
    """Restore only model and RNG while intentionally resetting optimizer state.

    This matches the public BADIT runner's repeated DeepSpeed ``Trainer.train``
    lifecycle under Transformers 4.30.2: every task calls ``deepspeed_init``
    again, creating a new optimizer and scheduler while retaining model state.
    """

    common = torch.load(
        checkpoint_dir / "common.pt", map_location=map_location, weights_only=False
    )
    local = torch.load(
        checkpoint_dir / f"rank_{rank}.pt",
        map_location=map_location,
        weights_only=False,
    )
    load_trainable_and_ability_state(model, common["model_state"])
    restore_rng_state(local["state"]["rng"])
    observed = {
        "model_state_sha256": state_sha256(trainable_and_ability_state(model)),
        "rng_state_sha256": state_sha256(capture_rng_state()),
    }
    expected = {
        "model_state_sha256": common["hashes"]["model_state_sha256"],
        "rng_state_sha256": local["hashes"]["rng_state_sha256"],
    }
    assertions = {key: observed[key] == expected[key] for key in observed}
    return common["metadata"], assertions, observed


def continual_metrics(
    evaluation_matrix: list[list[float]], baseline: list[float]
) -> dict[str, float]:
    """Match the upstream BADIT continual-learning metric definitions."""

    scores = np.asarray(evaluation_matrix, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("evaluation matrix must be square")
    if base.shape != (scores.shape[0],):
        raise ValueError("baseline length must equal task count")
    task_count = scores.shape[0]
    forget = [
        float(np.max(scores[:-1, index]) - scores[-1, index])
        for index in range(task_count - 1)
    ]
    return {
        "continual_score": float(np.mean(scores[-1])),
        "forget_rate": float(np.mean(forget)) if forget else 0.0,
        "forward": float(np.mean(np.diag(scores) - base)),
        "backward": float(np.mean(scores[-1] - np.diag(scores))),
    }


def topk_schedule_matches_protocol(
    *, dense_steps: int, expected_dense_steps: int
) -> bool:
    """Validate the frozen switch schedule, including source-compatible step zero."""

    return int(dense_steps) == int(expected_dense_steps)
