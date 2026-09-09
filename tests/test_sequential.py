import random

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from badit_tf.core import TFAbilityLinear
from badit_tf.sequential import (
    capture_rng_state,
    capture_trainable_anchor,
    continual_metrics,
    deterministic_rehearsal_rank_records,
    effective_task_step_budgets,
    load_trainable_and_ability_state,
    optimizer_updates_match_training_steps,
    restore_rng_state,
    restore_scheduler_initial_state,
    relative_squared_trainable_drift,
    scheduler_initial_state,
    state_sha256,
    topk_schedule_matches_protocol,
    trainable_and_ability_state,
)


def test_optimizer_update_audit_uses_loaded_bank_baseline():
    initial = {
        "layer.0": {"optimizer_updates": 1},
        "layer.1": {"optimizer_updates": 1},
    }
    current = {
        "layer.0": {"optimizer_updates": 3},
        "layer.1": {"optimizer_updates": 3},
    }
    assert optimizer_updates_match_training_steps(current, initial, 2)
    assert not optimizer_updates_match_training_steps(current, initial, 3)
    assert not optimizer_updates_match_training_steps(
        {"layer.0": {"optimizer_updates": 3}}, initial, 2
    )


def test_rehearsal_order_is_deterministic_balanced_and_train_only():
    tasks = []
    for task_index in range(3):
        tasks.append(
            {
                "epoch_orders": [
                    [
                        {
                            "sample_id": f"t{task_index}-{record_index}",
                            "task": f"task-{task_index}",
                        }
                        for record_index in range(16)
                    ]
                ]
            }
        )
    first = deterministic_rehearsal_rank_records(
        tasks, current_task_index=2, rank=0, world_size=8, seed=7
    )
    second = deterministic_rehearsal_rank_records(
        tasks, current_task_index=2, rank=0, world_size=8, seed=7
    )
    assert first == second
    assert [record["task"] for record in first] == [
        "task-0",
        "task-1",
        "task-0",
        "task-1",
    ]
    assert {record["sample_id"] for record in first} <= {
        "t0-0",
        "t0-8",
        "t1-0",
        "t1-8",
    }


def test_relative_trainable_drift_is_normalized_and_differentiable():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 4.0]]))
    anchor = capture_trainable_anchor(layer)
    assert relative_squared_trainable_drift(layer, anchor).item() == 0.0
    with torch.no_grad():
        layer.weight.add_(torch.tensor([[0.0, 5.0]]))
    drift = relative_squared_trainable_drift(layer, anchor)
    torch.testing.assert_close(drift, torch.tensor(1.0))
    drift.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_compact_state_round_trip_preserves_router_and_assignment():
    layer = TFAbilityLinear(
        torch.nn.Linear(24, 32, bias=False),
        num_experts=4,
        rank=2,
        top_k=2,
        lora_dropout=0.0,
    )
    layer.advance_optimizer_step()
    with torch.no_grad():
        layer.residual_router.weight.add_(0.125)
    saved = trainable_and_ability_state(layer)
    fingerprint = state_sha256(saved)
    with torch.no_grad():
        layer.residual_router.weight.zero_()
        layer.topk_enabled.zero_()
    load_trainable_and_ability_state(layer, saved)
    assert state_sha256(trainable_and_ability_state(layer)) == fingerprint
    assert bool(layer.topk_enabled.item())


def test_rng_round_trip_is_exact():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    observed = (random.random(), np.random.random(), torch.rand(3))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_continual_metrics_match_upstream_formulas():
    matrix = [[80.0, 10.0, 5.0], [70.0, 60.0, 8.0], [65.0, 55.0, 50.0]]
    metrics = continual_metrics(matrix, [20.0, 20.0, 20.0])
    assert metrics["continual_score"] == 170.0 / 3.0
    assert metrics["forget_rate"] == 10.0
    assert metrics["forward"] == 130.0 / 3.0
    assert metrics["backward"] == -20.0 / 3.0


def test_effective_task_step_budgets_are_task_specific():
    tasks = [
        {"task": "full", "epoch_orders": [[{}] * 16, [{}] * 16]},
        {"task": "short", "epoch_orders": [[{}] * 8, [{}] * 8]},
    ]
    assert effective_task_step_budgets(tasks, world_size=8) == [4, 2]
    assert effective_task_step_budgets(tasks, world_size=8, cap=3) == [3, 2]


def test_topk_schedule_accepts_locked_source_step_zero_and_rejects_drift():
    assert topk_schedule_matches_protocol(dense_steps=0, expected_dense_steps=0)
    assert topk_schedule_matches_protocol(dense_steps=1, expected_dense_steps=1)
    assert not topk_schedule_matches_protocol(dense_steps=0, expected_dense_steps=1)


def test_task_local_scheduler_restart_restores_lr_and_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD([parameter], lr=0.2)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / (step + 1))
    initial = scheduler_initial_state(scheduler)
    optimizer.step()
    scheduler.step()
    assert scheduler.state_dict() != initial
    restore_scheduler_initial_state(scheduler, initial)
    assert scheduler.state_dict() == initial
    assert optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]
