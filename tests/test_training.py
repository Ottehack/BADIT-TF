import numpy as np
import torch

from badit_tf.core import TFAbilityLinear
from badit_tf.training import (
    balanced_gradient_geometry_assignment,
    losses_are_finite_and_stable,
    nonzero_cosine_warmup_multiplier,
    public_badit_gradient_regroup,
    resolve_assignment_labels,
    resolve_code_commit,
    residual_router_optimizer_groups,
)


def test_loss_stability_uses_fixed_cold_start_burn_in():
    losses = np.concatenate(
        [np.array([15.0]), np.full(31, 1.0), np.linspace(2.0, 0.2, 100)]
    )
    assert losses_are_finite_and_stable(
        losses, explosion_factor=10.0, burn_in_steps=32
    )


def test_loss_stability_rejects_nonfinite_values_even_during_burn_in():
    losses = np.concatenate([np.array([np.nan]), np.ones(63)])
    assert not losses_are_finite_and_stable(
        losses, explosion_factor=10.0, burn_in_steps=32
    )


def test_loss_stability_rejects_post_burn_in_explosion():
    losses = np.concatenate([np.ones(32), np.array([10.01]), np.ones(31)])
    assert not losses_are_finite_and_stable(
        losses, explosion_factor=10.0, burn_in_steps=32
    )


def test_loss_stability_validates_short_traces_instead_of_skipping_them():
    assert not losses_are_finite_and_stable(
        [1.0] * 31 + [100.0], explosion_factor=10.0, burn_in_steps=32
    )


def assignment_payload():
    return {
        "layer_names": ["layer.0", "layer.1"],
        "contiguous": [0, 0, 1, 1],
        "random_balanced": [[1, 0, 1, 0]],
        "tf": {
            "0.1": {
                "layer.0": [0, 1, 0, 1],
                "layer.1": [1, 1, 0, 0],
            }
        },
        "raw_q": {
            "layer.0": [1, 0, 1, 0],
            "layer.1": [0, 1, 1, 0],
        },
        "tf_no_capacity": {
            "layer.0": [0, 0, 0, 1],
            "layer.1": [1, 0, 0, 0],
        },
    }


def test_resolve_assignment_labels_supports_frozen_methods():
    payload = assignment_payload()
    tf = resolve_assignment_labels(
        payload, method="tf", epsilon=0.1, num_experts=2, rank=2
    )
    contiguous = resolve_assignment_labels(
        payload, method="contiguous", epsilon=0.1, num_experts=2, rank=2
    )
    random = resolve_assignment_labels(
        payload,
        method="random_balanced",
        epsilon=0.1,
        num_experts=2,
        rank=2,
        random_index=0,
    )
    assert tf == payload["tf"]["0.1"]
    assert contiguous == {
        "layer.0": [0, 0, 1, 1],
        "layer.1": [0, 0, 1, 1],
    }
    assert random == {
        "layer.0": [1, 0, 1, 0],
        "layer.1": [1, 0, 1, 0],
    }
    raw_q = resolve_assignment_labels(
        payload, method="raw_q", epsilon=0.1, num_experts=2, rank=2
    )
    assert raw_q == payload["raw_q"]
    no_capacity = resolve_assignment_labels(
        payload,
        method="tf_no_capacity",
        epsilon=0.1,
        num_experts=2,
        rank=2,
        allow_unequal_capacity=True,
    )
    assert no_capacity == payload["tf_no_capacity"]
    public = resolve_assignment_labels(
        payload, method="public_contiguous", epsilon=0.1, num_experts=3, rank=2
    )
    assert public == {
        "layer.0": [0, 0, 1, 1, 2, 2],
        "layer.1": [0, 0, 1, 1, 2, 2],
    }


def test_contiguous_assignment_is_canonical_when_payload_omits_explicit_row():
    payload = assignment_payload()
    payload.pop("contiguous")
    contiguous = resolve_assignment_labels(
        payload, method="contiguous", epsilon=0.1, num_experts=2, rank=2
    )
    assert contiguous == {
        "layer.0": [0, 0, 1, 1],
        "layer.1": [0, 0, 1, 1],
    }


def test_resolve_assignment_labels_rejects_capacity_violation():
    payload = assignment_payload()
    payload["contiguous"] = [0, 0, 0, 1]
    try:
        resolve_assignment_labels(
            payload, method="contiguous", epsilon=0.1, num_experts=2, rank=2
        )
    except ValueError as error:
        assert "equal capacity" in str(error)
    else:
        raise AssertionError("capacity violation was accepted")


def test_residual_router_optimizer_groups_cover_trainables_once():
    layer = TFAbilityLinear(
        torch.nn.Linear(24, 32, bias=False),
        num_experts=4,
        rank=2,
        top_k=2,
        lora_dropout=0.0,
    )
    groups = residual_router_optimizer_groups(
        layer, learning_rate=2e-4, router_lr_multiplier=0.25
    )
    assert [group["group_name"] for group in groups] == [
        "expert",
        "residual_router",
    ]
    assert [group["lr"] for group in groups] == [2e-4, 5e-5]
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }
    assert len(grouped) == len({id(parameter) for parameter in grouped})


def test_balanced_gradient_geometry_assignment_is_exact_and_deterministic():
    generator = torch.Generator().manual_seed(7)
    gradients = torch.randn(32, 13, generator=generator)
    first, audit = balanced_gradient_geometry_assignment(
        gradients, num_experts=8, rank=4, seed=0
    )
    second, _ = balanced_gradient_geometry_assignment(
        gradients, num_experts=8, rank=4, seed=0
    )
    assert np.array_equal(first, second)
    assert np.array_equal(np.bincount(first, minlength=8), np.full(8, 4))
    assert audit["zero_gradient_primitives"] == 0


def test_public_regroup_reproduces_greedy_slot_contract():
    gradients = torch.randn(24, 11, generator=torch.Generator().manual_seed(9))
    clusters, audit = public_badit_gradient_regroup(
        gradients,
        num_experts=6,
        rank=4,
        rng=np.random.RandomState(1),
    )
    assert len(clusters) == 6
    assert all(len(cluster) == 4 for cluster in clusters)
    assert audit["unique_source_slots"] <= 24
    assert audit["duplicate_copies"] == audit["omitted_source_slots"]


def test_dynamic_regroup_moves_initial_bank_with_current_bank():
    base = torch.nn.Linear(40, 48, bias=False)
    layer = TFAbilityLinear(
        base, num_experts=4, rank=2, top_k=2, lora_dropout=0.0
    )
    before = layer.reconstructed_weight().detach().clone()
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    layer.apply_assignment(labels)
    after = layer.reconstructed_weight().detach()
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
    assert torch.equal(layer.primitive_indices.flatten().sort().values, torch.arange(8))


def test_warmup_first_update_is_nonzero():
    assert nonzero_cosine_warmup_multiplier(
        0, warmup_steps=44, total_steps=1468
    ) == 1 / 44
    assert nonzero_cosine_warmup_multiplier(
        0, warmup_steps=1, total_steps=2
    ) == 1.0
    assert nonzero_cosine_warmup_multiplier(
        1, warmup_steps=1, total_steps=2
    ) == 0.0


def test_code_commit_accepts_release_archive_override(monkeypatch):
    monkeypatch.setenv("BADIT_TF_CODE_COMMIT", "A" * 40)
    assert resolve_code_commit() == "a" * 40


def test_code_commit_rejects_malformed_release_override(monkeypatch):
    monkeypatch.setenv("BADIT_TF_CODE_COMMIT", "not-a-commit")
    try:
        resolve_code_commit()
    except RuntimeError as error:
        assert "invalid code commit" in str(error)
    else:
        raise AssertionError("malformed release commit was accepted")
