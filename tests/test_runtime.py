from pathlib import Path

import torch
from torch import nn

import pytest

from badit_tf.runtime import (
    apply_layer_assignments,
    assert_canonical_layer_assignments,
    inject_from_config,
    load_initial_bank_checkpoint,
)
from badit_tf.sequential import state_sha256, trainable_and_ability_state


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(6, 8, bias=False)


def config(path: Path | None = None) -> dict:
    result = {
        "target_modules": ["gate_proj"],
        "num_experts": 2,
        "rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.05,
        "top_k": 1,
        "dense_steps": 3,
        "router_mode": "residual_mask",
        "router_bias": True,
        "initialization_method": "kaiming_zero",
        "apply_lora_dropout": False,
        "svd_method": "exact",
        "residual_implementation": "paired_subtraction",
        "seed": 7,
        "svd_oversample": 0,
        "svd_niter": 0,
    }
    if path is not None:
        result["initial_bank_checkpoint"] = str(path)
    return result


def test_frozen_kaiming_bank_round_trip(tmp_path: Path):
    torch.manual_seed(7)
    source = TinyModel()
    inject_from_config(source, config())
    with torch.no_grad():
        source.gate_proj.lora_B.add_(0.25)
    state = trainable_and_ability_state(source)
    contract = {
        "initialization_method": "kaiming_zero",
        "num_experts": 2,
        "rank": 2,
        "lora_alpha": 4.0,
        "router_mode": "residual_mask",
        "router_bias": True,
        "apply_lora_dropout": False,
    }
    checkpoint = tmp_path / "bank.pt"
    torch.save(
        {
            "model_state": state,
            "metadata": {
                "bank_contract": contract,
                "model_state_sha256": state_sha256(state),
            },
        },
        checkpoint,
    )

    torch.manual_seed(7)
    target = TinyModel()
    inject_from_config(target, config(checkpoint))
    audit = load_initial_bank_checkpoint(target, config(checkpoint))
    assert audit is not None
    assert audit["model_state_sha256"] == state_sha256(
        trainable_and_ability_state(target)
    )
    assert torch.equal(target.gate_proj.lora_B, source.gate_proj.lora_B)


def test_frozen_bank_allows_explicit_runtime_dropout_override(tmp_path: Path):
    torch.manual_seed(7)
    source = TinyModel()
    inject_from_config(source, config())
    state = trainable_and_ability_state(source)
    checkpoint = tmp_path / "bank.pt"
    torch.save(
        {
            "model_state": state,
            "metadata": {
                "bank_contract": {
                    "initialization_method": "kaiming_zero",
                    "num_experts": 2,
                    "rank": 2,
                    "lora_alpha": 4.0,
                    "router_mode": "residual_mask",
                    "router_bias": True,
                    "apply_lora_dropout": False,
                },
                "model_state_sha256": state_sha256(state),
            },
        },
        checkpoint,
    )
    override = config(checkpoint)
    override["apply_lora_dropout"] = True
    override["allow_runtime_dropout_override"] = True
    target = TinyModel()
    inject_from_config(target, override)
    audit = load_initial_bank_checkpoint(target, override)
    assert audit["runtime_dropout_override"] is True
    assert audit["checkpoint_apply_lora_dropout"] is False
    assert audit["runtime_apply_lora_dropout"] is True


def test_canonical_initial_bank_contract_is_preserved_for_gg():
    """GG must start canonical; TF-style pre-regrouping is rejected."""

    model = TinyModel()
    inject_from_config(model, config())
    assert_canonical_layer_assignments(model)
    apply_layer_assignments(
        model,
        {"gate_proj": torch.tensor([0, 1, 0, 1])},
    )
    with pytest.raises(ValueError, match="not canonical"):
        assert_canonical_layer_assignments(model)
