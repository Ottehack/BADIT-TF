"""Shared construction and frozen-bank loading for BADIT-TF runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .core import AbilityInjectionResult, inject_ability_layers, iter_ability_layers
from .sequential import load_trainable_and_ability_state, state_sha256


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inject_from_config(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    assignments: Mapping[str, torch.Tensor] | None = None,
) -> AbilityInjectionResult:
    """Inject one ability bank while honoring the full frozen config contract."""

    return inject_ability_layers(
        model,
        target_suffixes=tuple(config["target_modules"]),
        assignments=assignments,
        num_experts=int(config["num_experts"]),
        rank=int(config["rank"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        top_k=int(config["top_k"]),
        dense_steps=int(config["dense_steps"]),
        router_mode=str(config.get("router_mode", "residual_mask")),
        router_bias=bool(config.get("router_bias", True)),
        initialization_method=str(config.get("initialization_method", "svd")),
        apply_lora_dropout=bool(config.get("apply_lora_dropout", True)),
        svd_method=str(config.get("svd_method", "randomized")),
        residual_implementation=str(
            config.get("residual_implementation", "paired_subtraction")
        ),
        cache_last_gradients=bool(config.get("public_epoch_regroup", False)),
        allow_unequal_capacity=bool(config.get("allow_unequal_capacity", False)),
        svd_seed=int(config["seed"]),
        svd_oversample=int(config.get("svd_oversample", 0)),
        svd_niter=int(config.get("svd_niter", 0)),
    )


def load_initial_bank_checkpoint(
    model: torch.nn.Module, config: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Load a frozen canonical warm-start bank, with hash and contract checks."""

    raw_path = config.get("initial_bank_checkpoint")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    observed_file_sha = file_sha256(path)
    expected_file_sha = config.get("initial_bank_checkpoint_sha256")
    if expected_file_sha and observed_file_sha != str(expected_file_sha):
        raise AssertionError("initial bank checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise ValueError("initial bank checkpoint lacks model_state")
    metadata = dict(payload.get("metadata", {}))
    expected_contract = {
        "initialization_method": str(config.get("initialization_method", "svd")),
        "num_experts": int(config["num_experts"]),
        "rank": int(config["rank"]),
        "lora_alpha": float(config["lora_alpha"]),
        "router_mode": str(config.get("router_mode", "residual_mask")),
        "router_bias": bool(config.get("router_bias", True)),
        "apply_lora_dropout": bool(config.get("apply_lora_dropout", True)),
    }
    observed_contract = metadata.get("bank_contract")
    contract_for_comparison = dict(observed_contract or {})
    runtime_dropout_override = bool(
        config.get("allow_runtime_dropout_override", False)
    )
    if runtime_dropout_override and observed_contract is not None:
        # Dropout owns no trainable state and is applied only during later SFT.
        # H1 preregisters it as a training hyperparameter, so a frozen bank may
        # be reused while retaining strict checks on every state-bearing field.
        contract_for_comparison["apply_lora_dropout"] = expected_contract[
            "apply_lora_dropout"
        ]
    if contract_for_comparison != expected_contract:
        raise AssertionError(
            f"initial bank contract mismatch: {observed_contract} != {expected_contract}"
        )
    load_trainable_and_ability_state(model, payload["model_state"])
    observed_state_sha = state_sha256(payload["model_state"])
    expected_state_sha = metadata.get("model_state_sha256")
    if expected_state_sha and observed_state_sha != expected_state_sha:
        raise AssertionError("initial bank model-state SHA256 mismatch")
    return {
        "path": str(path),
        "file_sha256": observed_file_sha,
        "model_state_sha256": observed_state_sha,
        "metadata": metadata,
        "runtime_dropout_override": runtime_dropout_override,
        "checkpoint_apply_lora_dropout": (
            observed_contract.get("apply_lora_dropout")
            if observed_contract is not None
            else None
        ),
        "runtime_apply_lora_dropout": expected_contract["apply_lora_dropout"],
    }


def apply_layer_assignments(
    model: torch.nn.Module, assignments: Mapping[str, torch.Tensor]
) -> None:
    """Regroup a loaded canonical bank without changing primitive identity."""

    layers = dict(iter_ability_layers(model))
    if set(layers) != set(assignments):
        raise ValueError("assignment layers do not match injected layers")
    for name, layer in layers.items():
        layer.apply_assignment(assignments[name].to(layer.lora_A.device))


def assert_canonical_layer_assignments(model: torch.nn.Module) -> None:
    """Fail closed unless a loaded bank still uses canonical primitive slots.

    A frozen warm-start bank is serialized in canonical primitive order.  The
    TF confirmation path deliberately regroups it with frozen TF labels, while
    the matched GG path must retain that canonical order until its final
    gradient-geometry regroup.  This check prevents an accidental empty or
    partial assignment map from silently changing that contract.
    """

    for name, layer in iter_ability_layers(model):
        expected = torch.arange(
            layer.total_primitives,
            device=layer.primitive_indices.device,
            dtype=layer.primitive_indices.dtype,
        ).reshape(layer.num_experts, layer.rank)
        if not torch.equal(layer.primitive_indices, expected):
            raise ValueError(
                f"loaded GG initial bank is not canonical for layer={name}"
            )
