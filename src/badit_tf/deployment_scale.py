"""Pure helpers for trained-router deployment-scale fidelity."""

from __future__ import annotations

import numpy as np


FROZEN_DEPLOYMENT_ETAS = (0.10, 0.25, 0.50, 1.00)


def validate_deployment_etas(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    etas = tuple(float(value) for value in values)
    if etas != FROZEN_DEPLOYMENT_ETAS:
        raise ValueError(
            f"deployment eta grid must be {FROZEN_DEPLOYMENT_ETAS}, got {etas}"
        )
    return etas


def canonical_primitive_delta(
    route_code: np.ndarray, primitive_indices: np.ndarray
) -> np.ndarray:
    """Expand expert ``c-1`` to immutable primitive order using P."""

    code = np.asarray(route_code, dtype=np.float64)
    indices = np.asarray(primitive_indices, dtype=np.int64)
    if code.ndim != 1 or indices.ndim != 2 or indices.shape[0] != code.size:
        raise ValueError("route code and primitive assignment shapes disagree")
    flat_indices = indices.reshape(-1)
    if sorted(flat_indices.tolist()) != list(range(flat_indices.size)):
        raise ValueError("primitive indices are not a permutation")
    physical = np.repeat(code - 1.0, indices.shape[1])
    canonical = np.empty_like(physical)
    canonical[flat_indices] = physical
    return canonical


def quadratic_loss_delta(
    q: np.ndarray, damped_fisher: np.ndarray, delta: np.ndarray
) -> float:
    """Evaluate Q(delta)-Q(0) for q=-dL/dz and diagonal Fisher."""

    q = np.asarray(q, dtype=np.float64)
    fisher = np.asarray(damped_fisher, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if q.shape != fisher.shape or q.shape != delta.shape:
        raise ValueError("q, Fisher, and delta shapes must match")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(fisher)) or not np.all(np.isfinite(delta)):
        raise FloatingPointError("non-finite deployment quadratic input")
    if np.any(fisher <= 0):
        raise ValueError("damped Fisher must be strictly positive")
    return float(-np.dot(q, delta) + 0.5 * np.dot(fisher, np.square(delta)))


def relative_prediction_error(predicted: float, observed: float) -> float:
    return float(abs(float(observed) - float(predicted)) / (abs(float(observed)) + 1e-8))
