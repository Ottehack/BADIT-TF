"""Pure helpers for the frozen M4 grouping/routing decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assignment import validate_assignment


@dataclass(frozen=True)
class Decomposition:
    grouping: float
    routing: float
    total: float
    direct_total: float
    orthogonality_error: float
    tied_coefficients: np.ndarray
    tied_primitive_delta: np.ndarray
    routed_primitive_delta: np.ndarray


def labels_from_primitive_indices(primitive_indices: np.ndarray) -> np.ndarray:
    """Convert expert-major primitive IDs to canonical primitive labels."""

    indices = np.asarray(primitive_indices, dtype=np.int64)
    if indices.ndim != 2:
        raise ValueError("primitive indices must be [experts, rank]")
    experts, rank = indices.shape
    flat = indices.reshape(-1)
    if sorted(flat.tolist()) != list(range(experts * rank)):
        raise ValueError("primitive indices are not a permutation")
    labels = np.empty(experts * rank, dtype=np.int64)
    for expert in range(experts):
        labels[indices[expert]] = expert
    validate_assignment(labels, experts, rank)
    return labels


def canonical_to_physical(
    canonical: np.ndarray, primitive_indices: np.ndarray
) -> np.ndarray:
    """Map immutable primitive order into a checkpoint's physical slot order."""

    values = np.asarray(canonical, dtype=np.float64)
    indices = np.asarray(primitive_indices, dtype=np.int64)
    if values.ndim != 1 or indices.ndim != 2 or values.size != indices.size:
        raise ValueError("canonical values and primitive indices disagree")
    if sorted(indices.reshape(-1).tolist()) != list(range(values.size)):
        raise ValueError("primitive indices are not a permutation")
    return values[indices.reshape(-1)]


def expand_expert_delta(route_delta: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Apply an analysis-only primitive-to-expert mapping P."""

    code = np.asarray(route_delta, dtype=np.float64)
    assignment = np.asarray(labels, dtype=np.int64)
    if code.ndim != 1 or assignment.ndim != 1:
        raise ValueError("route delta and labels must be vectors")
    if assignment.size == 0 or assignment.min() < 0 or assignment.max() >= code.size:
        raise ValueError("assignment references an invalid expert")
    return code[assignment]


def grouping_routing_decomposition(
    q: np.ndarray,
    damped_fisher: np.ndarray,
    labels: np.ndarray,
    route_delta: np.ndarray,
) -> Decomposition:
    """Compute grouping, routing and their exact weighted-norm decomposition.

    ``q`` and Fisher are in immutable primitive order. ``route_delta`` is the
    unchanged trained router's expert coefficient ``c_hat - 1``. Only
    ``labels`` changes between controlled assignment rows.
    """

    q_value = np.asarray(q, dtype=np.float64)
    fisher = np.asarray(damped_fisher, dtype=np.float64)
    assignment = np.asarray(labels, dtype=np.int64)
    code = np.asarray(route_delta, dtype=np.float64)
    if q_value.ndim != 1 or q_value.shape != fisher.shape or q_value.shape != assignment.shape:
        raise ValueError("q, Fisher and labels must have the same primitive shape")
    if code.ndim != 1:
        raise ValueError("route delta must be one-dimensional")
    rank = assignment.size // code.size
    validate_assignment(assignment, code.size, rank)
    if not np.all(np.isfinite(q_value)) or not np.all(np.isfinite(fisher)):
        raise FloatingPointError("non-finite decomposition input")
    if np.any(fisher <= 0):
        raise ValueError("damped Fisher must be strictly positive")

    free = q_value / fisher
    coefficients = np.empty(code.size, dtype=np.float64)
    tied = np.empty_like(free)
    for expert in range(code.size):
        selected = assignment == expert
        coefficients[expert] = q_value[selected].sum() / fisher[selected].sum()
        tied[selected] = coefficients[expert]
    routed = expand_expert_delta(code, assignment)
    grouping = 0.5 * float(np.sum(fisher * np.square(tied - free)))
    routing = 0.5 * float(np.sum(fisher * np.square(routed - tied)))
    direct = 0.5 * float(np.sum(fisher * np.square(routed - free)))
    total = grouping + routing
    error = abs(total - direct)
    tolerance = 1e-9 * max(1.0, abs(total), abs(direct))
    if error > tolerance:
        raise AssertionError(
            f"grouping/routing projection identity failed: error={error} tolerance={tolerance}"
        )
    return Decomposition(
        grouping=grouping,
        routing=routing,
        total=total,
        direct_total=direct,
        orthogonality_error=error,
        tied_coefficients=coefficients,
        tied_primitive_delta=tied,
        routed_primitive_delta=routed,
    )
