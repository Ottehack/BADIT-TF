"""Task-functional profile construction and balanced assignment.

For fixed centroids, equal capacity is solved exactly by expanding every
expert into ``rank`` identical slots and applying linear-sum assignment.  This
is an exact integral solution of the corresponding capacity-constrained
minimum-cost flow subproblem.  Alternation over centroids is non-convex, so the
multi-restart solver reports the best observed objective and never claims a
global optimum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RestartAudit:
    restart: int
    seed: int
    iterations: int
    converged: bool
    objective: float
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class AssignmentResult:
    labels: np.ndarray
    matrix: np.ndarray
    objective: float
    best_restart: int
    audits: tuple[RestartAudit, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "labels": self.labels.tolist(),
            "matrix": self.matrix.tolist(),
            "objective": self.objective,
            "best_restart": self.best_restart,
            "audits": [asdict(item) for item in self.audits],
        }


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        raise ValueError(f"{name} contains NaN/Inf at {bad[:8].tolist()}")
    return array


def validate_assignment(labels: np.ndarray, num_experts: int, rank: int) -> None:
    labels = np.asarray(labels, dtype=np.int64)
    expected = num_experts * rank
    if labels.shape != (expected,):
        raise ValueError(f"labels must have shape ({expected},), got {labels.shape}")
    if labels.min(initial=0) < 0 or labels.max(initial=-1) >= num_experts:
        raise ValueError("assignment contains an invalid expert index")
    counts = np.bincount(labels, minlength=num_experts)
    if not np.array_equal(counts, np.full(num_experts, rank, dtype=np.int64)):
        raise ValueError(f"assignment capacity mismatch: {counts.tolist()}")


def labels_to_matrix(labels: np.ndarray, num_experts: int, rank: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    validate_assignment(labels, num_experts, rank)
    matrix = np.zeros((num_experts * rank, num_experts), dtype=np.int8)
    matrix[np.arange(num_experts * rank), labels] = 1
    if not np.all(matrix.sum(axis=1) == 1):
        raise AssertionError("every primitive must be assigned exactly once")
    if not np.all(matrix.sum(axis=0) == rank):
        raise AssertionError("every expert must receive exactly rank primitives")
    return matrix


def build_natural_modulation_profiles(
    q: np.ndarray,
    fisher: np.ndarray,
    task_weights: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build stacked task-balanced natural-modulation profiles.

    Args:
        q: Per-probe negative gate gradients, shape ``[I, M]``.
        fisher: Undamped model-Fisher diagonal, shape ``[M]``.
        task_weights: Per-probe weights, shape ``[I]`` and positive.
        damping: Positive layer-specific rho.

    Returns:
        ``profiles[M, I]``, damped Fisher weights ``f[M]``, and natural
        modulations ``d[I, M]``.
    """

    q = _finite("q", q)
    fisher = _finite("fisher", fisher)
    task_weights = _finite("task_weights", task_weights)
    if q.ndim != 2:
        raise ValueError(f"q must be 2-D, got {q.shape}")
    if fisher.shape != (q.shape[1],):
        raise ValueError("fisher shape does not match q primitives")
    if task_weights.shape != (q.shape[0],):
        raise ValueError("task_weights shape does not match q probes")
    if damping <= 0 or not np.isfinite(damping):
        raise ValueError("damping must be finite and positive")
    if np.any(fisher < 0):
        raise ValueError("undamped Fisher entries must be nonnegative")
    if np.any(task_weights <= 0):
        raise ValueError("task weights must be positive")

    damped = fisher + float(damping)
    natural = q / damped[None, :]
    profiles = (np.sqrt(task_weights)[:, None] * natural).T
    return _finite("profiles", profiles), damped, natural


def _centroids(
    profiles: np.ndarray,
    primitive_weights: np.ndarray,
    labels: np.ndarray,
    num_experts: int,
) -> np.ndarray:
    centroids = np.empty((num_experts, profiles.shape[1]), dtype=np.float64)
    for expert in range(num_experts):
        selected = labels == expert
        weights = primitive_weights[selected]
        if not np.any(selected) or weights.sum() <= 0:
            raise ValueError(f"expert {expert} has no positive-weight primitive")
        centroids[expert] = np.average(profiles[selected], axis=0, weights=weights)
    return centroids


def _cost_matrix(
    profiles: np.ndarray,
    primitive_weights: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    squared = ((profiles[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return primitive_weights[:, None] * squared


def assignment_objective(
    profiles: np.ndarray,
    primitive_weights: np.ndarray,
    labels: np.ndarray,
    num_experts: int,
) -> float:
    centroids = _centroids(profiles, primitive_weights, labels, num_experts)
    cost = _cost_matrix(profiles, primitive_weights, centroids)
    return float(0.5 * cost[np.arange(labels.size), labels].sum())


def _solve_fixed_centroids(cost: np.ndarray, rank: int) -> np.ndarray:
    """Solve the exact equal-capacity assignment for fixed centroids."""

    slot_cost = np.repeat(cost, rank, axis=1)
    row_indices, slot_indices = linear_sum_assignment(slot_cost)
    if not np.array_equal(row_indices, np.arange(cost.shape[0])):
        raise AssertionError("linear assignment did not cover every primitive")
    return (slot_indices // rank).astype(np.int64, copy=False)


def _solve_fixed_centroids_nonempty(cost: np.ndarray) -> np.ndarray:
    """Solve the fixed-centroid problem with only a nonempty lower bound.

    Every primitive is initially sent to its cheapest expert.  Covering an
    expert with one distinct primitive then costs the excess over that
    primitive's cheapest destination.  A rectangular Hungarian solve chooses
    the exact minimum-cost set of representatives; all remaining primitives
    keep their individually cheapest destinations.
    """

    if cost.ndim != 2 or cost.shape[0] < cost.shape[1]:
        raise ValueError("nonempty assignment requires primitives >= experts")
    baseline = np.argmin(cost, axis=1).astype(np.int64, copy=False)
    excess = cost - np.min(cost, axis=1, keepdims=True)
    primitive_indices, expert_indices = linear_sum_assignment(excess)
    if len(expert_indices) != cost.shape[1] or set(expert_indices.tolist()) != set(
        range(cost.shape[1])
    ):
        raise AssertionError("nonempty assignment did not cover every expert")
    labels = baseline.copy()
    labels[primitive_indices] = expert_indices
    if np.any(np.bincount(labels, minlength=cost.shape[1]) == 0):
        raise AssertionError("nonempty assignment produced an empty expert")
    return labels


def solve_balanced_tf_assignment(
    profiles: np.ndarray,
    primitive_weights: np.ndarray,
    *,
    num_experts: int,
    rank: int,
    restarts: int = 5,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    seed: int = 1,
) -> AssignmentResult:
    """Alternate exact capacity assignments and weighted centroids."""

    profiles = _finite("profiles", profiles)
    primitive_weights = _finite("primitive_weights", primitive_weights)
    primitive_count = num_experts * rank
    if profiles.ndim != 2 or profiles.shape[0] != primitive_count:
        raise ValueError(
            f"profiles must have shape ({primitive_count}, I), got {profiles.shape}"
        )
    if primitive_weights.shape != (primitive_count,):
        raise ValueError("primitive_weights shape mismatch")
    if np.any(primitive_weights <= 0):
        raise ValueError("primitive weights must be strictly positive")
    if restarts < 1 or max_iterations < 1:
        raise ValueError("restarts and max_iterations must be positive")

    audits: list[RestartAudit] = []
    candidates: list[tuple[float, np.ndarray]] = []
    for restart in range(restarts):
        restart_seed = int(seed + 104729 * restart)
        rng = np.random.default_rng(restart_seed)
        labels = np.repeat(np.arange(num_experts, dtype=np.int64), rank)
        rng.shuffle(labels)
        history: list[float] = [
            assignment_objective(profiles, primitive_weights, labels, num_experts)
        ]
        converged = False

        for iteration in range(1, max_iterations + 1):
            centroids = _centroids(
                profiles, primitive_weights, labels, num_experts
            )
            updated = _solve_fixed_centroids(
                _cost_matrix(profiles, primitive_weights, centroids), rank
            )
            validate_assignment(updated, num_experts, rank)
            objective = assignment_objective(
                profiles, primitive_weights, updated, num_experts
            )
            history.append(objective)
            if objective > history[-2] + max(tolerance, abs(history[-2]) * 1e-12):
                raise AssertionError(
                    "alternating assignment objective increased: "
                    f"{history[-2]} -> {objective}"
                )
            unchanged = np.array_equal(updated, labels)
            improvement = history[-2] - history[-1]
            labels = updated
            if unchanged or improvement <= tolerance:
                converged = True
                break

        final_objective = assignment_objective(
            profiles, primitive_weights, labels, num_experts
        )
        audit = RestartAudit(
            restart=restart,
            seed=restart_seed,
            iterations=len(history) - 1,
            converged=converged,
            objective=final_objective,
            objective_history=tuple(float(item) for item in history),
        )
        audits.append(audit)
        candidates.append((final_objective, labels.copy()))

    best_restart = int(np.argmin([item[0] for item in candidates]))
    best_objective, best_labels = candidates[best_restart]
    validate_assignment(best_labels, num_experts, rank)
    return AssignmentResult(
        labels=best_labels,
        matrix=labels_to_matrix(best_labels, num_experts, rank),
        objective=float(best_objective),
        best_restart=best_restart,
        audits=tuple(audits),
    )


def solve_nonempty_tf_assignment(
    profiles: np.ndarray,
    primitive_weights: np.ndarray,
    *,
    num_experts: int,
    restarts: int = 5,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    seed: int = 1,
) -> AssignmentResult:
    """Alternate weighted centroids with the paper's nonempty-only constraint.

    This is the Table VI ``w/o equal capacity`` diagnostic: all primitives are
    assigned once and every expert is nonempty, but expert sizes may differ.
    The returned matrix therefore has unequal column sums and is intentionally
    not validated by :func:`validate_assignment`.
    """

    profiles = _finite("profiles", profiles)
    primitive_weights = _finite("primitive_weights", primitive_weights)
    primitive_count = profiles.shape[0]
    if profiles.ndim != 2 or primitive_count < int(num_experts):
        raise ValueError("invalid nonempty profile shape")
    if primitive_weights.shape != (primitive_count,) or np.any(
        primitive_weights <= 0
    ):
        raise ValueError("invalid nonempty primitive weights")
    if restarts < 1 or max_iterations < 1:
        raise ValueError("restarts and max_iterations must be positive")

    audits: list[RestartAudit] = []
    candidates: list[tuple[float, np.ndarray]] = []
    for restart in range(restarts):
        restart_seed = int(seed + 104729 * restart)
        rng = np.random.default_rng(restart_seed)
        representatives = rng.choice(
            primitive_count, size=num_experts, replace=False
        )
        labels = rng.integers(num_experts, size=primitive_count, dtype=np.int64)
        labels[representatives] = np.arange(num_experts, dtype=np.int64)
        history = [
            assignment_objective(profiles, primitive_weights, labels, num_experts)
        ]
        converged = False
        for _ in range(1, max_iterations + 1):
            centroids = _centroids(
                profiles, primitive_weights, labels, num_experts
            )
            updated = _solve_fixed_centroids_nonempty(
                _cost_matrix(profiles, primitive_weights, centroids)
            )
            objective = assignment_objective(
                profiles, primitive_weights, updated, num_experts
            )
            history.append(objective)
            if objective > history[-2] + max(
                tolerance, abs(history[-2]) * 1e-12
            ):
                raise AssertionError(
                    "nonempty alternating objective increased: "
                    f"{history[-2]} -> {objective}"
                )
            unchanged = np.array_equal(updated, labels)
            improvement = history[-2] - history[-1]
            labels = updated
            if unchanged or improvement <= tolerance:
                converged = True
                break
        final_objective = assignment_objective(
            profiles, primitive_weights, labels, num_experts
        )
        audits.append(
            RestartAudit(
                restart=restart,
                seed=restart_seed,
                iterations=len(history) - 1,
                converged=converged,
                objective=final_objective,
                objective_history=tuple(float(item) for item in history),
            )
        )
        candidates.append((final_objective, labels.copy()))

    best_restart = int(np.argmin([item[0] for item in candidates]))
    best_objective, best_labels = candidates[best_restart]
    counts = np.bincount(best_labels, minlength=num_experts)
    if np.any(counts == 0) or int(counts.sum()) != primitive_count:
        raise AssertionError("invalid final nonempty assignment")
    matrix = np.zeros((primitive_count, num_experts), dtype=np.int8)
    matrix[np.arange(primitive_count), best_labels] = 1
    return AssignmentResult(
        labels=best_labels,
        matrix=matrix,
        objective=float(best_objective),
        best_restart=best_restart,
        audits=tuple(audits),
    )
