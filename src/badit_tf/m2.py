"""Shared, auditable helpers for the frozen M2 fidelity experiment.

M2 deliberately evaluates only assignments that define a genuine equal-capacity
projection from the fixed primitive bank to experts.  The historical public
BADIT greedy repair is retained as a provenance diagnostic elsewhere, but can
duplicate/drop slots and therefore cannot be used as the ``P`` in the M2
finite-gap identity.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .assignment import (
    build_natural_modulation_profiles,
    solve_balanced_tf_assignment,
    validate_assignment,
)


def validate_fidelity_precision(config: dict[str, Any]) -> None:
    """Require a frozen FP32 numerical path for every M2 finite-loss run.

    P1's fixed-probe audit found the virtual-gate derivative and repeated
    forward check unreliable in BF16.  M2 reuses the same objective identity,
    so collection, assignment and finite-loss evaluation must use a frozen
    FP32 path.  SDPA is the default recovery backend.  Gemma2 is the one
    recorded model-specific exception: its prior SDPA diagnostic emitted an
    attention warning, while its H0/H1 bank was frozen with eager attention.
    This is a fail-closed protocol contract rather than a scale adjustment.
    """

    expected_attention = {
        "fp32_sdpa_fidelity_v1": "sdpa",
        "fp32_eager_fidelity_v1": "eager",
    }
    contract = config.get("precision_contract")
    if contract not in expected_attention:
        raise AssertionError("M2 must declare an approved FP32 fidelity precision contract")
    if config.get("torch_dtype") != "float32":
        raise AssertionError("M2 finite-loss fidelity requires float32 after the P1 BF16 audit")
    if config.get("attention_implementation") != expected_attention[contract]:
        raise AssertionError(
            f"M2 precision contract {contract} requires "
            f"{expected_attention[contract]} attention"
        )
    manifest_path = config.get("precision_contract_manifest")
    manifest_sha256 = config.get("precision_contract_manifest_sha256")
    if bool(manifest_path) != bool(manifest_sha256):
        raise AssertionError("M2 precision-contract manifest path and SHA256 must be specified together")
    if manifest_path:
        path = Path(str(manifest_path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if digest != manifest_sha256:
            raise AssertionError("M2 precision-contract manifest is missing or hash-mismatched")
        policy = json.loads(path.read_text(encoding="utf-8"))
        declared = policy.get("model_exceptions", {}).get(
            config.get("model_name"), policy.get("default_contract", {})
        )
        for key in ("precision_contract", "torch_dtype", "attention_implementation"):
            if config.get(key) != declared.get(key):
                raise AssertionError(f"M2 config disagrees with frozen model precision policy for {key}")


def task_prefix_indices(metadata: Iterable[dict[str, Any]], count: int) -> np.ndarray:
    """Return the frozen first ``count`` rows per task in manifest order."""

    rows = list(metadata)
    if count <= 0:
        raise ValueError("per-task count must be positive")
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[str(row["task"])].append(index)
    if not by_task:
        raise ValueError("metadata contains no tasks")
    selected: list[int] = []
    for task in sorted(by_task):
        indices = by_task[task]
        if len(indices) < count:
            raise ValueError(f"{task} has {len(indices)} rows but needs {count}")
        selected.extend(indices[:count])
    return np.asarray(selected, dtype=np.int64)


def task_balanced_weights(metadata: Iterable[dict[str, Any]]) -> np.ndarray:
    """Weights that give every observed task equal total mass."""

    rows = list(metadata)
    tasks = [str(row["task"]) for row in rows]
    if not tasks:
        raise ValueError("metadata contains no rows")
    counts = {task: tasks.count(task) for task in sorted(set(tasks))}
    return np.asarray(
        [1.0 / (len(counts) * counts[task]) for task in tasks], dtype=np.float64
    )


def canonical_gg_assignment(
    vectors: np.ndarray,
    *,
    num_experts: int,
    rank: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Matched BADIT-GG geometry with deterministic exact capacity.

    Import locally to keep this module usable in collection-only CPU tools.
    """

    import torch

    from .training import balanced_gradient_geometry_assignment

    value = torch.as_tensor(vectors, dtype=torch.float32)
    labels, audit = balanced_gradient_geometry_assignment(
        value, num_experts=num_experts, rank=rank, seed=seed
    )
    validate_assignment(labels, num_experts, rank)
    return labels, audit


def build_assignment_payload(
    *,
    q_assignment: np.ndarray,
    assignment_metadata: list[dict[str, Any]],
    fisher_scores: np.ndarray,
    fisher_metadata: list[dict[str, Any]],
    gg_mean_vectors: np.ndarray,
    layer_names: list[str],
    num_experts: int,
    rank: int,
    assignment_probes_per_task: int,
    fisher_probes_per_task: int,
    epsilon_f: float,
    random_assignments: int,
    restarts: int,
    max_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build all M2 assignments without reading fidelity examples.

    ``q_assignment`` has shape ``[I, layers, primitives]``; Fisher scores
    are per-sequence score gradients.  ``gg_mean_vectors`` is reduced only
    across the selected assignment role and has shape
    ``[layers, primitives, feature_dim]``.
    """

    q_assignment = np.asarray(q_assignment, dtype=np.float64)
    fisher_scores = np.asarray(fisher_scores, dtype=np.float64)
    gg_mean_vectors = np.asarray(gg_mean_vectors, dtype=np.float64)
    primitive_count = int(num_experts) * int(rank)
    if q_assignment.ndim != 3 or q_assignment.shape[2] != primitive_count:
        raise ValueError("q_assignment shape does not match primitive count")
    # H0 can legitimately select a different number of assignment and Fisher
    # probes per task (Gemma2-2B's locked H0-05 uses 16 and 8 respectively).
    # They must share the model's layer/primitive axes, not the sample axis.
    if fisher_scores.ndim != 3 or fisher_scores.shape[1:] != q_assignment.shape[1:]:
        raise ValueError("fisher_scores must match q_assignment layer/primitive shape")
    if len(layer_names) != q_assignment.shape[1]:
        raise ValueError("layer names do not match q_assignment")
    if gg_mean_vectors.ndim != 3 or gg_mean_vectors.shape[:2] != q_assignment.shape[1:]:
        raise ValueError("GG vector shape does not match q_assignment")
    if random_assignments < 20:
        raise ValueError("M2 requires at least 20 random balanced assignments")
    if epsilon_f <= 0:
        raise ValueError("epsilon_f must be positive")

    assignment_indices = task_prefix_indices(
        assignment_metadata, assignment_probes_per_task
    )
    fisher_indices = task_prefix_indices(fisher_metadata, fisher_probes_per_task)
    selected_q = q_assignment[assignment_indices]
    selected_meta = [assignment_metadata[index] for index in assignment_indices]
    selected_fisher = fisher_scores[fisher_indices]
    fisher = np.mean(np.square(selected_fisher), axis=0)
    if not np.all(np.isfinite(fisher)) or np.any(fisher < 0):
        raise FloatingPointError("M2 Fisher contains NaN/Inf or negative entries")
    weights = task_balanced_weights(selected_meta)

    contiguous = np.arange(primitive_count, dtype=np.int64) // int(rank)
    rng = np.random.default_rng(int(seed) + 911)
    random_labels: list[list[int]] = []
    for _ in range(random_assignments):
        labels = contiguous.copy()
        rng.shuffle(labels)
        validate_assignment(labels, num_experts, rank)
        random_labels.append(labels.tolist())

    payload: dict[str, Any] = {
        "schema_version": 1,
        "method_contract": {
            "contiguous": "canonical SVD primitive order",
            "random_balanced": "20-or-more independently shuffled exact-capacity assignments; all are averaged",
            "gg_dog": "matched gradient-geometry KMeans plus Hungarian exact capacity; public greedy duplicate/drop repair is audit-only and excluded",
            "raw_q": "unweighted exact-capacity clustering of direct q profiles",
            "tf": "task-balanced natural q/(F+rho) profiles with exact capacity",
        },
        "layer_names": list(layer_names),
        "contiguous": contiguous.tolist(),
        "random_balanced": random_labels,
        "gg_dog": {},
        "raw_q": {},
        "tf": {},
        "rho": {},
        "solver_audit": {"gg_dog": {}, "raw_q": {}, "tf": {}},
        "selected_counts": {
            "assignment_probes_per_task": int(assignment_probes_per_task),
            "fisher_probes_per_task": int(fisher_probes_per_task),
        },
    }
    for layer_index, layer_name in enumerate(layer_names):
        layer_fisher = fisher[layer_index]
        median = float(np.median(layer_fisher))
        if not np.isfinite(median) or median <= 0:
            raise FloatingPointError(f"invalid Fisher median for {layer_name}")
        rho = float(epsilon_f) * median
        profiles, damped, _ = build_natural_modulation_profiles(
            selected_q[:, layer_index, :], layer_fisher, weights, rho
        )
        tf = solve_balanced_tf_assignment(
            profiles,
            damped,
            num_experts=num_experts,
            rank=rank,
            restarts=restarts,
            max_iterations=max_iterations,
            tolerance=tolerance,
            seed=int(seed) + layer_index * 4099,
        )
        raw_q = solve_balanced_tf_assignment(
            selected_q[:, layer_index, :].T,
            np.ones(primitive_count, dtype=np.float64),
            num_experts=num_experts,
            rank=rank,
            restarts=restarts,
            max_iterations=max_iterations,
            tolerance=tolerance,
            seed=int(seed) + 50_000 + layer_index * 4099,
        )
        gg_labels, gg_audit = canonical_gg_assignment(
            gg_mean_vectors[layer_index],
            num_experts=num_experts,
            rank=rank,
            seed=layer_index,
        )
        payload["rho"][layer_name] = rho
        payload["tf"][layer_name] = tf.labels.tolist()
        payload["raw_q"][layer_name] = raw_q.labels.tolist()
        payload["gg_dog"][layer_name] = gg_labels.tolist()
        payload["solver_audit"]["tf"][layer_name] = tf.to_jsonable()
        payload["solver_audit"]["raw_q"][layer_name] = raw_q.to_jsonable()
        payload["solver_audit"]["gg_dog"][layer_name] = gg_audit

    arrays = {
        "fisher": fisher.astype(np.float64),
        "assignment_indices": assignment_indices,
        "fisher_indices": fisher_indices,
    }
    return payload, arrays
