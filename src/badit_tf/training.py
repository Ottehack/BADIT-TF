"""Training-time helpers shared by BADIT-TF and matched BADIT-GG."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def resolve_assignment_labels(
    payload: dict[str, Any],
    *,
    method: str,
    epsilon: float,
    num_experts: int,
    rank: int,
    random_index: int = 0,
    allow_unequal_capacity: bool = False,
) -> dict[str, list[int]]:
    """Resolve one frozen assignment method with strict layer/capacity checks."""

    layer_names = [str(name) for name in payload["layer_names"]]
    if len(layer_names) != len(set(layer_names)) or not layer_names:
        raise ValueError("assignment payload must contain unique layer_names")
    if method == "tf":
        resolved = payload["tf"][str(epsilon)]
        if set(resolved) != set(layer_names):
            raise ValueError("TF assignment layers do not match layer_names")
        by_layer = {name: list(resolved[name]) for name in layer_names}
    elif method == "contiguous":
        # Architecture-specific H0 payloads freeze the layer order and TF/
        # random assignments, but some omit this canonical control row.
        # Contiguous-SVD is definitionally the ordered equal-capacity layout,
        # so derive it deterministically when the redundant row is absent.
        labels = list(
            payload.get(
                "contiguous",
                np.repeat(np.arange(num_experts), rank).tolist(),
            )
        )
        by_layer = {name: list(labels) for name in layer_names}
    elif method == "public_contiguous":
        labels = np.repeat(np.arange(num_experts), rank).tolist()
        by_layer = {name: list(labels) for name in layer_names}
    elif method == "random_balanced":
        candidates = payload["random_balanced"]
        if random_index < 0 or random_index >= len(candidates):
            raise ValueError(
                f"assignment random_index={random_index} outside "
                f"[0,{len(candidates)})"
            )
        labels = list(candidates[random_index])
        by_layer = {name: list(labels) for name in layer_names}
    elif method in {"raw_q", "tf_no_capacity"}:
        resolved = payload[method]
        if set(resolved) != set(layer_names):
            raise ValueError(f"{method} assignment layers do not match layer_names")
        by_layer = {name: list(resolved[name]) for name in layer_names}
    else:
        raise ValueError(f"unsupported assignment method={method}")

    expected_primitives = int(num_experts) * int(rank)
    expected_capacity = np.full(int(num_experts), int(rank), dtype=np.int64)
    normalized: dict[str, list[int]] = {}
    for name, raw_labels in by_layer.items():
        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.shape != (expected_primitives,):
            raise ValueError(
                f"assignment {name} has shape={labels.shape}; "
                f"expected={(expected_primitives,)}"
            )
        if np.any(labels < 0) or np.any(labels >= int(num_experts)):
            raise ValueError(f"assignment {name} has out-of-range labels")
        capacity = np.bincount(labels, minlength=int(num_experts))
        unequal = allow_unequal_capacity and method == "tf_no_capacity"
        if unequal and np.any(capacity == 0):
            raise ValueError(
                f"assignment {name} violates nonempty constraint: {capacity.tolist()}"
            )
        if not unequal and not np.array_equal(capacity, expected_capacity):
            raise ValueError(
                f"assignment {name} violates equal capacity: {capacity.tolist()}"
            )
        normalized[name] = labels.tolist()
    return normalized


def residual_router_optimizer_groups(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    router_lr_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Separate expert and residual-router parameters for fair H1 screening.

    A multiplier of one is exactly the legacy optimizer layout semantically;
    exposing the two groups lets the preregistered router-LR factor vary
    without changing expert learning rate or any trainable parameter.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if router_lr_multiplier <= 0:
        raise ValueError("router_lr_multiplier must be positive")
    expert_parameters = []
    router_parameters = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in seen:
            raise RuntimeError(f"duplicate trainable parameter: {name}")
        seen.add(identifier)
        if ".residual_router." in name or name.startswith("residual_router."):
            router_parameters.append(parameter)
        else:
            expert_parameters.append(parameter)
    if not expert_parameters or not router_parameters:
        raise RuntimeError(
            "expected non-empty expert and residual-router parameter groups"
        )
    return [
        {
            "params": expert_parameters,
            "lr": float(learning_rate),
            "group_name": "expert",
        },
        {
            "params": router_parameters,
            "lr": float(learning_rate) * float(router_lr_multiplier),
            "group_name": "residual_router",
        },
    ]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def losses_are_finite_and_stable(
    losses: Any,
    *,
    explosion_factor: float,
    burn_in_steps: int = 32,
) -> bool:
    """Check finiteness globally and loss explosions after fixed burn-in.

    The first ``burn_in_steps`` establish the cold-start reference and are not
    themselves treated as an explosion.  Mixed-task order changes can make an
    early batch much harder than the rest, so comparing that same cold-start
    window's maximum with its median confounds batch difficulty with numerical
    instability.  Non-finite values remain forbidden over the full trace.
    """

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return False
    if explosion_factor <= 0 or burn_in_steps <= 0:
        raise ValueError("explosion_factor and burn_in_steps must be positive")
    if not np.isfinite(values).all():
        return False
    reference_end = min(int(burn_in_steps), int(values.size))
    reference = max(float(np.median(values[:reference_end])), 1e-6)
    monitored = values[reference_end:] if values.size > reference_end else values
    return bool(float(monitored.max()) <= float(explosion_factor) * reference)


def resolve_code_commit(*, cwd: str | Path | None = None) -> str:
    """Return a validated immutable source commit for result provenance.

    Release archives intentionally do not contain ``.git``.  Their launchers
    therefore pin ``BADIT_TF_CODE_COMMIT`` to the commit used to build the
    archive.  Interactive worktrees fall back to ``git rev-parse HEAD``.  An
    absent or malformed value is fatal rather than silently emitting an empty
    provenance field.
    """

    override = os.environ.get("BADIT_TF_CODE_COMMIT", "").strip()
    if override:
        commit = override
    else:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=None if cwd is None else Path(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "unable to resolve code commit; set BADIT_TF_CODE_COMMIT for "
                "git-archive releases"
            ) from error
        commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise RuntimeError(f"invalid code commit: {commit!r}")
    return commit.lower()


def nonzero_cosine_warmup_multiplier(
    current_step: int, *, warmup_steps: int, total_steps: int
) -> float:
    """Cosine schedule whose first optimizer update has nonzero LR."""

    if warmup_steps < 1 or total_steps < 1:
        raise ValueError("warmup_steps and total_steps must be positive")
    if current_step < warmup_steps:
        return float(current_step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(
        1.0,
        float(current_step + 1 - warmup_steps) / float(decay_steps),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def balanced_gradient_geometry_assignment(
    gradient_vectors: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Spherical K-means followed by exact equal-capacity assignment.

    This preserves BADIT-GG's last-backward gradient geometry while replacing
    the upstream random truncate/fill repair, which can duplicate or omit
    primitives, with a deterministic one-to-one Hungarian assignment.
    """

    expected = num_experts * rank
    if gradient_vectors.ndim != 2 or gradient_vectors.shape[0] != expected:
        raise ValueError(f"expected [{expected}, D] gradient vectors")
    vectors = gradient_vectors.detach().float().cpu()
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    zero_count = int((norms.squeeze(1) < 1e-12).sum().item())
    directions = vectors / norms.clamp_min(1e-12)
    array = directions.numpy()
    kmeans = KMeans(
        n_clusters=num_experts,
        random_state=int(seed),
        n_init=10,
        max_iter=100,
    ).fit(array)
    centers = kmeans.cluster_centers_
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(center_norms, 1e-12)
    slots = np.repeat(centers, rank, axis=0)
    cost = -(array @ slots.T)
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(expected)):
        raise AssertionError("Hungarian solver did not cover every primitive")
    labels = (columns // rank).astype(np.int64)
    counts = np.bincount(labels, minlength=num_experts)
    if not np.array_equal(counts, np.full(num_experts, rank)):
        raise AssertionError(f"capacity mismatch: {counts.tolist()}")
    objective = float((array * centers[labels]).sum())
    return labels, {
        "algorithm": "spherical_kmeans_plus_hungarian_capacity",
        "seed": int(seed),
        "zero_gradient_primitives": zero_count,
        "capacity": counts.tolist(),
        "cosine_objective": objective,
        "kmeans_inertia": float(kmeans.inertia_),
    }


def public_badit_gradient_regroup(
    gradient_vectors: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
    rng: Any,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Reproduce the frozen public BADIT KMeans plus greedy slot repair."""

    expected = int(num_experts) * int(rank)
    if gradient_vectors.ndim != 2 or gradient_vectors.shape[0] != expected:
        raise ValueError(f"expected [{expected}, D] gradient vectors")
    vectors = gradient_vectors.detach().float().cpu()
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    directions = vectors / torch.where(
        norms < 1e-12, torch.ones_like(norms), norms
    )
    labels = KMeans(
        n_clusters=num_experts,
        random_state=0,
        n_init=10,
        max_iter=100,
    ).fit_predict(directions.numpy())
    cluster_to_indices = {expert: [] for expert in range(num_experts)}
    for index, label in enumerate(labels):
        cluster_to_indices[int(label)].append(index)
    all_indices = list(range(expected))
    rng.shuffle(all_indices)
    clusters: list[list[int]] = []
    for expert in range(num_experts):
        current = list(cluster_to_indices[expert])
        if len(current) < rank:
            remaining = [
                index
                for index in all_indices
                if index not in current
                and not any(index in cluster for cluster in clusters)
            ]
            if len(remaining) < rank - len(current):
                remaining = all_indices
            current.extend(
                rng.choice(
                    remaining, rank - len(current), replace=False
                ).tolist()
            )
        elif len(current) > rank:
            current = rng.choice(current, rank, replace=False).tolist()
        clusters.append([int(index) for index in current[:rank]])
    flat = [index for cluster in clusters for index in cluster]
    counts = np.bincount(flat, minlength=expected)
    return clusters, {
        "algorithm": "public_spherical_kmeans_plus_random_greedy_repair",
        "zero_gradient_primitives": int((norms.squeeze(1) < 1e-12).sum()),
        "unique_source_slots": len(set(flat)),
        "duplicate_copies": int((counts - 1).clip(min=0).sum()),
        "omitted_source_slots": int((counts == 0).sum()),
        "clusters": clusters,
    }
