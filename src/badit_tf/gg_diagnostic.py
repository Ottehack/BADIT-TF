"""Pure analysis helpers for the P2-GG-D1 regroup-collapse diagnostic."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def labels_from_primitive_indices(indices: torch.Tensor) -> torch.Tensor:
    """Convert expert-major canonical primitive IDs back to assignment labels."""

    if indices.ndim != 2:
        raise ValueError("primitive indices must have shape [experts, rank]")
    num_experts, rank = indices.shape
    total = num_experts * rank
    flat = indices.detach().cpu().long().flatten()
    if sorted(flat.tolist()) != list(range(total)):
        raise ValueError("primitive indices are not a permutation")
    labels = torch.empty(total, dtype=torch.long)
    for expert in range(num_experts):
        labels[indices[expert].detach().cpu().long()] = expert
    return labels


def assignment_fragmentation(
    assignments: Mapping[str, Sequence[int]], *, num_experts: int, rank: int
) -> dict[str, Any]:
    """Measure how strongly a regroup fragments the original contiguous experts."""

    expected = num_experts * rank
    layer_rows = {}
    for layer, raw_labels in sorted(assignments.items()):
        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.shape != (expected,):
            raise ValueError(f"{layer}: expected {expected} labels")
        counts = np.bincount(labels, minlength=num_experts)
        if not np.array_equal(counts, np.full(num_experts, rank)):
            raise ValueError(f"{layer}: assignment is not equal capacity")
        overlap = np.zeros((num_experts, num_experts), dtype=np.int64)
        for primitive, new_expert in enumerate(labels):
            overlap[primitive // rank, new_expert] += 1
        purity = float(np.mean(overlap.max(axis=0) / rank))
        entropies = []
        for new_expert in range(num_experts):
            probabilities = overlap[:, new_expert] / rank
            positive = probabilities[probabilities > 0]
            entropies.append(
                float(-(positive * np.log(positive)).sum() / math.log(num_experts))
            )
        old_splits = (overlap > 0).sum(axis=1)
        original = np.arange(expected) // rank
        layer_rows[layer] = {
            "changed_primitive_fraction": float(np.mean(labels != original)),
            "new_expert_origin_purity": purity,
            "new_expert_origin_entropy": float(np.mean(entropies)),
            "mean_old_expert_split_count": float(np.mean(old_splits)),
            "max_old_expert_split_count": int(old_splits.max()),
            "overlap_old_by_new": overlap.tolist(),
        }
    scalar_keys = (
        "changed_primitive_fraction",
        "new_expert_origin_purity",
        "new_expert_origin_entropy",
        "mean_old_expert_split_count",
        "max_old_expert_split_count",
    )
    return {
        "per_layer": layer_rows,
        "aggregate": {
            key: float(np.mean([row[key] for row in layer_rows.values()]))
            for key in scalar_keys
        },
    }


def prediction_degeneracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return transparent string/token diagnostics without excluding outputs."""

    predictions = [" ".join(str(row["prediction"]).split()) for row in rows]
    token_rows = [text.split() for text in predictions]
    adjacent = []
    unique_ratios = []
    dominant_ratios = []
    for tokens in token_rows:
        if not tokens:
            adjacent.append(0.0)
            unique_ratios.append(0.0)
            dominant_ratios.append(0.0)
            continue
        adjacent.append(
            sum(left == right for left, right in zip(tokens, tokens[1:]))
            / max(len(tokens) - 1, 1)
        )
        counts = Counter(tokens)
        unique_ratios.append(len(counts) / len(tokens))
        dominant_ratios.append(max(counts.values()) / len(tokens))
    counts = Counter(predictions)
    n = max(len(predictions), 1)
    return {
        "records": float(len(predictions)),
        "empty_fraction": float(sum(not text for text in predictions) / n),
        "unique_prediction_fraction": float(len(counts) / n),
        "dominant_prediction_fraction": float(max(counts.values(), default=0) / n),
        "mean_whitespace_tokens": float(np.mean([len(x) for x in token_rows]))
        if token_rows
        else 0.0,
        "mean_adjacent_repeat_fraction": float(np.mean(adjacent)) if adjacent else 0.0,
        "mean_unique_token_fraction": float(np.mean(unique_ratios))
        if unique_ratios
        else 0.0,
        "mean_dominant_token_fraction": float(np.mean(dominant_ratios))
        if dominant_ratios
        else 0.0,
    }


def classify_collapse(
    rouge_l: Mapping[str, float], *, causal_margin_points: float
) -> dict[str, Any]:
    """Apply the frozen D1 causal classification rule."""

    required = {
        "post_regroup_top4",
        "canonical_top4",
        "post_regroup_dense",
        "canonical_dense",
        "post_regroup_all_one",
        "canonical_all_one",
    }
    missing = required.difference(rouge_l)
    if missing:
        raise ValueError(f"missing conditions: {sorted(missing)}")
    base = float(rouge_l["post_regroup_top4"])
    effects = {
        "undo_regroup_at_trained_top4": float(rouge_l["canonical_top4"] - base),
        "disable_top4_after_regroup": float(rouge_l["post_regroup_dense"] - base),
        "all_one_after_regroup": float(rouge_l["post_regroup_all_one"] - base),
        "undo_regroup_at_dense": float(
            rouge_l["canonical_dense"] - rouge_l["post_regroup_dense"]
        ),
    }
    supported = {
        "regroup_router_semantic_mismatch": (
            effects["undo_regroup_at_trained_top4"] >= causal_margin_points
        ),
        "topk_regroup_interaction": (
            effects["disable_top4_after_regroup"] >= causal_margin_points
        ),
        "learned_router_dependency": (
            effects["all_one_after_regroup"] >= causal_margin_points
        ),
    }
    if supported["regroup_router_semantic_mismatch"]:
        primary = "REGROUP_ROUTER_SEMANTIC_MISMATCH_SUPPORTED"
    elif supported["topk_regroup_interaction"]:
        primary = "TOPK_REGROUP_INTERACTION_SUPPORTED"
    elif supported["learned_router_dependency"]:
        primary = "LEARNED_ROUTER_DEPENDENCY_SUPPORTED"
    else:
        primary = "COLLAPSE_PREDATES_REGROUP_OR_UNRESOLVED"
    return {
        "causal_margin_points": float(causal_margin_points),
        "effects_rougeL_points": effects,
        "supported": supported,
        "primary": primary,
    }
