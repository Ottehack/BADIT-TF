import pytest
import torch

from badit_tf.gg_diagnostic import (
    assignment_fragmentation,
    classify_collapse,
    labels_from_primitive_indices,
    prediction_degeneracy,
)


def test_labels_from_primitive_indices_roundtrip():
    indices = torch.tensor([[0, 3], [1, 2]])
    assert labels_from_primitive_indices(indices).tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError):
        labels_from_primitive_indices(torch.tensor([[0, 0], [1, 2]]))


def test_assignment_fragmentation_contiguous_and_mixed():
    contiguous = assignment_fragmentation(
        {"layer": [0, 0, 1, 1]}, num_experts=2, rank=2
    )["aggregate"]
    assert contiguous["changed_primitive_fraction"] == 0.0
    assert contiguous["new_expert_origin_purity"] == 1.0
    mixed = assignment_fragmentation(
        {"layer": [0, 1, 0, 1]}, num_experts=2, rank=2
    )["aggregate"]
    assert mixed["changed_primitive_fraction"] == 0.5
    assert mixed["new_expert_origin_purity"] == 0.5
    assert mixed["mean_old_expert_split_count"] == 2.0


def test_prediction_degeneracy_retains_empty_and_repeated_outputs():
    rows = [
        {"prediction": "x x x"},
        {"prediction": ""},
        {"prediction": "x x x"},
        {"prediction": "a b"},
    ]
    metrics = prediction_degeneracy(rows)
    assert metrics["records"] == 4.0
    assert metrics["empty_fraction"] == 0.25
    assert metrics["unique_prediction_fraction"] == 0.75
    assert metrics["dominant_prediction_fraction"] == 0.5


def test_classify_collapse_uses_frozen_five_point_margin():
    result = classify_collapse(
        {
            "post_regroup_top4": 0.1,
            "canonical_top4": 8.0,
            "post_regroup_dense": 2.0,
            "canonical_dense": 8.5,
            "post_regroup_all_one": 7.0,
            "canonical_all_one": 7.0,
        },
        causal_margin_points=5.0,
    )
    assert result["primary"] == "REGROUP_ROUTER_SEMANTIC_MISMATCH_SUPPORTED"
    assert result["supported"]["regroup_router_semantic_mismatch"]
    assert not result["supported"]["topk_regroup_interaction"]
