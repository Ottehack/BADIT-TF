import hashlib
import json

import numpy as np

from badit_tf.assignment import validate_assignment
import pytest

from badit_tf.m2 import (
    build_assignment_payload,
    task_prefix_indices,
    validate_fidelity_precision,
)


def _rows(per_task: int):
    return [
        {"sample_id": f"{task}-{index}", "task": task}
        for task in ("a", "b")
        for index in range(per_task)
    ]


def test_task_prefix_indices_retains_frozen_manifest_order_per_task():
    rows = _rows(4)
    assert task_prefix_indices(rows, 2).tolist() == [0, 1, 4, 5]


def test_m2_fidelity_precision_contract_is_fp32_sdpa_and_fail_closed():
    validate_fidelity_precision(
        {
            "precision_contract": "fp32_sdpa_fidelity_v1",
            "torch_dtype": "float32",
            "attention_implementation": "sdpa",
        }
    )
    with pytest.raises(AssertionError, match="float32"):
        validate_fidelity_precision(
            {
                "precision_contract": "fp32_sdpa_fidelity_v1",
                "torch_dtype": "bfloat16",
                "attention_implementation": "sdpa",
            }
        )


def test_m2_precision_policy_manifest_is_hash_bound(tmp_path):
    policy = {
        "default_contract": {
            "precision_contract": "fp32_sdpa_fidelity_v1",
            "torch_dtype": "float32",
            "attention_implementation": "sdpa",
        },
        "model_exceptions": {
            "Gemma2-2B": {
                "precision_contract": "fp32_eager_fidelity_v1",
                "torch_dtype": "float32",
                "attention_implementation": "eager",
            }
        },
    }
    path = tmp_path / "precision.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    validate_fidelity_precision(
        {
            "model_name": "Gemma2-2B",
            "precision_contract": "fp32_eager_fidelity_v1",
            "torch_dtype": "float32",
            "attention_implementation": "eager",
            "precision_contract_manifest": str(path),
            "precision_contract_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    validate_fidelity_precision(
        {
            "precision_contract": "fp32_eager_fidelity_v1",
            "torch_dtype": "float32",
            "attention_implementation": "eager",
        }
    )
    with pytest.raises(AssertionError, match="requires eager"):
        validate_fidelity_precision(
            {
                "precision_contract": "fp32_eager_fidelity_v1",
                "torch_dtype": "float32",
                "attention_implementation": "sdpa",
            }
        )


def test_m2_payload_uses_only_exact_capacity_assignments():
    rng = np.random.default_rng(10)
    # Two layers, four experts, rank two, and four assignment/Fisher probes per
    # task.  The selected prefix below is two probes per task.
    q = rng.normal(size=(8, 2, 8))
    scores = rng.normal(size=(8, 2, 8))
    gg = rng.normal(size=(2, 8, 13))
    payload, arrays = build_assignment_payload(
        q_assignment=q,
        assignment_metadata=_rows(4),
        fisher_scores=scores,
        fisher_metadata=_rows(4),
        gg_mean_vectors=gg,
        layer_names=["layer.0", "layer.1"],
        num_experts=4,
        rank=2,
        assignment_probes_per_task=2,
        fisher_probes_per_task=2,
        epsilon_f=0.1,
        random_assignments=20,
        restarts=2,
        max_iterations=10,
        tolerance=1e-6,
        seed=5,
    )
    assert arrays["assignment_indices"].tolist() == [0, 1, 4, 5]
    assert len(payload["random_balanced"]) == 20
    for labels in payload["random_balanced"]:
        validate_assignment(np.asarray(labels), 4, 2)
    for method in ("gg_dog", "raw_q", "tf"):
        for labels in payload[method].values():
            validate_assignment(np.asarray(labels), 4, 2)
    assert "public greedy" in payload["method_contract"]["gg_dog"]


def test_m2_payload_allows_h0_locked_asymmetric_fisher_probe_count():
    rng = np.random.default_rng(11)
    q = rng.normal(size=(8, 2, 8))
    # Assignment role has four rows per task while frozen Fisher has two.
    fisher = rng.normal(size=(4, 2, 8))
    gg = rng.normal(size=(2, 8, 13))
    payload, _ = build_assignment_payload(
        q_assignment=q,
        assignment_metadata=_rows(4),
        fisher_scores=fisher,
        fisher_metadata=_rows(2),
        gg_mean_vectors=gg,
        layer_names=["layer.0", "layer.1"],
        num_experts=4,
        rank=2,
        assignment_probes_per_task=4,
        fisher_probes_per_task=2,
        epsilon_f=0.1,
        random_assignments=20,
        restarts=2,
        max_iterations=10,
        tolerance=1e-6,
        seed=5,
    )
    assert payload["selected_counts"] == {
        "assignment_probes_per_task": 4,
        "fisher_probes_per_task": 2,
    }
