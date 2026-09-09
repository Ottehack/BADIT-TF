import numpy as np
import pytest

from badit_tf.decomposition import (
    canonical_to_physical,
    grouping_routing_decomposition,
    labels_from_primitive_indices,
)


def test_labels_and_physical_round_trip() -> None:
    indices = np.asarray([[2, 0], [3, 1]])
    labels = labels_from_primitive_indices(indices)
    assert labels.tolist() == [0, 1, 0, 1]
    assert canonical_to_physical(np.asarray([10, 11, 12, 13]), indices).tolist() == [12, 10, 13, 11]


def test_grouping_routing_is_exact_projection_decomposition() -> None:
    q = np.asarray([1.0, 4.0, 2.0, 3.0])
    fisher = np.asarray([1.0, 2.0, 1.5, 0.5])
    labels = np.asarray([0, 1, 0, 1])
    result = grouping_routing_decomposition(q, fisher, labels, np.asarray([0.25, -0.5]))
    assert result.total == pytest.approx(result.grouping + result.routing)
    assert result.total == pytest.approx(result.direct_total)
    assert result.orthogonality_error < 1e-12


def test_invalid_capacity_fails_closed() -> None:
    with pytest.raises(ValueError):
        grouping_routing_decomposition(
            np.ones(4), np.ones(4), np.asarray([0, 0, 0, 1]), np.zeros(2)
        )
