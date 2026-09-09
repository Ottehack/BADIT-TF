import numpy as np
import pytest

from badit_tf.deployment_scale import (
    canonical_primitive_delta,
    quadratic_loss_delta,
    relative_prediction_error,
    validate_deployment_etas,
)


def test_deployment_eta_grid_is_exact_and_frozen():
    assert validate_deployment_etas([0.10, 0.25, 0.50, 1.00]) == (0.10, 0.25, 0.50, 1.00)
    with pytest.raises(ValueError, match="deployment eta grid"):
        validate_deployment_etas([0.10, 0.50, 1.00])


def test_route_delta_is_expanded_in_canonical_primitive_order():
    code = np.asarray([2.0, 0.5])
    physical_to_canonical = np.asarray([[2, 0], [3, 1]])
    assert canonical_primitive_delta(code, physical_to_canonical).tolist() == [1.0, -0.5, 1.0, -0.5]


def test_quadratic_loss_delta_uses_q_negative_loss_gradient_once():
    q = np.asarray([2.0, -1.0])
    fisher = np.asarray([4.0, 3.0])
    delta = np.asarray([0.5, -0.25])
    expected = -np.dot(q, delta) + 0.5 * np.dot(fisher, delta**2)
    assert quadratic_loss_delta(q, fisher, delta) == pytest.approx(expected)
    assert relative_prediction_error(expected, expected) == 0.0
