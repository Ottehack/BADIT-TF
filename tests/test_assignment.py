import numpy as np

from badit_tf.assignment import (
    build_natural_modulation_profiles,
    solve_balanced_tf_assignment,
    solve_nonempty_tf_assignment,
    validate_assignment,
)


def test_balanced_solver_is_deterministic_and_exact_capacity():
    rng = np.random.default_rng(7)
    profiles = rng.normal(size=(32, 12))
    fisher = rng.uniform(0.1, 2.0, size=32)
    first = solve_balanced_tf_assignment(
        profiles,
        fisher,
        num_experts=8,
        rank=4,
        restarts=3,
        seed=19,
    )
    second = solve_balanced_tf_assignment(
        profiles,
        fisher,
        num_experts=8,
        rank=4,
        restarts=3,
        seed=19,
    )
    validate_assignment(first.labels, 8, 4)
    assert np.array_equal(first.labels, second.labels)
    assert first.objective == second.objective
    assert np.all(first.matrix.sum(axis=1) == 1)
    assert np.all(first.matrix.sum(axis=0) == 4)
    for audit in first.audits:
        history = np.asarray(audit.objective_history)
        assert np.all(history[1:] <= history[:-1] + 1e-6)


def test_task_balanced_natural_profiles():
    q = np.asarray([[2.0, 4.0], [1.0, 3.0]])
    fisher = np.asarray([1.0, 2.0])
    weights = np.asarray([0.25, 1.0])
    profiles, damped, natural = build_natural_modulation_profiles(
        q, fisher, weights, damping=1.0
    )
    assert np.allclose(damped, [2.0, 3.0])
    assert np.allclose(natural, [[1.0, 4.0 / 3.0], [0.5, 1.0]])
    assert np.allclose(profiles[:, 0], np.sqrt(0.25) * natural[0])


def test_nonempty_solver_is_deterministic_and_allows_unequal_sizes():
    rng = np.random.default_rng(17)
    profiles = rng.normal(size=(32, 9))
    weights = rng.uniform(0.2, 1.7, size=32)
    first = solve_nonempty_tf_assignment(
        profiles, weights, num_experts=8, restarts=3, seed=29
    )
    second = solve_nonempty_tf_assignment(
        profiles, weights, num_experts=8, restarts=3, seed=29
    )
    assert np.array_equal(first.labels, second.labels)
    assert first.objective == second.objective
    assert np.all(first.matrix.sum(axis=1) == 1)
    assert np.all(first.matrix.sum(axis=0) >= 1)
    for audit in first.audits:
        history = np.asarray(audit.objective_history)
        assert np.all(history[1:] <= history[:-1] + 1e-6)
