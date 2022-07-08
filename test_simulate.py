"""Testing simulate.py."""

import numpy as np
import pytest

from simulate import run_cts_mc, run_discrete_mc


@pytest.mark.parametrize(
    "matrix, expected_eq_dist",
    [
        (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0.2, 0.2, 0.4, 0.2, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0.5, 0.5, 0],
                ],
                dtype=np.float64,
            ),
            np.array([1 / 4, 0, 3 / 20, 3 / 10, 3 / 10], dtype=np.float64),
        ),
        (
            np.array(
                [
                    [0, 1, 0],
                    [1 / 3, 0, 2 / 3],
                    [0, 1, 0],
                ],
                dtype=np.float64,
            ),
            np.array([1 / 6, 1 / 2, 1 / 3], dtype=np.float64),
        ),
    ],
)
def test_run_discrete_mc(matrix, expected_eq_dist):
    x_zero = 2
    num_steps = 100
    num_reps = 100

    actual_eq_dist = run_discrete_mc(
        matrix=matrix,
        x_zero=x_zero,
        num_steps=num_steps,
        num_reps=num_reps,
    )

    assert np.allclose(actual_eq_dist, expected_eq_dist, atol=0.05)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array(
            [
                [-2, 2, 0],
                [6, -8, 2],
                [0, 3, -3],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_run_cts_mc(matrix):
    x_zero = 3
    max_time = 100
    num_reps = 100

    actual_eq_dist, pi_q = run_cts_mc(
        matrix=matrix,
        x_zero=x_zero,
        max_time=max_time,
        num_reps=num_reps,
    )

    assert np.allclose(pi_q, 0, atol=0.05)
