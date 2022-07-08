"""Markov chain simulation."""

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Tuple

import numpy as np
from scipy import stats


class MarkovChain(ABC):
    """Markov chain base class."""

    def __init__(self, matrix: np.ndarray) -> None:
        """Base initialiser."""
        self.matrix = matrix
        self.num_states = len(self.matrix)
        self.state_space = set(range(1, self.num_states + 1))

    @abstractmethod
    def simulate(self) -> None:
        """Simulate the chain."""
        raise NotImplementedError()


class DiscreteTimeMarkovChain(MarkovChain):
    """Discrete-time Markov chain."""

    def __init__(
        self,
        x_zero: int,
        num_steps: int,
        matrix: np.ndarray,
    ) -> None:
        """Initialise the object.

        :param x_zero: state at timestep zero.
        :param num_steps: number of steps to run the chain for.
        :param matrix: transition matrix.
        """
        super().__init__(matrix=matrix)
        self.x_zero = x_zero
        self.num_steps = num_steps
        self.num_visits = np.zeros(self.num_states, dtype=int)

    def simulate(self) -> None:
        """Simulate the Markov chain."""
        x_curr = self.x_zero
        self.num_visits[x_curr - 1] += 1

        for _ in range(self.num_steps - 1):
            row = self.matrix[x_curr - 1, :]
            discrete_dist = stats.rv_discrete(values=(list(self.state_space), row))
            x_curr = discrete_dist.rvs(size=1)[0]
            self.num_visits[x_curr - 1] += 1


def run_discrete_mc(
    matrix: np.ndarray,
    x_zero: int,
    num_steps: int,
    num_reps: int,
) -> np.ndarray:
    """Run the discrete-time Markov chain.

    :param matrix: Transition matrix.
    :param x_zero: state at timestep zero.
    :param num_steps: number of timesteps to run the chain for.
    :param num_reps: number of repetitions to perform.
    :return: estimate of the equilibrium distribution.
    """
    markov_chain = DiscreteTimeMarkovChain(
        matrix=matrix, x_zero=x_zero, num_steps=num_steps
    )

    for _ in range(num_reps):
        markov_chain.simulate()

    return markov_chain.num_visits / np.sum(markov_chain.num_visits)


@dataclass
class ContinuousTimeMarkovChain(MarkovChain):
    """Continuous-time Markov chain."""

    def __init__(
        self,
        x_zero: int,
        max_time: int,
        num_reps: int,
        matrix: np.ndarray,
    ) -> None:
        """Initialise the object.

        :param x_zero: state at timestep zero.
        :param max_time: maximum amount of time to run the chain for.
        :param num_reps: number of repetitions to perform.
        :param matrix: Q matrix.
        """
        super().__init__(matrix=matrix)
        self.x_zero = x_zero
        self.max_time = max_time
        self.num_reps = num_reps
        self.durations = np.zeros(self.num_states, dtype=float)

    @property
    def jump_matrix(self) -> np.ndarray:
        """Jump matrix."""
        matrix_diag = np.resize(
            np.diag(self.matrix), (self.num_states, self.num_states)
        ).T

        id_matrix = np.eye(self.num_states, dtype=bool)
        jump_matrix = np.where(~id_matrix, self.matrix / (-matrix_diag + 1e-10), 0)

        np.fill_diagonal(jump_matrix, (np.diag(self.matrix) == 0).astype(int))
        return jump_matrix

    def simulate(self) -> None:
        """Simulate the Markov chain."""
        x_curr = self.x_zero
        timer = 0

        while timer < self.max_time:
            row_curr = self.matrix[x_curr - 1, :]
            possible_states = np.where(row_curr > 0)[0] + 1
            rates = row_curr[possible_states - 1]
            samples = np.random.exponential(1 / rates)
            time = np.min(samples)

            self.durations[x_curr - 1] += time
            timer += time

            x_curr = possible_states[np.argmin(samples)]


def run_cts_mc(
    matrix: np.ndarray,
    x_zero: int,
    max_time: int,
    num_reps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run continuous-time Markov chain.

    :param matrix: Transition matrix.
    :param x_zero: state at timestep zero.
    :param max_time: maximum amount of time to run the chain for.
    :param num_reps: number of repetitions to perform.
    :return: estimate of the equilibrium distribution.
    """
    markov_chain = ContinuousTimeMarkovChain(
        matrix=matrix,
        x_zero=x_zero,
        max_time=max_time,
        num_reps=num_reps,
    )

    for _ in range(num_reps):
        markov_chain.simulate()

    eq_dist = markov_chain.durations / np.sum(markov_chain.durations)
    pi_q = eq_dist @ matrix

    return eq_dist, pi_q
