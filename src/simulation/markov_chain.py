"""4-state operating mode Markov chain simulator (spec §10.4)."""

import numpy as np

from src.config.constants import OPERATING_MODES, TRANSITION_MATRIX, WINDOWS_PER_DAY


class MarkovChainSimulator:
    """Simulates operating mode transitions at 60-second intervals."""

    def __init__(self, transition_matrix: np.ndarray = TRANSITION_MATRIX):
        self.P = transition_matrix
        self.n_states = len(OPERATING_MODES)

        # Validate transition matrix
        assert self.P.shape == (self.n_states, self.n_states), (
            f"Transition matrix shape {self.P.shape} != ({self.n_states}, {self.n_states})"
        )
        assert np.allclose(self.P.sum(axis=1), 1.0), (
            f"Transition matrix rows must sum to 1.0, got {self.P.sum(axis=1)}"
        )
        assert np.all(self.P >= 0), "Transition matrix must have non-negative entries"

    def simulate_day(self, rng: np.random.Generator, initial_state: int = 0) -> np.ndarray:
        """Generate operating mode indices for one day (1440 windows).

        Args:
            rng: Random number generator.
            initial_state: Starting state index (0=idle, 1=city, 2=cruise, 3=heavy).

        Returns:
            Array of shape (1440,) with integer state indices.
        """
        states = np.empty(WINDOWS_PER_DAY, dtype=np.int32)
        state = initial_state
        for i in range(WINDOWS_PER_DAY):
            states[i] = state
            state = rng.choice(self.n_states, p=self.P[state])
        return states

    def stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution of the Markov chain."""
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        return pi
