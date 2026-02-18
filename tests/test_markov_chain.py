"""Tests for Markov chain operating mode simulator."""

import numpy as np
import pytest

from src.simulation.markov_chain import MarkovChainSimulator


class TestMarkovChain:
    @pytest.fixture
    def sim(self):
        return MarkovChainSimulator()

    def test_row_normalization(self, sim):
        """Transition matrix rows must sum to 1.0."""
        np.testing.assert_allclose(sim.P.sum(axis=1), 1.0)
        assert np.all(sim.P >= 0)

    def test_invalid_matrix_rejected(self):
        """Constructor should reject invalid transition matrices."""
        bad_matrix = np.array([
            [0.5, 0.3, 0.1, 0.05],  # sums to 0.95
            [0.1, 0.6, 0.2, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])
        with pytest.raises(AssertionError):
            MarkovChainSimulator(bad_matrix)

    def test_output_shape(self, sim, rng):
        states = sim.simulate_day(rng)
        assert states.shape == (1440,)

    def test_valid_states(self, sim, rng):
        states = sim.simulate_day(rng)
        assert np.all((states >= 0) & (states < 4))

    def test_stationary_distribution(self, sim):
        """Stationary distribution should roughly match:
        Idle ~7%, City ~17%, Cruise ~66%, Heavy ~10%."""
        pi = sim.stationary_distribution()
        assert pi.shape == (4,)
        assert abs(pi.sum() - 1.0) < 1e-6
        # Cruise should dominate
        assert pi[2] > 0.45  # cruise dominates (~49%)
        # Idle should be smallest or close to heavy
        assert pi[0] < 0.15  # idle < 15%

    def test_long_run_matches_stationary(self, sim):
        """Long simulation should converge to stationary distribution."""
        rng = np.random.default_rng(123)
        # Run for many days
        all_states = np.concatenate([sim.simulate_day(rng) for _ in range(100)])
        empirical = np.bincount(all_states, minlength=4) / len(all_states)
        pi = sim.stationary_distribution()
        np.testing.assert_allclose(empirical, pi, atol=0.03)

    def test_deterministic_with_seed(self, sim):
        """Same seed produces same output."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        states1 = sim.simulate_day(rng1)
        states2 = sim.simulate_day(rng2)
        np.testing.assert_array_equal(states1, states2)
