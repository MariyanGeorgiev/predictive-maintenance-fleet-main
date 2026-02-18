"""Exponential degradation with Wiener process (spec §10.2.3).

fault_severity(t) = base_severity(t) + σ * noise(t)

Where base_severity is a monotonically increasing exponential from 0 to 1 over
the total life, and the Wiener noise adds stochastic variation (bounded to keep
severity in [0, 1] and overall trend monotonic).
"""

import numpy as np


class DegradationModel:
    """Models fault severity progression over time."""

    def __init__(
        self,
        severity_0: float,
        lambda_rate: float,
        sigma: float,
        total_hours: int,
        seed: int,
    ):
        self.severity_0 = severity_0
        self.lambda_rate = lambda_rate
        self.sigma = sigma
        self.total_hours = total_hours

        # Precompute noise path at hourly resolution
        # Use bounded random walk (mean-reverting) so it doesn't dominate the trend
        rng = np.random.default_rng(seed)
        n = total_hours + 1
        noise = np.zeros(n)
        for i in range(1, n):
            # Mean-reverting: pulls back toward 0
            noise[i] = 0.95 * noise[i - 1] + rng.normal(0, 1.0)
        # Normalize to [-1, 1] range
        max_abs = max(abs(noise.min()), abs(noise.max()), 1e-8)
        self._noise = noise / max_abs

    def severity_at(self, t_hours: float) -> float:
        """Compute fault severity at time t (hours since onset).

        Base severity follows exponential curve from ~0 to 1 over total_hours.
        Wiener noise adds ±sigma variation around the base.

        Returns value in [0, 1] range (clamped).
        """
        if t_hours <= 0:
            return 0.0
        if t_hours >= self.total_hours:
            return 1.0

        # Base severity: exponential growth from 0 to 1
        # Use logistic-like curve: (exp(k*t/T) - 1) / (exp(k) - 1)
        k = 5.0  # steepness (controls how late the acceleration happens)
        t_frac = t_hours / self.total_hours
        base = (np.exp(k * t_frac) - 1.0) / (np.exp(k) - 1.0)

        # Interpolate noise
        idx = int(t_hours)
        frac = t_hours - idx
        if idx >= len(self._noise) - 1:
            noise_val = self._noise[-1]
        else:
            noise_val = self._noise[idx] + frac * (self._noise[idx + 1] - self._noise[idx])

        # Apply noise as fraction of base (bounded)
        raw = base + self.sigma * noise_val * base * 0.5
        return float(np.clip(raw, 0.0, 1.0))

    def stage_at(self, t_hours: float, total_life_hours: float) -> str:
        """Map time to degradation stage based on life percentage.

        Args:
            t_hours: Hours since fault onset.
            total_life_hours: Total hours from onset to end of Stage 4.

        Returns:
            One of "healthy", "stage2", "stage3", "stage4".
        """
        if t_hours <= 0:
            return "healthy"

        life_pct = t_hours / total_life_hours if total_life_hours > 0 else 1.0

        if life_pct < 0.60:
            return "healthy"  # Stage 1
        elif life_pct < 0.75:
            return "stage2"
        elif life_pct < 0.95:
            return "stage3"
        else:
            return "stage4"
