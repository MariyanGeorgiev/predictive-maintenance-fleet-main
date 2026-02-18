"""RPM and load generation per operating mode (spec §10.4)."""

import numpy as np

from src.config.constants import LOAD_RANGES, OPERATING_MODES, RPM_RANGES


def generate_rpm_load(
    operating_modes: np.ndarray,
    engine_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate RPM and load arrays for a sequence of operating modes.

    RPM is sampled from the mode-specific range with smoothing between transitions.
    Load is sampled from mode-specific range, correlated with RPM.

    Args:
        operating_modes: Array of integer mode indices (1440,).
        engine_type: "modern" or "older".
        rng: Random number generator.

    Returns:
        (rpm_array, load_array) each of shape (1440,).
    """
    n = len(operating_modes)
    rpm = np.empty(n, dtype=np.float64)
    load = np.empty(n, dtype=np.float64)

    for i in range(n):
        mode = OPERATING_MODES[operating_modes[i]]
        rpm_lo, rpm_hi = RPM_RANGES[mode][engine_type]
        load_lo, load_hi = LOAD_RANGES[mode]

        # Sample target RPM and load from truncated normal within range
        rpm_mid = (rpm_lo + rpm_hi) / 2.0
        rpm_std = (rpm_hi - rpm_lo) / 4.0  # ~95% within range
        rpm_target = np.clip(rng.normal(rpm_mid, rpm_std), rpm_lo, rpm_hi)

        load_mid = (load_lo + load_hi) / 2.0
        load_std = (load_hi - load_lo) / 4.0
        load_target = np.clip(rng.normal(load_mid, load_std), load_lo, load_hi)

        rpm[i] = rpm_target
        load[i] = load_target

    # Apply first-order smoothing (tau ~ 5 windows = 5 minutes effective)
    alpha = 1.0 / 5.0
    for i in range(1, n):
        rpm[i] = rpm[i - 1] + alpha * (rpm[i] - rpm[i - 1])
        load[i] = load[i - 1] + alpha * (load[i] - load[i - 1])

    return rpm, load
