"""Conditioning features: RPM estimate and load proxy (spec §9.3)."""

import numpy as np


def compute_conditioning_features(
    rpm: float,
    load: float,
    t3_mean: float,
    engine_type: str,
    rng: np.random.Generator,
) -> dict:
    """Compute 2 conditioning features.

    Args:
        rpm: True RPM (will add estimation noise).
        load: True normalized load.
        t3_mean: Mean T3 temperature for this window.
        engine_type: "modern" or "older".
        rng: Random number generator.

    Returns:
        Dict with 'rpm_est' and 'load_proxy'.
    """
    # RPM estimate from vibration dominant frequency (E11: ~2-5% relative error)
    rpm_noise = rng.normal(0, rpm * 0.03)
    rpm_est = rpm + rpm_noise

    # Load proxy = (mean_T3 - baseline_T3) / (cruise_T3 - baseline_T3)
    # Baselines from spec
    if engine_type == "modern":
        baseline_t3 = 175.0  # midpoint of idle range (150-200)
        cruise_t3 = 400.0    # midpoint of cruise range (315-482)
    else:
        baseline_t3 = 185.0  # midpoint of idle range (160-210)
        cruise_t3 = 400.0    # midpoint of cruise range (300-500)

    denominator = cruise_t3 - baseline_t3
    if denominator > 0:
        load_proxy = (t3_mean - baseline_t3) / denominator
    else:
        load_proxy = load  # fallback

    return {
        "rpm_est": rpm_est,
        "load_proxy": load_proxy,
    }
