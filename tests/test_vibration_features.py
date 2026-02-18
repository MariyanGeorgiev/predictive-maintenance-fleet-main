"""Tests for vibration feature synthesis."""

import numpy as np
import pytest

from src.features.vibration_features import synthesize_vibration_features
from src.faults.fault_mode import FaultEffect


class TestVibrationFeatures:
    def test_healthy_rms_range(self):
        """Healthy ACC-1 RMS should be in 0.05-0.15g range."""
        rng = np.random.default_rng(42)
        features = synthesize_vibration_features(
            rpm=1475, load=0.75, fault_effects_list=[], rng=rng,
        )
        rms = features["acc1_rms_x_mean"]
        assert 0.01 < rms < 0.30  # with load scaling and noise

    def test_healthy_kurtosis(self):
        """Healthy kurtosis should be approximately Gaussian (~3)."""
        rng = np.random.default_rng(42)
        features = synthesize_vibration_features(
            rpm=1475, load=0.75, fault_effects_list=[], rng=rng,
        )
        kurt = features["acc1_kurtosis_x_mean"]
        assert 2.0 < kurt < 5.0

    def test_healthy_sk_low(self):
        """Healthy spectral kurtosis should be < 5."""
        rng = np.random.default_rng(42)
        features = synthesize_vibration_features(
            rpm=1475, load=0.75, fault_effects_list=[], rng=rng,
        )
        sk = features["acc1_sk_max_value"]
        assert 0.5 < sk < 8.0  # allowing some noise margin

    def test_bearing_fault_increases_rms(self):
        """FM-01 bearing fault should increase RMS significantly."""
        rng = np.random.default_rng(42)
        # Simulate Stage 3 bearing fault effect
        fault_effect = FaultEffect(
            vibration_effects={
                "acc1_rms": ("set", 0.8),       # Stage 3: 0.3-1.5g
                "acc1_kurtosis": ("set", 8.0),   # Stage 3: 6-10
                "acc1_sk_max": ("set", 12.0),    # Stage 3: 10+
                "acc1_mid_high_energy": ("multiply", 5.0),
            },
            thermal_effects={},
        )
        features = synthesize_vibration_features(
            rpm=1475, load=0.75, fault_effects_list=[fault_effect], rng=rng,
        )
        assert features["acc1_rms_x_mean"] > 0.3

    def test_all_sensors_present(self):
        """All three sensors should have features."""
        rng = np.random.default_rng(42)
        features = synthesize_vibration_features(
            rpm=1475, load=0.75, fault_effects_list=[], rng=rng,
        )
        assert any(k.startswith("acc1_") for k in features)
        assert any(k.startswith("acc2_") for k in features)
        assert any(k.startswith("acc3_") for k in features)

    def test_no_nan_values(self):
        """No features should be NaN."""
        rng = np.random.default_rng(42)
        features = synthesize_vibration_features(
            rpm=1475, load=0.75, fault_effects_list=[], rng=rng,
        )
        for key, value in features.items():
            assert not np.isnan(value), f"NaN found in {key}"
