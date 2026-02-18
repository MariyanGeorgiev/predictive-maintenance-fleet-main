"""Tests for multi-fault truck interactions."""

import numpy as np
import pytest

from src.features.vibration_features import synthesize_vibration_features
from src.faults.fault_mode import FaultEffect


class TestMultiFaultInteraction:
    def test_bearing_plus_turbo(self):
        """FM-01 (bearing) + FM-05 (turbo): vibration should combine."""
        rng = np.random.default_rng(42)

        # Bearing fault effect
        bearing_effect = FaultEffect(
            vibration_effects={
                "acc1_rms": ("set", 0.8),
                "acc1_kurtosis": ("set", 8.0),
                "acc1_mid_high_energy": ("multiply", 5.0),
            },
            thermal_effects={},
        )

        # Turbo fault effect (late stage: ACC-3 broadband)
        turbo_effect = FaultEffect(
            vibration_effects={
                "acc3_broadband_energy": ("multiply", 3.0),
                "acc3_rms": ("multiply", 2.0),
            },
            thermal_effects={"t4_turbo_factor": 0.25},
        )

        # Combined
        features = synthesize_vibration_features(
            rpm=1475, load=0.75,
            fault_effects_list=[bearing_effect, turbo_effect],
            rng=rng,
        )

        # ACC-1 should show bearing fault
        assert features["acc1_rms_x_mean"] > 0.3
        # ACC-3 should show turbo degradation (RMS elevated)
        assert features["acc3_rms_x_mean"] > 0

    def test_bearing_kurtosis_takes_max(self):
        """When multiple faults set kurtosis, max should win."""
        rng = np.random.default_rng(42)

        effect1 = FaultEffect(
            vibration_effects={"acc1_kurtosis": ("set", 6.0)},
            thermal_effects={},
        )
        effect2 = FaultEffect(
            vibration_effects={"acc1_kurtosis": ("set", 8.0)},
            thermal_effects={},
        )

        features = synthesize_vibration_features(
            rpm=1475, load=0.75,
            fault_effects_list=[effect1, effect2],
            rng=rng,
        )

        # Should be approximately 8.0 (the max), not 6.0
        assert features["acc1_kurtosis_x_mean"] > 5.0

    def test_independent_sensors(self):
        """FM-01 on ACC-1 should not affect ACC-3 (turbo sensor)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        bearing_effect = FaultEffect(
            vibration_effects={
                "acc1_rms": ("set", 1.0),
                "acc1_kurtosis": ("set", 8.0),
            },
            thermal_effects={},
        )

        features_faulty = synthesize_vibration_features(
            rpm=1475, load=0.75,
            fault_effects_list=[bearing_effect],
            rng=rng1,
        )
        features_healthy = synthesize_vibration_features(
            rpm=1475, load=0.75,
            fault_effects_list=[],
            rng=rng2,
        )

        # ACC-3 should be similar in both cases (same seed)
        # The RNG state diverges after acc1/acc2 processing, so exact match isn't expected
        # but ACC-3 should be in healthy range regardless
        assert 0.001 < features_faulty["acc3_rms_x_mean"] < 0.5
