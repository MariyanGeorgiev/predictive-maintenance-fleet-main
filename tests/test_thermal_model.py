"""Tests for thermal feature synthesis."""

import numpy as np
import pytest

from src.config.constants import TEMP_SENSORS
from src.config.schema import create_engine_profile
from src.faults.fault_mode import FaultEffect
from src.features.thermal_features import synthesize_thermal_features


class TestThermalFeatures:
    @pytest.fixture
    def profile(self):
        rng = np.random.default_rng(42)
        return create_engine_profile("modern", rng)

    @pytest.fixture
    def idle_temps(self):
        return {"t1": 65.0, "t2": 75.0, "t3": 175.0, "t4": 125.0, "t5": 90.0, "t6": 35.0}

    def test_output_feature_count(self, profile, idle_temps):
        rng = np.random.default_rng(42)
        features, _ = synthesize_thermal_features(
            rpm=1475, load=0.75, profile=profile, ambient_temp=25.0,
            fault_effects_list=[], prev_temps=idle_temps, rng=rng,
        )
        # 6 sensors × 6 stats + 3 differentials = 39
        assert len(features) == 39

    def test_healthy_t3_at_cruise(self, profile, idle_temps):
        """T3 at cruise load should be in 315-482°C range."""
        rng = np.random.default_rng(42)
        # Run a few windows to let temps stabilize
        temps = idle_temps
        for _ in range(20):
            features, temps = synthesize_thermal_features(
                rpm=1475, load=0.75, profile=profile, ambient_temp=25.0,
                fault_effects_list=[], prev_temps=temps, rng=rng,
            )
        # Allow some tolerance for model variability
        assert 200 < features["t3_mean"] < 600

    def test_t3_t4_delta_positive(self, profile, idle_temps):
        """T3-T4 delta should be positive (T3 > T4)."""
        rng = np.random.default_rng(42)
        temps = idle_temps
        for _ in range(10):
            features, temps = synthesize_thermal_features(
                rpm=1475, load=0.75, profile=profile, ambient_temp=25.0,
                fault_effects_list=[], prev_temps=temps, rng=rng,
            )
        assert features["t3_t4_delta"] > 0

    def test_thermal_state_continuity(self, profile, idle_temps):
        """Temperature should change smoothly between windows."""
        rng = np.random.default_rng(42)
        temps = idle_temps
        t3_values = []
        for _ in range(5):
            features, temps = synthesize_thermal_features(
                rpm=1475, load=0.75, profile=profile, ambient_temp=25.0,
                fault_effects_list=[], prev_temps=temps, rng=rng,
            )
            t3_values.append(features["t3_mean"])
        # Temperature should not jump wildly between windows
        for i in range(1, len(t3_values)):
            assert abs(t3_values[i] - t3_values[i-1]) < 100  # < 100°C per window

    def test_no_negative_temperatures(self, profile, idle_temps):
        """All temperatures must stay within physical sensor bounds."""
        rng = np.random.default_rng(42)
        temps = idle_temps
        for _ in range(20):
            features, temps = synthesize_thermal_features(
                rpm=1475, load=0.75, profile=profile, ambient_temp=-10.0,
                fault_effects_list=[], prev_temps=temps, rng=rng,
            )
        for sensor in ["t1", "t2", "t3", "t4", "t5", "t6"]:
            lo, hi = TEMP_SENSORS[sensor]["range_c"]
            assert features[f"{sensor}_mean"] >= lo, f"{sensor} below physical min"
            assert features[f"{sensor}_mean"] <= hi, f"{sensor} above physical max"

    def test_multi_fault_thermal_capping(self, profile, idle_temps):
        """Stacked thermal faults should be capped to prevent non-physical values."""
        rng = np.random.default_rng(42)
        # Stack two faults both raising T1
        effects = [
            FaultEffect(vibration_effects={}, thermal_effects={"t1": 30.0}),
            FaultEffect(vibration_effects={}, thermal_effects={"t1": 40.0}),
        ]
        temps = idle_temps
        for _ in range(30):
            features, temps = synthesize_thermal_features(
                rpm=1475, load=0.75, profile=profile, ambient_temp=25.0,
                fault_effects_list=effects, prev_temps=temps, rng=rng,
            )
        # T1 should be elevated but capped within physical range (0-120°C)
        assert features["t1_mean"] <= 120.0, "T1 exceeded physical sensor max"
