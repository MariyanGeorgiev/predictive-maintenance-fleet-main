"""Integration test: generate 1 truck × 1 day and validate end-to-end."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.config.constants import WINDOWS_PER_DAY
from src.config.schema import create_engine_profile
from src.faults.degradation_model import DegradationModel
from src.faults.fm01_bearing import BearingWearFault
from src.features.feature_vector import FEATURE_COLUMNS
from src.generator.truck_day_generator import TruckDayGenerator
from src.labels.ground_truth import GroundTruthLabel
from src.storage.parquet_writer import ParquetWriter
from src.storage.thermal_state import default_idle_temps


class TestIntegration:
    def test_healthy_truck_one_day(self):
        """Generate 1 healthy truck × 1 day. Validate output shape and ranges."""
        rng = np.random.default_rng(42)
        profile = create_engine_profile("modern", rng)

        generator = TruckDayGenerator()
        features, labels, final_temps = generator.generate(
            profile=profile,
            engine_type="modern",
            day_index=0,
            faults=[],
            initial_temps=default_idle_temps("modern"),
            seed=42,
        )

        # Shape checks
        assert len(features) == WINDOWS_PER_DAY
        assert len(labels) == WINDOWS_PER_DAY

        # All labels should be healthy
        for label in labels:
            assert label.fault_mode == "HEALTHY"
            assert label.path_a_label == "NORMAL"

        # Feature keys should match expected columns
        for feat in features[:3]:
            for col in ["rpm_est", "load_proxy", "t1_mean", "acc1_rms_x_mean"]:
                assert col in feat, f"Missing feature: {col}"

        # No NaN values
        for i, feat in enumerate(features):
            for key, val in feat.items():
                assert not np.isnan(val), f"NaN at window {i}, feature {key}"

        # Final temps should be dict with all 6 sensors
        assert len(final_temps) == 6
        for sensor in ["t1", "t2", "t3", "t4", "t5", "t6"]:
            assert sensor in final_temps

    def test_faulty_truck_one_day(self):
        """Generate 1 faulty truck (FM-01 bearing) × 1 day."""
        rng = np.random.default_rng(42)
        profile = create_engine_profile("modern", rng)

        # Create a bearing fault that starts early and is in Stage 3 by day 0
        deg = DegradationModel(0.01, 0.0002, 0.10, 500, seed=42)
        fault = BearingWearFault(
            onset_hours=-400.0,  # Started 400h ago, should be in Stage 3
            degradation=deg,
            total_life_hours=500.0,
            affected_sensor="acc1",
        )

        generator = TruckDayGenerator()
        features, labels, _ = generator.generate(
            profile=profile,
            engine_type="modern",
            day_index=0,
            faults=[fault],
            initial_temps=default_idle_temps("modern"),
            seed=42,
        )

        assert len(features) == WINDOWS_PER_DAY

        # Some labels should show fault
        fault_labels = [l for l in labels if l.fault_mode != "HEALTHY"]
        assert len(fault_labels) > 0

    def test_write_and_validate_parquet(self):
        """Generate + write Parquet, then read back and validate."""
        rng = np.random.default_rng(42)
        profile = create_engine_profile("modern", rng)

        generator = TruckDayGenerator()
        features, labels, _ = generator.generate(
            profile=profile,
            engine_type="modern",
            day_index=0,
            faults=[],
            initial_temps=default_idle_temps("modern"),
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(Path(tmpdir))
            path = writer.write_truck_day(
                truck_id=1, engine_type="modern", day_index=0,
                features=features, labels=labels,
            )

            # Read back
            df = pq.read_table(path).to_pandas()
            assert len(df) == WINDOWS_PER_DAY
            assert df["truck_id"].iloc[0] == 1
            assert df["engine_type"].iloc[0] == "modern"

            # Check that feature columns have non-zero values
            assert df["rpm_est"].mean() > 0
            assert df["t1_mean"].mean() > 0

    def test_reproducibility_same_seed(self):
        """Same seed must produce identical output (critical for ML benchmarking)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        profile1 = create_engine_profile("modern", rng1)
        profile2 = create_engine_profile("modern", rng2)

        gen1 = TruckDayGenerator()
        gen2 = TruckDayGenerator()

        features1, labels1, temps1 = gen1.generate(
            profile1, "modern", 0, [], default_idle_temps("modern"), seed=42,
        )
        features2, labels2, temps2 = gen2.generate(
            profile2, "modern", 0, [], default_idle_temps("modern"), seed=42,
        )

        # Every feature value must be bit-identical
        for i in range(len(features1)):
            for key in features1[i]:
                assert features1[i][key] == features2[i][key], (
                    f"Non-deterministic at window {i}, feature {key}: "
                    f"{features1[i][key]} != {features2[i][key]}"
                )

        # Labels must match
        for i in range(len(labels1)):
            assert labels1[i].fault_mode == labels2[i].fault_mode
            assert labels1[i].rul_hours == labels2[i].rul_hours

        # Final temps must match
        for sensor in temps1:
            assert temps1[sensor] == temps2[sensor]

    def test_reproducibility_with_faults(self):
        """Reproducibility must hold for faulty trucks too (including FM-07 leak events)."""
        from src.faults.fm07_egr import EGRCoolerFault

        rng = np.random.default_rng(99)
        profile = create_engine_profile("modern", rng)

        deg = DegradationModel(0.01, 0.0003, 0.12, 800, seed=77)
        fault = EGRCoolerFault(
            onset_hours=-600.0, degradation=deg, total_life_hours=800.0,
            delta_t5_max=40.0, leak_t1_spike=20.0, leak_t5_spike=55.0, seed=77,
        )

        gen1 = TruckDayGenerator()
        gen2 = TruckDayGenerator()

        # Same fault object used twice — must be stateless
        features1, _, _ = gen1.generate(
            profile, "modern", 0, [fault], default_idle_temps("modern"), seed=55,
        )
        features2, _, _ = gen2.generate(
            profile, "modern", 0, [fault], default_idle_temps("modern"), seed=55,
        )

        for i in range(len(features1)):
            for key in features1[i]:
                assert features1[i][key] == features2[i][key], (
                    f"FM-07 non-deterministic at window {i}, feature {key}"
                )
