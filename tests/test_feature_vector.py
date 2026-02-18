"""Tests for feature vector assembly."""

import numpy as np
import pytest

from src.features.feature_vector import (
    ALL_COLUMNS,
    CONDITIONING_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    N_FEATURES,
    THERMAL_COLUMNS,
    VIBRATION_COLUMNS,
    assemble_feature_dict,
)


class TestFeatureVector:
    def test_total_column_count(self):
        """Should have 229 total columns (4 meta + 221 features + 4 labels)."""
        assert len(ALL_COLUMNS) == len(METADATA_COLUMNS) + len(FEATURE_COLUMNS) + len(LABEL_COLUMNS)
        assert len(ALL_COLUMNS) == 229

    def test_feature_count_contract(self):
        """Feature count must be exactly 221 (2 cond + 180 vib + 39 thermal)."""
        assert N_FEATURES == 221

    def test_metadata_columns(self):
        assert METADATA_COLUMNS == ["timestamp", "truck_id", "engine_type", "day_index"]

    def test_label_columns(self):
        assert LABEL_COLUMNS == ["fault_mode", "fault_severity", "rul_hours", "path_a_label"]

    def test_conditioning_count(self):
        assert len(CONDITIONING_COLUMNS) == 2

    def test_thermal_count(self):
        """39 thermal features: 6 sensors × 6 stats + 3 differentials."""
        assert len(THERMAL_COLUMNS) == 39

    def test_no_duplicate_columns(self):
        assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS))

    def test_assemble_fills_missing(self):
        """Missing features should be filled with 0.0."""
        result = assemble_feature_dict({}, {}, {})
        assert len(result) == len(FEATURE_COLUMNS)
        for val in result.values():
            assert val == 0.0

    def test_assemble_preserves_values(self):
        cond = {"rpm_est": 1475.0, "load_proxy": 0.8}
        thermal = {"t1_mean": 85.0}
        result = assemble_feature_dict(cond, {}, thermal)
        assert result["rpm_est"] == 1475.0
        assert result["load_proxy"] == 0.8
        assert result["t1_mean"] == 85.0
