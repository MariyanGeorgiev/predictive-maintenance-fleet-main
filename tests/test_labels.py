"""Tests for ground truth label computation."""

import pytest

from src.faults.degradation_model import DegradationModel
from src.faults.fm01_bearing import BearingWearFault
from src.labels.ground_truth import compute_label


class TestGroundTruth:
    @pytest.fixture
    def bearing_fault(self):
        deg = DegradationModel(0.01, 0.0002, 0.10, 3000, seed=42)
        return BearingWearFault(
            onset_hours=100.0,
            degradation=deg,
            total_life_hours=3000.0,
            affected_sensor="acc1",
        )

    def test_healthy_label(self):
        """No faults should produce HEALTHY label."""
        label = compute_label(100.0, [])
        assert label.fault_mode == "HEALTHY"
        assert label.fault_severity == "HEALTHY"
        assert label.path_a_label == "NORMAL"

    def test_before_onset(self, bearing_fault):
        """Before fault onset, label should still reflect healthy for that fault."""
        label = compute_label(50.0, [bearing_fault])
        assert label.fault_mode == "HEALTHY"
        assert label.path_a_label == "NORMAL"

    def test_stage2_label(self, bearing_fault):
        """Stage 2 should be NORMAL."""
        # onset=100, total_life=3000
        # Stage 2 starts at 60% of life = 1800h after onset = t=1900
        t = 100.0 + 3000.0 * 0.65  # middle of stage 2
        label = compute_label(t, [bearing_fault])
        assert label.fault_severity == "STAGE_2"
        assert label.path_a_label == "NORMAL"

    def test_stage3_imminent(self, bearing_fault):
        """Early Stage 3 should be IMMINENT."""
        t = 100.0 + 3000.0 * 0.80  # early stage 3
        label = compute_label(t, [bearing_fault])
        assert label.fault_severity == "STAGE_3"
        assert label.path_a_label == "IMMINENT"

    def test_stage4_critical(self, bearing_fault):
        """Stage 4 should be CRITICAL."""
        t = 100.0 + 3000.0 * 0.97  # stage 4
        label = compute_label(t, [bearing_fault])
        assert label.fault_severity == "STAGE_4"
        assert label.path_a_label == "CRITICAL"

    def test_rul_decreases(self, bearing_fault):
        """RUL should decrease over time."""
        t1 = 100.0 + 3000.0 * 0.5
        t2 = 100.0 + 3000.0 * 0.8
        label1 = compute_label(t1, [bearing_fault])
        label2 = compute_label(t2, [bearing_fault])
        assert label2.rul_hours < label1.rul_hours

    def test_no_label_leakage(self, bearing_fault):
        """Labels must depend only on fault internal state, not on features.

        Verify that compute_label uses fault.current_stage/current_rul
        (deterministic from onset_hours + degradation model), not any
        generated feature values.
        """
        import inspect
        source = inspect.getsource(compute_label)
        # Label function should reference fault state methods, not feature dicts
        assert "current_stage" in source or "current_severity" in source
        assert "features" not in source
        assert "vibration" not in source
        assert "thermal" not in source
