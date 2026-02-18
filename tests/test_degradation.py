"""Tests for degradation model and fault modes."""

import numpy as np
import pytest

from src.faults.degradation_model import DegradationModel


class TestDegradationModel:
    @pytest.fixture
    def model(self):
        return DegradationModel(
            severity_0=0.01, lambda_rate=0.0002, sigma=0.10,
            total_hours=3000, seed=42,
        )

    def test_zero_time(self, model):
        assert model.severity_at(0.0) == 0.0

    def test_severity_increases(self, model):
        """Severity should generally increase over time (average over samples)."""
        # With Wiener process noise, individual paths can decrease,
        # so test that the trend is upward by sampling many seeds
        late_higher = 0
        for seed in range(50):
            m = DegradationModel(0.01, 0.0002, 0.10, 3000, seed=seed)
            s_early = m.severity_at(500)
            s_late = m.severity_at(2500)
            if s_late > s_early:
                late_higher += 1
        # At least 60% of paths should show increasing severity
        assert late_higher >= 30

    def test_severity_clamped(self, model):
        """Severity should be clamped to [0, 1]."""
        s = model.severity_at(model.total_hours)
        assert 0.0 <= s <= 1.0

    def test_stage_transitions(self, model):
        """Stages should progress: healthy -> stage2 -> stage3 -> stage4."""
        total = 3000.0
        assert model.stage_at(0, total) == "healthy"
        assert model.stage_at(total * 0.3, total) == "healthy"  # 30% < 60%
        assert model.stage_at(total * 0.65, total) == "stage2"  # 60-75%
        assert model.stage_at(total * 0.85, total) == "stage3"  # 75-95%
        assert model.stage_at(total * 0.97, total) == "stage4"  # 95-100%

    def test_different_seeds(self):
        """Different seeds should produce different paths."""
        m1 = DegradationModel(0.01, 0.0002, 0.10, 3000, seed=1)
        m2 = DegradationModel(0.01, 0.0002, 0.10, 3000, seed=2)
        # At same time, severities should differ
        s1 = m1.severity_at(1500)
        s2 = m2.severity_at(1500)
        assert s1 != s2

    def test_deterministic(self):
        """Same parameters produce same results."""
        m1 = DegradationModel(0.01, 0.0002, 0.10, 3000, seed=42)
        m2 = DegradationModel(0.01, 0.0002, 0.10, 3000, seed=42)
        assert m1.severity_at(1500) == m2.severity_at(1500)
