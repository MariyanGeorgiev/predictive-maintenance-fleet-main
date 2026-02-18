"""Pydantic models for all configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config.constants import (
    BEARING_GEOMETRY_MODERN,
    BEARING_GEOMETRY_OLDER,
    THERMAL_BASELINES,
    TURBO_DELTA_BASELINE,
)


@dataclass(frozen=True)
class BearingGeometry:
    """Bearing geometry parameters for fault frequency computation."""

    n_balls: int
    ball_dia_mm: float
    pitch_dia_mm: float
    contact_angle_deg: float = 0.0

    @classmethod
    def modern_default(cls) -> BearingGeometry:
        return cls(**BEARING_GEOMETRY_MODERN)

    @classmethod
    def older_default(cls) -> BearingGeometry:
        return cls(**BEARING_GEOMETRY_OLDER)


@dataclass(frozen=True)
class ThermalBaseline:
    """Per-sensor thermal baseline parameters (sampled from spec ranges)."""

    idle_temp: float       # T at idle (°C)
    cruise_temp: float     # T at cruise (°C)
    delta_load: float      # ΔT from idle to cruise (°C)
    tau: float             # Thermal time constant (seconds)


@dataclass(frozen=True)
class EngineProfile:
    """Engine profile with all physical parameters."""

    name: str                          # "modern" or "older"
    displacement_range: Tuple[float, float]  # liters
    n_main_bearings: int
    cruise_rpm_range: Tuple[float, float]
    bearing_geometry: BearingGeometry
    thermal_baselines: Dict[str, ThermalBaseline]  # t1..t6
    turbo_delta_baseline: Tuple[float, float]       # T3-T4 at cruise


@dataclass
class TruckConfig:
    """Individual truck configuration."""

    truck_id: int
    engine_type: str          # "modern" or "older"
    profile: EngineProfile
    seed: int                 # RNG seed for reproducibility
    split: str                # "train", "val", or "test"


@dataclass
class FaultAssignment:
    """A fault assigned to a specific truck."""

    fault_id: str             # "FM-01" through "FM-08"
    onset_hours: float        # When the fault starts progressing
    lambda_rate: float        # Base degradation rate
    sigma: float              # Stochasticity parameter
    total_life_hours: float   # Total time from onset to Stage 4 end


@dataclass
class FleetMetadata:
    """Metadata about the generated fleet."""

    total_trucks: int
    modern_count: int
    older_count: int
    train_ids: List[int]
    val_ids: List[int]
    test_ids: List[int]
    seed: int


def sample_thermal_baselines(
    engine_type: str, rng: np.random.Generator
) -> Dict[str, ThermalBaseline]:
    """Sample thermal baseline parameters from spec ranges for an engine type.

    Samples idle_temp and delta_load independently, then derives cruise_temp
    to ensure the delta_load range from the spec is respected.
    """
    baselines = {}
    for sensor, params in THERMAL_BASELINES[engine_type].items():
        idle_temp = rng.uniform(*params["idle"])
        delta_load = rng.uniform(*params["delta_load"])
        cruise_temp = idle_temp + delta_load
        tau = rng.uniform(*params["tau"])
        baselines[sensor] = ThermalBaseline(
            idle_temp=idle_temp,
            cruise_temp=cruise_temp,
            delta_load=delta_load,
            tau=tau,
        )
    return baselines


def create_engine_profile(
    engine_type: str, rng: np.random.Generator
) -> EngineProfile:
    """Create an engine profile with sampled parameters."""
    if engine_type == "modern":
        return EngineProfile(
            name="modern",
            displacement_range=(12.7, 15.0),
            n_main_bearings=7,
            cruise_rpm_range=(1400, 1550),
            bearing_geometry=BearingGeometry.modern_default(),
            thermal_baselines=sample_thermal_baselines("modern", rng),
            turbo_delta_baseline=TURBO_DELTA_BASELINE["modern"],
        )
    else:
        return EngineProfile(
            name="older",
            displacement_range=(10.4, 14.3),
            n_main_bearings=7,
            cruise_rpm_range=(1500, 1700),
            bearing_geometry=BearingGeometry.older_default(),
            thermal_baselines=sample_thermal_baselines("older", rng),
            turbo_delta_baseline=TURBO_DELTA_BASELINE["older"],
        )
