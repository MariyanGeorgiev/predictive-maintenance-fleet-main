"""Synthesize 39 thermal features (spec §9.2).

Per sensor (T1-T6): mean, std, max, min, range, slope = 6 features × 6 sensors = 36
Differentials: T3-T4 delta, T1-T5 delta, T3 exceedance duration = 3
Total: 39 features per 60-second window.

Uses first-order lag model for thermal inertia:
    dT/dt = (T_target - T_current) / τ
Simulates 60 one-second steps within each window.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.config.constants import (
    AMBIENT_T_REF,
    T3_EXCEEDANCE_THRESHOLD,
    TEMP_SENSORS,
    THERMAL_NOISE_STD,
)
from src.config.schema import EngineProfile, ThermalBaseline
from src.faults.fault_mode import FaultEffect

# Physical temperature bounds per sensor (from spec)
_TEMP_BOUNDS = {k: v["range_c"] for k, v in TEMP_SENSORS.items()}

# Maximum total thermal offset from stacked faults (prevents non-physical values)
_MAX_THERMAL_OFFSET = {
    "t1": 50.0,   # coolant: max ~50°C above baseline from combined faults
    "t2": 50.0,   # oil
    "t3": 250.0,  # EGT pre-turbo: can swing large due to DPF/injector
    "t4": 200.0,  # EGT post-turbo
    "t5": 100.0,  # EGR outlet
    "t6": 30.0,   # intake manifold
}


TEMP_SENSORS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def _compute_target_temp(
    sensor: str,
    baseline: ThermalBaseline,
    load: float,
    ambient_temp: float,
) -> float:
    """Compute target temperature from thermal model (spec §10.3.1).

    T_target = T_base + ΔT_load × load + ΔT_ambient × (T_ambient - T_ref)
    """
    # Interpolate between idle and cruise based on load
    t_target = baseline.idle_temp + baseline.delta_load * load
    # Ambient correction (small effect, ~0.5°C per °C ambient deviation)
    t_target += 0.5 * (ambient_temp - AMBIENT_T_REF)
    return t_target


def synthesize_thermal_features(
    rpm: float,
    load: float,
    profile: EngineProfile,
    ambient_temp: float,
    fault_effects_list: List[FaultEffect],
    prev_temps: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Synthesize 39 thermal features for one 60-second window.

    Args:
        rpm: Current RPM.
        load: Current normalized load (0-1.2).
        profile: Engine profile with thermal baselines.
        ambient_temp: Current ambient temperature °C.
        fault_effects_list: Active fault effects.
        prev_temps: Previous window's final temperatures (state continuity).
        rng: Random number generator.

    Returns:
        (features_dict, updated_temps) where updated_temps is the
        end-of-window state for next window initialization.
    """
    # Merge thermal effects: additive offsets with dampening cap
    thermal_offsets: Dict[str, float] = {}
    turbo_degradation_factor = 0.0

    for fe in fault_effects_list:
        for key, value in fe.thermal_effects.items():
            if key == "t4_turbo_factor":
                turbo_degradation_factor = max(turbo_degradation_factor, value)
            else:
                thermal_offsets[key] = thermal_offsets.get(key, 0.0) + value

    # Cap stacked fault offsets to prevent non-physical temperatures
    for key in thermal_offsets:
        cap = _MAX_THERMAL_OFFSET.get(key, 100.0)
        thermal_offsets[key] = np.clip(thermal_offsets[key], -cap, cap)

    # Simulate 60 one-second steps for each sensor
    sensor_traces: Dict[str, np.ndarray] = {}

    for sensor in TEMP_SENSORS:
        baseline = profile.thermal_baselines[sensor]
        tau = baseline.tau  # thermal time constant in seconds

        # Compute target temperature
        target = _compute_target_temp(sensor, baseline, load, ambient_temp)

        # Apply fault thermal offsets
        if sensor in thermal_offsets:
            target += thermal_offsets[sensor]

        # Special handling for T4 with turbo degradation (FM-05)
        # T4 = T3 - baseline_delta × (1 - degradation_factor)
        # We handle this after T3 is computed

        # Starting temperature (from previous window or initial)
        current = prev_temps.get(sensor, baseline.idle_temp)

        # Simulate 60 one-second steps with first-order lag
        trace = np.empty(60)
        for s in range(60):
            dt = 1.0  # 1 second
            current += (target - current) * (dt / tau) if tau > 0 else target
            # Add sensor noise
            current += rng.normal(0, THERMAL_NOISE_STD)
            trace[s] = current

        # Clamp to physical bounds (no negative temps, no exceeding sensor range)
        lo, hi = _TEMP_BOUNDS[sensor]
        np.clip(trace, lo, hi, out=trace)
        sensor_traces[sensor] = trace

    # Post-process T4 for turbo degradation
    if turbo_degradation_factor > 0 and "t3" in sensor_traces and "t4" in sensor_traces:
        t3_mean = np.mean(sensor_traces["t3"])
        t4_trace = sensor_traces["t4"]
        baseline_delta = t3_mean - np.mean(t4_trace)
        if baseline_delta > 0:
            # T4 rises (delta shrinks) proportional to turbo degradation
            delta_reduction = baseline_delta * turbo_degradation_factor
            sensor_traces["t4"] = t4_trace + delta_reduction

    # Compute 6 statistics per sensor
    features: Dict[str, float] = {}
    for sensor in TEMP_SENSORS:
        trace = sensor_traces[sensor]
        features[f"{sensor}_mean"] = float(np.mean(trace))
        features[f"{sensor}_std"] = float(np.std(trace))
        features[f"{sensor}_max"] = float(np.max(trace))
        features[f"{sensor}_min"] = float(np.min(trace))
        features[f"{sensor}_range"] = float(np.max(trace) - np.min(trace))
        # Slope via linear regression
        x = np.arange(60, dtype=np.float64)
        slope = np.polyfit(x, trace, 1)[0]
        features[f"{sensor}_slope"] = float(slope)

    # Differential features
    features["t3_t4_delta"] = features["t3_mean"] - features["t4_mean"]
    features["t1_t5_delta"] = features["t1_mean"] - features["t5_mean"]
    # T3 exceedance duration: count of seconds where T3 > 677°C
    t3_exceedance = float(np.sum(sensor_traces["t3"] > T3_EXCEEDANCE_THRESHOLD))
    features["t3_exceedance_duration"] = t3_exceedance

    # Updated temps for next window (final values)
    updated_temps = {sensor: float(sensor_traces[sensor][-1]) for sensor in TEMP_SENSORS}

    return features, updated_temps
