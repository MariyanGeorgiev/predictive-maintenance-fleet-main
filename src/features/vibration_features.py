"""Synthesize 132 vibration features directly from physics models (spec §9.1).

Per sensor per axis:
- Time-domain: RMS, Peak, Crest Factor, Kurtosis (4 features × 3 axes = 12)
- Freq-domain: Band energy, Peak freq, Spectral centroid per band
  ACC-1/ACC-2: 4 bands × 3 features × 3 axes = 36
  ACC-3: 2 bands × 3 features × 3 axes = 18
- Spectral Kurtosis: max SK value + freq of max SK = 2 per sensor

Totals: ACC-1: 50, ACC-2: 50, ACC-3: 32 = 132 features

After 60-second aggregation (mean + max-kurtosis + std-RMS):
150 vibration features total (see aggregation.py).
But we track the full 132 base features and handle aggregation as part of synthesis.
"""

from typing import Dict, List

import numpy as np

from src.config.constants import (
    ACC12_BANDS,
    ACC3_BANDS,
    AXES,
    HEALTHY_VIBRATION,
    VIBRATION_NOISE_FRACTION,
    WINDOWS_PER_AGG_ACC12,
    WINDOWS_PER_AGG_ACC3,
)
from src.faults.fault_mode import FaultEffect


def _apply_effect(base: float, effects: Dict, key: str) -> float:
    """Apply a fault effect to a base feature value."""
    if key not in effects:
        return base
    mode, value = effects[key]
    if mode == "set":
        return value
    elif mode == "multiply":
        return base * value
    elif mode == "add":
        return base + value
    return base


def _synthesize_sensor_features(
    sensor: str,
    rpm: float,
    load: float,
    fault_effects: Dict[str, tuple],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Synthesize all vibration features for one sensor.

    Returns dict with keys like 'acc1_rms_x_mean', 'acc1_kurtosis_x_max', etc.
    """
    params = HEALTHY_VIBRATION[sensor]
    rms_lo, rms_hi = params["rms_base"]
    cf_lo, cf_hi = params["crest_factor_base"]

    # Base RMS scales with load
    rms_base = rng.uniform(rms_lo, rms_hi) * (0.7 + 0.3 * load)

    is_acc3 = sensor == "acc3"
    bands = ACC3_BANDS if is_acc3 else ACC12_BANDS
    n_sub_windows = WINDOWS_PER_AGG_ACC3 if is_acc3 else WINDOWS_PER_AGG_ACC12

    features = {}

    for axis in AXES:
        # --- Time-domain features ---
        rms_val = _apply_effect(rms_base, fault_effects, f"{sensor}_rms")
        # Add per-axis variation
        rms_val *= (1.0 + rng.normal(0, 0.05))
        rms_val = max(rms_val, 0.001)

        kurtosis_base = params["kurtosis_base"] + rng.normal(0, 0.2)
        kurtosis_val = _apply_effect(kurtosis_base, fault_effects, f"{sensor}_kurtosis")
        kurtosis_val = max(kurtosis_val, 2.0)

        crest_factor = rng.uniform(cf_lo, cf_hi)
        crest_factor = _apply_effect(crest_factor, fault_effects, f"{sensor}_crest_factor")

        peak_val = rms_val * crest_factor

        # Aggregation over sub-windows:
        # mean of RMS is approximately the single-window RMS
        # max of kurtosis over N windows follows extreme value distribution
        # std of RMS is small (central limit)
        noise = VIBRATION_NOISE_FRACTION

        features[f"{sensor}_rms_{axis}_mean"] = rms_val * (1 + rng.normal(0, noise * 0.3))
        features[f"{sensor}_rms_{axis}_std"] = rms_val * abs(rng.normal(0.05, 0.02))
        features[f"{sensor}_peak_{axis}_mean"] = peak_val * (1 + rng.normal(0, noise))
        features[f"{sensor}_crest_factor_{axis}_mean"] = crest_factor * (1 + rng.normal(0, noise * 0.5))

        # Kurtosis: mean and max (max is diagnostic)
        features[f"{sensor}_kurtosis_{axis}_mean"] = kurtosis_val * (1 + rng.normal(0, noise * 0.3))
        # Max kurtosis over many windows is higher than mean (extreme value)
        features[f"{sensor}_kurtosis_{axis}_max"] = kurtosis_val * (1.0 + 0.15 * np.log(n_sub_windows) * rng.uniform(0.5, 1.5))

        # --- Frequency-domain features (per band) ---
        # Distribute total energy across bands
        total_energy = rms_val ** 2  # RMS^2 ~ total vibration energy

        for band_name, (f_lo, f_hi) in bands.items():
            # Base energy fraction: roughly equal with 1/f roll-off
            band_center = (f_lo + f_hi) / 2.0
            base_fraction = 1.0 / (1.0 + band_center / 1000.0)  # 1/f-ish

            band_energy = total_energy * base_fraction
            # Apply fault-specific band energy effects
            band_key = f"{sensor}_{band_name}_energy"
            band_energy = _apply_effect(band_energy, fault_effects, band_key)
            band_energy = max(band_energy, 1e-8)

            # Band energy (normalized by bandwidth for density)
            bandwidth = f_hi - f_lo
            energy_density = band_energy / bandwidth

            features[f"{sensor}_band_{band_name}_energy_{axis}_mean"] = energy_density * (1 + rng.normal(0, noise))
            features[f"{sensor}_band_{band_name}_energy_{axis}_std"] = energy_density * abs(rng.normal(0.1, 0.03))

            # Peak frequency in band
            # Healthy: roughly centered; with fault, shifts toward fault frequency
            peak_freq = rng.uniform(f_lo + bandwidth * 0.2, f_hi - bandwidth * 0.2)
            shift_key = f"{sensor}_{band_name}_peak_shift"
            if shift_key in fault_effects:
                # Fault present: peak locks to a specific frequency in band
                peak_freq = f_lo + bandwidth * 0.4  # approximate fault freq location
            features[f"{sensor}_band_{band_name}_peak_freq_{axis}_mean"] = peak_freq

            # Spectral centroid
            centroid = (f_lo + f_hi) / 2.0 + rng.normal(0, bandwidth * 0.05)
            features[f"{sensor}_band_{band_name}_centroid_{axis}_mean"] = np.clip(centroid, f_lo, f_hi)

    # --- Spectral Kurtosis (2 features per sensor) ---
    sk_base = rng.uniform(1.0, 5.0)
    sk_val = _apply_effect(sk_base, fault_effects, f"{sensor}_sk_max")
    features[f"{sensor}_sk_max_value"] = sk_val * (1 + rng.normal(0, noise))

    # SK max frequency: if bearing fault, aligns with bearing fault band
    if f"{sensor}_mid_high_peak_shift" in fault_effects:
        sk_freq = rng.uniform(2000, 10000)  # bearing fault band
    elif f"{sensor}_broadband_energy" in fault_effects:
        sk_freq = rng.uniform(1000, 5000)  # turbo band
    else:
        sk_freq = rng.uniform(500, 5000)  # random
    features[f"{sensor}_sk_max_freq"] = sk_freq

    return features


def synthesize_vibration_features(
    rpm: float,
    load: float,
    fault_effects_list: List[FaultEffect],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Synthesize all 132+ vibration features for one 60-second window.

    Args:
        rpm: Current engine RPM.
        load: Current normalized load.
        fault_effects_list: List of FaultEffect from active fault modes.
        rng: Random number generator.

    Returns:
        Dict of feature_name -> value.
    """
    # Merge fault effects: energy features add, shape features take max
    merged: Dict[str, tuple] = {}
    for fe in fault_effects_list:
        for key, (mode, value) in fe.vibration_effects.items():
            if key not in merged:
                merged[key] = (mode, value)
            else:
                existing_mode, existing_val = merged[key]
                if mode == "set":
                    # Shape features (kurtosis, SK, crest): take max
                    if "kurtosis" in key or "sk" in key or "crest" in key:
                        merged[key] = (mode, max(existing_val, value))
                    else:
                        merged[key] = (mode, value)  # last wins
                elif mode == "multiply":
                    # Energy features: combine multiplicatively
                    merged[key] = (mode, existing_val * value)
                elif mode == "add":
                    merged[key] = (mode, existing_val + value)

    all_features = {}
    for sensor in ["acc1", "acc2", "acc3"]:
        sensor_features = _synthesize_sensor_features(sensor, rpm, load, merged, rng)
        all_features.update(sensor_features)

    return all_features
