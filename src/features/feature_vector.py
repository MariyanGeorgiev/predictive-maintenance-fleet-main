"""Assembles the full feature vector with canonical column ordering (spec §9.6).

The spec defines 191 base features. Our implementation produces 221 features
because we include aggregation statistics (std, max) beyond the base mean values:
- 2 conditioning (rpm_est, load_proxy)
- 180 vibration (includes per-axis std + max-kurtosis aggregation stats)
- 39 thermal (6 sensors × 6 stats + 3 differentials)

Column layout (229 total):
- 4 metadata: timestamp, truck_id, engine_type, day_index
- 221 features (2 conditioning + 180 vibration + 39 thermal)
- 4 labels: fault_mode, fault_severity, rul_hours, path_a_label
"""

from typing import Dict, List

AXES = ["x", "y", "z"]

# =============================================================================
# Build canonical feature column names
# =============================================================================


def _vibration_columns() -> List[str]:
    """Generate vibration feature column names."""
    cols = []

    for sensor in ["acc1", "acc2", "acc3"]:
        is_acc3 = sensor == "acc3"
        bands = ["low", "broadband"] if is_acc3 else ["low", "mid_low", "mid_high", "high"]

        for axis in AXES:
            # Time-domain (mean aggregated)
            cols.append(f"{sensor}_rms_{axis}_mean")
            cols.append(f"{sensor}_rms_{axis}_std")
            cols.append(f"{sensor}_peak_{axis}_mean")
            cols.append(f"{sensor}_crest_factor_{axis}_mean")
            cols.append(f"{sensor}_kurtosis_{axis}_mean")
            cols.append(f"{sensor}_kurtosis_{axis}_max")

            # Frequency-domain (per band)
            for band in bands:
                cols.append(f"{sensor}_band_{band}_energy_{axis}_mean")
                cols.append(f"{sensor}_band_{band}_energy_{axis}_std")
                cols.append(f"{sensor}_band_{band}_peak_freq_{axis}_mean")
                cols.append(f"{sensor}_band_{band}_centroid_{axis}_mean")

        # Spectral kurtosis (per sensor, not per axis)
        cols.append(f"{sensor}_sk_max_value")
        cols.append(f"{sensor}_sk_max_freq")

    return cols


def _thermal_columns() -> List[str]:
    """Generate thermal feature column names."""
    cols = []
    for sensor in ["t1", "t2", "t3", "t4", "t5", "t6"]:
        for stat in ["mean", "std", "max", "min", "range", "slope"]:
            cols.append(f"{sensor}_{stat}")

    # Differentials
    cols.append("t3_t4_delta")
    cols.append("t1_t5_delta")
    cols.append("t3_exceedance_duration")

    return cols


# Canonical column ordering
METADATA_COLUMNS = ["timestamp", "truck_id", "engine_type", "day_index"]
CONDITIONING_COLUMNS = ["rpm_est", "load_proxy"]
VIBRATION_COLUMNS = _vibration_columns()
THERMAL_COLUMNS = _thermal_columns()
FEATURE_COLUMNS = CONDITIONING_COLUMNS + VIBRATION_COLUMNS + THERMAL_COLUMNS
LABEL_COLUMNS = ["fault_mode", "fault_severity", "rul_hours", "path_a_label"]

ALL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS

# Total feature columns (conditioning + vibration + thermal)
N_FEATURES = len(FEATURE_COLUMNS)

# Contract: enforce exact feature count to catch silent drift
assert N_FEATURES == 221, (
    f"Feature count contract violation: expected 221, got {N_FEATURES}. "
    f"(2 conditioning + 180 vibration + 39 thermal)"
)


def assemble_feature_dict(
    conditioning: Dict[str, float],
    vibration: Dict[str, float],
    thermal: Dict[str, float],
) -> Dict[str, float]:
    """Assemble a complete feature dict from component dicts.

    Missing features are filled with 0.0 and a warning could be logged.
    """
    result = {}
    for col in FEATURE_COLUMNS:
        if col in conditioning:
            result[col] = conditioning[col]
        elif col in vibration:
            result[col] = vibration[col]
        elif col in thermal:
            result[col] = thermal[col]
        else:
            result[col] = 0.0

    assert len(result) == N_FEATURES, (
        f"Assembled feature vector has {len(result)} features, expected {N_FEATURES}"
    )
    return result
