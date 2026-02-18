"""Feature range validation against spec Table 10.6 (±20% tolerance)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

from src.config.constants import VALIDATION_RANGES, VALIDATION_TOLERANCE

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    feature: str
    expected_range: Tuple[float, float]
    actual_range: Tuple[float, float]
    condition: str
    passed: bool
    message: str = ""


@dataclass
class ValidationReport:
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self) -> str:
        lines = [f"Validation: {self.n_passed} passed, {self.n_failed} failed"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  [{status}] {r.condition}/{r.feature}: "
                f"expected {r.expected_range}, got {r.actual_range} {r.message}"
            )
        return "\n".join(lines)


def _check_range(
    values: pd.Series,
    expected: Tuple[float, float],
    tolerance: float,
    feature: str,
    condition: str,
) -> ValidationResult:
    """Check if feature values fall within expected range ± tolerance."""
    exp_lo = expected[0] * (1 - tolerance)
    exp_hi = expected[1] * (1 + tolerance)

    actual_lo = float(values.min())
    actual_hi = float(values.max())
    actual_mean = float(values.mean())

    # Check: mean should be within the tolerance-expanded range
    passed = exp_lo <= actual_mean <= exp_hi

    return ValidationResult(
        feature=feature,
        expected_range=expected,
        actual_range=(actual_lo, actual_hi),
        condition=condition,
        passed=passed,
        message=f"mean={actual_mean:.3f}" if not passed else "",
    )


def validate_parquet_dir(
    data_dir: Path,
    tolerance: float = VALIDATION_TOLERANCE,
    sample_trucks: Optional[int] = None,
) -> ValidationReport:
    """Validate features from Parquet files against spec ranges.

    Args:
        data_dir: Directory containing truck_XXX/day_XXX.parquet files.
        tolerance: Acceptable tolerance (default 20%).
        sample_trucks: If set, only validate this many trucks.

    Returns:
        ValidationReport with pass/fail for each check.
    """
    report = ValidationReport()
    data_path = Path(data_dir)

    # Find all parquet files
    parquet_files = sorted(data_path.glob("truck_*/day_*.parquet"))
    if sample_trucks:
        # Get unique truck dirs and sample
        truck_dirs = sorted(set(f.parent for f in parquet_files))[:sample_trucks]
        parquet_files = [f for f in parquet_files if f.parent in truck_dirs]

    if not parquet_files:
        logger.warning(f"No parquet files found in {data_dir}")
        return report

    # Read all files into one DataFrame
    dfs = []
    for f in parquet_files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)

    # Cruise-specific checks use load_proxy > 0.5 to filter operating conditions
    has_load = "load_proxy" in all_data.columns
    cruise_keys = {"t3_mean_cruise", "t3_t4_delta"}

    # Validate healthy trucks
    healthy = all_data[all_data["fault_mode"] == "HEALTHY"]
    if len(healthy) > 0 and "healthy" in VALIDATION_RANGES:
        ranges = VALIDATION_RANGES["healthy"]
        for feat_key, expected in ranges.items():
            if isinstance(expected, str):
                continue  # Skip qualitative checks like "low"/"high"

            col = _map_feature_key(feat_key, healthy.columns)
            if col and col in healthy.columns:
                subset = healthy
                if feat_key in cruise_keys and has_load:
                    subset = healthy[healthy["load_proxy"] > 0.5]
                if len(subset) == 0:
                    continue
                result = _check_range(subset[col], expected, tolerance, feat_key, "healthy")
                report.results.append(result)

    # Validate FM-01 Stage 3
    fm01_s3 = all_data[
        (all_data["fault_mode"] == "FM-01") & (all_data["fault_severity"] == "STAGE_3")
    ]
    if len(fm01_s3) > 0 and "fm01_stage3" in VALIDATION_RANGES:
        ranges = VALIDATION_RANGES["fm01_stage3"]
        for feat_key, expected in ranges.items():
            if isinstance(expected, str):
                continue
            col = _map_feature_key(feat_key, fm01_s3.columns)
            if col and col in fm01_s3.columns:
                subset = fm01_s3
                if feat_key in cruise_keys and has_load:
                    subset = fm01_s3[fm01_s3["load_proxy"] > 0.5]
                if len(subset) == 0:
                    continue
                result = _check_range(subset[col], expected, tolerance, feat_key, "fm01_stage3")
                report.results.append(result)

    # Validate FM-06 degraded
    fm06 = all_data[
        (all_data["fault_mode"] == "FM-06") & (all_data["fault_severity"].isin(["STAGE_3", "STAGE_4"]))
    ]
    if len(fm06) > 0 and "fm06_degraded" in VALIDATION_RANGES:
        ranges = VALIDATION_RANGES["fm06_degraded"]
        for feat_key, expected in ranges.items():
            if isinstance(expected, str):
                continue
            col = _map_feature_key(feat_key, fm06.columns)
            if col and col in fm06.columns:
                subset = fm06
                if feat_key in cruise_keys and has_load:
                    subset = fm06[fm06["load_proxy"] > 0.5]
                if len(subset) == 0:
                    continue
                result = _check_range(subset[col], expected, tolerance, feat_key, "fm06_degraded")
                report.results.append(result)

    return report


def _map_feature_key(key: str, columns: pd.Index) -> Optional[str]:
    """Map a validation key to actual column name."""
    mapping = {
        "acc1_rms": "acc1_rms_x_mean",
        "acc1_kurtosis": "acc1_kurtosis_x_mean",
        "acc1_sk_max": "acc1_sk_max_value",
        "t3_mean_cruise": "t3_mean",
        "t3_t4_delta": "t3_t4_delta",
        "acc1_high_band_energy": "acc1_band_high_energy_x_mean",
    }
    return mapping.get(key)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.validation.range_checks <data_dir> [--sample N]")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    sample = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--sample" else None

    report = validate_parquet_dir(data_dir, sample_trucks=sample)
    print(report.summary())
    sys.exit(0 if report.passed else 1)
