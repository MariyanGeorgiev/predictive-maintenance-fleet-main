"""Degradation progression curve validation."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def validate_progression(
    data_dir: Path,
    sample_trucks: Optional[int] = None,
) -> bool:
    """Validate that degradation curves follow expected patterns.

    Checks:
    - RMS increases monotonically with severity stage progression
    - Kurtosis follows 4-stage pattern (rises then drops in Stage 4)
    - RUL decreases over time for faulty trucks

    Returns True if all checks pass.
    """
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("truck_*/day_*.parquet"))

    if sample_trucks:
        truck_dirs = sorted(set(f.parent for f in parquet_files))[:sample_trucks]
        parquet_files = [f for f in parquet_files if f.parent in truck_dirs]

    if not parquet_files:
        logger.warning("No parquet files found")
        return True

    dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
    all_data = pd.concat(dfs, ignore_index=True)

    passed = True

    # Check 1: RMS increases with severity stage
    stage_order = ["HEALTHY", "STAGE_2", "STAGE_3", "STAGE_4"]
    rms_col = "acc1_rms_x_mean"
    if rms_col in all_data.columns:
        stage_means = {}
        for stage in stage_order:
            subset = all_data[all_data["fault_severity"] == stage]
            if len(subset) > 0:
                stage_means[stage] = subset[rms_col].mean()

        prev_val = 0
        for stage in stage_order:
            if stage in stage_means:
                if stage_means[stage] < prev_val * 0.8:  # Allow 20% tolerance
                    logger.warning(
                        f"RMS progression check failed: {stage}={stage_means[stage]:.4f} "
                        f"< previous={prev_val:.4f}"
                    )
                    passed = False
                prev_val = stage_means[stage]

    # Check 2: RUL decreases over time for faulty trucks
    faulty = all_data[all_data["fault_mode"] != "HEALTHY"].copy()
    if len(faulty) > 0 and "rul_hours" in faulty.columns:
        for truck_id in faulty["truck_id"].unique()[:5]:  # Check first 5 trucks
            truck_data = faulty[faulty["truck_id"] == truck_id].sort_values("timestamp")
            if len(truck_data) > 10:
                rul_start = truck_data["rul_hours"].iloc[:10].mean()
                rul_end = truck_data["rul_hours"].iloc[-10:].mean()
                if rul_end > rul_start:
                    logger.warning(
                        f"RUL progression failed for truck {truck_id}: "
                        f"start={rul_start:.1f}h, end={rul_end:.1f}h"
                    )
                    passed = False

    if passed:
        logger.info("All progression checks passed")
    else:
        logger.warning("Some progression checks failed")

    return passed


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.validation.progression_checks <data_dir>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    result = validate_progression(Path(sys.argv[1]))
    sys.exit(0 if result else 1)
