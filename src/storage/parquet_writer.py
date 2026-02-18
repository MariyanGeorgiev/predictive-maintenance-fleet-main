"""Write per-truck-per-day Parquet files."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.feature_vector import ALL_COLUMNS, FEATURE_COLUMNS
from src.labels.ground_truth import GroundTruthLabel
from src.storage.schema_definition import PARQUET_SCHEMA


class ParquetWriter:
    """Writes truck-day data to Parquet files."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def write_truck_day(
        self,
        truck_id: int,
        engine_type: str,
        day_index: int,
        features: List[Dict[str, float]],
        labels: List[GroundTruthLabel],
        base_timestamp: datetime = datetime(2025, 1, 1),
    ) -> Path:
        """Write one Parquet file for a truck-day.

        Args:
            truck_id: Truck identifier.
            engine_type: "modern" or "older".
            day_index: Day number (0-182).
            features: List of 1440 feature dicts.
            labels: List of 1440 GroundTruthLabel objects.
            base_timestamp: Simulation start time.

        Returns:
            Path to the written Parquet file.
        """
        n_windows = len(features)

        # Build row data
        rows = []
        day_start = base_timestamp + timedelta(days=day_index)

        for i in range(n_windows):
            ts = day_start + timedelta(seconds=i * 60)
            feat = features[i]
            label = labels[i]

            row = {
                "timestamp": ts,
                "truck_id": truck_id,
                "engine_type": engine_type,
                "day_index": day_index,
            }

            # Feature values
            for col in FEATURE_COLUMNS:
                row[col] = feat.get(col, 0.0)

            # Labels
            row["fault_mode"] = label.fault_mode
            row["fault_severity"] = label.fault_severity
            row["rul_hours"] = label.rul_hours if np.isfinite(label.rul_hours) else -1.0
            row["path_a_label"] = label.path_a_label

            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure column order matches schema
        df = df[[c for c in ALL_COLUMNS if c in df.columns]]

        # Write Parquet
        truck_dir = self.output_dir / f"truck_{truck_id:03d}"
        truck_dir.mkdir(parents=True, exist_ok=True)
        output_path = truck_dir / f"day_{day_index:03d}.parquet"

        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
        pq.write_table(table, output_path, compression="snappy")

        return output_path
