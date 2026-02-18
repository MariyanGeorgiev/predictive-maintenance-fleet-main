"""Tests for Parquet output."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.features.feature_vector import ALL_COLUMNS, FEATURE_COLUMNS
from src.labels.ground_truth import GroundTruthLabel
from src.storage.parquet_writer import ParquetWriter


class TestParquetWriter:
    def test_write_and_read_back(self):
        """Written Parquet should be readable with correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(Path(tmpdir))

            # Create minimal test data (3 windows)
            features = []
            labels = []
            for i in range(3):
                feat = {col: float(i) for col in FEATURE_COLUMNS}
                features.append(feat)
                labels.append(GroundTruthLabel(
                    fault_mode="HEALTHY",
                    fault_severity="HEALTHY",
                    rul_hours=float("inf"),
                    path_a_label="NORMAL",
                ))

            path = writer.write_truck_day(
                truck_id=1, engine_type="modern", day_index=0,
                features=features, labels=labels,
            )

            assert path.exists()

            # Read back
            table = pq.read_table(path)
            df = table.to_pandas()
            assert len(df) == 3
            assert "truck_id" in df.columns
            assert "fault_mode" in df.columns
            assert "rpm_est" in df.columns

    def test_file_naming(self):
        """Output file should be at truck_001/day_000.parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetWriter(Path(tmpdir))

            features = [{col: 0.0 for col in FEATURE_COLUMNS}]
            labels = [GroundTruthLabel("HEALTHY", "HEALTHY", float("inf"), "NORMAL")]

            path = writer.write_truck_day(1, "modern", 0, features, labels)
            assert path.name == "day_000.parquet"
            assert path.parent.name == "truck_001"
