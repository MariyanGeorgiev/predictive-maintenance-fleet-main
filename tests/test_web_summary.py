"""Tests for fullstack live summary payload."""

from pathlib import Path

from src.features.feature_vector import FEATURE_COLUMNS
from src.labels.ground_truth import GroundTruthLabel
from src.storage.parquet_writer import ParquetWriter
from src.web.fullstack_server import _build_live_summary_from_output


def _sample_data(n: int):
    features = []
    labels = []
    for i in range(n):
        features.append({col: float(i) for col in FEATURE_COLUMNS})
        labels.append(GroundTruthLabel(
            fault_mode="FM-01" if i % 2 else "HEALTHY",
            fault_severity="STAGE_2" if i % 2 else "HEALTHY",
            rul_hours=100.0,
            path_a_label="ALERT" if i % 2 else "NORMAL",
        ))
    return features, labels


def test_summary_demo_when_no_parquet(tmp_path: Path):
    payload = _build_live_summary_from_output(tmp_path)
    assert payload["source"] == "demo"
    assert payload["data_files"] == 0
    assert payload["rows_total"] == 0


def test_summary_live_from_generated_parquet(tmp_path: Path):
    writer = ParquetWriter(tmp_path)

    features, labels = _sample_data(5)
    writer.write_truck_day(1, "modern", 0, features, labels)
    writer.write_truck_day(2, "older", 0, features, labels)

    payload = _build_live_summary_from_output(tmp_path)

    assert payload["source"] == "live"
    assert payload["fleet_size"] == 2
    assert payload["data_files"] == 2
    assert payload["rows_total"] == 10
    assert payload["failure_modes"] >= 1
