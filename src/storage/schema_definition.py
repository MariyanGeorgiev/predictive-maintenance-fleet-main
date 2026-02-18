"""PyArrow schema definition for Parquet output (spec §9.6)."""

import pyarrow as pa

from src.features.feature_vector import (
    ALL_COLUMNS,
    CONDITIONING_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    THERMAL_COLUMNS,
    VIBRATION_COLUMNS,
)


def build_parquet_schema() -> pa.Schema:
    """Build the PyArrow schema for truck-day Parquet files.

    201 columns total:
    - 4 metadata (timestamp, truck_id, engine_type, day_index)
    - 2 conditioning + 132+ vibration + 39 thermal = ~191 features (float32)
    - 4 labels (fault_mode, fault_severity, rul_hours, path_a_label)
    """
    fields = []

    # Metadata
    fields.append(pa.field("timestamp", pa.timestamp("s")))
    fields.append(pa.field("truck_id", pa.int32()))
    fields.append(pa.field("engine_type", pa.string()))
    fields.append(pa.field("day_index", pa.int32()))

    # Feature columns (all float32)
    for col in FEATURE_COLUMNS:
        fields.append(pa.field(col, pa.float32()))

    # Label columns
    fields.append(pa.field("fault_mode", pa.string()))
    fields.append(pa.field("fault_severity", pa.string()))
    fields.append(pa.field("rul_hours", pa.float32()))
    fields.append(pa.field("path_a_label", pa.string()))

    return pa.schema(fields)


PARQUET_SCHEMA = build_parquet_schema()
