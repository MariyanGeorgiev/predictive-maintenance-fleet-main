"""Persist end-of-day thermal state for next-day initialization."""

import json
from pathlib import Path
from typing import Dict, Optional

from src.config.constants import THERMAL_BASELINES


def default_idle_temps(engine_type: str) -> Dict[str, float]:
    """Return default idle temperatures from spec §10.3.1."""
    baselines = THERMAL_BASELINES[engine_type]
    return {
        sensor: (params["idle"][0] + params["idle"][1]) / 2.0
        for sensor, params in baselines.items()
    }


def save_thermal_state(
    output_dir: Path, truck_id: int, day_index: int, temps: Dict[str, float]
) -> None:
    """Save end-of-day thermal state."""
    state_dir = Path(output_dir) / f"truck_{truck_id:03d}" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"thermal_day_{day_index:03d}.json"
    path.write_text(json.dumps(temps, indent=2) + "\n")


def load_thermal_state(
    output_dir: Path, truck_id: int, day_index: int, engine_type: str
) -> Dict[str, float]:
    """Load previous day's thermal state, or defaults if day_index=0."""
    if day_index <= 0:
        return default_idle_temps(engine_type)

    path = Path(output_dir) / f"truck_{truck_id:03d}" / "state" / f"thermal_day_{day_index - 1:03d}.json"
    if path.exists():
        return json.loads(path.read_text())

    return default_idle_temps(engine_type)
