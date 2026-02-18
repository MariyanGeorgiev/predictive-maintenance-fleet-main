"""Parallel batch generation across the fleet.

Days are generated sequentially per truck (for thermal continuity).
Trucks are processed in parallel using multiprocessing.
"""

import json
import logging
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

from src.config.constants import SIMULATION_DAYS
from src.faults.fault_mode import FaultMode
from src.fleet.truck import Truck
from src.generator.truck_day_generator import TruckDayGenerator
from src.storage.parquet_writer import ParquetWriter
from src.storage.thermal_state import load_thermal_state, save_thermal_state

logger = logging.getLogger(__name__)


def _generate_truck_all_days(
    truck: Truck,
    faults: List[FaultMode],
    output_dir: Path,
    n_days: int,
    skip_existing: bool,
) -> None:
    """Generate all days for a single truck (called in worker process).

    Seed flow (all deterministic, no mutable RNG state crosses process boundary):
        master_seed -> truck.seed = master_seed + truck_id  (set at fleet creation)
        truck.seed  -> day_seed = truck.seed * 1000 + day   (set here per day)
        day_seed    -> np.random.default_rng(day_seed)      (created fresh per day in generator)

    The per-day RNG is created inside generate() and threaded through all synthesis
    functions. Fault modes use either the passed-in RNG or deterministic hash functions
    (no internal mutable RNG state), ensuring identical output regardless of
    multiprocessing fork order.
    """
    generator = TruckDayGenerator()
    writer = ParquetWriter(output_dir)

    for day in range(n_days):
        # Check if already generated
        output_path = output_dir / f"truck_{truck.truck_id:03d}" / f"day_{day:03d}.parquet"
        if skip_existing and output_path.exists():
            continue

        # Load thermal state from previous day
        initial_temps = load_thermal_state(
            output_dir, truck.truck_id, day, truck.engine_type,
        )

        # Generate truck-day seed: deterministic from truck seed + day
        day_seed = truck.seed * 1000 + day

        features, labels, final_temps = generator.generate(
            profile=truck.profile,
            engine_type=truck.engine_type,
            day_index=day,
            faults=faults,
            initial_temps=initial_temps,
            seed=day_seed,
        )

        # Write Parquet
        writer.write_truck_day(
            truck_id=truck.truck_id,
            engine_type=truck.engine_type,
            day_index=day,
            features=features,
            labels=labels,
        )

        # Save thermal state for next day
        save_thermal_state(output_dir, truck.truck_id, day, final_temps)

    logger.info(f"Truck {truck.truck_id:03d}: completed {n_days} days")


class BatchGenerator:
    """Generates data for the entire fleet."""

    def __init__(
        self,
        trucks: List[Truck],
        fault_schedule: Dict[int, List[FaultMode]],
        output_dir: Path,
        n_workers: int = 8,
        skip_existing: bool = True,
    ):
        self.trucks = trucks
        self.fault_schedule = fault_schedule
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        self.skip_existing = skip_existing

    def generate_all(self, n_days: int = SIMULATION_DAYS) -> None:
        """Generate full dataset for all trucks."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting generation: {len(self.trucks)} trucks × {n_days} days "
            f"({self.n_workers} workers, skip_existing={self.skip_existing})"
        )

        # Prepare work items
        args = [
            (truck, self.fault_schedule.get(truck.truck_id, []),
             self.output_dir, n_days, self.skip_existing)
            for truck in self.trucks
        ]

        if self.n_workers <= 1:
            # Sequential mode (useful for debugging)
            for arg in args:
                _generate_truck_all_days(*arg)
        else:
            with Pool(self.n_workers) as pool:
                pool.starmap(_generate_truck_all_days, args)

        # Write generation manifest
        self._write_manifest(n_days)
        logger.info("Generation complete.")

    def generate_single_truck(
        self, truck_id: int, day_index: Optional[int] = None, n_days: int = SIMULATION_DAYS,
    ) -> None:
        """Generate data for a single truck (for testing/validation)."""
        truck = next((t for t in self.trucks if t.truck_id == truck_id), None)
        if truck is None:
            raise ValueError(f"Truck {truck_id} not found in fleet")

        faults = self.fault_schedule.get(truck_id, [])

        if day_index is not None:
            # Single day
            generator = TruckDayGenerator()
            writer = ParquetWriter(self.output_dir)
            initial_temps = load_thermal_state(
                self.output_dir, truck_id, day_index, truck.engine_type,
            )
            features, labels, final_temps = generator.generate(
                truck.profile, truck.engine_type, day_index, faults,
                initial_temps, truck.seed * 1000 + day_index,
            )
            writer.write_truck_day(truck_id, truck.engine_type, day_index, features, labels)
            save_thermal_state(self.output_dir, truck_id, day_index, final_temps)
        else:
            _generate_truck_all_days(truck, faults, self.output_dir, n_days, self.skip_existing)

    def _write_manifest(self, n_days: int) -> None:
        """Write generation metadata."""
        meta_dir = self.output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "generation_date": datetime.now().isoformat(),
            "spec_version": "1.0",
            "num_trucks": len(self.trucks),
            "num_days": n_days,
            "total_windows": len(self.trucks) * n_days * 1440,
            "fault_distribution": {
                "healthy": sum(1 for t in self.trucks if not self.fault_schedule.get(t.truck_id)),
                "with_faults": sum(1 for t in self.trucks if self.fault_schedule.get(t.truck_id)),
            },
        }

        (meta_dir / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
