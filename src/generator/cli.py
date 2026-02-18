"""Command-line interface for the synthetic data generator."""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import click
import numpy as np

from src.config.constants import SIMULATION_DAYS
from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultMode
from src.faults.fault_schedule import assign_faults
from src.faults.fm01_bearing import BearingWearFault
from src.faults.fm05_turbo import TurboDegradationFault
from src.faults.fm06_injector import InjectorWearFault
from src.fleet.fleet_factory import create_fleet
from src.fleet.truck import Truck
from src.generator.batch_generator import BatchGenerator
from src.generator.truck_day_generator import TruckDayGenerator
from src.storage.parquet_writer import ParquetWriter
from src.storage.thermal_state import default_idle_temps


def _build_validation_schedule(
    trucks: List[Truck],
) -> Dict[int, List[FaultMode]]:
    """Build controlled fault schedule for validation checkpoint.

    10 trucks with specific fault configurations:
    - Trucks 0-1: Healthy (no faults)
    - Trucks 2-3: FM-01 Stage 3 bearing wear
    - Trucks 4-5: FM-05 turbo degradation
    - Trucks 6-7: FM-06 injector wear
    - Trucks 8-9: Multi-fault FM-01 + FM-05
    """
    schedule: Dict[int, List[FaultMode]] = {}

    for i, truck in enumerate(trucks[:10]):
        if i < 2:
            # Healthy
            schedule[truck.truck_id] = []

        elif i < 4:
            # FM-01 Stage 3: onset_hours=-400 with total_life=500
            # At t=0 (day 0), time_since_onset=400h = 80% of life → Stage 3
            deg = DegradationModel(0.01, 0.0002, 0.10, 600, seed=100 + i)
            fault = BearingWearFault(
                onset_hours=-400.0, degradation=deg,
                total_life_hours=500.0, affected_sensor="acc1",
            )
            schedule[truck.truck_id] = [fault]

        elif i < 6:
            # FM-05 turbo degradation: onset=-500, life=700 → 71% = Stage 2-3
            deg = DegradationModel(0.01, 0.0003, 0.10, 800, seed=200 + i)
            fault = TurboDegradationFault(
                onset_hours=-500.0, degradation=deg,
                total_life_hours=700.0, degradation_factor_max=0.3,
            )
            schedule[truck.truck_id] = [fault]

        elif i < 8:
            # FM-06 injector wear: onset=-600, life=800 → 75% = Stage 3
            deg = DegradationModel(0.01, 0.0002, 0.08, 900, seed=300 + i)
            fault = InjectorWearFault(
                onset_hours=-600.0, degradation=deg,
                total_life_hours=800.0, delta_t3_max=55.0,
                delta_t_injector=75.0,
            )
            schedule[truck.truck_id] = [fault]

        else:
            # Multi-fault: FM-01 + FM-05
            deg1 = DegradationModel(0.01, 0.0002, 0.10, 600, seed=400 + i)
            bearing = BearingWearFault(
                onset_hours=-400.0, degradation=deg1,
                total_life_hours=500.0, affected_sensor="acc1",
            )
            deg2 = DegradationModel(0.01, 0.0003, 0.10, 800, seed=500 + i)
            turbo = TurboDegradationFault(
                onset_hours=-500.0, degradation=deg2,
                total_life_hours=700.0, degradation_factor_max=0.3,
            )
            schedule[truck.truck_id] = [bearing, turbo]

    return schedule


@click.command()
@click.option("--trucks", default=200, help="Number of trucks to generate.")
@click.option("--days", default=SIMULATION_DAYS, help="Number of days to simulate.")
@click.option("--seed", default=42, help="Master RNG seed.")
@click.option("--output-dir", default="output/", help="Output directory.")
@click.option("--workers", default=1, help="Number of parallel workers.")
@click.option("--single-truck", default=None, type=int, help="Generate only this truck ID.")
@click.option("--single-day", default=None, type=int, help="Generate only this day index.")
@click.option("--skip-existing/--no-skip-existing", default=True, help="Skip already generated files.")
@click.option("--validation-checkpoint", is_flag=True, help="Run validation checkpoint (10 trucks × 1 day).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(trucks, days, seed, output_dir, workers, single_truck, single_day,
         skip_existing, validation_checkpoint, verbose):
    """Synthetic data generator for predictive maintenance system."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)

    # Create fleet
    logger.info(f"Creating fleet (seed={seed})...")
    fleet, metadata = create_fleet(seed=seed, output_dir=output_path)

    if validation_checkpoint:
        # Step 8: 10 trucks × 1 day with controlled fault states
        logger.info("Running validation checkpoint (10 trucks × 1 day, controlled faults)...")
        val_schedule = _build_validation_schedule(fleet)

        generator = TruckDayGenerator()
        writer = ParquetWriter(output_path)

        for truck in fleet[:10]:
            faults = val_schedule.get(truck.truck_id, [])
            fault_desc = ", ".join(f.fault_id for f in faults) or "HEALTHY"
            logger.info(f"  Truck {truck.truck_id:03d}: {fault_desc}")

            features, labels, _ = generator.generate(
                profile=truck.profile,
                engine_type=truck.engine_type,
                day_index=0,
                faults=faults,
                initial_temps=default_idle_temps(truck.engine_type),
                seed=truck.seed * 1000,
            )
            writer.write_truck_day(
                truck.truck_id, truck.engine_type, 0, features, labels,
            )

        logger.info(f"Validation data written to {output_path}")
        return

    # Limit fleet size if requested
    if trucks < len(fleet):
        fleet = fleet[:trucks]

    # Assign faults
    logger.info("Assigning fault schedules...")
    fault_schedule = assign_faults(fleet, seed=seed)

    # Create batch generator
    gen = BatchGenerator(
        trucks=fleet,
        fault_schedule=fault_schedule,
        output_dir=output_path,
        n_workers=workers,
        skip_existing=skip_existing,
    )

    if single_truck is not None:
        logger.info(f"Generating truck {single_truck}...")
        gen.generate_single_truck(single_truck, day_index=single_day, n_days=days)
    else:
        logger.info(f"Generating {len(fleet)} trucks × {days} days...")
        gen.generate_all(n_days=days)

    logger.info("Done.")


if __name__ == "__main__":
    main()
