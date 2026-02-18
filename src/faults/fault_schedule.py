"""Assigns fault modes and onset times to trucks (spec §10.7)."""

from typing import Dict, List

import numpy as np

from src.config.constants import (
    BEARING_DEGRADATION,
    FAULT_DISTRIBUTION,
    SIMULATION_DAYS,
    THERMAL_FAULT_PARAMS,
    VALVE_TRAIN_PARAMS,
)
from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultMode
from src.faults.fm01_bearing import BearingWearFault
from src.faults.fm02_cooling import CoolingDegradationFault
from src.faults.fm03_valve_train import ValveTrainWearFault
from src.faults.fm04_oil import OilDegradationFault
from src.faults.fm05_turbo import TurboDegradationFault
from src.faults.fm06_injector import InjectorWearFault
from src.faults.fm07_egr import EGRCoolerFault
from src.faults.fm08_dpf import DPFBlockageFault
from src.fleet.truck import Truck

TOTAL_SIM_HOURS = SIMULATION_DAYS * 24  # 4392 hours

# All fault type IDs for balanced assignment
FAULT_IDS = ["FM-01", "FM-02", "FM-03", "FM-04", "FM-05", "FM-06", "FM-07", "FM-08"]


def _create_fault(
    fault_id: str,
    onset_hours: float,
    engine_type: str,
    rng: np.random.Generator,
) -> FaultMode:
    """Create a specific fault mode instance with sampled parameters."""
    seed = int(rng.integers(0, 2**31))

    if fault_id == "FM-01":
        params = BEARING_DEGRADATION[engine_type]
        lam = rng.uniform(*params["lambda_range"])
        sigma = rng.uniform(*params["sigma_range"])
        t_stage2 = rng.uniform(*params["t_stage2_hours"])
        dt_23 = rng.uniform(*params["dt_23_hours"])
        dt_34 = rng.uniform(*params["dt_34_hours"])
        total_life = t_stage2 + dt_23 + dt_34

        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=lam, sigma=sigma,
            total_hours=int(total_life) + 100, seed=seed,
        )
        sensor = rng.choice(["acc1", "acc2"])
        return BearingWearFault(onset_hours, degradation, total_life, sensor)

    elif fault_id == "FM-02":
        p = THERMAL_FAULT_PARAMS["fm02_cooling"]
        total_life = rng.uniform(*p["progression_hours"])
        delta = rng.uniform(*p["delta_t1_max"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0002, sigma=0.08,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return CoolingDegradationFault(onset_hours, degradation, total_life, delta)

    elif fault_id == "FM-03":
        p = VALVE_TRAIN_PARAMS
        total_life = rng.uniform(*p["progression_hours"])
        energy_mult = rng.uniform(*p["energy_multiplier_max"])
        kurt_inc = rng.uniform(*p["kurtosis_increase_max"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0002, sigma=0.10,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return ValveTrainWearFault(onset_hours, degradation, total_life, energy_mult, kurt_inc)

    elif fault_id == "FM-04":
        p = THERMAL_FAULT_PARAMS["fm04_oil"]
        total_life = rng.uniform(*p["progression_hours"])
        delta = rng.uniform(*p["delta_t2_max"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0002, sigma=0.08,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return OilDegradationFault(onset_hours, degradation, total_life, delta)

    elif fault_id == "FM-05":
        p = THERMAL_FAULT_PARAMS["fm05_turbo"]
        total_life = rng.uniform(*p["progression_hours"])
        deg_max = rng.uniform(*p["degradation_factor_max"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0003, sigma=0.10,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return TurboDegradationFault(onset_hours, degradation, total_life, deg_max)

    elif fault_id == "FM-06":
        p = THERMAL_FAULT_PARAMS["fm06_injector"]
        total_life = rng.uniform(*p["progression_hours"])
        delta_t3 = rng.uniform(*p["delta_t3_max"])
        delta_inj = rng.uniform(*p["delta_t_injector_full"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0002, sigma=0.08,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return InjectorWearFault(onset_hours, degradation, total_life, delta_t3, delta_inj)

    elif fault_id == "FM-07":
        p_foul = THERMAL_FAULT_PARAMS["fm07_egr_fouling"]
        p_leak = THERMAL_FAULT_PARAMS["fm07_egr_leak"]
        total_life = rng.uniform(*p_foul["progression_hours"])
        delta_t5 = rng.uniform(*p_foul["delta_t5_max"])
        leak_t1 = rng.uniform(*p_leak["delta_t1_spike"])
        leak_t5 = rng.uniform(*p_leak["delta_t5_spike"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0003, sigma=0.12,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return EGRCoolerFault(
            onset_hours, degradation, total_life, delta_t5, leak_t1, leak_t5,
            seed=seed,
        )

    elif fault_id == "FM-08":
        p = THERMAL_FAULT_PARAMS["fm08_dpf"]
        total_life = rng.uniform(*p["progression_hours"])
        delta_t3 = rng.uniform(*p["delta_t3_max"])
        regen_int = rng.uniform(*p["regen_interval_hours"])
        degradation = DegradationModel(
            severity_0=0.01, lambda_rate=0.0005, sigma=0.15,
            total_hours=int(total_life) + 100, seed=seed,
        )
        return DPFBlockageFault(
            onset_hours, degradation, total_life, delta_t3, regen_int,
            seed=seed,
        )

    raise ValueError(f"Unknown fault_id: {fault_id}")


def assign_faults(
    trucks: List[Truck],
    seed: int = 42,
) -> Dict[int, List[FaultMode]]:
    """Assign fault modes to each truck in the fleet.

    Distribution: ~30% healthy, ~40% single fault, ~20% double, ~10% triple.
    Ensures balanced representation of all 8 fault modes.

    Args:
        trucks: List of Truck objects.
        seed: RNG seed.

    Returns:
        Dict mapping truck_id to list of FaultMode instances.
    """
    rng = np.random.default_rng(seed + 1000)
    n = len(trucks)

    # Determine number of faults per truck
    n_healthy = int(n * FAULT_DISTRIBUTION["healthy"])
    n_single = int(n * FAULT_DISTRIBUTION["single_fault"])
    n_double = int(n * FAULT_DISTRIBUTION["double_fault"])
    n_triple = n - n_healthy - n_single - n_double

    fault_counts = (
        [0] * n_healthy +
        [1] * n_single +
        [2] * n_double +
        [3] * n_triple
    )
    rng.shuffle(fault_counts)

    schedule: Dict[int, List[FaultMode]] = {}
    fault_type_counter = 0  # Round-robin across fault types for balance

    for truck, n_faults in zip(trucks, fault_counts):
        faults = []
        used_ids = set()

        for _ in range(n_faults):
            # Round-robin fault type selection (skip duplicates within same truck)
            for attempt in range(len(FAULT_IDS)):
                fid = FAULT_IDS[(fault_type_counter + attempt) % len(FAULT_IDS)]
                if fid not in used_ids:
                    break
            else:
                fid = rng.choice([f for f in FAULT_IDS if f not in used_ids])

            used_ids.add(fid)
            fault_type_counter = (fault_type_counter + 1) % len(FAULT_IDS)

            # Onset time: early enough for fault to reach at least Stage 2
            # onset must be before total_sim_hours - 0.60 * total_life
            # Use conservative estimate: onset in first 70% of simulation
            max_onset = TOTAL_SIM_HOURS * 0.70
            onset = rng.uniform(0, max_onset)

            fault = _create_fault(fid, onset, truck.engine_type, rng)
            faults.append(fault)

        schedule[truck.truck_id] = faults

    return schedule
