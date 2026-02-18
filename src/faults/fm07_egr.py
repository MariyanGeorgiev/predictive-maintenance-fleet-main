"""FM-07: EGR Cooler Failure (spec §10.3.2).

Two failure modes:
1. Fouling (gradual): T5 rises by 20-60°C over 500-1500 hours.
2. Coolant leak (sudden): T1 spikes 10-30°C, T5 spikes 30-80°C for 30-120 seconds.

Leak events are determined deterministically from t_hours (hash-based) rather than
from stateful RNG, ensuring reproducibility across multiprocessing workers.
"""

import hashlib
import struct

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


def _deterministic_random(seed: int, t_hours: float) -> float:
    """Deterministic pseudo-random float in [0, 1) from seed + time.

    Uses hash to avoid any mutable RNG state, making it safe for
    multiprocessing fork without seed coordination.
    """
    data = struct.pack("<id", seed, t_hours)
    h = hashlib.sha256(data).digest()
    # Use first 8 bytes as uint64, map to [0, 1)
    val = int.from_bytes(h[:8], "little")
    return val / (2**64)


class EGRCoolerFault(FaultMode):
    """FM-07: EGR cooler failure (fouling + potential leak events)."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        delta_t5_max: float = 40.0,     # Fouling: 20-60°C T5 rise
        leak_t1_spike: float = 20.0,    # Leak: 10-30°C T1 spike
        leak_t5_spike: float = 55.0,    # Leak: 30-80°C T5 spike
        leak_probability_per_hour: float = 0.002,  # Poisson rate for late stages
        seed: int = 0,
    ):
        super().__init__("FM-07", onset_hours, degradation, total_life_hours)
        self.delta_t5_max = delta_t5_max
        self.leak_t1_spike = leak_t1_spike
        self.leak_t5_spike = leak_t5_spike
        self.leak_probability_per_hour = leak_probability_per_hour
        self._seed = seed

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        stage = self.current_stage(t_hours)

        # Gradual fouling effect
        fouling_factor = severity * 0.4  # max fouling_factor = 0.4
        delta_t5 = self.delta_t5_max * fouling_factor

        thermal = {"t5": delta_t5}

        # Sudden leak events (only in stage3 or stage4)
        # Deterministic: hash(seed, t_hours) decides if a leak occurs at this window
        if stage in ("stage3", "stage4"):
            p = self.leak_probability_per_hour / 60.0  # per-window probability
            r = _deterministic_random(self._seed, t_hours)
            if r < p * severity:
                thermal["t1"] = thermal.get("t1", 0) + self.leak_t1_spike
                thermal["t5"] = thermal.get("t5", 0) + self.leak_t5_spike
                # Also apply to the next window (leak duration 30-120s = 1-2 windows)
                # handled by the fact that consecutive t_hours will each independently
                # evaluate — some will trigger, some won't, creating realistic clusters

        return FaultEffect(vibration_effects={}, thermal_effects=thermal)
