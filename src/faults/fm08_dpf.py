"""FM-08: DPF Blockage (spec §10.3.2).

Sustained T3 elevation when blocked. Normal regen cycles (~538°C transient) are NOT the fault.
dpf_blockage(t) = soot_accumulation(t) - regen_events(t)
T3 remains elevated at 650-750°C continuously when blocked.
Regen cycle clears blockage partially every 200-400 hours.
"""

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class DPFBlockageFault(FaultMode):
    """FM-08: DPF blockage causing sustained T3 elevation."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        delta_t3_max: float = 150.0,   # 100-200°C sustained T3 rise
        regen_interval_hours: float = 300.0,  # 200-400 hours between regens
        regen_clearance: float = 0.3,   # Fraction of blockage cleared per regen
        seed: int = 0,
    ):
        super().__init__("FM-08", onset_hours, degradation, total_life_hours)
        self.delta_t3_max = delta_t3_max
        self.regen_interval_hours = regen_interval_hours
        self.regen_clearance = regen_clearance

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        dt = self.time_since_onset(t_hours)

        # Effective blockage considers partial clearing from regen cycles
        n_regens = int(dt / self.regen_interval_hours)
        effective_severity = severity * (1.0 - self.regen_clearance) ** n_regens
        # But severity keeps growing, so net effect still increases
        effective_severity = min(severity, max(effective_severity, severity * 0.5))

        # Sustained T3 elevation proportional to blockage
        delta_t3 = self.delta_t3_max * effective_severity

        return FaultEffect(
            vibration_effects={},
            thermal_effects={"t3": delta_t3},
        )
