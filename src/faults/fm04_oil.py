"""FM-04: Oil Degradation - rising T2 at load (spec §10.3.2)."""

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class OilDegradationFault(FaultMode):
    """FM-04: Oil degradation. Rising oil temperature (T2) proportional to load."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        delta_t2_max: float = 20.0,
    ):
        super().__init__("FM-04", onset_hours, degradation, total_life_hours)
        self.delta_t2_max = delta_t2_max

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        delta_t2 = self.delta_t2_max * severity * load

        return FaultEffect(
            vibration_effects={},
            thermal_effects={"t2": delta_t2},
        )
