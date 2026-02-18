"""FM-02: Cooling System Degradation - rising T1 (spec §10.3.2)."""

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class CoolingDegradationFault(FaultMode):
    """FM-02: Cooling degradation. Rising coolant temperature (T1) over weeks."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        delta_t1_max: float = 20.0,  # Max T1 increase at full severity
    ):
        super().__init__("FM-02", onset_hours, degradation, total_life_hours)
        self.delta_t1_max = delta_t1_max

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        # T1 rises proportionally to severity and load
        delta_t1 = self.delta_t1_max * severity * (0.5 + 0.5 * load)

        return FaultEffect(
            vibration_effects={},
            thermal_effects={"t1": delta_t1},
        )
