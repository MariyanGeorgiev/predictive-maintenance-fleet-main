"""FM-05: Turbocharger Degradation (spec §10.3.2).

Primary: T3-T4 delta shrinks as turbo_efficiency = 1.0 - degradation_factor.
Late-stage: Broadband vibration increase at ACC-3 (1-5 kHz).
"""

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class TurboDegradationFault(FaultMode):
    """FM-05: Turbocharger degradation."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        degradation_factor_max: float = 0.3,  # Max efficiency loss (0.2-0.4)
    ):
        super().__init__("FM-05", onset_hours, degradation, total_life_hours)
        self.degradation_factor_max = degradation_factor_max

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        stage = self.current_stage(t_hours)

        # Turbo efficiency loss reduces T3-T4 delta
        # T4 = T3 - baseline_delta * (1 - degradation_factor)
        # This means T4 rises (less heat extracted), so delta shrinks
        # We express this as a T4 increase (thermal effect)
        degradation_factor = severity * self.degradation_factor_max

        # Vibration: late-stage only (Stage 3+) broadband increase in ACC-3
        vib_effects = {}
        if stage in ("stage3", "stage4"):
            vib_effects = {
                "acc3_broadband_energy": ("multiply", 1.0 + severity * 3.0),
                "acc3_rms": ("multiply", 1.0 + severity * 1.5),
            }

        return FaultEffect(
            vibration_effects=vib_effects,
            thermal_effects={
                "t4_turbo_factor": degradation_factor,  # Special key for turbo model
            },
        )
