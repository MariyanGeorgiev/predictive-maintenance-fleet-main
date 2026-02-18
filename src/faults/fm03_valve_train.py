"""FM-03: Valve Train Wear - increased 500-2000 Hz energy (spec §10.2)."""

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class ValveTrainWearFault(FaultMode):
    """FM-03: Valve train wear. Increases impact energy in 500-2000 Hz band."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        energy_multiplier_max: float = 5.0,
        kurtosis_increase_max: float = 2.0,
    ):
        super().__init__("FM-03", onset_hours, degradation, total_life_hours)
        self.energy_multiplier_max = energy_multiplier_max
        self.kurtosis_increase_max = kurtosis_increase_max

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        effects = {
            # Energy increase in mid-low band (500-2000 Hz) for ACC-1 and ACC-2
            "acc1_mid_low_energy": ("multiply", 1.0 + severity * self.energy_multiplier_max),
            "acc2_mid_low_energy": ("multiply", 1.0 + severity * self.energy_multiplier_max),
            # Kurtosis increase from impact events
            "acc1_kurtosis": ("add", severity * self.kurtosis_increase_max),
            "acc2_kurtosis": ("add", severity * self.kurtosis_increase_max),
            # Slight RMS increase
            "acc1_rms": ("multiply", 1.0 + severity * 0.5),
            "acc2_rms": ("multiply", 1.0 + severity * 0.5),
        }

        return FaultEffect(vibration_effects=effects, thermal_effects={})
