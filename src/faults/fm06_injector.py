"""FM-06: Fuel Injector Wear (spec §10.3.2).

T3 rises + vibration energy increases in 10-25 kHz band.
T3(t) = T3_baseline(load) + ΔT_injector × wear(t)
wear(t) increases from 0.0 to 0.15-0.30 at replacement threshold.
"""

from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class InjectorWearFault(FaultMode):
    """FM-06: Fuel injector wear."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        delta_t3_max: float = 55.0,    # 30-80°C T3 rise
        delta_t_injector: float = 75.0, # 50-100°C full failure penalty
    ):
        super().__init__("FM-06", onset_hours, degradation, total_life_hours)
        self.delta_t3_max = delta_t3_max
        self.delta_t_injector = delta_t_injector

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        severity = self.current_severity(t_hours)
        if severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        # Wear factor (0 to 0.15-0.30)
        wear = severity * 0.22  # midpoint of 0.15-0.30

        # T3 rises with wear
        delta_t3 = self.delta_t_injector * wear

        # Vibration: increased energy in 10-25 kHz (high band) for ACC-1 and ACC-2
        vib_effects = {
            "acc1_high_energy": ("multiply", 1.0 + severity * 5.0),
            "acc2_high_energy": ("multiply", 1.0 + severity * 5.0),
            # Slight RMS increase
            "acc1_rms": ("multiply", 1.0 + severity * 0.3),
            "acc2_rms": ("multiply", 1.0 + severity * 0.3),
            # Mild kurtosis increase
            "acc1_kurtosis": ("add", severity * 1.0),
            "acc2_kurtosis": ("add", severity * 1.0),
        }

        return FaultEffect(
            vibration_effects=vib_effects,
            thermal_effects={"t3": delta_t3},
        )
