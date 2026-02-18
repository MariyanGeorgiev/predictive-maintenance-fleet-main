"""FM-01: Bearing Wear - 4-stage vibration model (spec §10.2.3).

Stage 1 (Healthy, 0-60%): RMS 0.05-0.15g, Kurtosis ~3, SK <5
Stage 2 (Early, 60-75%): RMS 0.15-0.3g, Kurtosis 4-6, SK 5-8
Stage 3 (Propagation, 75-95%): RMS 0.3-1.5g, Kurtosis 6-10, SK 10+
Stage 4 (Accelerated, 95-100%): RMS 1.5-5g, Kurtosis 3-5 (drops), SK 5-8 (drops)
"""

import numpy as np

from src.config.constants import BEARING_STAGES
from src.faults.degradation_model import DegradationModel
from src.faults.fault_mode import FaultEffect, FaultMode


class BearingWearFault(FaultMode):
    """FM-01: Bearing wear affecting ACC-1 and ACC-2."""

    def __init__(
        self,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
        affected_sensor: str = "acc1",  # "acc1" or "acc2"
    ):
        super().__init__("FM-01", onset_hours, degradation, total_life_hours)
        self.affected_sensor = affected_sensor

    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        stage = self.current_stage(t_hours)
        severity = self.current_severity(t_hours)

        if stage == "healthy" or severity <= 0:
            return FaultEffect(vibration_effects={}, thermal_effects={})

        # Get stage-specific target values
        stage_params = BEARING_STAGES[stage]
        rms_lo, rms_hi = stage_params["rms"]
        kurt_lo, kurt_hi = stage_params["kurtosis"]
        sk_lo, sk_hi = stage_params["sk"]

        # Interpolate within stage based on severity
        # severity grows through the stage, use it to blend within the range
        frac = min(1.0, severity)
        rms_target = rms_lo + frac * (rms_hi - rms_lo)
        kurt_target = kurt_lo + frac * (kurt_hi - kurt_lo)
        sk_target = sk_lo + frac * (sk_hi - sk_lo)

        # Load scaling: vibration energy increases with load
        load_factor = 0.7 + 0.3 * load

        sensor = self.affected_sensor
        effects = {
            f"{sensor}_rms": ("set", rms_target * load_factor),
            f"{sensor}_kurtosis": ("set", kurt_target),
            f"{sensor}_sk_max": ("set", sk_target),
            f"{sensor}_crest_factor": ("set", rms_target * 3.0 / max(rms_target, 0.01)),
            # Concentrate energy in mid-high band (2-10 kHz bearing fault band)
            f"{sensor}_mid_high_energy": ("multiply", 1.0 + severity * 10.0),
            # Peak frequency in mid-high band aligns with fault frequency
            f"{sensor}_mid_high_peak_shift": ("set", 1.0),  # flag for feature synth
        }

        return FaultEffect(vibration_effects=effects, thermal_effects={})
