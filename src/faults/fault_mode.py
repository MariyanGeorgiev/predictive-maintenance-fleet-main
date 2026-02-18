"""Base fault mode abstract class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from src.faults.degradation_model import DegradationModel


@dataclass
class FaultEffect:
    """Container for fault effects on features.

    vibration_effects: Dict mapping feature keys to (mode, value) where mode is
        "multiply" (multiply base value) or "add" (add to base value) or "set" (override).
    thermal_effects: Dict mapping sensor name to temperature offset in °C.
    """
    vibration_effects: Dict[str, tuple]  # key -> ("multiply"|"add"|"set", value)
    thermal_effects: Dict[str, float]     # sensor -> temp offset °C


class FaultMode(ABC):
    """Abstract base class for all 8 failure modes."""

    def __init__(
        self,
        fault_id: str,
        onset_hours: float,
        degradation: DegradationModel,
        total_life_hours: float,
    ):
        self.fault_id = fault_id
        self.onset_hours = onset_hours
        self.degradation = degradation
        self.total_life_hours = total_life_hours

    def time_since_onset(self, t_hours: float) -> float:
        """Hours elapsed since fault onset."""
        return max(0.0, t_hours - self.onset_hours)

    def current_severity(self, t_hours: float) -> float:
        """Current fault severity [0, 1]."""
        dt = self.time_since_onset(t_hours)
        if dt <= 0:
            return 0.0
        return self.degradation.severity_at(dt)

    def current_stage(self, t_hours: float) -> str:
        """Current degradation stage."""
        dt = self.time_since_onset(t_hours)
        return self.degradation.stage_at(dt, self.total_life_hours)

    def current_rul(self, t_hours: float) -> float:
        """Remaining useful life in hours (time until Stage 4 end)."""
        end_time = self.onset_hours + self.total_life_hours
        return max(0.0, end_time - t_hours)

    @abstractmethod
    def get_effects(self, t_hours: float, rpm: float, load: float) -> FaultEffect:
        """Compute fault effects on features at current time.

        Args:
            t_hours: Absolute simulation time in hours.
            rpm: Current engine RPM.
            load: Current normalized load (0-1.2).

        Returns:
            FaultEffect with vibration and thermal modifications.
        """
        ...

    def path_a_label(self, t_hours: float) -> str:
        """Path A classification label based on current stage."""
        stage = self.current_stage(t_hours)
        if stage in ("healthy", "stage2"):
            return "NORMAL"
        elif stage == "stage3":
            # Early vs late Stage 3
            dt = self.time_since_onset(t_hours)
            life_pct = dt / self.total_life_hours if self.total_life_hours > 0 else 1.0
            if life_pct < 0.85:
                return "IMMINENT"
            else:
                return "CRITICAL"
        else:  # stage4
            return "CRITICAL"
