"""Ground truth label computation (spec §10.7)."""

from dataclasses import dataclass
from typing import List

from src.faults.fault_mode import FaultMode


@dataclass
class GroundTruthLabel:
    fault_mode: str       # "HEALTHY" or "FM-01" through "FM-08" (worst active)
    fault_severity: str   # "HEALTHY", "STAGE_2", "STAGE_3", "STAGE_4"
    rul_hours: float      # Remaining useful life (of worst fault)
    path_a_label: str     # "NORMAL", "IMMINENT", "CRITICAL"


def compute_label(t_hours: float, active_faults: List[FaultMode]) -> GroundTruthLabel:
    """Compute ground truth label from active faults at time t.

    For multi-fault trucks, the worst (most advanced) fault drives the label.
    """
    if not active_faults:
        return GroundTruthLabel(
            fault_mode="HEALTHY",
            fault_severity="HEALTHY",
            rul_hours=float("inf"),
            path_a_label="NORMAL",
        )

    # Find the worst fault (most advanced stage, then lowest RUL)
    worst_fault = None
    worst_stage_rank = -1
    worst_rul = float("inf")

    stage_ranks = {"healthy": 0, "stage2": 1, "stage3": 2, "stage4": 3}

    for fault in active_faults:
        stage = fault.current_stage(t_hours)
        rank = stage_ranks.get(stage, 0)
        rul = fault.current_rul(t_hours)

        if rank > worst_stage_rank or (rank == worst_stage_rank and rul < worst_rul):
            worst_fault = fault
            worst_stage_rank = rank
            worst_rul = rul

    if worst_fault is None or worst_stage_rank == 0:
        return GroundTruthLabel(
            fault_mode="HEALTHY",
            fault_severity="HEALTHY",
            rul_hours=worst_rul if worst_fault else float("inf"),
            path_a_label="NORMAL",
        )

    stage = worst_fault.current_stage(t_hours)
    severity_map = {
        "healthy": "HEALTHY",
        "stage2": "STAGE_2",
        "stage3": "STAGE_3",
        "stage4": "STAGE_4",
    }

    return GroundTruthLabel(
        fault_mode=worst_fault.fault_id,
        fault_severity=severity_map.get(stage, "HEALTHY"),
        rul_hours=worst_fault.current_rul(t_hours),
        path_a_label=worst_fault.path_a_label(t_hours),
    )
