"""Bearing fault frequency calculations (spec §10.2.2).

Formulas:
    BPFO = (N_b/2) * f_shaft * [1 - (B_d/P_d) * cos(β)]
    BPFI = (N_b/2) * f_shaft * [1 + (B_d/P_d) * cos(β)]
    BSF  = (P_d/(2*B_d)) * f_shaft * [1 - (B_d/P_d)² * cos²(β)]
    FTF  = (f_shaft/2) * [1 - (B_d/P_d) * cos(β)]
"""

import math

from src.config.schema import BearingGeometry


def _ratio_cos(bg: BearingGeometry) -> float:
    """Compute (B_d / P_d) * cos(β)."""
    return (bg.ball_dia_mm / bg.pitch_dia_mm) * math.cos(math.radians(bg.contact_angle_deg))


def shaft_frequency(rpm: float) -> float:
    """Convert RPM to shaft frequency in Hz."""
    return rpm / 60.0


def bpfo(bg: BearingGeometry, rpm: float) -> float:
    """Ball Pass Frequency Outer race (Hz)."""
    f_shaft = shaft_frequency(rpm)
    return (bg.n_balls / 2.0) * f_shaft * (1.0 - _ratio_cos(bg))


def bpfi(bg: BearingGeometry, rpm: float) -> float:
    """Ball Pass Frequency Inner race (Hz)."""
    f_shaft = shaft_frequency(rpm)
    return (bg.n_balls / 2.0) * f_shaft * (1.0 + _ratio_cos(bg))


def bsf(bg: BearingGeometry, rpm: float) -> float:
    """Ball Spin Frequency (Hz)."""
    f_shaft = shaft_frequency(rpm)
    rc = _ratio_cos(bg)
    return (bg.pitch_dia_mm / (2.0 * bg.ball_dia_mm)) * f_shaft * (1.0 - rc * rc)


def ftf(bg: BearingGeometry, rpm: float) -> float:
    """Fundamental Train (cage) Frequency (Hz)."""
    f_shaft = shaft_frequency(rpm)
    return (f_shaft / 2.0) * (1.0 - _ratio_cos(bg))
