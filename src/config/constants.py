"""Physical constants, frequency ranges, thresholds, and transition matrices from the spec."""

import numpy as np

# =============================================================================
# Operating Modes
# =============================================================================

OPERATING_MODES = ["idle", "city", "cruise", "heavy"]

# Markov chain transition matrix (spec §10.4)
# Rows: from state, Columns: to state
# Order: idle, city, cruise, heavy
TRANSITION_MATRIX = np.array([
    [0.70, 0.25, 0.04, 0.01],  # from idle
    [0.10, 0.60, 0.25, 0.05],  # from city
    [0.02, 0.15, 0.75, 0.08],  # from cruise
    [0.05, 0.20, 0.70, 0.05],  # from heavy
])

# RPM ranges per operating mode (spec §10.4)
RPM_RANGES = {
    "idle":   {"modern": (600, 800),   "older": (600, 800)},
    "city":   {"modern": (1000, 1400), "older": (1000, 1400)},
    "cruise": {"modern": (1400, 1550), "older": (1500, 1700)},
    "heavy":  {"modern": (1600, 2100), "older": (1600, 2100)},
}

# Load ranges (normalized) per operating mode (spec §10.4)
LOAD_RANGES = {
    "idle":   (0.0, 0.1),
    "city":   (0.2, 0.5),
    "cruise": (0.6, 0.9),
    "heavy":  (0.9, 1.2),
}

# =============================================================================
# Sensor Configuration
# =============================================================================

# Vibration sensors
ACC_SENSORS = {
    "acc1": {"location": "front_main_bearing", "sample_rate_hz": 50_000, "axes": 3},
    "acc2": {"location": "rear_main_bearing",  "sample_rate_hz": 50_000, "axes": 3},
    "acc3": {"location": "turbocharger",       "sample_rate_hz": 10_000, "axes": 3},
}

# Temperature sensors
TEMP_SENSORS = {
    "t1": {"measurement": "engine_coolant_outlet", "range_c": (0, 120)},
    "t2": {"measurement": "engine_oil",            "range_c": (0, 150)},
    "t3": {"measurement": "egt_pre_turbo",         "range_c": (0, 900)},
    "t4": {"measurement": "egt_post_turbo",        "range_c": (0, 700)},
    "t5": {"measurement": "egr_cooler_outlet",     "range_c": (0, 600)},
    "t6": {"measurement": "intake_manifold",       "range_c": (0, 200)},
}

AXES = ["x", "y", "z"]

# =============================================================================
# Vibration Feature Frequency Bands (spec §9.1)
# =============================================================================

# ACC-1, ACC-2 bands (50 kHz sampling)
ACC12_BANDS = {
    "low":      (0, 500),       # Shaft imbalance, misalignment
    "mid_low":  (500, 2000),    # Valve train (FM-03)
    "mid_high": (2000, 10000),  # Bearing faults (FM-01)
    "high":     (10000, 25000), # Injector faults (FM-06)
}

# ACC-3 bands (10 kHz sampling)
ACC3_BANDS = {
    "low":       (0, 1000),    # Shaft imbalance
    "broadband": (1000, 5000), # Turbo journal bearing (FM-05)
}

# =============================================================================
# Window Parameters (spec §9)
# =============================================================================

VIBRATION_WINDOW_SIZE = 2048       # samples (power of 2 for FFT)
VIBRATION_OVERLAP = 0.5            # 50%
THERMAL_WINDOW_SIZE_SEC = 60       # seconds
THERMAL_OVERLAP = 0.0              # no overlap

# Windows per 60-second aggregation period
# ACC-1/ACC-2: 50kHz, hop = 1024 samples, 60s = 3_000_000 samples -> ~2929 windows
# ACC-3: 10kHz, hop = 1024 samples, 60s = 600_000 samples -> ~585 windows
WINDOWS_PER_AGG_ACC12 = 2929
WINDOWS_PER_AGG_ACC3 = 585

WINDOWS_PER_DAY = 1440  # 24h * 60min = 1440 sixty-second windows

# =============================================================================
# Bearing Geometry Defaults (spec §10.2.2)
# =============================================================================

BEARING_GEOMETRY_MODERN = {
    "n_balls": 12,
    "ball_dia_mm": 20.0,
    "pitch_dia_mm": 120.0,
    "contact_angle_deg": 0.0,
}

BEARING_GEOMETRY_OLDER = {
    "n_balls": 10,
    "ball_dia_mm": 18.0,
    "pitch_dia_mm": 110.0,
    "contact_angle_deg": 0.0,
}

# =============================================================================
# Thermal Baselines (spec §10.3.1)
# T_base at idle, ΔT_load to cruise, thermal time constant τ
# =============================================================================

THERMAL_BASELINES = {
    "modern": {
        "t1": {"idle": (60, 70),   "cruise": (85, 95),   "delta_load": (25, 35),  "tau": (60, 120)},
        "t2": {"idle": (70, 80),   "cruise": (95, 110),  "delta_load": (25, 40),  "tau": (90, 180)},
        "t3": {"idle": (150, 200), "cruise": (315, 482),  "delta_load": (240, 350), "tau": (15, 30)},
        "t4": {"idle": (100, 130), "cruise": (110, 160),  "delta_load": (5, 30),    "tau": (20, 40)},
        "t5": {"idle": (80, 100),  "cruise": (150, 250),  "delta_load": (70, 180),  "tau": (30, 60)},
        "t6": {"idle": (30, 40),   "cruise": (50, 80),    "delta_load": (20, 50),   "tau": (10, 20)},
    },
    "older": {
        "t1": {"idle": (65, 75),   "cruise": (85, 95),   "delta_load": (25, 35),  "tau": (60, 120)},
        "t2": {"idle": (80, 90),   "cruise": (90, 110),  "delta_load": (25, 40),  "tau": (90, 180)},
        "t3": {"idle": (160, 210), "cruise": (300, 500),  "delta_load": (240, 350), "tau": (15, 30)},
        "t4": {"idle": (110, 140), "cruise": (120, 170),  "delta_load": (5, 30),    "tau": (20, 40)},
        "t5": {"idle": (90, 110),  "cruise": (160, 240),  "delta_load": (70, 180),  "tau": (30, 60)},
        "t6": {"idle": (35, 45),   "cruise": (50, 75),    "delta_load": (20, 50),   "tau": (10, 20)},
    },
}

# Turbo delta baselines (T3 - T4) at cruise (spec §10.2.1)
TURBO_DELTA_BASELINE = {
    "modern": (200, 280),  # degrees C
    "older":  (150, 250),
}

# =============================================================================
# Degradation Parameters (spec §10.2.3)
# =============================================================================

# Bearing degradation (FM-01)
BEARING_DEGRADATION = {
    "modern": {
        "lambda_range": (0.0001, 0.0003),  # base rate per hour
        "sigma_range":  (0.05, 0.15),       # stochasticity
        "t_stage2_hours": (2000, 4000),     # time to Stage 2
        "dt_23_hours":    (200, 500),       # Stage 2→3 duration
        "dt_34_hours":    (50, 150),        # Stage 3→4 duration
    },
    "older": {
        "lambda_range": (0.0002, 0.0005),
        "sigma_range":  (0.10, 0.20),
        "t_stage2_hours": (1500, 3000),
        "dt_23_hours":    (150, 400),
        "dt_34_hours":    (30, 100),
    },
}

# Bearing vibration stages (spec §10.2.3)
BEARING_STAGES = {
    "stage1": {"life_pct": (0.0, 0.60), "rms": (0.05, 0.15), "kurtosis": (2.5, 3.5), "sk": (1.0, 5.0)},
    "stage2": {"life_pct": (0.60, 0.75), "rms": (0.15, 0.30), "kurtosis": (4.0, 6.0), "sk": (5.0, 8.0)},
    "stage3": {"life_pct": (0.75, 0.95), "rms": (0.30, 1.50), "kurtosis": (6.0, 10.0), "sk": (10.0, 20.0)},
    "stage4": {"life_pct": (0.95, 1.00), "rms": (1.50, 5.00), "kurtosis": (3.0, 5.0), "sk": (5.0, 8.0)},
}

# Thermal fault progression rates (spec §10.3.2, E16)
THERMAL_FAULT_PARAMS = {
    "fm02_cooling": {"delta_t1_max": (10, 30), "progression_hours": (500, 1500)},
    "fm04_oil":     {"delta_t2_max": (10, 30), "progression_hours": (500, 1500)},
    "fm05_turbo":   {"degradation_factor_max": (0.2, 0.4), "progression_hours": (500, 1000)},
    "fm06_injector": {"delta_t3_max": (30, 80), "delta_t_injector_full": (50, 100), "progression_hours": (1000, 2000)},
    "fm07_egr_fouling": {"delta_t5_max": (20, 60), "progression_hours": (500, 1500)},
    "fm07_egr_leak":    {"delta_t1_spike": (10, 30), "delta_t5_spike": (30, 80), "duration_sec": (30, 120)},
    "fm08_dpf":     {"delta_t3_max": (100, 200), "regen_interval_hours": (200, 400), "progression_hours": (100, 500)},
}

# Valve train vibration params (FM-03)
VALVE_TRAIN_PARAMS = {
    "band": "mid_low",  # 500-2000 Hz
    "energy_multiplier_max": (3.0, 8.0),
    "kurtosis_increase_max": (1.0, 3.0),
    "progression_hours": (1000, 3000),
}

# =============================================================================
# Vibration Baselines (spec §10.2)
# =============================================================================

HEALTHY_VIBRATION = {
    "acc1": {"rms_base": (0.05, 0.15), "kurtosis_base": 3.0, "crest_factor_base": (2.5, 4.0)},
    "acc2": {"rms_base": (0.05, 0.15), "kurtosis_base": 3.0, "crest_factor_base": (2.5, 4.0)},
    "acc3": {"rms_base": (0.02, 0.08), "kurtosis_base": 3.0, "crest_factor_base": (2.5, 4.0)},
}

# Vibration resonance parameters for bearing fault model
BEARING_RESONANCE_FREQ = (3000, 8000)   # Hz
BEARING_DAMPING_RATIO = (0.05, 0.15)

# =============================================================================
# Fleet Configuration
# =============================================================================

FLEET_SIZE = 200
MODERN_FRACTION = 0.80  # 160 modern, 40 older
SIMULATION_DAYS = 183   # 6 months
HOURS_PER_DAY = 24
DUTY_CYCLE_HOURS = 15   # hours of operation per day (E3)

# Train/val/test split (spec §9.5.1, E19)
SPLIT_RATIOS = {"train": 120, "val": 50, "test": 30}

# Fault assignment distribution
FAULT_DISTRIBUTION = {
    "healthy": 0.30,
    "single_fault": 0.40,
    "double_fault": 0.20,
    "triple_fault": 0.10,
}

# =============================================================================
# Noise Parameters (spec E15)
# =============================================================================

VIBRATION_NOISE_FRACTION = 0.10  # 5-15% relative noise on features
THERMAL_NOISE_STD = 1.0          # degrees C sensor noise

# =============================================================================
# EGT Reference Thresholds (spec §10.2.1)
# =============================================================================

EGT_ALARM_THRESHOLD = 677     # °C - max safe sustained T3
DPF_REGEN_TEMP = 538           # °C - normal transient during regen
T3_EXCEEDANCE_THRESHOLD = 677  # °C - for T3 exceedance duration feature

# =============================================================================
# Ambient Temperature Model
# =============================================================================

AMBIENT_TEMP_MEAN = 15.0       # °C annual mean (temperate climate)
AMBIENT_TEMP_SEASONAL_AMP = 15.0  # °C seasonal amplitude
AMBIENT_TEMP_DAILY_AMP = 5.0   # °C daily amplitude
AMBIENT_T_REF = 25.0           # °C reference temperature for thermal model

# =============================================================================
# Feature Validation Thresholds (spec §10.6)
# =============================================================================

VALIDATION_TOLERANCE = 0.20  # ±20%

VALIDATION_RANGES = {
    "healthy": {
        "acc1_rms": (0.05, 0.15),
        "acc1_kurtosis": (2.5, 4.0),
        "acc1_sk_max": (1.0, 5.0),
        "acc1_high_band_energy": "low",
        "t3_mean_cruise": (315, 482),
        "t3_t4_delta": (200, 280),
    },
    "fm01_stage3": {
        "acc1_rms": (0.30, 1.50),
        "acc1_kurtosis": (6.0, 10.0),
        "acc1_sk_max": (10.0, 20.0),
        "t3_mean_cruise": (315, 482),  # unchanged
    },
    "fm06_degraded": {
        "acc1_rms": (0.10, 0.25),
        "acc1_kurtosis": (3.0, 5.0),
        "acc1_high_band_energy": "high",
        "t3_mean_cruise": (400, 530),
    },
}
