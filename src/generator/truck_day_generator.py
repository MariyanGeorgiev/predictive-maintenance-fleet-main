"""Orchestrate generation of one truck × one day (1440 windows)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config.constants import HOURS_PER_DAY, WINDOWS_PER_DAY
from src.config.schema import EngineProfile
from src.faults.fault_mode import FaultMode
from src.features.conditioning import compute_conditioning_features
from src.features.feature_vector import assemble_feature_dict
from src.features.thermal_features import synthesize_thermal_features
from src.features.vibration_features import synthesize_vibration_features
from src.labels.ground_truth import GroundTruthLabel, compute_label
from src.simulation.ambient import AmbientTemperatureModel
from src.simulation.markov_chain import MarkovChainSimulator
from src.simulation.operating_state import generate_rpm_load

logger = logging.getLogger(__name__)


class TruckDayGenerator:
    """Generates feature data for one truck on one day."""

    def __init__(self, ambient_model: Optional[AmbientTemperatureModel] = None):
        self.markov = MarkovChainSimulator()
        self.ambient = ambient_model or AmbientTemperatureModel()

    def generate(
        self,
        profile: EngineProfile,
        engine_type: str,
        day_index: int,
        faults: List[FaultMode],
        initial_temps: Dict[str, float],
        seed: int,
    ) -> Tuple[List[Dict[str, float]], List[GroundTruthLabel], Dict[str, float]]:
        """Generate 1440 feature windows for one truck-day.

        Args:
            profile: Engine profile.
            engine_type: "modern" or "older".
            day_index: Day number (0-182).
            faults: Active fault modes for this truck.
            initial_temps: T1-T6 starting temperatures.
            seed: RNG seed for this truck-day.

        Returns:
            (features_list, labels_list, final_temps) where:
            - features_list: 1440 feature dicts
            - labels_list: 1440 GroundTruthLabel objects
            - final_temps: End-of-day thermal state
        """
        rng = np.random.default_rng(seed)

        # 1. Generate operating mode sequence
        # Start with cruise (most common) unless day_index=0
        initial_state = 2 if day_index > 0 else 0
        modes = self.markov.simulate_day(rng, initial_state=initial_state)

        # 2. Generate RPM and load
        rpm_array, load_array = generate_rpm_load(modes, engine_type, rng)

        # 3. Generate features for each 60-second window
        features_list = []
        labels_list = []
        prev_temps = dict(initial_temps)

        for window_idx in range(WINDOWS_PER_DAY):
            t_hours = day_index * HOURS_PER_DAY + window_idx / 60.0
            rpm = rpm_array[window_idx]
            load = load_array[window_idx]
            second_of_day = window_idx * 60

            # Ambient temperature
            ambient_temp = self.ambient.get_temperature(day_index, second_of_day)

            # Get fault effects
            fault_effects = [f.get_effects(t_hours, rpm, load) for f in faults]

            # Synthesize vibration features (132+)
            vib_features = synthesize_vibration_features(rpm, load, fault_effects, rng)

            # Synthesize thermal features (39)
            thermal_features, prev_temps = synthesize_thermal_features(
                rpm, load, profile, ambient_temp, fault_effects, prev_temps, rng,
            )

            # Conditioning features (2)
            cond_features = compute_conditioning_features(
                rpm, load, thermal_features.get("t3_mean", 300.0), engine_type, rng,
            )

            # Assemble full feature vector
            feature_dict = assemble_feature_dict(cond_features, vib_features, thermal_features)
            features_list.append(feature_dict)

            # Compute ground truth label from fault internal state (severity timeline),
            # NOT from generated features — this ordering prevents label leakage.
            label = compute_label(t_hours, faults)
            labels_list.append(label)

        return features_list, labels_list, prev_temps
