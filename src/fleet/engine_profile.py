"""Engine profile factory functions."""

import numpy as np

from src.config.schema import EngineProfile, create_engine_profile


def modern_diesel_profile(rng: np.random.Generator) -> EngineProfile:
    """Create a modern turbocharged inline-6 diesel profile."""
    return create_engine_profile("modern", rng)


def older_diesel_profile(rng: np.random.Generator) -> EngineProfile:
    """Create an older turbocharged inline-6 diesel profile."""
    return create_engine_profile("older", rng)
