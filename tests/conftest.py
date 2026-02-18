"""Shared test fixtures."""

import numpy as np
import pytest

from src.config.schema import BearingGeometry, create_engine_profile
from src.fleet.truck import Truck


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def modern_profile(rng):
    return create_engine_profile("modern", rng)


@pytest.fixture
def older_profile(rng):
    return create_engine_profile("older", rng)


@pytest.fixture
def modern_truck(modern_profile):
    return Truck(
        truck_id=1,
        engine_type="modern",
        profile=modern_profile,
        seed=42,
        split="train",
    )


@pytest.fixture
def older_truck(older_profile):
    return Truck(
        truck_id=161,
        engine_type="older",
        profile=older_profile,
        seed=203,
        split="train",
    )


@pytest.fixture
def modern_bearing():
    return BearingGeometry.modern_default()


@pytest.fixture
def older_bearing():
    return BearingGeometry.older_default()
