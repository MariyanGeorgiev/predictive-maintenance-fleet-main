"""Truck dataclass representing a single truck in the fleet."""

from dataclasses import dataclass

from src.config.schema import EngineProfile


@dataclass
class Truck:
    truck_id: int
    engine_type: str          # "modern" or "older"
    profile: EngineProfile
    seed: int                 # Deterministic RNG seed for this truck
    split: str                # "train", "val", or "test"
