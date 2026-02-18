"""Fleet factory: creates 200 trucks with stratified train/val/test splits."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config.constants import FLEET_SIZE, MODERN_FRACTION, SPLIT_RATIOS
from src.config.schema import FleetMetadata
from src.fleet.engine_profile import modern_diesel_profile, older_diesel_profile
from src.fleet.truck import Truck


def create_fleet(
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[List[Truck], FleetMetadata]:
    """Create a fleet of 200 trucks with stratified train/val/test splits.

    Stratification ensures each split maintains ~80/20 modern/older engine ratio.

    Args:
        seed: Master RNG seed for reproducibility.
        output_dir: If provided, writes split metadata files.

    Returns:
        (trucks, metadata) tuple.
    """
    rng = np.random.default_rng(seed)

    n_modern = int(FLEET_SIZE * MODERN_FRACTION)
    n_older = FLEET_SIZE - n_modern

    # Create engine types array and shuffle
    engine_types = ["modern"] * n_modern + ["older"] * n_older

    # Stratified split: maintain engine ratio in each split
    modern_ids = list(range(1, n_modern + 1))
    older_ids = list(range(n_modern + 1, FLEET_SIZE + 1))
    rng.shuffle(modern_ids)
    rng.shuffle(older_ids)

    # Split modern trucks proportionally (120/50/30 = 96/40/24 modern)
    total = sum(SPLIT_RATIOS.values())
    m_train = int(n_modern * SPLIT_RATIOS["train"] / total)
    m_val = int(n_modern * SPLIT_RATIOS["val"] / total)
    m_test = n_modern - m_train - m_val

    # Split older trucks proportionally (24/10/6 older)
    o_train = SPLIT_RATIOS["train"] - m_train
    o_val = SPLIT_RATIOS["val"] - m_val
    o_test = SPLIT_RATIOS["test"] - m_test

    # Assign splits
    split_map = {}
    idx = 0
    for tid in modern_ids[:m_train]:
        split_map[tid] = "train"
    for tid in modern_ids[m_train:m_train + m_val]:
        split_map[tid] = "val"
    for tid in modern_ids[m_train + m_val:]:
        split_map[tid] = "test"

    for tid in older_ids[:o_train]:
        split_map[tid] = "train"
    for tid in older_ids[o_train:o_train + o_val]:
        split_map[tid] = "val"
    for tid in older_ids[o_train + o_val:]:
        split_map[tid] = "test"

    # Create truck objects
    trucks = []
    for truck_id in range(1, FLEET_SIZE + 1):
        engine_type = "modern" if truck_id <= n_modern else "older"
        truck_rng = np.random.default_rng(seed + truck_id)

        if engine_type == "modern":
            profile = modern_diesel_profile(truck_rng)
        else:
            profile = older_diesel_profile(truck_rng)

        trucks.append(Truck(
            truck_id=truck_id,
            engine_type=engine_type,
            profile=profile,
            seed=seed + truck_id,
            split=split_map[truck_id],
        ))

    # Build metadata
    train_ids = sorted(t.truck_id for t in trucks if t.split == "train")
    val_ids = sorted(t.truck_id for t in trucks if t.split == "val")
    test_ids = sorted(t.truck_id for t in trucks if t.split == "test")

    metadata = FleetMetadata(
        total_trucks=FLEET_SIZE,
        modern_count=n_modern,
        older_count=n_older,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=seed,
    )

    # Write metadata files if output_dir provided
    if output_dir is not None:
        meta_dir = Path(output_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            (meta_dir / f"{split_name}_trucks.txt").write_text(
                "\n".join(str(i) for i in ids) + "\n"
            )

        stratification = {
            "train": {"total": len(train_ids),
                      "modern": sum(1 for t in trucks if t.split == "train" and t.engine_type == "modern"),
                      "older": sum(1 for t in trucks if t.split == "train" and t.engine_type == "older")},
            "val":   {"total": len(val_ids),
                      "modern": sum(1 for t in trucks if t.split == "val" and t.engine_type == "modern"),
                      "older": sum(1 for t in trucks if t.split == "val" and t.engine_type == "older")},
            "test":  {"total": len(test_ids),
                      "modern": sum(1 for t in trucks if t.split == "test" and t.engine_type == "modern"),
                      "older": sum(1 for t in trucks if t.split == "test" and t.engine_type == "older")},
        }
        (meta_dir / "fleet_stratification.json").write_text(
            json.dumps(stratification, indent=2) + "\n"
        )

    return trucks, metadata
