# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive Maintenance System for a fleet of 200 commercial diesel trucks. Uses edge computing (real-time classification) and cloud analytics (remaining useful life prediction) to detect 8 failure modes from vibration and temperature sensor data.

The complete technical specification lives in `docs/Predictive Maintenance System for Commercial Truck Fleet CLEAN.docx`. All implementation decisions should be grounded in this document.

## Build & Run Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Run all tests (60 tests, ~32s)
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_thermal_model.py -v

# Run a single test
.venv/bin/python -m pytest tests/test_integration.py::TestIntegration::test_healthy_truck_one_day -v

# Generate 1 truck × 1 day (smoke test)
.venv/bin/python -m src.generator.cli --single-truck 1 --single-day 0 --output-dir output/test/

# Validation checkpoint (10 trucks × 1 day, controlled faults)
.venv/bin/python -m src.generator.cli --validation-checkpoint --output-dir output/validation/
.venv/bin/python -m src.validation.range_checks output/validation/
.venv/bin/python -m src.validation.progression_checks output/validation/
.venv/bin/python -m src.validation.cross_feature output/validation/

# Full fleet generation (200 trucks × 183 days), resumable
.venv/bin/python -m src.generator.cli --trucks 200 --days 183 --seed 42 --output-dir output/full/ --workers 8 --skip-existing
```

## Current Progress

### Phase 1: Synthetic Data Generator

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project setup (requirements, constants, schema) | DONE |
| 2 | Fleet foundation (bearing geometry, profiles, trucks, factory) | DONE |
| 3 | Operating state simulation (Markov chain, RPM/load, ambient) | DONE |
| 4 | Fault modeling (degradation, 8 fault modes, schedule) | DONE |
| 5 | Feature synthesis (vibration, thermal, conditioning, vector) | DONE |
| 6 | Labels and storage (ground truth, Parquet, thermal state) | DONE |
| 7 | Orchestration (generator, batch, CLI, validation, tests) | DONE |
| 8 | Validation checkpoint (10 trucks × 1 day, 3 validators) | DONE — all pass |
| 9 | Full fleet generation (200 trucks × 183 days) | **NEXT** |

**Tests:** 60/60 passing. **Validation:** 12/12 range checks + progression + cross-feature all pass.

**Step 9 command:**
```bash
.venv/bin/python -m src.generator.cli --trucks 200 --days 183 --seed 42 --output-dir output/full/ --workers 8 --skip-existing
```

### Phases 2-4: Not started

- **Phase 2:** Full dataset generation, XGBoost (Path A) + CNN-LSTM-Attention (Path B) training, hyperparameter tuning
- **Phase 3:** Sacred test set evaluation, ONNX export, production artifacts
- **Phase 4:** Optional field pilot

## Architecture

### Dual-Path ML System

- **Path A (Edge):** XGBoost classifier for real-time fault detection. Must run <50ms latency. Exported via ONNX + TensorRT INT8 quantization.
- **Path B (Cloud):** CNN-LSTM-Attention network for Remaining Useful Life (RUL) prediction. Target RMSE <15 hours.

### Synthetic Data Generator (Phase 1)

Generates 221-feature vectors per 60-second window for 200 trucks over 183 days.

**Feature breakdown:** 2 conditioning (RPM est, load proxy) + 180 vibration (3 sensors × per-axis time/freq/SK with aggregation stats) + 39 thermal (6 sensors × 6 stats + 3 differentials) = 221 features. Output Parquet has 229 columns (4 metadata + 221 features + 4 labels).

**Seed flow (deterministic):** `master_seed → truck.seed (master + truck_id) → day_seed (truck_seed * 1000 + day) → np.random.default_rng(day_seed)` created fresh per generate() call. No mutable RNG state crosses process boundaries.

**Thermal model:** First-order lag with physical clamping. End-of-day state persisted as JSON for next-day continuity. Multi-fault thermal offsets capped per sensor.

**Fault effects:** Vibration energy features multiply, shape features (kurtosis, SK) take max. FM-07 leak events use deterministic hash (not mutable RNG) for multiprocessing safety.

### 8 Failure Modes (FM-01 through FM-08)

Bearing wear, cooling degradation, valve train wear, oil degradation, turbo degradation, injector wear, EGR cooler failure, DPF blockage.

### 3-Tier Storage

1. **Edge:** Ring buffer with 24h raw sensor retention per truck
2. **Cloud:** TimescaleDB feature store with 6-month history (~90% compression target)
3. **Archive:** Optional long-term raw data

## Key Conventions

- **Evaluation splits are truck-based** (120 train / 50 val / 30 test sacred), not time-based
- **Extrapolations E1-E19** mark estimates that need field validation — track these in code
- **Feature extraction tolerance:** ±20% vs specification values
- **Performance targets:** Path A F1 ≥ 0.85, Path B RMSE < 15 hours, edge latency < 50ms
- **Feature count contract:** 221 features enforced by assertion in `feature_vector.py`
- **Label safety:** Labels computed from fault internal state (severity/stage/RUL), never from generated features

## Key Bugs Fixed (reference for future debugging)

- **`sample_thermal_baselines`** was ignoring the `delta_load` spec range — derived it from `cruise - idle` instead of sampling it. Fixed to sample `delta_load` from spec range directly.
- **FM-07 mutable RNG** caused non-determinism under multiprocessing fork. Replaced with deterministic hash function.
- **Degradation model** original Wiener process had noise dominating drift. Rewrote to logistic growth curve + bounded mean-reverting noise.
