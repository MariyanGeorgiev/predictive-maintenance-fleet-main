# Phase 2: Model Training Pipeline - Implementation Plan

## Context

Phase 1 data must be **regenerated** with the maintenance lifecycle model (§10.10) before
Phase 2 training begins. The original dataset (36,600 files, 62/36/2 distribution) has no
repair cycles and is invalid for training.

**Post-regeneration expected data:**
- ~34,800–35,600 Parquet files (200 trucks x 183 days minus repair gaps)
- ~50–51M samples (fewer due to missing repair days)
- ~61–63 GB (snappy compressed)
- Class distribution: NORMAL ~95%, IMMINENT ~4%, CRITICAL ~1%
- Schema: 230 columns (5 metadata incl. episode_id + 221 features + 4 labels)
- Truck-based splits: 120 train / 50 val / 30 test
- Per-truck `maintenance_log.json` in metadata directory

Phase 2 builds two ML models:
- **Path A (Edge):** XGBoost 3-class classifier (NORMAL/IMMINENT/CRITICAL). Target F1 >= 0.85, latency < 50ms. Asymmetric loss alpha=10 (missed failures 10x costlier).
- **Path B (Cloud):** CNN-LSTM-Attention for RUL prediction. Target RMSE < 15 hours. Quantile regression (10th/50th/90th percentiles).

## Prerequisites: Data Regeneration

Before any Phase 2 work, the Phase 1 generator must be updated to implement §10.10:
1. Add maintenance lifecycle state machine (detection, inspection, repair, monitor)
2. Add `episode_id` column to Parquet schema (230 columns)
3. Generate `maintenance_log.json` per truck
4. Skip Parquet generation for repair days (missing data gaps)
5. Handle post-repair thermal state reset and new fault assignment
6. Validate class distribution: NORMAL 93–96%, IMMINENT 3–5%, CRITICAL 0.5–2%
7. Full fleet regeneration (200 trucks x 183 days, ~3.5 hours)

## File Structure

```
src/ml/
├── __init__.py
├── config.py                  # Hyperparameters, paths, constants
├── data/
│   ├── __init__.py
│   ├── dataset_registry.py    # Reads split metadata, lists truck Parquet paths
│   ├── scaler.py              # Welford online StandardScaler (single-pass, no full-data load)
│   ├── path_a_dataset.py      # Flat tabular loader for XGBoost (row-level)
│   └── path_b_dataset.py      # Episode-aware sequence loader for CNN-LSTM (50-window sequences)
├── path_a/
│   ├── __init__.py
│   ├── trainer.py             # XGBoost training with TimeSeriesSplit CV
│   ├── grid_search.py         # Hyperparameter grid search
│   └── export.py              # ONNX export
├── path_b/
│   ├── __init__.py
│   ├── model.py               # CNN-LSTM-Attention architecture (PyTorch)
│   ├── quantile_loss.py       # Pinball loss for quantile regression
│   ├── trainer.py             # Training loop with early stopping
│   └── export.py              # ONNX export
├── evaluate/
│   ├── __init__.py
│   ├── metrics.py             # F1, confusion matrix, RMSE, MAE, coverage
│   └── sacred_test.py         # One-shot evaluation on 30 sacred test trucks
└── cli.py                     # Click CLI: train-a, train-b, evaluate, export
```

## Implementation Steps

### Step 1: Dependencies & Config
- Update `requirements.txt`: add xgboost, torch, onnx, onnxruntime, scikit-learn, matplotlib
- Create `src/ml/config.py`: data paths, feature columns (import from feature_vector.py), hyperparameter defaults, label mappings, episode_id column name

**Files:** `requirements.txt`, `src/ml/__init__.py`, `src/ml/config.py`

### Step 2: Dataset Registry & Scaler
- `dataset_registry.py`: Read `train_trucks.txt`, `val_trucks.txt`, `test_trucks.txt` from metadata dir. Return dict of split -> list of Parquet file paths. Handle missing day files (repair gaps) gracefully — gaps are expected, not errors.
- `scaler.py`: Welford's online algorithm — iterate over training Parquet files one at a time, accumulate running mean/variance for 221 features. `fit()` does single pass, `transform()` applies (x-mean)/std. Save/load as JSON.

**Files:** `src/ml/data/dataset_registry.py`, `src/ml/data/scaler.py`

### Step 3: Path A Data Loading
- `path_a_dataset.py`: Load Parquet files for a split, extract 221 feature columns + `path_a_label`. Map labels to integers (NORMAL=0, IMMINENT=1, CRITICAL=2). Return X (numpy array) and y (numpy array). Lazy loading — iterate files, concat in chunks to limit memory.
- Compute class weights from training set for asymmetric loss. With ~95/4/1 distribution, IMMINENT and CRITICAL are true minority classes — alpha=10 weighting is critical.
- Path A is row-level: episode boundaries do not affect data loading (each row is independent).

**Files:** `src/ml/data/path_a_dataset.py`

### Step 4: Path A Training (XGBoost)
- `trainer.py`: Train XGBoost with `multi:softprob`, custom sample weights (alpha=10 for fault classes). TimeSeriesSplit 5-fold CV on training trucks (split by truck groups, not random). Log per-fold metrics.
- `grid_search.py`: Grid over max_depth (4,6,8), n_estimators (200,500,1000), learning_rate (0.01,0.05,0.1), subsample (0.7,0.8), colsample_bytree (0.7,0.8). Select by macro F1 on val set.
- Train final model on full training set with best params, evaluate on val set.

**Files:** `src/ml/path_a/trainer.py`, `src/ml/path_a/grid_search.py`

### Step 5: Path A Export
- `export.py`: Export trained XGBoost to ONNX format. Verify ONNX inference matches XGBoost predictions.

**Files:** `src/ml/path_a/export.py`

### Step 6: Path B Data Loading
- `path_b_dataset.py`: PyTorch IterableDataset. **Episode-aware**: for each truck, read `episode_id` column and segment data into episodes. Within each episode, create sliding windows of 50 consecutive 60-second samples (50-min lookback). **Sequences must NEVER cross episode boundaries** (§9.6.1). Episodes shorter than 50 windows are either padded with masking or discarded.
- Target: RUL value at the last window step. Yield (sequence_tensor [50, 221], rul_target).
- Shuffle across trucks, sequential within episodes.
- Handle RUL sentinel value (99999.0 = infinity for healthy/improving trucks).

**Files:** `src/ml/data/path_b_dataset.py`

### Step 7: Path B Model Architecture
- `model.py`: CNN-LSTM-Attention
  - CNN block: 1D Conv (kernel=3, channels 221->128->64) with BatchNorm + ReLU, applied along the time axis
  - LSTM: 2-layer bidirectional LSTM (hidden=128)
  - Attention: Self-attention over LSTM outputs to weight important timesteps
  - Head: FC layers -> 3 outputs (10th, 50th, 90th percentile RUL)
- `quantile_loss.py`: Pinball loss for quantiles [0.1, 0.5, 0.9]
- Model must handle improving trajectories (RUL = infinity) — consider capping RUL target at a max value (e.g., 2000h) or treating infinity as a separate class.

**Files:** `src/ml/path_b/model.py`, `src/ml/path_b/quantile_loss.py`

### Step 8: Path B Training
- `trainer.py`: Training loop with AdamW, cosine LR schedule, early stopping on val RMSE (patience=5). Mixed precision (torch.amp). Gradient clipping. Checkpoint best model.
- Train on 120 training trucks, validate on 50 val trucks.
- RUL target handling: Cap at max_rul (e.g., 2000h). Samples with RUL = 99999.0 are either capped or excluded from RMSE calculation (they represent healthy/improving — no meaningful RUL to predict).

**Files:** `src/ml/path_b/trainer.py`

### Step 9: Path B Export
- ONNX export of the PyTorch model.

**Files:** `src/ml/path_b/export.py`

### Step 10: Evaluation Suite
- `metrics.py`: Per-class precision/recall/F1, macro F1, confusion matrix (Path A). RMSE, MAE, quantile coverage (Path B).
- `sacred_test.py`: Load 30 test trucks, run both models ONCE, report all metrics. This is the final evaluation — no iteration allowed after.

**Files:** `src/ml/evaluate/metrics.py`, `src/ml/evaluate/sacred_test.py`

### Step 11: CLI & Tests
- `cli.py`: Click commands: `fit-scaler`, `train-a`, `train-b`, `evaluate`, `export`
- Tests for scaler, data loading (incl. episode boundary enforcement), model forward pass, metrics computation

**Files:** `src/ml/cli.py`, `tests/test_ml_*.py`

## Key Design Decisions

1. **Memory management**: ~62 GB of Parquet won't fit in RAM. Welford scaler streams files one at a time. Path A loads in chunks. Path B uses IterableDataset.
2. **Class imbalance**: With ~95/4/1, IMMINENT (~4%) and CRITICAL (~1%) are true minority classes. Alpha=10 asymmetric loss in XGBoost is essential. Consider SMOTE or focal loss if F1 < 0.85.
3. **Episode segmentation**: Path B sequences respect episode_id boundaries. No lookback across repair events. This prevents the model learning "failure causes improvement."
4. **Repair gap handling**: Missing Parquet files for repair days are expected, not errors. Dataset registry handles sparse day numbering. Maintenance logs provide ground truth for gaps.
5. **RUL for healthy/improving**: RUL = 99999.0 (infinity sentinel). Path B training caps RUL at a practical maximum or excludes these samples from regression loss.
6. **Truck-based splits**: Never mix windows from the same truck across train/val/test. The split files from Phase 1 metadata enforce this.
7. **Sacred test set**: 30 trucks, evaluated ONCE at the very end. No hyperparameter tuning on test set.
8. **Quantile regression**: Path B outputs 3 RUL estimates (10th/50th/90th percentile) for uncertainty quantification.
9. **Feature reuse**: Import `FEATURE_COLUMNS` from `src/features/feature_vector.py` — single source of truth for the 221 feature names.

## Verification

```bash
# Fit scaler on training data
.venv/bin/python -m src.ml.cli fit-scaler --data-dir data-nvme/full/

# Train Path A (XGBoost)
.venv/bin/python -m src.ml.cli train-a --data-dir data-nvme/full/

# Train Path B (CNN-LSTM-Attention)
.venv/bin/python -m src.ml.cli train-b --data-dir data-nvme/full/ --epochs 30

# Sacred test evaluation (ONCE)
.venv/bin/python -m src.ml.cli evaluate --data-dir data-nvme/full/

# Export to ONNX
.venv/bin/python -m src.ml.cli export --format onnx

# Run ML tests
.venv/bin/python -m pytest tests/test_ml_*.py -v
```

## Data Summary (Post-Regeneration Expected)

| Property | Value |
|----------|-------|
| Total files | ~34,800–35,600 Parquet (repair gaps reduce count) |
| Total samples | ~50–51M |
| Storage | ~61–63 GB (snappy compressed) |
| Features | 221 (2 conditioning + 180 vibration + 39 thermal) |
| Columns per file | 230 (5 metadata incl. episode_id + 221 features + 4 labels) |
| Train trucks | 120 (96 modern, 24 older) |
| Val trucks | 50 (40 modern, 10 older) |
| Test trucks | 30 (24 modern, 6 older) |
| NORMAL | ~95% (93–96% acceptable) |
| IMMINENT | ~4% (3–5% acceptable) |
| CRITICAL | ~1% (0.5–2% acceptable) |
| Metadata | maintenance_log.json per truck, split files |
