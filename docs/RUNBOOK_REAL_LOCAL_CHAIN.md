# P39 Production-Style Runbook — Real Local Chain

## Overview

End-to-end command sequence for training, backtesting, and fusing day-ahead electricity price predictions locally. All artifacts are stored under `.local_artifacts/p31_p40_multimodel_fusion/`.

## Prerequisites

```bash
# Ensure dependencies
pip install lightgbm catboost pandas numpy
```

## Phase-by-Phase Commands

### P31 — Train Multi-Model Pool

Trains all 5 models (cfg05 + 4 candidates) and saves artifacts.

```bash
python -m scripts.run_p31_train_dayahead_model_pool \
    --target-day 2026-06-30 \
    --force \
    --strict
```

**Output**: `.local_artifacts/p31_p40_multimodel_fusion/models/<model_name>/`
**Statuses**: REAL_24H_READY | TRAINED_BUT_NOT_24H | DEP_MISSING | MODEL_TRAIN_FAILED

### P32 — 30-Day Backtest

Runs predictions for each model across June 2026.

```bash
python -m scripts.run_p32_multimodel_30d_backtest \
    --start-day 2026-06-01 \
    --end-day 2026-06-30 \
    --force \
    --strict
```

**Output**: `.local_artifacts/p31_p40_multimodel_fusion/ledger/predictions_<model>_30d.csv`

### P33 — Prediction Ledger

Consolidates per-model predictions into unified ledger.

```bash
python -m scripts.run_p33_multimodel_prediction_ledger --force
```

**Output**: `prediction_ledger_30d.csv` (3600 rows = 5 models × 30 days × 24 hours)

### P34 — Actual Ledger

Extracts y_true from raw data aligned to the backtest period.

```bash
python -m scripts.run_p34_actual_ledger_alignment --force
```

**Output**: `actual_ledger_30d.csv` (720 rows = 30 days × 24 hours)

### P35 — Period BGEW Weight Learning

Computes per-period Bayesian-Gamma-Exponential-Weighted fusion weights.

```bash
python -m scripts.train_p35_period_bgew_multimodel --alpha 0.5 --force
```

**Output**: `period_bgew_weights.json`

### P36 — Fusion Backtest

Compares BGEW fusion vs cfg05-alone. Measures sMAPE_floor50, MAE, RMSE.

```bash
python -m scripts.run_p36_fusion_backtest --json
```

**Output**: `fusion_backtest_30d.csv`

### P37 — Regime Analysis

Analyzes negative and low-price hours across the backtest period.

```bash
python -m scripts.analyze_p37_negative_low_price_regime --low-threshold 100
```

### P38 — Full Chain Fused Prediction

Single-day end-to-end fused prediction (all models → fuse).

```bash
python -m scripts.run_p38_fused_full_chain --target-day 2026-06-30 --json
```

**Output**: `fused_full_chain_output.csv`

## Quick-Start (All Phases)

```bash
python -m scripts.run_p31_train_dayahead_model_pool --force
python -m scripts.run_p32_multimodel_30d_backtest --force
python -m scripts.run_p33_multimodel_prediction_ledger --force
python -m scripts.run_p34_actual_ledger_alignment --force
python -m scripts.train_p35_period_bgew_multimodel --force
python -m scripts.run_p36_fusion_backtest --json
python -m scripts.run_p38_fused_full_chain --target-day 2026-06-30 --json
```

## P41-P45: Trusted Fusion Delivery (Default Profile)

The delivery default is the **trusted_no_stage3** profile. Run in order:

```bash
# P41 — Model trust gate (flags SUSPECT_LEAKAGE models)
python -m scripts.run_p41_model_trust_gate --json --strict

# P42 — Trusted fusion backtest (excludes quarantined models)
python -m scripts.run_p42_trusted_fusion_backtest --json

# P43 — Rolling weight validation (no-lookahead verification)
python -m scripts.run_p43_rolling_weight_fusion_validation --json --strict

# P44 — Delivery readiness packager (assembles P41-P43)
python -m scripts.run_p44_delivery_readiness_packager --json
```

### Profiles

| Profile | Models | Delivery Allowed | Use Case |
|---------|--------|-----------------|----------|
| `trusted_no_stage3` | cfg05 + catboost_spike_residual | ✅ Yes | **Default production** |
| `research_all_models` | All 5 (incl. stage3) | ❌ No | Research reproduction |

### Important Notes

- **stage3_business_fixed** excluded due to source-repo training leakage (sMAPE=0.39%, 82.5% within 1%)
- **best_two_average** and **catboost_sota** excluded (corr > 0.995) — good models but conservative gate
- Trusted pool is only 2 models, so fusion improvement is modest (~6.79%)
- Research results (69.96%, 2.97%) are NOT delivery claims

## Artifact Layout

```
.local_artifacts/p31_p40_multimodel_fusion/
├── models/
│   ├── cfg05_dayahead_lgbm/       # cfg05 LightGBM model + predictions
│   ├── best_two_average/           # Average of 2 LightGBM trials
│   ├── stage3_business_fixed/      # Stage-3 LightGBM
│   ├── catboost_sota/              # CatBoost SOTA baseline
│   └── catboost_spike_residual/    # CatBoost spike-robust
├── ledger/
│   ├── prediction_ledger_30d.csv   # Unified prediction ledger
│   ├── actual_ledger_30d.csv       # Actual price ledger
│   └── predictions_*_30d.csv       # Per-model predictions
├── period_bgew_weights.json        # BGEW fusion weights
├── fusion_backtest_30d.csv         # 30-day fusion results
└── fused_full_chain_output.csv     # Single-day fused output
```

## Health Checks

```bash
# Check any prediction CSV for 24-hour completeness
python -m scripts.check_cfg05_hour24_completeness \
    --input .local_artifacts/p31_p40_multimodel_fusion/ledger/predictions_cfg05_dayahead_lgbm_30d.csv \
    --target-day 2026-06-30

# Run full test suite
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

## Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `MODEL_TRAIN_FAILED` | CatBoost not installed | `pip install catboost` |
| `23H` completeness | Old filter (exclusive end) | Use `filter_dayahead()` from `artifacts.dayahead_window` |
| Feature shape mismatch | v3 column fill missing | Re-run P31 with `--force` |
| NaN weights | All models have NaN y_true | Check actual_ledger has valid values |
| `P36_FUSION_NOT_IMPROVED` | Single model dominates | Review weights, adjust alpha |
