# P39 Production-Style Runbook — Real Local Chain

## Overview

End-to-end command sequence for training, backtesting, and fusing day-ahead electricity price predictions locally. All artifacts are stored under `.local_artifacts/`.

**Default delivery profile**: `trusted_delivery` (cfg05 + catboost_spike_residual)

## Quick Start — One-Command Runner

The simplest way to run the full delivery pipeline:

```bash
python -m scripts.run_delivery_local_chain \
    --raw-data ../data/shandong_pmos_hourly.csv \
    --source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --fusion-engine regime_bgew \
    --required-training-days 30 \
    --allow-degraded \
    --strict-no-leakage \
    --start-day 2026-06-01 \
    --end-day 2026-06-30 \
    --work-dir .local_artifacts/delivery_run \
    --json --strict
```

## P57 CLI Flags (Safety Supervisor)

The runner supports additional safety supervisor flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--fusion-engine NAME` | `period_bgew` | Fusion engine to use: `regime_bgew`, `period_bgew`, `equal_weight`, `cfg05` |
| `--required-training-days N` | 30 | Required complete training days for P52 adaptive selector |
| `--max-lookback-days N` | 180 | Maximum calendar days to scan backwards for training days |
| `--min-days-for-degraded N` | 7 | Minimum days to qualify as DEGRADED (below this = INSUFFICIENT) |
| `--allow-degraded` | off | Allow delivery with DEGRADED_MIN_DAYS training day status |
| `--strict-no-leakage` | off | Fail immediately if ANY leakage check triggers (P53 strict mode) |

Example with full P57 safety config:

```bash
python -m scripts.run_delivery_local_chain \
    --raw-data ../data/shandong_pmos_hourly.csv \
    --source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --fusion-engine regime_bgew \
    --required-training-days 30 \
    --max-lookback-days 180 \
    --min-days-for-degraded 7 \
    --allow-degraded \
    --strict-no-leakage \
    --start-day 2026-06-01 \
    --end-day 2026-06-30 \
    --work-dir .local_artifacts/delivery_run \
    --json --strict
```

## P57 Safety Supervisor Pipeline

The runner executes these steps in order (P57 additions in **bold**):

1. Raw data check (existing)
2. Source repo check (existing)
3. **Safety preflight** (P53 leakage sentinel)
4. **Adaptive training days** (P52)
5. Trust gate (P41)
6. Actual ledger (P34)
7. Trusted fusion (P42) — if 2+ trusted models
8. Rolling validation (P43) — if 2+ trusted models
9. **Fallback ladder** (P54)
10. **Postflight validation** (P55)
11. Delivery summary (P44)
12. Forbidden file check
13. Claim guard (P46)

## Prerequisites

```bash
# Ensure dependencies
pip install lightgbm catboost pandas numpy pyyaml
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

## P41-P49: Delivery Gate & Guard

### Trust Gate (P41)

```bash
python -m scripts.run_p41_model_trust_gate --json --strict
```

### Trusted Fusion Backtest (P42)

```bash
python -m scripts.run_p42_trusted_fusion_backtest --json
```

### Rolling Validation (P43)

```bash
python -m scripts.run_p43_rolling_weight_fusion_validation --json --strict
```

### Delivery Packager (P44)

```bash
python -m scripts.run_p44_delivery_readiness_packager --json
```

### Claim Guard (P46)

```bash
python -m scripts.validate_delivery_claims --json --strict
```

### Final Audit (P49)

```bash
python -m scripts.run_p49_final_delivery_audit --json --strict
```

## Delivery Profiles

| Profile | Models | Delivery Allowed | Default |
|---------|--------|-----------------|---------|
| `trusted_delivery` | cfg05 + catboost_spike_residual | ✅ Yes | ✅ Yes |
| `balanced_candidate` | cfg05 + best_two_average + catboost_sota + catboost_spike_residual | ❌ Manual review | ❌ |
| `research_all_models` | All 5 (incl. stage3) | ❌ No | ❌ |

## Important Notes

- **stage3_business_fixed** excluded due to source-repo training leakage (sMAPE=0.39%, 82.5% within 1%)
- **best_two_average** and **catboost_sota** excluded (corr > 0.995) — conservative safety gate
- Trusted pool is only 2 models, so fusion improvement is modest (~6.79%)
- **Research results (69.96%, 2.97%) are NOT delivery claims**

## Output Files

The runner produces the following output files under `<work-dir>/`:

| File | Description |
|------|-------------|
| `delivery_summary.json` | Full delivery summary with metrics and status (P44) |
| `final_output.csv` | Fused predictions (24-hour day-ahead prices) |
| `metrics.json` | Extracted performance metrics (sMAPE, improvement) |
| `run_manifest.json` | **New (P55):** Delivery run manifest with run ID, timestamps, profile, training days, fusion method, fallback info, postflight results, warnings, and errors |
| `delivery_report.md` | **New (P55):** Human-readable markdown delivery report with per-check pass/fail table and metrics |
| `delivery_report.json` | **New (P55):** Machine-readable JSON delivery report for programmatic consumption |

### run_manifest.json Schema

```json
{
    "run_id": "delivery-20260705-001",
    "target_day": "2026-07-05",
    "profile": "trusted_delivery",
    "started_at": "2026-07-05T08:00:00",
    "completed_at": "2026-07-05T08:30:00",
    "status": "PASS",
    "delivery_status": "NORMAL",
    "selected_training_days": 30,
    "trusted_models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
    "quarantined_models": ["stage3_business_fixed"],
    "fusion_method": "regime_bgew",
    "fallback": {"fallback_used": false, "fallback_method": ""},
    "postflight": { ... },
    "metrics": {"sMAPE": 9.23, "MAE": 15.4},
    "warnings": [],
    "errors": []
}
```

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
