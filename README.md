# electricity_forecast_model3.0 — Multi-Model Fusion for Electricity Price Forecasting

> **Project Status**: FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS (v3.0.0)
> **Default Profile**: `trusted_delivery`
> **Date**: 2026-07-05

---

## 1. What This Is

A production-oriented multi-model fusion system for day-ahead electricity price forecasting in the Shandong PMOS market. This is a **new consolidated repo** — not a patch on older 2.x versions.

**Core capability**: Combine 2+ trusted ML models via BGEW (Bayesian-Gamma-Exponential-Weighted) fusion to produce more accurate day-ahead price predictions than any single model.

## 2. Project Status

| Area | Status |
|------|--------|
| Pipeline | DELIVERY_READY (trusted_no_stage3) |
| Tests | 1053 passing, 0 failing |
| Default profile | `trusted_delivery` |
| Delivery freeze | P50 complete — `DELIVERY_FREEZE_READY` |

## 3. Metrics (trusted_delivery profile)

| Metric | Value | Notes |
|--------|-------|-------|
| cfg05 baseline sMAPE | 9.90% | 30-day backtest (June 2026) |
| **Trusted BGEW fusion sMAPE** | **9.23%** | **+6.79% improvement** |
| Equal-weight fusion | 9.94% | Reference baseline |
| Split OOS fusion | 10.12% vs cfg05 10.49% | Train 20d / test 9d |
| Rolling OOS fusion | 10.08% vs cfg05 10.76% | 22 rolling days |

### What This Means

The trusted fusion pipeline consistently improves over cfg05 by 3.6–6.8% across all validation approaches (in-sample, train/test split, rolling expanding window). The improvement holds out-of-sample.

## 4. Data Preparation

### Raw CSV Format

The raw data should be a Chinese electricity market CSV with columns:

```
时刻,日前电价,实时电价,直调负荷预测值,风电总加预测值,光伏总加预测值,联络线受电负荷预测值,竞价空间预测值
```

Validate your CSV:

```bash
python -m scripts.check_cfg05_raw_data_contract --raw-data path/to/data.csv
```

### Source Repository

The cfg05 model requires features built from the source epf-sota-experiment repo:

```bash
git clone https://github.com/disdorqin/epf-sota-experiment.git \
    .local_artifacts/source_repos/epf-sota-experiment
```

## 5. Quick Start — One-Command Runner

The simplest way to run the full delivery pipeline:

```bash
python -m scripts.run_delivery_local_chain \
    --raw-data ../data/shandong_pmos_hourly.csv \
    --source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --start-day 2026-06-01 \
    --end-day 2026-06-30 \
    --work-dir .local_artifacts/delivery_run \
    --json --strict
```

Outputs:

- `<work-dir>/delivery_summary.json` — Full delivery summary
- `<work-dir>/final_output.csv` — Fused predictions (if available)
- `<work-dir>/metrics.json` — Extracted performance metrics

## 6. Phase-by-Phase Pipeline

Run individual phases for debugging or inspection:

```bash
# P31 — Train multi-model pool
python -m scripts.run_p31_train_dayahead_model_pool --target-day 2026-06-30 --force --strict

# P32 — 30-day backtest
python -m scripts.run_p32_multimodel_30d_backtest --start-day 2026-06-01 --end-day 2026-06-30 --force

# P33-P34 — Ledger materialization
python -m scripts.run_p33_multimodel_prediction_ledger --force
python -m scripts.run_p34_actual_ledger_alignment --force

# P35 — BGEW weight learning
python -m scripts.train_p35_period_bgew_multimodel --force

# P36-P38 — Fusion backtest & full chain
python -m scripts.run_p36_fusion_backtest --json
python -m scripts.run_p38_fused_full_chain --target-day 2026-06-30 --json
```

**Delivery gate (P41-P44)**:

```bash
python -m scripts.run_p41_model_trust_gate --json --strict
python -m scripts.run_p42_trusted_fusion_backtest --json
python -m scripts.run_p43_rolling_weight_fusion_validation --json --strict
python -m scripts.run_p44_delivery_readiness_packager --json
```

## 7. Delivery Profiles

### `trusted_delivery` ✅ (Default)

| Property | Value |
|----------|-------|
| Allowed models | `lightgbm_cfg05_dayahead`, `catboost_spike_residual` |
| Delivery claims | ✅ Allowed |
| sMAPE | 9.23% (fusion) |

### `balanced_candidate` ⚠️ (Manual Review)

| Property | Value |
|----------|-------|
| Allowed models | cfg05, best_two_average, catboost_sota, catboost_spike_residual |
| Delivery claims | ❌ Requires manual review |
| sMAPE | ~5% (estimated) |

### `research_all_models` 🔬 (Research Only)

| Property | Value |
|----------|-------|
| Allowed models | All 5 (incl. stage3_business_fixed) |
| Delivery claims | ❌ Strictly forbidden |
| sMAPE | 2.97% (with stage3 leakage artifact) |

## 8. Quarantined Models

| Model | Reason | sMAPE | Detail |
|-------|--------|-------|--------|
| `stage3_business_fixed` | SUSPECT_LEAKAGE | 0.39% | Source-repo training leakage confirmed (82.5% within 1% of actual, corr=0.9999) |
| `best_two_average` | CONSERVATIVE_CORR_GATE | 4.94% | corr(y_pred, y_true)=0.9962 > 0.995 threshold |
| `catboost_sota` | CONSERVATIVE_CORR_GATE | 4.06% | corr(y_pred, y_true)=0.9965 > 0.995 threshold |

The corr > 0.995 gate is deliberately conservative. `best_two_average` and `catboost_sota` are good models (4-5% sMAPE) with no evidence of actual leakage. They are excluded from delivery as a safety margin.

## 9. Forbidden Claims

The following claims must **never** appear in delivery context:

| Claim | Reason |
|-------|--------|
| "2.97% production sMAPE" | Research result with leakage-suspect stage3 |
| "69.96% production improvement" | Same — stage3 dominated fusion |
| "stage3 production readiness" | Confirmed source-repo training leakage |
| "Source 11.48% reproduction" | Not verified on trusted pool |

These claims may appear in **research** context only if accompanied by: `research only`, `not delivery`, `stage3 leakage caveat`.

## 10. Claim Guard

An automated claim guard scans all docs/reports and README for forbidden claims:

```bash
python -m scripts.validate_delivery_claims --json --strict
```

## 11. Architecture

```
raw CSV → feature builder → model pool (5 models) → prediction ledger
                                                          ↓
actual ledger ← data source ←──────────────────── actual prices
                                                          ↓
                                              BGEW weight learner
                                                          ↓
                                              trusted fusion (P41 gated)
                                                          ↓
                                              rolling validation (P43)
                                                          ↓
                                              delivery packager (P44)
```

## 12. Test Suite

```bash
python -m pytest tests/ -v --tb=short
```

## 13. Safety Supervisor (P52-P57)

The safety supervisor is a multi-layered runtime guard that ensures delivery output integrity through progressive fallback, runtime leakage detection, and postflight validation. It integrates into the P47 delivery runner as P57.

### Components

- **P53 — Leakage Sentinel**: Runtime guard that detects y_true leakage via correlation, sMAPE, and MAE checks against actuals. Classifies models as TRUSTED, CONSERVATIVE_QUARANTINE, or SUSPECT_LEAKAGE. Thresholds: CORR_THRESHOLD=0.995, WITHIN_1PCT_THRESHOLD=0.80, sMAPE_floor50<2.0, MAE<10.0 CNY.

- **P52 — Adaptive Training Days**: Scans backwards from D-1 to find complete training days. Four status levels: COMPLETE_30D (>=30 days), DEGRADED (>=7 days), INSUFFICIENT (<7 days), NO_VALID_DAYS (0 days). Configurable via `--required-training-days`, `--max-lookback-days`, `--min-days-for-degraded`.

- **P54 — Fallback Ladder**: 6-level progressive fallback: trusted_bgew_fusion (NORMAL) → trusted_equal_weight (DEGRADED) → best_trusted_single_model (DEGRADED) → cfg05_baseline (DEGRADED) → historical_same_hour_median (DEGRADED) → FAILED_NO_DELIVERY.

- **P55 — Postflight Validation**: 12 checks on delivery output: file exists, 24 rows, hour_business range 1..24, no duplicate hours, no NaN, business_day consistency, profile delivery allowed, no quarantined models, claim guard pass, no git-tracked artifacts, hour-24 convention, no merge suffixes.

- **P56 — Regime BGEW Fusion**: 4-regime adaptive weighting (normal/low_price/negative_risk/high_spike) with trust gating, cfg05 floor (30%), min/max weight bounds (5%/75%), and internal 3-level fallback chain (regime_bgew → period_bgew → equal_weight).

### Three-Tier Delivery Status

| Status | Exit Code | Condition |
|--------|-----------|-----------|
| NORMAL | 0 | Level 1 fallback succeeds AND postflight PASS; safe to deliver |
| DEGRADED_DELIVERED | 2 | Levels 2-5 produce valid 24H output; delivery allowed with warnings |
| FAILED_NO_DELIVERY | 1 | All 6 fallback levels failed; no delivery possible |

## 14. Safety & No-Leakage Policy (Cont.)

1. **No y_true in prediction ledger**: Prediction and actual ledgers are kept separate
2. **No forward-looking features**: All feature builder operations use `shift(1+)` and backward-looking windows
3. **No target as feature**: Day-ahead price is never used as an input feature
4. **Rolling weight validation**: Fusion weights are validated with no-lookahead
5. **Claim guard**: Automated scanning prevents delivery-context misuse of research results
6. **No data, model, or ledger files committed**: Git-tracked files contain only code, config, and documentation
7. **Cannot commit delivery artifacts**: Runner explicitly avoids committing `delivery_summary.json`, `final_output.csv`, `metrics.json`

## 15. File Layout

```
config/fusion_profiles.yaml      Profile definitions
data/                             Raw data (gitignored)
docs/                             Reports and documentation
models/                           Model adapters
pipelines/                        Pipeline orchestration
scripts/                          CLI entry points
tests/                            Test suite
.local_artifacts/                 Artifacts (gitignored)
```

## 16. Deliverables

- ✅ P41 model trust gate (scripts/run_p41_model_trust_gate.py)
- ✅ P42 trusted fusion backtest (scripts/run_p42_trusted_fusion_backtest.py)
- ✅ P43 rolling weight validation (scripts/run_p43_rolling_weight_fusion_validation.py)
- ✅ P44 delivery packager (scripts/run_p44_delivery_readiness_packager.py)
- ✅ P46 profile registry + claim guard (config/fusion_profiles.yaml, scripts/validate_delivery_claims.py)
- ✅ P47 one-command runner (scripts/run_delivery_local_chain.py)
- ✅ P49 final audit (scripts/run_p49_final_delivery_audit.py)
- ✅ P50 delivery freeze report (docs/reports/p50_final_delivery_freeze_report.md)
- ✅ **P52 adaptive training days** (fusion/adaptive_training_days.py)
- ✅ **P53 leakage sentinel** (safety/leakage_sentinel.py)
- ✅ **P54 fallback ladder** (delivery/fallback_ladder.py)
- ✅ **P55 postflight + manifest + report** (delivery/postflight.py, delivery/manifest.py, delivery/report.py)
- ✅ **P56 regime BGEW fusion** (fusion/trust_gated_regime_bgew.py)
- ✅ **P57 safety supervisor integration** (scripts/run_delivery_local_chain.py)

## 17. Realtime Prediction Status (P91-P95)

### DA-Safe Realtime Baseline — Official Default

The realtime prediction is now an **official DA-Safe Baseline** (not a fallback):

```
rt_pred = da_anchor  (DA-Safe Baseline)
         +
SGDFNet Assist / sidecar (optional enhancement)
da_error_prob, residual_direction_prob, uncertainty_score,
correction_permission, reason_codes
```

### Status Rules

| Condition | Status |
|---|---|
| rt_da_anchor available + realtime_price no NaN + safety PASS | **REALTIME_READY_DA_SAFE_ONLY** |
| SGDFNet available + ledger + learner PASS | **REALTIME_HYBRID_READY** |
| realtime_price NaN | **NO_GO** |

### Learner Policy

```python
learner_policy = {
    "dayahead": "period_regime_bgew",    # period x regime dimensional
    "realtime": "pooled_30d_bgew",        # pooled 24H, 30-day lookback
}
```

### Key Files

| File | Purpose |
|---|---|
| `models/adapters/sgdfnet_assist_adapter.py` | SGDFNet assist adapter (P92) |
| `ledgers/realtime_prediction_ledger.py` | 2-candidate realtime ledger (P93) |
| `fusion/unified_weight_learner.py` | Pooled 30D BGEW learner (P94) |
| `models/realtime_state.py` | Realtime state constants |
| `scripts/run_p92_sgdfnet_assist_adapter.py` | Run SGDFNet assist |
| `scripts/run_p93_realtime_two_candidate_ledger.py` | Build realtime ledger |
| `scripts/run_p94_realtime_pooled_learner.py` | Run realtime learner |

### What Remains Future Work

- Retrain stage3 with proper temporal CV in source repo
- Integrate real-time (RT) model assist
- Expand 30-day validation window to 90+ days
- Seasonal validation (summer/winter/spring/autumn)
- Integrate residual plugins (P5M negative valley)
