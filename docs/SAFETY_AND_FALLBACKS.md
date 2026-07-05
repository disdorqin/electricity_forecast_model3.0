# Safety Supervisor Architecture

> **Phases**: P52-P57
> **Status**: IMPLEMENTED
> **Updated**: 2026-07-05

---

## 1. Overview

The Safety Supervisor is a multi-layered runtime guard that protects the delivery pipeline from producing unreliable or contaminated output. It consists of five components that form a progressive safety chain:

```
Raw pipeline output
    |
    v
[P56 Regime BGEW Fusion]   -- adaptive weighting with trust gating
    |
    v
[P53 Leakage Sentinel]      -- runtime leakage detection per model
    |
    v
[P52 Adaptive Training Days] -- verifies training data completeness
    |
    v
[P54 Fallback Ladder]       -- 6-level progressive fallback
    |
    v
[P55 Postflight Validation] -- 12 checks on final output
    |
    v
Three-tier delivery status: NORMAL / DEGRADED_DELIVERED / FAILED_NO_DELIVERY
```

### Integration in the Pipeline

The safety supervisor integrates into the P47 delivery runner as P57. The pipeline step order is:

1. Raw data check
2. Source repo check
3. **P53 — Safety preflight** (leakage sentinel)
4. **P52 — Adaptive training days**
5. P41 — Trust gate
6. P34 — Actual ledger
7. P42 — Trusted fusion (if 2+ models)
8. P43 — Rolling validation (if 2+ models)
9. **P54 — Fallback ladder**
10. **P55 — Postflight validation + manifest + report**
11. P44 — Delivery summary
12. Forbidden file check
13. P46 — Claim guard

---

## 2. Three-Tier Delivery Status

The final delivery status is determined by the fallback ladder and postflight validation:

| Status | Exit Code | Meaning | Condition |
|--------|-----------|---------|-----------|
| **NORMAL** | 0 | Fully trusted delivery | Level 1 (trusted_bgew_fusion) succeeds AND postflight PASS |
| **DEGRADED_DELIVERED** | 2 | Degraded but usable | Levels 2-5 produce valid 24H output |
| **FAILED_NO_DELIVERY** | 1 | No output possible | All 6 fallback levels failed |

### Exit Code Convention

- **Exit 0 (NORMAL)**: The output is fully trusted. BGEW fusion succeeded and all postflight checks passed. This is the ideal state.
- **Exit 2 (DEGRADED_DELIVERED)**: The output is usable but the pipeline fell back from the primary fusion engine. Common reasons: BGEW weight computation failed, insufficient training data, or missing model predictions. Delivery is still allowed but consumers receive a warning.
- **Exit 1 (FAILED_NO_DELIVERY)**: No output was produced. All 6 fallback levels were exhausted. The pipeline cannot deliver predictions for the target day.

---

## 3. P53 — Leakage Sentinel

**File**: `safety/leakage_sentinel.py`

The Leakage Sentinel is a runtime guard that checks each model on every run for data leakage indicators. It is invoked during the safety preflight step (step 3) of the delivery pipeline.

### Public API

```python
from safety.leakage_sentinel import check_model_leakage, run_leakage_sentinel, is_delivery_allowed

# Check a single model
result = check_model_leakage(
    model_name="lightgbm_cfg05_dayahead",
    prediction_ledger_path="ledgers/prediction_ledger_30d.csv",
    actual_ledger_path="ledgers/actual_ledger_30d.csv",
)

# Check all trusted models
summary = run_leakage_sentinel(
    trusted_models=["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
    prediction_ledger_path="ledgers/prediction_ledger_30d.csv",
    actual_ledger_path="ledgers/actual_ledger_30d.csv",
)

# Query delivery eligibility
allowed = is_delivery_allowed(
    model_name="lightgbm_cfg05_dayahead",
    sentinel_result=summary,
    profile_name="trusted_delivery",
)
```

### Thresholds

| Constant | Value | Used For | Triggers |
|----------|-------|----------|----------|
| `CORR_THRESHOLD` | 0.995 | Pearson correlation y_pred vs y_true | CONSERVATIVE_QUARANTINE if > 0.995 |
| `WITHIN_1PCT_THRESHOLD` | 0.80 | Ratio of predictions within 1% of actual | CONSERVATIVE_QUARANTINE if > 80% |
| `SMAPE_FLOOR50_TOO_GOOD` | 2.0 (%) | sMAPE with floor of 50 | SUSPECT_LEAKAGE if < 2% |
| `MAE_TOO_GOOD` | 10.0 (CNY) | Mean Absolute Error | SUSPECT_LEAKAGE if < 10 CNY |

### 11 Checks

| # | Check Name | What It Detects | Result if FAIL |
|---|------------|-----------------|----------------|
| 1 | `no_y_true_in_prediction_ledger` | Prediction ledger must NOT contain a `y_true`, `target`, or `日前电价` column | INVALID_SCHEMA |
| 2 | `no_target_in_features` | Feature columns must not include target column names | INVALID_SCHEMA |
| 3 | `sufficient_eval_rows` | Model must have >= 24 eval rows after merge + NaN drop | INVALID_24H |
| 4 | `within_1pct_ratio` | Ratio of predictions within 1% of actual value | CONSERVATIVE_QUARANTINE |
| 5 | `corr_y_pred_y_true` | Pearson correlation between prediction and actual | CONSERVATIVE_QUARANTINE |
| 6 | `sMAPE_floor50` | sMAPE with floor of 50 | SUSPECT_LEAKAGE |
| 7 | `MAE` | Mean Absolute Error in CNY | SUSPECT_LEAKAGE |
| 8 | `no_future_timestamps` | Any prediction timestamp in the future | SUSPECT_LEAKAGE |
| 9 | `no_target_day_overlap` | Target-day overlap (informational) | Informational (always passes) |
| 10 | `no_duplicate_keys` | Duplicate `(business_day, hour_business)` rows | SUSPECT_LEAKAGE |
| 11 | `24h_completeness` | All 24 hours (1..24) present in prediction | INVALID_24H |

### Status Determination Priority

```
INVALID_SCHEMA > INVALID_24H > SUSPECT_LEAKAGE > CONSERVATIVE_QUARANTINE > TRUSTED
```

### Action Matrix

| Status | `trusted_delivery` | `balanced_candidate` | `research_all_models` |
|--------|-------------------|---------------------|----------------------|
| TRUSTED | ALLOWED | ALLOWED | ALLOWED |
| CONSERVATIVE_QUARANTINE | BLOCKED | ALLOWED | ALLOWED |
| SUSPECT_LEAKAGE | BLOCKED | BLOCKED | BLOCKED |
| INVALID_SCHEMA | BLOCKED | BLOCKED | BLOCKED |
| INVALID_24H | BLOCKED | BLOCKED | BLOCKED |

---

## 4. P52 — Adaptive Training Days

**File**: `fusion/adaptive_training_days.py`

Scans backwards from `target_date - 1` to find complete training days for weight learning. A day is considered complete when:

1. **Prediction ledger**: Every trusted model has hour_business 1..24, no NaN in y_pred, no duplicate keys on `(task, model_name, target_day, business_day, hour_business)`.
2. **Actual ledger**: hour_business 1..24, no NaN in y_true, no duplicate keys on `(task, target_day, business_day, hour_business)`.

### Status Levels

| Status | Condition | Meaning |
|--------|-----------|---------|
| COMPLETE_30D | >= required_days (default 30) | Sufficient training data for full BGEW weight learning |
| DEGRADED_MIN_DAYS | >= min_days_for_degraded (default 7) | Marginal but usable; trigger degraded delivery mode |
| INSUFFICIENT_DAYS | > 0 but < min_days_for_degraded | Too few days for reliable weight learning |
| NO_VALID_DAYS | 0 days, or ledgers not found/empty | No training data available |

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `required_days` | 30 | Number of complete training days desired |
| `max_lookback_days` | 180 | Maximum calendar days to scan backwards |
| `min_days_for_degraded` | 7 | Minimum days for DEGRADED status |

### Output Example

```python
{
    "status": "COMPLETE_30D",
    "selected_days": ["2026-06-30", "2026-06-29", ...],
    "selected_count": 30,
    "skipped_days": [
        ("2026-05-01", "models_with_nan_y_pred=['model_b']"),
        ("2026-04-28", "actual_incomplete_hours; missing_hours=[23, 24]")
    ],
    "errors": [],
    "warnings": ["Skipped 3 day(s) during scan"],
    "latest_selected_day": "2026-06-30",
    "oldest_selected_day": "2026-06-01",
    "training_rows": 1440,   # 30 * n_models * 24
    "actual_rows": 720,       # 30 * 24
}
```

### CLI Flags

```
--required-training-days N    (default: 30)
--max-lookback-days N         (default: 180)
--min-days-for-degraded N     (default: 7)
--allow-degraded              (allow DEGRADED_MIN_DAYS delivery)
```

---

## 5. P54 — Fallback Ladder

**File**: `delivery/fallback_ladder.py`

The fallback ladder provides a 6-level progressive degradation path. Each level attempts to produce 24 rows of hourly day-ahead delivery prices. The first level that passes validation is selected.

### 6 Levels

| Level | Method | Description | Delivery Status |
|-------|--------|-------------|-----------------|
| 1 | `trusted_bgew_fusion` | BGEW inverse-error weighting over trusted models | NORMAL |
| 2 | `trusted_equal_weight` | Simple average across all trusted models | DEGRADED_DELIVERED |
| 3 | `best_trusted_single_model` | Best single model by recent MAE against actuals | DEGRADED_DELIVERED |
| 4 | `cfg05_baseline` | cfg05 champion model output (always available) | DEGRADED_DELIVERED |
| 5 | `historical_same_hour_median` | Per-hour median of historical raw prices (leakage-aware) | DEGRADED_DELIVERED |
| 6 | `FAILED_NO_DELIVERY` | All levels failed | FAILED_NO_DELIVERY |

### When Each Level Triggers

| Level | Trigger Condition |
|-------|------------------|
| 1 | Prediction and actual ledgers loaded, trusted models > 0, BGEW weight computation succeeds, output validates to 24 rows |
| 2 | Level 1 fails; prediction ledger has data for trusted models; equal-weight average produces valid 24 rows |
| 3 | Levels 1-2 fail; at least one trusted model has predictions for target date; best model selected by historical MAE |
| 4 | Levels 1-3 fail; cfg05 model found in prediction ledger by case-insensitive name match |
| 5 | Levels 1-4 fail; raw data file available; historical same-hour median computes 24 valid rows |
| 6 | All above levels fail; no output possible |

### Key Design Decisions

1. **BGEW weight reuse** — Level 1 delegates to `fusion.weights.bgew_skeleton` for inverse-error weighting.
2. **Best-model selection** — Level 3 selects the model with lowest historical MAE against actuals.
3. **Leakage-aware historical median** — Level 5 filters to `business_day < target_date` before computing medians.
4. **cfg05 champion guarantee** — Level 4 always finds cfg05 by case-insensitive name match.
5. **Empty-tolerance** — All helpers return `None` or failure dicts; no exceptions propagate.

### Delivery Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `business_day` | datetime | Business day |
| `ds` | datetime | Wall-clock timestamp |
| `hour_business` | int | Hour 1..24 |
| `period` | str | Period bucket (`1_8`, `9_16`, `17_24`) |
| `dayahead_price` | float | Predicted day-ahead price |
| `realtime_price` | float or None | Predicted real-time price (None for dayahead) |

---

## 6. P55 — Postflight Validation

**File**: `delivery/postflight.py`

The postflight module runs 12 quality and safety checks on the final delivery output CSV.

### 12 Checks

| # | Check Name | What It Validates | Critical? |
|---|------------|-------------------|-----------|
| 1 | `file_exists_readable` | Final output exists and is a readable CSV | Yes |
| 2 | `twenty_four_rows` | Exactly 24 data rows (one per business hour) | Yes |
| 3 | `hour_business_range` | `hour_business` column contains values 1..24 | Yes |
| 4 | `no_duplicate_hours` | No duplicate `hour_business` values | Yes |
| 5 | `no_nan_in_predictions` | No NaN in `y_pred` / price / forecast columns | Yes |
| 6 | `business_day_consistency` | `business_day` column contains the target date | Yes |
| 7 | `profile_delivery_allowed` | Profile has `delivery_allowed: True` | Yes |
| 8 | `no_quarantined_models` | No quarantined models in `allowed_models` | Yes |
| 9 | `claim_guard_pass` | Calls `validate_delivery_claims.run_claim_guard()` | Yes |
| 10 | `no_git_tracked_artifacts` | Work directory is not tracked by git | No (informational) |
| 11 | `hour24_convention` | Hour 24 follows D+1 00:00:00 convention | No (convention) |
| 12 | `no_merge_suffixes` | No `_x`/`_y` suffix columns from bad merges | Yes |

### Status Determination

| Condition | Status | Action |
|-----------|--------|--------|
| All 12 checks passed | PASS | Full confidence delivery |
| 1-2 checks failed | WARN | Degraded but usable; check warnings |
| 3+ checks failed | FAIL | Block delivery; inspect errors |

### Postflight Output Example

```python
{
    "status": "PASS",
    "target_date": "2026-07-05",
    "checks": {
        "file_exists_readable": {"passed": True, "detail": "File exists and readable"},
        "twenty_four_rows": {"passed": True, "detail": "Exactly 24 rows (24)"},
        "no_nan_in_predictions": {"passed": True, "detail": "No NaN in prediction columns"},
        # ... 9 more checks
    },
    "errors": [],
    "warnings": [],
    "summary": {"total": 12, "passed": 12, "failed": 0, "warned": 0},
}
```

### Manifest + Report (P55)

Alongside postflight validation, P55 generates three additional artifacts:

- **`run_manifest.json`**: Machine-readable delivery run metadata including run ID, timestamps, profile, trusted models, fusion method, fallback info, postflight results, and metrics.
- **`delivery_report.md`**: Human-readable markdown report with per-check pass/fail table, metrics, and warnings.
- **`delivery_report.json`**: JSON version of the delivery report for programmatic consumption.

---

## 7. P56 — Regime BGEW Fusion

**File**: `fusion/trust_gated_regime_bgew.py`

The Regime BGEW fusion engine extends the P35 period BGEW concept with regime-aware adaptive weighting, trust gating, and safety constraints.

### 4-Regime Classification

| Regime | Condition | Priority |
|--------|-----------|----------|
| `negative_risk` | `historical_actual_median < 0` OR `ensemble_median < 0` | 1 (highest) |
| `low_price` | `ensemble_median < 100` (CNY) | 2 |
| `high_spike` | `ensemble_median > recent_p90` | 3 |
| `normal` | Everything else | 4 (default) |

Classification is per-day (not per-hour) and first-match-wins.

### Weight Formula

```
base_score[m]     = exp(-alpha * period_smape[m, P])
stability_score[m]= exp(-beta * rmse_volatility[m, P])
regime_score[m]   = exp(-alpha * regime_smape[m, P, R])  or 1.0 (neutral)
trust_score[m]    = 1.0  (all models passed trust gate)

score[m]          = base_score[m] * stability_score[m] * regime_score[m]
weight[m]         = normalize(score, min=0.05, max=0.75, cfg05_floor=0.30)
```

Where `P` = period (1_8, 9_16, 17_24), `R` = regime, `alpha` = 5.0, `beta` = 3.0.

### Trust Gating

| Trust State | `trusted_delivery` | `balanced_candidate` |
|-------------|------------------|-------------------|
| TRUSTED | Allowed | Allowed |
| DELIVERY_ALLOWED | Allowed | Allowed |
| COMPLETE_24H | Allowed | Allowed |
| CONSERVATIVE_QUARANTINE | **Blocked** | Allowed |
| SUSPECT_LEAKAGE | Blocked | Blocked |
| DRY_RUN / STUB / DATA_MISSING / INVALID_24H | Blocked | Blocked |

### Safety Constraints

| Constraint | Value | Condition |
|------------|-------|-----------|
| `min_weight` | 0.05 (5%) | Always applied |
| `max_weight` | 0.75 (75%) | Always applied |
| `cfg05_floor` | 0.30 (30%) | `trusted_delivery` profile only |

### Internal Fallback Chain

| Level | Method | Threshold | Description |
|-------|--------|-----------|-------------|
| 1 | `regime_bgew` | >= 10 training days | Full regime + period weights |
| 2 | `period_bgew` | >= 5 training days | Period-only weights |
| 3 | `equal_weight` | always | Simple average |
| 4 | `failed` | — | Return failure, caller falls back to P54 |

### Usage

```python
from fusion.trust_gated_regime_bgew import run_trust_gated_regime_bgew

result = run_trust_gated_regime_bgew(
    target_date="2026-07-05",
    trusted_models=["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
    prediction_ledger_path="ledgers/prediction_ledger.csv",
    actual_ledger_path="ledgers/actual_ledger.csv",
    profile_name="trusted_delivery",
)
if result["success"]:
    output_df = result["output"]    # 24-row DataFrame
    print(result["regime"])         # e.g. "normal"
    print(result["weights"])        # per-period weight dict
```

### CLI Integration

```
--fusion-engine regime_bgew    (use P56 as the fusion engine)
--fusion-engine period_bgew    (use P35 period BGEW, default)
--fusion-engine equal_weight   (use simple equal weighting)
--fusion-engine cfg05          (use cfg05 baseline only)
```

---

## 8. How They Chain Together

The full safety chain in the P47/P57 runner:

```
Step 3: P53 Leakage Sentinel
  - Loads prediction + actual ledgers
  - Runs 11 checks per trusted model
  - If SUSPECT_LEAKAGE detected and --strict-no-leakage: FAIL
  - Otherwise: WARNING (model quarantined but pipeline continues)
        |
        v
Step 4: P52 Adaptive Training Days
  - Scans backwards from D-1 for complete days
  - Returns COMPLETE_30D / DEGRADED / INSUFFICIENT / NO_VALID_DAYS
  - If --allow-degraded: DEGRADED treated as PASSED
        |
        v
Steps 5-8: Trust Gate + Fusion + Rolling Validation
  - P41 trust gate selects eligible models
  - P42/P43 fusion and validation (skipped if < 2 trusted models)
        |
        v
Step 9: P54 Fallback Ladder
  - Attempts levels 1-5 in order
  - First valid 24-row output wins
  - Sets delivery_status to NORMAL / DEGRADED_DELIVERED / FAILED_NO_DELIVERY
        |
        v
Step 10: P55 Postflight Validation
  - 12 checks on final_output.csv
  - Generates manifest + report files
  - PASS -> NORMAL, WARN -> DEGRADED, FAIL -> FAILED_NO_DELIVERY
        |
        v
Step 11-13: Summary + Forbidden File Check + Claim Guard
  - Final audit before delivery claim
```

### Fallback Cascade Diagram

```
P56 Regime BGEW (if --fusion-engine regime_bgew)
  |
  +-- success? --> NORMAL delivery
  |
  +-- fail --> P54 Fallback Ladder Level 1 (trusted_bgew_fusion)
                |
                +-- success? --> NORMAL
                |
                +-- fail --> Level 2 (trusted_equal_weight)
                              |
                              +-- success? --> DEGRADED_DELIVERED
                              |
                              +-- fail --> Level 3 (best_trusted_single_model)
                                            |
                                            +-- success? --> DEGRADED_DELIVERED
                                            |
                                            +-- fail --> Level 4 (cfg05_baseline)
                                                          |
                                                          +-- success? --> DEGRADED_DELIVERED
                                                          |
                                                          +-- fail --> Level 5 (historical_median)
                                                                        |
                                                                        +-- success? --> DEGRADED_DELIVERED
                                                                        |
                                                                        +-- fail --> Level 6 (FAILED_NO_DELIVERY)
```

---

## 9. File Inventory

| Component | File | Description |
|-----------|------|-------------|
| P52 Adaptive Training Days | `fusion/adaptive_training_days.py` | Complete training day selector |
| P53 Leakage Sentinel | `safety/leakage_sentinel.py` | Runtime leakage detection |
| P54 Fallback Ladder | `delivery/fallback_ladder.py` | 6-level progressive fallback |
| P55 Postflight | `delivery/postflight.py` | 12-check output validation |
| P55 Manifest | `delivery/manifest.py` | Run manifest create/write/read/validate |
| P55 Report | `delivery/report.py` | Markdown + JSON delivery report generator |
| P56 Regime BGEW | `fusion/trust_gated_regime_bgew.py` | Regime-aware adaptive fusion |
| P57 Runner | `scripts/run_delivery_local_chain.py` | Pipeline orchestrator with safety integration |

### Test Files

| Component | Test File | Test Count |
|-----------|-----------|------------|
| P52 | `tests/test_p52_adaptive_training_days.py` | 27 |
| P53 | `tests/test_p53_leakage_sentinel.py` | 25 |
| P54 | `tests/test_p54_fallback_ladder.py` | 31 |
| P55 | `tests/test_p55_postflight_manifest_report.py` | 21 |
| P56 | `tests/test_p56_trust_gated_regime_bgew.py` | 39 |

---

## 10. Configuration Reference

### CLI Flags (P57 Runner)

| Flag | Default | Description |
|------|---------|-------------|
| `--fusion-engine` | `period_bgew` | `regime_bgew` / `period_bgew` / `equal_weight` / `cfg05` |
| `--required-training-days` | 30 | Complete training days required |
| `--max-lookback-days` | 180 | Calendar days to scan |
| `--min-days-for-degraded` | 7 | Minimum for degraded mode |
| `--allow-degraded` | off | Accept DEGRADED training status |
| `--strict-no-leakage` | off | Fail on any leakage trigger |
| `--profile` | `trusted_delivery` | Delivery profile |
| `--strict` | off | Exit non-zero on any step failure |

### Threshold Constants

| Constant | Value | Component |
|----------|-------|-----------|
| `CORR_THRESHOLD` | 0.995 | P53 Leakage Sentinel |
| `WITHIN_1PCT_THRESHOLD` | 0.80 | P53 Leakage Sentinel |
| `SMAPE_FLOOR50_TOO_GOOD` | 2.0% | P53 Leakage Sentinel |
| `MAE_TOO_GOOD` | 10.0 CNY | P53 Leakage Sentinel |
| `required_days` | 30 | P52 Adaptive Training Days |
| `max_lookback_days` | 180 | P52 Adaptive Training Days |
| `min_days_for_degraded` | 7 | P52 Adaptive Training Days |
| `alpha` | 5.0 | P56 Regime BGEW |
| `beta` | 3.0 | P56 Regime BGEW |
| `min_weight` | 0.05 | P56 Weight normalization |
| `max_weight` | 0.75 | P56 Weight normalization |
| `cfg05_floor` | 0.30 | P56 (trusted_delivery profile) |
| `min_training_days_for_regime` | 10 | P56 Regime BGEW level |
| `min_training_days_for_period` | 5 | P56 Period BGEW level |
