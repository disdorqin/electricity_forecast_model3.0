# P53 Leakage Sentinel Runtime Guard Report

> **Generated**: 2026-07-05
> **Phase**: P53
> **Status**: IMPLEMENTED

---

## 1. Implementation Overview

The Leakage Sentinel Runtime Guard (`safety/leakage_sentinel.py`) is a standalone safety module that checks every model on every run for data leakage indicators. It is designed specifically for day-ahead price forecasting (3.0's domain) and does **not** check realtime forecasting since 3.0 is DA_ONLY/DRY_RUN.

### Module Structure

| Component | File | Description |
|-----------|------|-------------|
| Package init | `safety/__init__.py` | Package marker |
| Sentinel logic | `safety/leakage_sentinel.py` | Core check, batch runner, delivery gate |
| Test suite | `tests/test_p53_leakage_sentinel.py` | 19 tests covering all statuses and edge cases |

### Public API

- **`check_model_leakage(model_name, prediction_ledger_path, actual_ledger_path, feature_columns)`** — Run all 11 leakage checks for a single model. Returns dict with `status`, `checks` dict, `details`, and `suspicion_reasons`.
- **`run_leakage_sentinel(trusted_models, prediction_ledger_path, actual_ledger_path)`** — Run the sentinel on all trusted models. Returns summary dict with per-model results and cross-model eval-row consistency check.
- **`is_delivery_allowed(model_name, sentinel_result, profile_name)`** — Check if a model is allowed for delivery based on sentinel result and profile. Implements the action matrix.

---

## 2. Check Types and Thresholds

### Eleven Runtime Checks

| # | Check | What it detects | Result if FAIL |
|---|-------|-----------------|----------------|
| 1 | `no_y_true_in_prediction_ledger` | Prediction ledger must NOT contain a `y_true`, `target`, or `日前电价` column | `INVALID_SCHEMA` |
| 2 | `no_target_in_features` | Feature columns must not include target column names | `INVALID_SCHEMA` |
| 3 | `sufficient_eval_rows` / eval rows consistency | Model must have >= 24 eval rows after merge + NaN drop; all models should have consistent row counts | `INVALID_24H` / warning |
| 4 | `within_1pct_ratio` | Ratio of predictions within 1% of actual value | `CONSERVATIVE_QUARANTINE` if > 80% |
| 5 | `corr_y_pred_y_true` | Pearson correlation between prediction and actual | `CONSERVATIVE_QUARANTINE` if > 0.995 |
| 6 | `sMAPE_floor50` | sMAPE with floor of 50 | `SUSPECT_LEAKAGE` if < 2% |
| 7 | `MAE` | Mean Absolute Error in CNY | `SUSPECT_LEAKAGE` if < 10 CNY |
| 8 | `no_future_timestamps` | Any prediction timestamp in the future | `SUSPECT_LEAKAGE` |
| 9 | `no_target_day_overlap` | Target-day overlap (informational; training overlap cannot be detected from ledgers alone) | Informational (always passes) |
| 10 | `no_duplicate_keys` | Duplicate `(business_day, hour_business)` rows | `SUSPECT_LEAKAGE` |
| 11 | `24h_completeness` | All 24 hours (1..24) present in prediction | `INVALID_24H` |

### Threshold Constants

| Constant | Value | Used for |
|----------|-------|----------|
| `CORR_THRESHOLD` | 0.995 | Check 5 |
| `WITHIN_1PCT_THRESHOLD` | 0.80 | Check 4 |
| `SMAPE_FLOOR50_TOO_GOOD` | 2.0 (%) | Check 6 |
| `MAE_TOO_GOOD` | 10.0 (CNY) | Check 7 |

### Merge Strategy

Prediction and actual ledgers are merged on `(business_day, hour_business)` for metric computation. This aligns with the spec for day-ahead evaluation where each business day has 24 hourly predictions (D+01:00 through D+1 00:00).

---

## 3. Status Determination

### Priority Order

Models are assigned exactly one status according to this priority (highest first):

1. **`INVALID_SCHEMA`** — Structural issue (missing columns, y_true in prediction ledger, target in features, model not found)
2. **`INVALID_24H`** — Incomplete 24-hour coverage
3. **`SUSPECT_LEAKAGE`** — Clear leakage evidence (sMAPE < 2%, MAE < 10, future timestamps, duplicate keys)
4. **`CONSERVATIVE_QUARANTINE`** — Borderline indicators (within-1% ratio > 80%, correlation > 0.995)
5. **`TRUSTED`** — All checks pass

### Status Assignment Logic

```
if any INVALID_SCHEMA condition:
    status = INVALID_SCHEMA
elif not 24h_completeness:
    status = INVALID_24H
elif any SUSPECT_LEAKAGE trigger:
    status = SUSPECT_LEAKAGE
elif any CONSERVATIVE_QUARANTINE trigger:
    status = CONSERVATIVE_QUARANTINE
else:
    status = TRUSTED
```

---

## 4. Action Matrix

### `is_delivery_allowed` Rules

| Status | `trusted_delivery` | `balanced_candidate` | `research_all_models` |
|--------|-------------------|---------------------|----------------------|
| `TRUSTED` | ALLOWED | ALLOWED | ALLOWED |
| `CONSERVATIVE_QUARANTINE` | BLOCKED | ALLOWED | ALLOWED |
| `SUSPECT_LEAKAGE` | BLOCKED | BLOCKED | BLOCKED |
| `INVALID_SCHEMA` | BLOCKED | BLOCKED | BLOCKED |
| `INVALID_24H` | BLOCKED | BLOCKED | BLOCKED |

### Impact on Fusion Pipeline

- **SUSPECT_LEAKAGE** models are quarantined: they cannot enter the delivery fusion pool regardless of profile.
- **CONSERVATIVE_QUARANTINE** models are excluded from `trusted_delivery` but remain accessible in research profiles with caveats.
- **INVALID_SCHEMA** and **INVALID_24H** models are treated as data errors and blocked everywhere until fixed.

---

## 5. Test Summary

### 19 Tests in `test_p53_leakage_sentinel.py`

| # | Test | Status |
|---|------|--------|
| 1 | `TestTrustedModel::test_trusted_status` | TRUSTED status assigned |
| 2 | `TestTrustedModel::test_trusted_all_checks_pass` | All checks True for trusted model |
| 3 | `TestTrustedModel::test_trusted_metrics_populated` | Metrics populated correctly |
| 4 | `TestSuspectLeakage::test_within_1pct_ratio_triggers_conservative` | High within_1pct -> CONSERVATIVE_QUARANTINE |
| 5 | `TestSuspectLeakage::test_smape_too_good_triggers_suspect` | Low sMAPE -> SUSPECT_LEAKAGE |
| 6 | `TestSuspectLeakage::test_mae_too_good_triggers_suspect` | Low MAE -> SUSPECT_LEAKAGE |
| 7 | `TestSuspectLeakage::test_future_timestamp_triggers_suspect` | Future ds -> SUSPECT_LEAKAGE |
| 8 | `TestSuspectLeakage::test_duplicate_keys_triggers_suspect` | Duplicate keys -> SUSPECT_LEAKAGE |
| 9 | `TestConservativeQuarantine::test_corr_high_triggers_conservative_quarantine` | High corr -> CONSERVATIVE_QUARANTINE |
| 10 | `TestInvalidSchema::test_missing_model_name_column` | Missing model_name -> INVALID_SCHEMA |
| 11 | `TestInvalidSchema::test_y_true_in_prediction_ledger` | y_true in pred -> INVALID_SCHEMA |
| 12 | `TestInvalidSchema::test_missing_merge_keys` | Missing merge keys -> INVALID_SCHEMA |
| 13 | `TestInvalid24H::test_not_24_hours` | < 24 hours -> INVALID_24H |
| 14 | `TestEdgeCases::test_missing_ledger_file_graceful` | Missing file -> INVALID_SCHEMA |
| 15 | `TestEdgeCases::test_nan_y_pred_detected` | NaN y_pred -> warning |
| 16 | `TestEdgeCases::test_model_name_mismatch` | Model not found -> INVALID_SCHEMA |
| 17 | `TestEdgeCases::test_empty_prediction_ledger` | Empty ledger -> INVALID_SCHEMA |
| 18 | `TestEdgeCases::test_target_in_feature_columns_detected` | target in features -> INVALID_SCHEMA |
| 19 | `TestDeliveryAllowed::test_delivery_allows_trusted` | TRUSTED allowed for delivery |
| 20 | `TestDeliveryAllowed::test_delivery_blocks_suspect_leakage` | SUSPECT_LEAKAGE blocked everywhere |
| 21 | `TestDeliveryAllowed::test_delivery_blocks_conservative_for_delivery_profile` | CONSERVATIVE blocked for delivery |
| 22 | `TestDeliveryAllowed::test_delivery_allows_conservative_for_research_profile` | CONSERVATIVE allowed for research |
| 23 | `TestDeliveryAllowed::test_delivery_blocks_invalid_schema` | INVALID_SCHEMA blocked everywhere |
| 24 | `TestDeliveryAllowed::test_delivery_blocks_unknown_model` | Unknown model blocked |
| 25 | `TestRunLeakageSentinel::test_sentinel_summary_structure` | Batch runner structure verified |

---

## 6. Design Decisions

### Why separate from P41 (Model Trust Gate)?

P41 is a one-time evaluation script that reads CSV files from `.local_artifacts`. P53 is a **runtime guard** that:
- Is a reusable Python module (not a script)
- Reads parquet ledgers (production format)
- Has more status categories (INVALID_SCHEMA, INVALID_24H, CONSERVATIVE_QUARANTINE vs P41's TRUSTED/SUSPECT_LEAKAGE)
- Integrates with the profile-based delivery gate
- Uses stricter thresholds appropriate for runtime checking

### Why CSV fallback in `_load_ledger`?

While the spec mandates parquet reading, the sentinel falls back to CSV for development environments where pyarrow may not be installed. This makes the module testable without additional dependencies.

### Why is `no_target_day_overlap` always marked as pass?

Training-data overlap cannot be detected from ledgers alone (ledgers only contain prediction results). Suspiciously good metrics that would result from training leakage are caught by checks 4-7 (within_1pct, corr, sMAPE, MAE). The check is retained as informational for future enhancement when training metadata becomes available.

---

## 7. Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `safety/__init__.py` | 1 | Package marker |
| `safety/leakage_sentinel.py` | ~300 | Leakage Sentinel implementation |
| `tests/test_p53_leakage_sentinel.py` | ~400 | 25 test cases in 8 test classes |
| `docs/reports/p53_leakage_sentinel_report.md` | ~180 | This report |
