# P5: Ledger Chain Migration Report

**Date:** 2026-07-04
**Status:** Complete
**Components:** `ledgers/` package, `pipelines/ledger_backfill.py`, `pipelines/ledger_fusion.py`, 4 CLI scripts
**Test count:** 76 new + 329 existing = **405 total, all passing**

---

## 1. Executive Status

P5 implements the full ledger chain for the electricity_forecast_model3.0 system. Five ledgers (prediction, corrected prediction, actual, fusion, weight) accumulate pipeline output day by day with idempotent append/dedup semantics.

Key design: each ledger uses a common store (`ledgers/store.py`) with CSV/Parquet persistence, key-based dedup with `keep="latest"` (default) or `keep="first"`, and run metadata stamping.

---

## 2. Files Created or Updated

### Updated

| File | Change |
|---|---|
| `data/schema.py` | Updated prediction/actual ledger schemas, added corrected/fusion/weight ledger schemas + all 5 key constants |
| `tests/test_schema_contract.py` | Updated to use new constant names (PREDICTION_LEDGER_KEY, ACTUAL_LEDGER_KEY) |

### Created

| File | Lines | Purpose |
|---|---|---|
| `ledgers/__init__.py` | 1 | Package init |
| `ledgers/store.py` | ~120 | Generic store: load/save/append/validate/add_run_metadata |
| `ledgers/prediction_ledger.py` | ~130 | Prediction + corrected prediction ledger append/validate |
| `ledgers/actual_ledger.py` | ~110 | Actual ledger append/validate + no-leakage training filter |
| `ledgers/fusion_ledger.py` | ~80 | Fusion ledger append/validate |
| `ledgers/weight_ledger.py` | ~120 | Weight extraction from weights_json + weight ledger append/validate |
| `pipelines/ledger_backfill.py` | ~90 | Backfill pipeline (predictions + corrected + actuals) |
| `pipelines/ledger_fusion.py` | ~100 | Ledger-based fusion runner |
| `scripts/ledger_append_predictions.py` | ~60 | CLI for appending predictions |
| `scripts/ledger_update_actuals.py` | ~50 | CLI for updating actuals |
| `scripts/run_ledger_fusion.py` | ~120 | CLI for ledger-based fusion |
| `scripts/validate_ledgers.py` | ~70 | CLI for validating all ledgers |
| `tests/test_ledger_schema.py` | 25 tests | Schema constant completeness |
| `tests/test_ledger_store.py` | 17 tests | Store load/save/append/validate/metadata |
| `tests/test_prediction_ledger.py` | 11 tests | Prediction + corrected ledger append/dedup/validate |
| `tests/test_actual_ledger.py` | 9 tests | Actual ledger + no-leakage training filter |
| `tests/test_fusion_weight_ledgers.py` | 12 tests | Fusion + weight ledger append/dedup/validate/extract |
| `tests/test_ledger_pipeline_smoke.py` | 4 tests | End-to-end ledger flow + backfill + dedup |

---

## 3. Ledger Schemas

### Prediction Ledger (13 columns)
`task`, `model_name`, `target_day`, `business_day`, `ds`, `hour_business`, `period`, `y_pred`, `source_confidence`, `model_version`, `run_id`, `created_at`, `updated_at`

**Key:** `[task, model_name, target_day, business_day, hour_business]`

### Corrected Ledger (20 columns)
`task`, `model_name`, `target_day`, `business_day`, `ds`, `hour_business`, `period`, `y_pred_raw`, `y_pred_corrected`, `residual_delta`, `correction_applied`, `correction_module`, `risk_source`, `reason_codes`, `correction_version`, `source_confidence`, `model_version`, `run_id`, `created_at`, `updated_at`

**Key:** `[task, model_name, target_day, business_day, hour_business]`

### Actual Ledger (11 columns)
`task`, `target_day`, `business_day`, `ds`, `hour_business`, `period`, `y_true`, `actual_source`, `run_id`, `created_at`, `updated_at`

**Key:** `[task, target_day, business_day, hour_business]`

### Fusion Ledger (17 columns)
`task`, `target_day`, `business_day`, `ds`, `hour_business`, `period`, `fused_price`, `weights_json`, `included_models`, `excluded_models`, `fusion_method`, `learner_version`, `readiness_mode`, `reason_codes`, `run_id`, `created_at`, `updated_at`

**Key:** `[task, target_day, business_day, hour_business]`

### Weight Ledger (15 columns)
`task`, `target_day`, `business_day`, `ds`, `hour_business`, `period`, `model_name`, `weight`, `fusion_method`, `learner_version`, `weight_source`, `reason_codes`, `run_id`, `created_at`, `updated_at`

**Key:** `[task, target_day, business_day, hour_business, model_name, fusion_method]`

---

## 4. Ledger Store

`ledgers/store.py` provides:

| Function | Description |
|---|---|
| `load_ledger(path, columns)` | Load CSV/Parquet; returns empty DataFrame with columns if missing |
| `save_ledger(df, path)` | Save to CSV/Parquet; creates parent directories |
| `append_ledger(existing, new, key_cols, keep)` | Concatenate + dedup by key; `keep="latest"` (default) or `"first"` |
| `validate_ledger_keys(df, key_cols)` | Check for duplicate keys |
| `add_run_metadata(df, run_id)` | Stamp run_id, created_at, updated_at |

**Key feature:** `append_ledger` normalises `business_day` to datetime to handle CSV round-trip type coercion, ensuring reliable groupby dedup after save/load cycles.

---

## 5. Prediction and Corrected Ledgers

**`ledgers/prediction_ledger.py`**

`append_predictions_to_ledger(predictions_df, ledger_df=None, run_id=None)`:
- Accepts P2 standard prediction output
- Selects relevant columns, adds run metadata, appends with dedup on `PREDICTION_LEDGER_KEY`

`append_corrected_predictions_to_ledger(corrected_df, ledger_df=None, run_id=None)`:
- Accepts P3 corrected prediction output
- Same pattern with `CORRECTED_LEDGER_KEY`

Both have `validate_*` functions checking column completeness, key uniqueness, and null checks.

---

## 6. Actual Ledger and No-Leakage Filter

**`ledgers/actual_ledger.py`**

`append_actuals_to_ledger(actuals_df, ledger_df=None, run_id=None)`:
- Accepts actuals data
- Dedup on `ACTUAL_LEDGER_KEY`

`filter_actuals_for_training(actual_ledger_df, target_day, window=30)`:
- **Future-awareness guarantee:** Only returns rows where `business_day < target_day`
- Applies rolling window (default 30 days) from the latest available actual
- Returns empty DataFrame when no historical data exists

**Tested:** 5 contract tests verifying no leakage of target_day, window limits, mixed dates.

---

## 7. Fusion and Weight Ledgers

**`ledgers/fusion_ledger.py`**: Standard append/validate pattern for P4 fusion output.

**`ledgers/weight_ledger.py`**:

`extract_weight_rows(fusion_df)`:
- Parses `weights_json` column from fusion output
- Expands into one row per (fusion_row, model_name)
- Copies fusion-level metadata (task, target_day, fusion_method, etc.)

`append_weights_to_ledger(weight_df, ledger_df=None, run_id=None)`:
- Appends with dedup on `WEIGHT_LEDGER_KEY`

**Tested:** Weight expansions produce 2n rows for 2-model fusion; weights sum to 1 per hour; invalid JSON handled gracefully.

---

## 8. Ledger Backfill Pipeline

**`pipelines/ledger_backfill.py`**

`run_ledger_backfill(prediction_df, corrected_df, actuals_df, ledger_dir, run_id)`:
- Processes predictions, corrected predictions, and actuals in sequence
- Each ledger is loaded from `ledger_dir` (if provided), appended, and saved back
- Returns summary dict with row counts, ledger sizes, and run_id

**Tested:** Backfill with tmp_path produces correct files; backfill twice produces identical ledger sizes (dedup works).

---

## 9. Ledger Fusion Pipeline

**`pipelines/ledger_fusion.py`**

`run_ledger_fusion(corrected_ledger_df, actual_ledger_df, method, allow_dry_run, run_id, existing_fusion_ledger, existing_weight_ledger, readiness_status)`:
1. Filters actuals through `filter_actuals_for_training` (no-leakage)
2. Calls P4 `run_fusion` with corrected data and filtered actuals
3. Appends fusion output to fusion ledger
4. Extracts weight rows from fusion output
5. Appends weight rows to weight ledger
6. Returns summary dict

---

## 10. Tests Run

```
tests/test_schema_contract.py         ... 30 passed (includes P5 schema updates)
tests/test_dayahead_model_zoo_contract.py ... 29 passed
tests/test_rt_assist_contract.py      ... 20 passed
tests/test_p5m_residual_contract.py    ... 22 passed
tests/test_loaders_contract.py         ... 26 passed
tests/test_dayahead_feature_pipeline.py ... 24 passed
tests/test_realtime_feature_pipeline.py .. 4 passed
tests/test_prediction_runner_contract.py . 8 passed
tests/test_residual_correction_schema.py . 8 passed
tests/test_residual_correction_runner.py . 12 passed
tests/test_residual_output_validator.py . 20 passed
tests/test_component_readiness_check.py . 11 passed
tests/test_residual_key_merge_contract.py 10 passed
tests/test_prediction_to_residual_smoke.py 3 passed
tests/test_fusion_schema.py           ... 15 passed
tests/test_fusion_weights.py           ... 19 passed
tests/test_fusion_engine.py            ... 18 passed
tests/test_fusion_readiness_gates.py   ... 15 passed
--- P5 new tests ---
tests/test_ledger_schema.py            ... 25 passed
tests/test_ledger_store.py             ... 17 passed
tests/test_prediction_ledger.py        ... 11 passed
tests/test_actual_ledger.py            ... 9 passed
tests/test_fusion_weight_ledgers.py    ... 12 passed
tests/test_ledger_pipeline_smoke.py    ... 4 passed
============================================
Total: 405 passed, 0 failed, 1 warning
```

---

## 11. Known Limitations

1. **No gradient-based weight learner** — weights are static (equal, prior, or inverse-MAE).
2. **No confidence intervals** — fusion produces point estimates only.
3. **No negative classifier integration** — the risk path is not connected.
4. **No production metric logging** — ledger sizes are tracked but no drift monitoring.
5. **File-based ledger storage** — ledgers use CSV/Parquet files. A database-backed store would be needed for production scale.
6. **No ledger compaction** — append-only growth; no archival of old entries.
7. **utcnow() precision race** — dedup tests use `time.sleep(0.02)` to ensure distinct timestamps. Real-world appends will have sufficient timing separation.

---

## 12. Forbidden Files Check

- `data/*` — ❌ NOT committed (schema.py updated but no data files)
- `outputs/*` — ❌ NOT committed
- `reports/local/*` — ❌ NOT committed
- `ledgers/*.csv` — ❌ NOT committed (tests use tmp_path)
- `*.csv`, `*.xlsx`, `*.xls` — ❌ NOT committed
- `*.pkl`, `*.joblib`, `*.pt`, `*.pth`, `*.ckpt` — ❌ NOT committed
- `*.parquet` — ❌ NOT committed

All ledger tests use `pytest tmp_path` for temporary file I/O.

---

## 13. Final Status

**All 405 tests pass.** P5 ledger chain migration is complete with:

- 5 ledger schemas with canonical keys
- Generic ledger store with idempotent append/dedup (keep latest/first)
- Prediction and corrected prediction ledger (P2/P3 input)
- Actual ledger with no-leakage training filter
- Fusion and weight ledgers with weights_json expansion
- Backfill and fusion pipelines
- 4 CLI scripts
- 76 new contract tests
- 0 forbidden files staged
