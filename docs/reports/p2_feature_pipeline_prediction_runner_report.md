# P2 Feature Pipeline Prediction Runner Report

## 1. Executive status

P2 establishes the data loading layer, feature engineering pipelines, prediction runners, and output validation for the 3.0 system. All 75 new contract tests pass alongside the 96 P1 tests (171 total, 0 failures). No fusion, no 2.5 ledger chain migration, no negative classifier, no training.

| Component | Status |
|-----------|--------|
| Unified data loader (`data/loaders.py`) | DONE — 23 tests |
| Day-ahead feature pipeline (`data/features/dayahead_features.py`) | DONE — 15 tests |
| Realtime feature pipeline (`data/features/realtime_features.py`) | DONE — 15 tests |
| Day-ahead model zoo runner (`scripts/run_dayahead_model_zoo.py`) | DONE — 8 tests |
| Realtime assist runner (`scripts/run_realtime_assist.py`) | DONE — 3 tests |
| Prediction output validator (`scripts/validate_prediction_output.py`) | DONE — 11 tests |

---

## 2. Files created or updated

| File | Action | Description |
|------|--------|-------------|
| `data/loaders.py` | **NEW** | Unified data loading with encoding fallback & Chinese column mapping |
| `data/features/__init__.py` | **NEW** | Package init |
| `data/features/dayahead_features.py` | **NEW** | cfg05 / day-ahead feature builder |
| `data/features/realtime_features.py` | **NEW** | Realtime assist input pipeline |
| `scripts/run_dayahead_model_zoo.py` | **NEW** | CLI runner for day-ahead model zoo |
| `scripts/run_realtime_assist.py` | **NEW** | CLI runner for DA-Safe Realtime Assist |
| `scripts/validate_prediction_output.py` | **NEW** | Prediction output schema validator |
| `src/registry/dayahead_models.py` | **UPDATED** | Added `feature_columns` to cfg05 config |
| `tests/test_loaders_contract.py` | **NEW** | 23 loader contract tests |
| `tests/test_dayahead_feature_pipeline.py` | **NEW** | 15 day-ahead feature tests |
| `tests/test_realtime_feature_pipeline.py` | **NEW** | 15 realtime feature tests |
| `tests/test_prediction_runner_contract.py` | **NEW** | 22 runner & validator tests |
| `docs/reports/p2_feature_pipeline_prediction_runner_report.md` | **NEW** | This report |

---

## 3. Loader layer (`data/loaders.py`)

### `load_table()` function

**Encoding fallback**: `utf-8` → `utf-8-sig` (BOM detection) → `gbk` → `gb18030`

**Chinese column mapping** (18 mappings):

| Chinese | Canonical |
|---------|-----------|
| 时间, 日期时间, timestamp, times | `ds` |
| 日前价格, 日前电价, da_price | `da_anchor` |
| 实时价格, 实时电价, rt_price | `rt_actual` |
| 预测价格, 预测值 | `y_pred` |
| 实际价格, 实际电价, actual_price | `y_true` |
| 负荷, load_actual | `load` |
| 风电, wind_actual | `wind` |
| 光伏, solar_actual | `solar` |
| 竞价空间, bidding | `bidding_space` |
| 净负荷 | `net_load` |

**Returns**: `(DataFrame, metadata_dict)` — metadata includes encoding_used, cn_mappings rows count, and an errors list.

### Validated (23 tests)

- utf-8, utf-8-sig (BOM), gbk all read correctly
- Chinese → canonical mapping for all 10 mapped pairs
- Unsupported extension raises `ValueError`
- Missing file raises `FileNotFoundError`
- `add_business_time=True` adds business_day, hour_business, period
- Midnight (00:00) maps to business_day D-1, hour 24
- Metadata keys completeness

---

## 4. Day-ahead feature pipeline (`data/features/dayahead_features.py`)

### Functions

| Function | Description |
|----------|-------------|
| `build_dayahead_features(df, model_id, fill_strategy)` | Build feature matrix for a day-ahead model |
| `get_dayahead_feature_columns(model_id)` | Return canonical feature column list |
| `validate_dayahead_feature_frame(df, model_id, strict)` | Validate feature DataFrame |
| `report_missing_features(df, model_id)` | Structured feature availability report |

### Design

- **42 feature columns** for cfg05 (sourced from registry config and adapter definition)
- **Business columns preserved**: `business_day`, `ds`, `hour_business`, `period`
- **Deny list**: `y_true`, `residual`, `error`, `abs_error` — any match in column names raises `ValueError`
- Missing features filled with 0 (default) or NaN
- `report_missing_features()` returns `{model_id, total_features, present[], missing[], ratio}`
- Invalid model raises `ValueError` immediately

### Validated (15 tests)

- cfg05 has 42 features
- Business columns preserved in output
- No y_true/residual/error/abs_error in output
- Invalid model raises ValueError
- Missing features filled with zero or NaN
- `validate_dayahead_feature_frame` detects denied columns
- Strict mode raises on denied columns
- Structured missing feature report

---

## 5. Realtime feature pipeline (`data/features/realtime_features.py`)

### Functions

| Function | Description |
|----------|-------------|
| `normalize_realtime_columns(df)` | Normalise column aliases → canonical names |
| `build_realtime_assist_input(df, production)` | Build standardised RT assist input |
| `validate_realtime_assist_input(df, production)` | Validate RT input completeness |

### Column normalisation

| Input alias | Canonical | Reason code |
|-------------|-----------|-------------|
| `times`, `timestamp`, `date_time` | `ds` | TIMESTAMP_FROM_... |
| `da_price` | `da_anchor` | DA_ANCHOR_FROM_DA_PRICE |
| `forecast_price` | `da_anchor` | DA_ANCHOR_FROM_FORECAST_PRICE |
| `rt_price`, `realtime_price` | `rt_actual` | RT_ACTUAL_FROM_RT_PRICE |

### Design

- Production mode: `rt_actual` optional (reason: `RT_ACTUAL_MISSING_FOR_PRODUCTION`)
- Eval mode: `rt_actual` required (error if missing)
- Business-time columns added if missing (reason: `BUSINESS_TIME_ADDED`)
- Reason codes tracked in metadata for audit trail

### Validated (15 tests)

- da_price → da_anchor mapping with reason code
- forecast_price → da_anchor fallback with reason
- rt_price → rt_actual mapping
- times/timestamp → ds mapping
- Business-time columns added when missing
- Production mode: missing rt_actual does not crash
- Eval mode: missing rt_actual reported in errors
- Missing da_anchor detected
- Missing ds detected

---

## 6. Day-ahead model zoo runner (`scripts/run_dayahead_model_zoo.py`)

### CLI

```
python scripts/run_dayahead_model_zoo.py --dry-run --out predictions.csv
python scripts/run_dayahead_model_zoo.py --models cfg05,best_two_average --dry-run
python scripts/run_dayahead_model_zoo.py --input data.csv --models cfg05 --model-dir ./weights
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--models` | Model IDs (comma-separated) or `default` for DEFAULT_FUSION_POOL |
| `--input` | Input CSV path |
| `--out` | Output CSV path |
| `--dry-run` | Synthetic predictions (no model weights needed) |
| `--allow-missing-model-artifacts` | Skip missing models instead of error |
| `--model-dir` | Model weight directory |
| `--target-date` | Target date (YYYY-MM-DD) |
| `--verbose` | Debug logging |

### Dry-run behavior

- Uses `da_anchor` or `y_pred` from input with small noise
- Falls back to random uniform(80, 200) if no anchor available
- Model version tagged as `dry_run`
- All models in DEFAULT_FUSION_POOL produce output

### Validated (8 tests + CLI tests)

- Dry-run produces standard schema
- Dry-run produces all 5 DEFAULT_FUSION_POOL models
- No NaN in y_pred, no y_true
- hour_business in 1..24
- Rejects invalid model
- `--models default` resolves to 5 configs

---

## 7. Realtime assist runner (`scripts/run_realtime_assist.py`)

### CLI

```
python scripts/run_realtime_assist.py --dry-run
python scripts/run_realtime_assist.py --input data.csv --out predictions.csv
python scripts/run_realtime_assist.py --input data.csv --enable-safe-correction --model-dir ./rt_assist_pack
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--input` | Input CSV path |
| `--out` | Output CSV path |
| `--model-dir` | RT assist pack directory |
| `--start` | Start date filter |
| `--end` | End date filter |
| `--dry-run` | Synthetic mode |
| `--enable-safe-correction` | Enable residual correction |

### Design

- Default: `rt_pred = da_anchor` via `DASafeRealtimeAssistAdapter`
- With `--enable-safe-correction`: loads model pack and applies correction
- Dry-run generates synthetic 48h data with da_anchor

### Validated (3 tests)

- Dry-run rt_pred == da_anchor
- Dry-run standard schema
- No NaN, no y_true

---

## 8. Prediction output validator (`scripts/validate_prediction_output.py`)

### CLI

```
python scripts/validate_prediction_output.py predictions.csv
python scripts/validate_prediction_output.py predictions.csv --require-24h
python scripts/validate_prediction_output.py predictions.csv --no-production
```

### Validation checks (10 total)

| # | Check | Error if |
|---|-------|----------|
| 1 | Input not empty | Empty DataFrame |
| 2 | Required columns | Missing any of PREDICTION_OUTPUT_COLUMNS |
| 3 | Eval-only columns | y_true present in production mode |
| 4 | hour_business range | Outside [1, 24] |
| 5 | period values | Not in 1_8 / 9_16 / 17_24 |
| 6 | y_pred NaN | Any NaN in y_pred |
| 7 | Key column NaN | Null in task/model_name/business_day/hour_business |
| 8 | Duplicate keys | Same (task, model_name, business_day, hour_business) |
| 9 | 24 rows per day | Optional (`--require-24h`) |
| 10 | Task values | Not dayahead or realtime |

### Validated (11 tests)

- Valid DataFrame passes all checks
- Duplicate keys detected
- NaN y_pred detected
- y_true rejected in production mode
- y_true allowed in eval mode
- hour_business out of range detected
- Invalid period detected
- Missing required column detected
- Empty DataFrame detected
- `--require-24h` catches partial days

---

## 9. Tests run

| Test suite | Tests | Passed | Failed |
|---|---|---|---|
| `test_schema_contract.py` (P1) | 31 | 31 | 0 |
| `test_dayahead_model_zoo_contract.py` (P1) | 23 | 23 | 0 |
| `test_rt_assist_contract.py` (P1) | 21 | 21 | 0 |
| `test_p5m_residual_contract.py` (P1) | 21 | 21 | 0 |
| `test_loaders_contract.py` (P2) | 23 | 23 | 0 |
| `test_dayahead_feature_pipeline.py` (P2) | 15 | 15 | 0 |
| `test_realtime_feature_pipeline.py` (P2) | 15 | 15 | 0 |
| `test_prediction_runner_contract.py` (P2) | 22 | 22 | 0 |
| **Total** | **171** | **171** | **0** |

---

## 10. Known limitations

1. **cf05 feature count**: The source review reported 44 features, but the actual cfg05 champion feature list contains 42 features. Verified against the adapter definition (`models/adapters/cfg05_dayahead_lgbm.py:CFG05_FEATURE_COLUMNS`). This is the correct count — the source review overcounted.

2. **Feature engineering is not actual feature generation**: The day-ahead feature pipeline maps column names and validates feature frames. Actual feature computation (lags, rolling stats, calendar features, interaction terms) requires real market data and feature computation logic. This is the engineering interface only.

3. **Real-time feature pipeline does not compute features**: The realtime pipeline normalises column names and validates input. Deep model features (residual history, regime features, model disagreement) require the RT model pack.

4. **Zoo runner does not wire adapter for non-cfg05 models**: Real execution for `best_two_average`, `stage3_business_fixed`, `catboost_spike_residual`, and `catboost_sota` requires adapter implementations. Only cfg05 has a wired adapter. All models work in dry-run mode.

5. **SGDFNet realtime runner not included**: SGDFNet is a CANDIDATE adapter (no weights). The realtime runner covers the DA-Safe Realtime Assist path.

6. **No safe correction without model pack**: `--enable-safe-correction` requires an rt_assist_pack directory. Without it, the adapter falls back to DA-only.

---

## 11. Forbidden files check

All 171 tests use synthetic tiny DataFrames. No real data files, model weights, CSVs, Excel files, or pickle files are tracked. Test fixtures using CSV use `pytest tmp_path` exclusively.

```
data/*       → NOT committed
outputs/*    → NOT committed
reports/local/* → NOT committed
*.csv        → NOT committed (except test fixtures via tmp_path)
*.xlsx       → NOT committed
*.pkl        → NOT committed
*.joblib     → NOT committed
*.pt / *.pth → NOT committed
*.ckpt       → NOT committed
*.parquet    → NOT committed
```

**Result**: PASS

---

## 12. Final status

```
P2 Execution Summary

1. Files created:      11 (loaders.py, features/*, scripts/*, 4 tests, 1 report)
2. Files updated:      1  (src/registry/dayahead_models.py — added feature_columns)
3. Tests added:        75 (23+15+15+22)
4. Tests run:          171 total (96 P1 + 75 P2), all pass
5. Day-ahead pipeline: DONE — 42 features, deny-list enforcement, missing feature reporting
6. Realtime pipeline:  DONE — column normalisation, reason codes, production/eval modes
7. Prediction validator: DONE — 10 checks, CLI and programmatic API
8. Known limitations:  See §10 — 6 items documented
9. Forbidden files:    PASS
10. Commit:            Pending (P2 commit)
11. Final status:      COMPLETE — Ready for P3 (Fusion Engine)
```
