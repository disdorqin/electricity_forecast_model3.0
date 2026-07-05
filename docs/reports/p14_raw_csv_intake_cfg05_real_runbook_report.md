# P14 Raw CSV Intake + cfg05 REAL Runbook Report
> **Research Only** — This report documents pre-leakage-discovery work. Results are not delivery claims. "not delivery"

> **Phase**: P14 — One-command raw CSV intake, schema inspection, and cfg05 REAL run
> **Generated**: 2026-07-04
> **Test count**: 675 total (656 prior + 19 P14 new), 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Raw CSV intake | **CFG05_RAW_DATA_VALID** |
| Schema inspector | **Implemented** (redacted by default) |
| P14 one-command wrapper | **Implemented** |
| cfg05 local train/export | **EXPORTED** |
| cfg05 REAL smoke | **REAL_READY** |
| Final status | **CFG05_REAL_READY_LOCAL: ACHIEVED** |

## 2. Raw CSV Intake Status

| Field | Value |
|-------|-------|
| Path | `electricity_forecast_model2.1/data/shandong_pmos_hourly.csv` |
| Status | **CFG05_RAW_DATA_VALID** |
| Rows | 39,408 |
| Columns | 23 total, 8 required Chinese columns present |
| Time range | 2022-01-01 01:00:00 ~ 2026-07-01 00:00:00 |
| Encoding | GBK |

### Required Columns Verified

| # | Column | Status |
|---|--------|--------|
| 1 | 时刻 | ✅ |
| 2 | 日前电价 | ✅ (24 nulls) |
| 3 | 实时电价 | ✅ (48 nulls) |
| 4 | 直调负荷预测值 | ✅ |
| 5 | 风电总加预测值 | ✅ |
| 6 | 光伏总加预测值 | ✅ |
| 7 | 联络线受电负荷预测值 | ✅ |
| 8 | 竞价空间预测值 | ✅ |

All columns are float64 dtype. Minor null counts in 日前电价 (24/39408) and 实时电价 (48/39408) are acceptable for training.

## 3. Schema Inspection Result

**Schema inspector** (`scripts/inspect_cfg05_raw_csv_schema.py`):
- Reads CSV (GBK → UTF-8 fallback)
- Reuses `check_cfg05_raw_data_contract` for column/timestamp/numeric validation
- Default: **values redacted** (`--no-redact-values` to show)
- Outputs: status, rows, columns, dtypes, null_counts, time_min/max, missing_columns
- Strict mode: exits non-zero on CFG05_RAW_DATA_INVALID/MISSING

## 4. cfg05 Local Train/Export Result

| Field | Value |
|-------|-------|
| Target day | 2026-06-30 |
| Training window | 90 days (2026-04-01 ~ 2026-06-29) |
| Training rows | 2,159 |
| Feature rows | 23 |
| Model path | `.local_artifacts/p14_cfg05/cfg05_model.txt` (9 MB) |
| Feature CSV path | `.local_artifacts/p14_cfg05/cfg05_features_2026-06-30.csv` (11 KB) |
| cfg05 artifact status | **LOADABLE** |
| cfg05 input status | **SCHEMA_READY** |

## 5. cfg05 REAL Smoke Result

| Field | Value |
|-------|-------|
| REAL smoke attempted | **YES** |
| Readiness label | **REAL_READY** |
| Prediction rows | **23** |
| Validator passed | **TRUE** |
| Overall status | **PASS** |

Notes:
- 23 prediction rows (not 24) due to adapter filter: `ds >= D+01:00` and `ds < D+1 00:00` — excludes the D+1 00:00 boundary row
- No claim of "11.48% reproduced" — real y_true evaluation has not been run on predictions
- All generated artifacts are under `.local_artifacts/` (gitignored)

## 6. Readiness Matrix

| Gate | Status |
|------|--------|
| Raw Chinese CSV | ✅ CFG05_RAW_DATA_VALID (39,408 rows) |
| Source repo | ✅ PRESENT |
| 42 CFG05_FEATURE_COLUMNS | ✅ All present after feature build |
| cfg05 model artifact | ✅ LOADABLE |
| cfg05 feature input | ✅ SCHEMA_READY (23 rows) |
| REAL smoke | ✅ REAL_READY (23 rows, validator passed) |

## 7. Local Artifact Files Created (ignored only)

```
.local_artifacts/p14_cfg05/
├── cfg05_model.txt              (9,058,607 bytes — trained LightGBM)
└── cfg05_features_2026-06-30.csv (11,025 bytes — 23 rows × 43 columns)
```

All files are under `.local_artifacts/` which is `.gitignore`d.

## 8. Tests Run

```
675 passed in 14.80s
```

### P14 Test Breakdown (19 tests)

**TestSchemaInspector** (8 tests):
- `test_missing_raw_data_returns_missing` — no path → MISSING
- `test_redacts_values_by_default` — values redacted in output
- `test_reports_dtypes_and_null_counts` — dtypes and null_count reported
- `test_reports_time_range` — time_min/time_max populated
- `test_invalid_csv_returns_invalid` — bad CSV → INVALID
- `test_valid_csv_passes` — valid Chinese CSV → VALID
- `test_non_strict_exit_0_on_missing` — exit 0 non-strict
- `test_strict_exit_nonzero_on_missing` — exit non-zero strict

**TestP14Wrapper** (9 tests):
- `test_missing_raw_data_returns_missing_and_exit_0` — exit 0 non-strict
- `test_missing_raw_data_exits_nonzero_strict` — exit non-zero strict
- `test_does_not_train_if_raw_data_invalid` — no training on bad data
- `test_uses_local_artifacts_p14_by_default` — default paths under p14
- `test_unsafe_work_dir_rejected` — outputs/ rejected
- `test_real_smoke_not_attempted_without_gates` — no smoke without gates
- `test_summary_contains_required_keys` — 15 keys present
- `test_nonexistent_raw_data_path_returns_missing` — bad path → MISSING
- `test_missing_source_repo_still_reported` — missing repo reported

**TestForbiddenFiles** (2 tests):
- `test_forbidden_files_check` — no forbidden extensions in untracked
- `test_no_generated_artifacts_in_repo` — no generated files tracked

## 9. Forbidden Files Check

**PASS** — no `.csv`, `.pkl`, `.joblib`, `.parquet`, `.feather`, `.pt`, `.pth`, `.ckpt` files tracked or untracked.

## 10. User Runbook Commands

### Inspect a raw CSV (values redacted by default):
```bash
python -m scripts.inspect_cfg05_raw_csv_schema \
  --raw-data /path/to/shandong_pmos_hourly.csv --json
```

### Run full P14 pipeline:
```bash
python -m scripts.run_p14_raw_csv_intake_cfg05 \
  --raw-data /path/to/shandong_pmos_hourly.csv \
  --target-day YYYY-MM-DD \
  --run-real-smoke --json --strict
```

### Run individual steps:
```bash
# Contract check only
python -m scripts.check_cfg05_raw_data_contract \
  --raw-data /path/to/data.csv

# Train and export only
python -m scripts.train_export_cfg05_local \
  --raw-data /path/to/data.csv --target-day YYYY-MM-DD --force

# P13 orchestration
python -m scripts.run_p13_cfg05_raw_data_to_real_smoke \
  --raw-data /path/to/data.csv --target-day YYYY-MM-DD --run-real-smoke
```

### Successful run used:
```bash
python -m scripts.run_p14_raw_csv_intake_cfg05 \
  --raw-data "D:/.../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv" \
  --target-day 2026-06-30 \
  --run-real-smoke --json --strict
```

## 11. P15 Recommendation

**Phase 15 should focus on evaluation and validation:**

1. **Evaluate prediction accuracy**: Compare cfg05 predictions against actual y_true for historical days where actuals are known (e.g., 2026-06-01 through 2026-06-29). This enables claiming reproduction of the 11.48% sMAPE.

2. **Multi-day backtest**: Run predictions for 30+ days and compute sMAPE/MAE/RMSE metrics.

3. **Feature importance analysis**: Extract feature importance from the trained model to validate that the 42 CFG05 features are meaningful.

4. **Prediction output ledger**: Generate prediction CSVs matching the standard schema and store them in `.local_artifacts/p14_cfg05/predictions/`.
