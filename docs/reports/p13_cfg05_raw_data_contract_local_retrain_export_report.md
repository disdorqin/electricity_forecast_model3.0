# P13 cfg05 Raw Data Contract + Local Retrain/Export Report

> **Phase**: P13 — Raw Chinese CSV contract, local LightGBM training and export
> **Generated**: 2026-07-04
> **Test count**: 656 total (630 prior + 26 P13 new), 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Raw data contract checker | **Implemented** |
| Local train/export wrapper | **Implemented** |
| P13 orchestration | **Implemented** |
| Tests | **26 passed** |
| Raw data status | **CFG05_RAW_DATA_MISSING** (no raw Chinese CSV provided) |
| SoURCE repo | **PRESENT** (cloned in P12) |
| Model export | **NOT_ATTEMPTED** (requires raw data) |
| Feature export | **NOT_ATTEMPTED** (requires raw data) |
| REAL smoke | **NOT_ATTEMPTED** (requires raw data) |

**Final status**: CFG05_RAW_DATA_MISSING — raw Chinese CSV data is the sole blocker.

## 2. Raw Chinese CSV Contract

### Required Columns (8)

| # | Chinese Column | English Meaning |
|---|----------------|-----------------|
| 1 | 时刻 | Timestamp |
| 2 | 日前电价 | Day-ahead price (target) |
| 3 | 实时电价 | Real-time price |
| 4 | 直调负荷预测值 | Load forecast |
| 5 | 风电总加预测值 | Wind generation forecast |
| 6 | 光伏总加预测值 | Solar generation forecast |
| 7 | 联络线受电负荷预测值 | Interconnect load forecast |
| 8 | 竞价空间预测值 | Bidding space forecast |

### Contract Checker: `scripts/check_cfg05_raw_data_contract.py`

**Behavior**:
1. No `--raw-data` → status=`CFG05_RAW_DATA_MISSING`, exit 0 (non-strict)
2. File not found → status=`CFG05_RAW_DATA_MISSING`
3. Read CSV (GBK → UTF-8 fallback)
4. Validate 8 required Chinese columns
5. Validate timestamp column (时刻 → datetime parse)
6. Validate numeric columns (coerce to float, count NaN)
7. Return structured summary with rows, time range, null counts, reason codes

**Status codes**: `CFG05_RAW_DATA_MISSING`, `CFG05_RAW_DATA_INVALID`, `CFG05_RAW_DATA_VALID`

## 3. Source Training/Export Route

### Data Flow

```
Raw Chinese CSV
  → src.common.data_loader.load_data(raw, target="dayahead")
    → DataFrame with ds, y, load, wind, solar, interconnect, bidding_space_raw
  → src.common.feature_builder_dayahead.build_features_dayahead(df)
    → Base features (24 cols) + Extended features (18 cols) = 42 CFG05_FEATURE_COLUMNS
  → Verify all 42 columns present
  → No-leakage split:
    Train:  [target - window_days, target - 1 hour]
    Input:  [target + 1 hour, target + 1 day)
  → LightGBM train with CFG05_PARAMS
  → Save model → cfg05_model.txt
  → Save features → cfg05_features_{target_day}.csv
  → Readiness checks on both outputs
```

### Script: `scripts/train_export_cfg05_local.py`

**Key features**:
- Imports source repo modules via `importlib.util.spec_from_file_location` (no sys.path pollution)
- Uses 3.0 `CFG05_PARAMS` from `models.adapters.cfg05_dayahead_lgbm`
- Dynamic feature count via `len(CFG05_FEATURE_COLUMNS)` — no hardcoding
- Requires ≥100 train rows, else `CFG05_LOCAL_TRAIN_FAILED`
- All outputs under `.local_artifacts/p13_cfg05/` (gitignored)
- `--force` flag to overwrite existing files
- Post-export readiness checks via `check_cfg05_artifact` and `check_cfg05_input`

### Possible Final Statuses

| Status | Condition |
|--------|-----------|
| `CFG05_REAL_READY_LOCAL` | Retrain + export + feature input + REAL smoke validator all pass |
| `CFG05_RAW_DATA_MISSING` | No raw Chinese CSV provided |
| `CFG05_RAW_DATA_INVALID` | CSV exists but missing columns / bad data |
| `CFG05_LOCAL_TRAIN_FAILED` | Raw data valid but training fails |
| `CFG05_LOCAL_EXPORT_FAILED` | Model trained but save_model fails |
| `CFG05_INPUT_EXPORT_FAILED` | Feature CSV export fails |
| `CFG05_REAL_SMOKE_FAILED` | Artifact + input exported but prediction/validator fails |

## 4. Local cfg05 Model Export Result

**NOT ATTEMPTED** — requires raw Chinese CSV data.

Expected output: `.local_artifacts/p13_cfg05/cfg05_model.txt`

## 5. Local cfg05 Feature Input Export Result

**NOT ATTEMPTED** — requires raw Chinese CSV data.

Expected output: `.local_artifacts/p13_cfg05/cfg05_features_{target_day}.csv` (ds + 42 columns)

## 6. cfg05 REAL Smoke Result

**NOT ATTEMPTED** — requires both model artifact and feature input.

## 7. Readiness Matrix

| Gate | Required | Status |
|------|----------|--------|
| Source repo | `.local_artifacts/source_repos/epf-sota-experiment/` | ✅ PRESENT |
| Raw Chinese CSV | External data file | ❌ MISSING |
| 42 feature columns | `CFG05_FEATURE_COLUMNS` | ✅ Defined (42) |
| cfg05 model artifact | Trained LightGBM `.txt` | ❌ NOT_ATTEMPTED |
| cfg05 feature input | CSV with ds + 42 columns | ❌ NOT_ATTEMPTED |
| REAL smoke | Predict + validate | ❌ NOT_ATTEMPTED |

## 8. Local Artifact Files Created (ignored only)

No local artifacts were generated because raw data is missing. When raw data is provided:
- `.local_artifacts/p13_cfg05/cfg05_model.txt`
- `.local_artifacts/p13_cfg05/cfg05_features_{target_day}.csv`

## 9. Tests Run

```
656 passed in 13.16s
```

### P13 Test Breakdown (26 tests)

**TestRawDataContract** (11 tests):
- `test_missing_raw_data_returns_missing` — no path → CFG05_RAW_DATA_MISSING
- `test_nonexistent_file_returns_missing` — bad path → CFG05_RAW_DATA_MISSING
- `test_missing_chinese_columns_returns_invalid` — missing cols → INVALID
- `test_invalid_timestamp_reported` — bad timestamps reported
- `test_non_numeric_columns_reported` — bad numeric data reported
- `test_valid_csv_contract_passes` — valid CSV → CFG05_RAW_DATA_VALID
- `test_utf8_encoding_works` — UTF-8 CSV readable
- `test_contract_result_contains_required_keys` — 9 keys present
- `test_non_strict_exit_0_on_missing` — non-strict exit 0
- `test_strict_exit_nonzero_on_missing` — strict exit non-zero
- `test_strict_exit_nonzero_on_invalid` — strict exit non-zero

**TestTrainExportCfg05Local** (5 tests):
- `test_source_repo_missing_returns_blocker` — no crash
- `test_raw_data_missing_skips_training` — no training without data
- `test_summary_contains_required_keys` — 12 keys present
- `test_model_features_paths_under_local_workdir` — paths under ignored dir
- `test_unsafe_work_dir_rejected` — outputs/ rejected

**TestP13Orchestration** (9 tests):
- `test_no_raw_data_returns_missing` — CFG05_RAW_DATA_MISSING
- `test_invalid_raw_data_returns_invalid` — CFG05_RAW_DATA_INVALID
- `test_source_repo_missing_returns_blocker` — no crash
- `test_placeholder_model_not_real_ready` — no false REAL_READY
- `test_real_smoke_not_attempted_without_gates` — no smoke without gates
- `test_non_strict_exit_0_on_blocker` — exit 0
- `test_strict_exit_nonzero_on_blocker` — exit non-zero
- `test_summary_contains_required_keys` — 15 keys present
- `test_mock_export_avoids_false_real_ready` — placeholder avoids REAL_READY

**TestForbiddenFiles** (1 test):
- `test_forbidden_files_check` — no csv/pkl/joblib in untracked

## 10. Forbidden Files Check

**PASS** — no `.csv`, `.pkl`, `.joblib`, `.parquet`, `.feather`, `.pt`, `.pth`, `.ckpt` files in untracked.

## 11. Blockers and Exact Next Commands

### Blocker: CFG05_RAW_DATA_MISSING

Raw Chinese CSV required for training. The CSV must have the 8 required Chinese columns.

**To use this data, the user must:**

1. Place the CSV at an accessible path (e.g., `C:/path/to/shandong_pmos_hourly.csv`)
2. Run the full P13 pipeline:

```bash
python -m scripts.run_p13_cfg05_raw_data_to_real_smoke \
  --raw-data /path/to/shandong_pmos_hourly.csv \
  --target-day 2026-07-01 \
  --run-real-smoke --json --strict
```

3. Or run individual steps:

```bash
# Check contract only
python -m scripts.check_cfg05_raw_data_contract \
  --raw-data /path/to/data.csv --json

# Train and export only
python -m scripts.train_export_cfg05_local \
  --raw-data /path/to/data.csv --target-day 2026-07-01 --force --json
```

## 12. P14 Recommendation

**Immediate**: Provide raw Chinese CSV with the 8 required columns to unblock training.

**After raw data provided**: Run P13 orchestration with `--run-real-smoke`. Expected progression:
1. Raw data valid → training proceeds
2. Training succeeds → model exported (CFG05_LOCAL_EXPORT_DONE)
3. Feature CSV passes schema → CFG05_READY_FOR_SMOKE
4. REAL smoke pipeline runs → CFG05_REAL_READY_LOCAL or CFG05_REAL_SMOKE_FAILED

**Do not claim 11.48% reproduced** unless real y_true evaluation runs on actual predictions.
