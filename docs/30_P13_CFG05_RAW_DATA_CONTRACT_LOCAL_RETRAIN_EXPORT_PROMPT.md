# P13 cfg05 Raw Data Contract + Local Retrain/Export Prompt

You are executing P13 for electricity_forecast_model3.0.

P12 is complete. It successfully cloned `disdorqin/epf-sota-experiment` into `.local_artifacts/source_repos/epf-sota-experiment/`, confirmed cfg05 as the day-ahead champion, and confirmed there are no pre-saved cfg05 model artifacts in the source repo. P12 also found that external Chinese raw CSV training data is required to produce cfg05 artifacts and 42-column feature input.

P13 is not another structural smoke phase. P13 should close the real blocker by defining the raw-data contract and adding a local-only retrain/export route that can create a cfg05 LightGBM text model and feature input CSV under `.local_artifacts/` when the user supplies the external Chinese CSV.

Hard scope:
- Do not commit raw data.
- Do not commit generated model artifacts.
- Do not commit generated feature CSVs, predictions, ledgers, parquet, pkl, joblib, pt, pth, ckpt.
- Do not claim cfg05 source 11.48% is reproduced unless a real evaluation with y_true is run and explicitly reported.
- Do not claim cfg05 REAL unless generated artifact + feature input + prediction + validator pass.
- All local generated files must live under `.local_artifacts/` or another ignored local path.

Facts from source repo:
- `scripts/run_dayahead_tabular_model_search.py` accepts `--data-path`, `--start`, `--end`, `--models`, `--windows`, and builds day-ahead features via `src.common.feature_builder_dayahead.build_features_dayahead`.
- Source `src.common.data_loader.load_data` expects Chinese raw columns including: `时刻`, `日前电价`, `实时电价`, `直调负荷预测值`, `风电总加预测值`, `光伏总加预测值`, `联络线受电负荷预测值`, `竞价空间预测值`.
- Source LightGBM adapter exposes `adapter.model` after training, so a wrapper can call `adapter.model.save_model(...)` locally after training.
- Existing source script saves predictions/metrics/reports but not cfg05 model weights.

P13 goals:
1. Add a raw Chinese CSV contract checker.
2. Add a local-only cfg05 retrain/export wrapper that uses source repo data loader + feature builder + LightGBM training, then saves a LightGBM text model under `.local_artifacts/p13_cfg05/`.
3. Add a local-only cfg05 feature input exporter for a target day under `.local_artifacts/p13_cfg05/`.
4. Immediately verify exported model with `check_cfg05_artifact` and exported input with `check_cfg05_input`.
5. If both pass, run `run_cfg05_real_smoke_pipeline`.
6. If raw data is missing, return `CFG05_RAW_DATA_MISSING` with exact required columns and command examples.

Final statuses:
- `CFG05_REAL_READY_LOCAL` — local retrain/export + input + smoke validator pass.
- `CFG05_RAW_DATA_MISSING` — no raw Chinese CSV provided/found.
- `CFG05_RAW_DATA_INVALID` — CSV exists but required Chinese columns missing/unparseable.
- `CFG05_LOCAL_TRAIN_FAILED` — raw data valid but training failed.
- `CFG05_LOCAL_EXPORT_FAILED` — model trained but save/export failed.
- `CFG05_INPUT_EXPORT_FAILED` — feature input could not be built/exported.
- `CFG05_REAL_SMOKE_FAILED` — artifact/input exported but adapter/validator failed.

Required files to create or update:
- `scripts/check_cfg05_raw_data_contract.py`
- `scripts/train_export_cfg05_local.py`
- `scripts/run_p13_cfg05_raw_data_to_real_smoke.py`
- `tests/test_p13_cfg05_raw_data_contract_export.py`
- `docs/reports/p13_cfg05_raw_data_contract_local_retrain_export_report.md`

Optional updates:
- `docs/LOCAL_ARTIFACTS.md` — add P13 examples.
- `.gitignore` only if additional local ignored paths are needed.

1. Raw data contract checker

File: `scripts/check_cfg05_raw_data_contract.py`

CLI args:
- `--raw-data <path>`
- `--json`
- `--strict`
- `--verbose`

Required Chinese columns:
- `时刻`
- `日前电价`
- `实时电价`
- `直调负荷预测值`
- `风电总加预测值`
- `光伏总加预测值`
- `联络线受电负荷预测值`
- `竞价空间预测值`

Behavior:
- Try GBK then UTF-8 encoding, same as source loader.
- Validate `时刻` parseable as datetime.
- Validate numeric conversion for price/load/forecast columns.
- Report min/max timestamp, row count, missing columns, null counts.
- Non-strict missing/invalid returns structured report exit 0.
- Strict missing/invalid exits nonzero.

Summary keys:
- `raw_data_status`
- `raw_data_path`
- `rows`
- `columns_present`
- `missing_columns`
- `time_min`
- `time_max`
- `reason_codes`

2. Local cfg05 train/export wrapper

File: `scripts/train_export_cfg05_local.py`

CLI args:
- `--source-repo .local_artifacts/source_repos/epf-sota-experiment`
- `--raw-data <path>`
- `--target-day YYYY-MM-DD`
- `--train-window-days 90`
- `--work-dir .local_artifacts/p13_cfg05`
- `--model-out .local_artifacts/p13_cfg05/cfg05_model.txt`
- `--features-out .local_artifacts/p13_cfg05/cfg05_features_<target-day>.csv`
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Verify source repo exists and contains source modules.
- Verify raw data contract via `check_cfg05_raw_data_contract`.
- Import source modules by prepending source repo to `sys.path`:
  - `src.common.data_loader.load_data`
  - `src.common.feature_builder_dayahead.build_features_dayahead`
  - `src.models.lightgbm_dayahead_adapter.LightGBMDayaheadAdapter`
- Load raw data with `target="dayahead"`.
- Build source day-ahead features.
- Dynamically import 3.0 `CFG05_FEATURE_COLUMNS` and `CFG05_PARAMS` from `models.adapters.cfg05_dayahead_lgbm`.
- Validate built features contain all `CFG05_FEATURE_COLUMNS`.
- Train with target-day no-leakage split:
  - target_dt = target_day
  - train_start = target_dt - train_window_days
  - train_end = target_dt - 1 hour
  - train rows: `ds >= train_start and ds < train_end`
  - feature input rows: `ds >= target_day + 1h and ds < target_day + 1d`
- Require enough training rows, default at least 100.
- Train LightGBM using cfg05 params. You may use source `LightGBMDayaheadAdapter` or direct `lightgbm.train`, but export must be standard LightGBM text model.
- Save model via `booster.save_model(model_out)` only under ignored local work-dir.
- Save feature input CSV with all `CFG05_FEATURE_COLUMNS` + `ds` only under ignored local work-dir.
- Validate model with `check_cfg05_artifact(model_out)`.
- Validate features with `check_cfg05_input(features_out)`.
- Do not commit any generated files.

Important:
- Do not overwrite existing files unless `--force` is supplied, or write timestamped filenames.
- If LightGBM is missing, return `CFG05_LOCAL_TRAIN_FAILED` with `LIGHTGBM_NOT_INSTALLED`.

3. P13 orchestration script

File: `scripts/run_p13_cfg05_raw_data_to_real_smoke.py`

CLI args:
- `--source-repo <path>`
- `--raw-data <path>`
- `--target-day YYYY-MM-DD`
- `--work-dir .local_artifacts/p13_cfg05`
- `--train-window-days 90`
- `--run-real-smoke`
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Run raw data checker.
- Run local train/export wrapper if raw data valid.
- Run `scripts.run_cfg05_real_smoke_pipeline` if model + features pass readiness gates.
- Return final status from the allowed P13 statuses.
- In non-strict mode, blockers return structured status with exit 0.
- In strict mode, blockers exit nonzero.

Summary keys:
- `source_repo_status`
- `raw_data_status`
- `model_export_status`
- `feature_export_status`
- `cfg05_artifact_status`
- `cfg05_input_status`
- `real_smoke_attempted`
- `readiness_label`
- `prediction_rows`
- `validator_passed`
- `final_status`
- `model_out`
- `features_out`
- `reason_codes`
- `forbidden_files_check`

4. Tests

Create `tests/test_p13_cfg05_raw_data_contract_export.py` using tmp_path only. Do not require real Chinese data or internet.

Required tests:
1. missing raw data returns `CFG05_RAW_DATA_MISSING`, not crash.
2. raw CSV missing required Chinese columns returns `CFG05_RAW_DATA_INVALID`.
3. raw CSV with required Chinese columns validates contract.
4. invalid timestamp in `时刻` is reported.
5. non-numeric price/load columns are reported.
6. source repo missing returns blocker.
7. work-dir outside `.local_artifacts/` or tmp safe path is rejected.
8. train/export does not run if raw data invalid.
9. train/export summary contains required keys.
10. generated model/features paths, when mocked, must be under ignored local work-dir.
11. placeholder/invalid model never becomes `REAL_READY`.
12. real smoke is not attempted unless artifact + input gates pass.
13. non-strict exits 0 on blocker.
14. strict exits nonzero on blocker.
15. forbidden files check passes.

If possible, add one lightweight mocked-success test that monkeypatches training/export to create placeholder outputs and confirms the orchestration still refuses REAL_READY unless `check_cfg05_artifact` and `check_cfg05_input` pass.

Run:
- `python -m pytest tests/test_p13_cfg05_raw_data_contract_export.py`
- `python -m pytest`

5. Report

Write:
`docs/reports/p13_cfg05_raw_data_contract_local_retrain_export_report.md`

Report format:
# P13 cfg05 Raw Data Contract + Local Retrain/Export Report

## 1. Executive status
## 2. Raw Chinese CSV contract
## 3. Source training/export route
## 4. Local cfg05 model export result
## 5. Local cfg05 feature input export result
## 6. cfg05 REAL smoke result
## 7. Readiness matrix
## 8. Local artifact files created (ignored only)
## 9. Tests run
## 10. Forbidden files check
## 11. Blockers and exact next commands
## 12. P14 recommendation

Report must state one of the P13 final statuses.

Hard wording:
- If no real raw data was supplied, write `CFG05_RAW_DATA_MISSING` and do not call this a model failure.
- If raw data valid but training/export fails, write exact exception/reason.
- If REAL_READY achieved, write `CFG05_REAL_READY_LOCAL: ACHIEVED`, artifact path, feature path, prediction_rows, validator result, and confirm no files committed.
- Do not claim 11.48% reproduced unless evaluation is run with y_true.

Final response format:
P13 cfg05 Raw Data Contract + Local Retrain/Export Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Raw data status:
6. Source repo status:
7. Model export status:
8. Feature export status:
9. REAL smoke status:
10. Readiness matrix:
11. Local artifact files:
12. Forbidden files check:
13. Blockers / next commands:
14. Commit:
15. Final status:
