# P14 Raw CSV Intake + cfg05 REAL Runbook Prompt

You are executing P14 for electricity_forecast_model3.0.

P13 is complete. The code path for raw Chinese CSV contract checking, local cfg05 training/export, feature CSV export, and cfg05 REAL smoke orchestration is implemented. The only blocker is missing raw Chinese CSV data.

P14 should not add more modeling layers. P14 should make the raw CSV intake and one-command real run workflow foolproof, then run it if the user provides a CSV path.

Current P13 status:
- Source repo: PRESENT at `.local_artifacts/source_repos/epf-sota-experiment/`
- Raw Chinese CSV: MISSING
- Model export: NOT_ATTEMPTED
- Feature export: NOT_ATTEMPTED
- REAL smoke: NOT_ATTEMPTED
- Final status: `CFG05_RAW_DATA_MISSING`

Hard scope:
- Do not commit raw data.
- Do not commit generated model artifacts.
- Do not commit generated feature CSVs or predictions.
- Do not commit parquet, pkl, joblib, pt, pth, ckpt, or ledgers.
- Do not claim cfg05 REAL unless `CFG05_REAL_READY_LOCAL` is achieved and validator passes.
- Do not claim 11.48% reproduced unless real y_true evaluation is run.

P14 goals:
1. Add a clear raw CSV intake/runbook wrapper for users who have the external Chinese CSV.
2. Add optional redacted sample/column-only validator so the user can confirm schema without exposing data.
3. Add one-command pipeline that calls P13 with safe local paths.
4. If a real raw CSV path is supplied, run contract → train/export → feature export → REAL smoke.
5. If no CSV path is supplied, final status must remain `CFG05_RAW_DATA_MISSING` with exact instructions.

Required files to create or update:
- `scripts/run_p14_raw_csv_intake_cfg05.py`
- `scripts/inspect_cfg05_raw_csv_schema.py`
- `tests/test_p14_raw_csv_intake_cfg05.py`
- `docs/reports/p14_raw_csv_intake_cfg05_real_runbook_report.md`
- Update `docs/LOCAL_ARTIFACTS.md` with P14 command examples if needed.

1. Schema inspection script

File: `scripts/inspect_cfg05_raw_csv_schema.py`

CLI args:
- `--raw-data <path>`
- `--sample-rows 5`
- `--redact-values` default true
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Load CSV with GBK → UTF-8 fallback.
- Print/report only column names, dtypes, null counts, timestamp range, row count.
- If `--redact-values` is true, do not print actual data values.
- Call or reuse `check_cfg05_raw_data_contract` logic.
- Return `CFG05_RAW_DATA_VALID`, `CFG05_RAW_DATA_INVALID`, or `CFG05_RAW_DATA_MISSING`.
- Strict mode exits nonzero on missing/invalid.

Summary keys:
- `raw_data_status`
- `rows`
- `columns`
- `dtypes`
- `null_counts`
- `time_min`
- `time_max`
- `missing_columns`
- `redacted`
- `reason_codes`

2. P14 one-command wrapper

File: `scripts/run_p14_raw_csv_intake_cfg05.py`

CLI args:
- `--raw-data <path>`
- `--source-repo .local_artifacts/source_repos/epf-sota-experiment`
- `--target-day YYYY-MM-DD`
- `--work-dir .local_artifacts/p14_cfg05`
- `--train-window-days 90`
- `--run-real-smoke`
- `--force`
- `--json`
- `--strict`
- `--verbose`

Behavior:
- If raw data path missing: return `CFG05_RAW_DATA_MISSING` with exact required columns and command examples.
- If raw data exists: run schema inspection + P13 orchestration.
- Ensure work-dir is ignored/local before writing.
- Use `.local_artifacts/p14_cfg05/` for generated model/features/output.
- Do not write anything unless raw data passes contract.
- Do not call train/export if raw data invalid.
- Do not call REAL smoke unless model artifact and feature input gates pass.
- Non-strict blocker exits 0 with structured summary.
- Strict blocker exits nonzero.

Summary keys:
- `raw_data_status`
- `source_repo_status`
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

Allowed final statuses:
- `CFG05_REAL_READY_LOCAL`
- `CFG05_RAW_DATA_MISSING`
- `CFG05_RAW_DATA_INVALID`
- `CFG05_LOCAL_TRAIN_FAILED`
- `CFG05_LOCAL_EXPORT_FAILED`
- `CFG05_INPUT_EXPORT_FAILED`
- `CFG05_REAL_SMOKE_FAILED`

3. Tests

Create `tests/test_p14_raw_csv_intake_cfg05.py` using tmp_path only.

Required tests:
1. missing raw data returns `CFG05_RAW_DATA_MISSING` and exit 0 non-strict.
2. missing raw data exits nonzero strict.
3. schema inspector redacts values by default.
4. schema inspector reports columns/dtypes/null counts/time range.
5. invalid raw CSV returns `CFG05_RAW_DATA_INVALID`.
6. valid minimal Chinese CSV passes contract in inspector.
7. P14 wrapper does not train/export if raw data invalid.
8. P14 wrapper uses `.local_artifacts/p14_cfg05/` by default.
9. unsafe work-dir is rejected.
10. real smoke is not attempted unless gates pass.
11. summary JSON contains required keys.
12. forbidden files check passes.
13. no generated forbidden files are tracked/untracked in repo.

Run:
- `python -m pytest tests/test_p14_raw_csv_intake_cfg05.py`
- `python -m pytest`

4. Report

Write:
`docs/reports/p14_raw_csv_intake_cfg05_real_runbook_report.md`

Report format:
# P14 Raw CSV Intake + cfg05 REAL Runbook Report

## 1. Executive status
## 2. Raw CSV intake status
## 3. Schema inspection result
## 4. cfg05 local train/export result
## 5. cfg05 REAL smoke result
## 6. Readiness matrix
## 7. Local artifact files created (ignored only)
## 8. Tests run
## 9. Forbidden files check
## 10. User runbook commands
## 11. P15 recommendation

If no raw CSV was supplied:
- Final status: `CFG05_RAW_DATA_MISSING`
- Report should say no model/export/smoke was attempted.
- Provide exact commands:

```bash
python -m scripts.inspect_cfg05_raw_csv_schema \
  --raw-data /path/to/shandong_pmos_hourly.csv --json

python -m scripts.run_p14_raw_csv_intake_cfg05 \
  --raw-data /path/to/shandong_pmos_hourly.csv \
  --target-day 2026-07-01 \
  --run-real-smoke --json --strict
```

If raw CSV was supplied and valid:
- Record row count/time range.
- Record generated local paths under `.local_artifacts/p14_cfg05/`.
- If REAL_READY achieved, write `CFG05_REAL_READY_LOCAL: ACHIEVED`.
- If not, write exact failure status and reason codes.

Hard wording:
- Do not claim REAL unless final status is `CFG05_REAL_READY_LOCAL`.
- Do not claim metric reproduction unless evaluation runs.
- Never include raw data rows/values in report unless explicitly requested; default report should be metadata-only.

Final response format:
P14 Raw CSV Intake + cfg05 REAL Runbook Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Raw CSV status:
6. Schema inspection status:
7. Model export status:
8. Feature export status:
9. REAL smoke status:
10. Readiness matrix:
11. Local artifact files:
12. Forbidden files check:
13. User runbook commands:
14. Commit:
15. Final status:
