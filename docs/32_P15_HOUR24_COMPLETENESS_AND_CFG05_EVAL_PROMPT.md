# P15 Hour-24 Completeness + cfg05 Evaluation Prompt

You are executing P15 for electricity_forecast_model3.0.

P14 achieved `CFG05_REAL_READY_LOCAL: ACHIEVED` for the first time. This is a major milestone: raw Chinese CSV was valid, a local cfg05 LightGBM model was trained/exported, cfg05 feature input was exported, and the REAL smoke validator passed.

However, P14 produced only **23 prediction rows** for target day `2026-06-30`. The P14 report explains this is due to the adapter/filter condition:

`ds >= D+01:00` and `ds < D+1 00:00`

This excludes the `D+1 00:00` boundary row, which should represent business hour 24 for business day D. Therefore P15 must fix and validate hour-24 completeness before claiming metric reproduction or moving to 30-day evaluation.

Hard scope:
- Do not commit raw data.
- Do not commit generated model artifacts.
- Do not commit generated feature CSVs/predictions/ledgers.
- Do not claim 11.48% reproduced until a complete 24-row-per-day evaluation is run against y_true.
- Do not accept 23 rows as production-ready for day-ahead daily output unless explicitly labeled `INCOMPLETE_HOUR24`.

P15 goals:
1. Fix target-day feature/prediction filtering so business day D includes 24 rows: D 01:00 through D+1 00:00.
2. Add row completeness validation for cfg05 feature input and prediction output.
3. Re-run local cfg05 train/export/smoke for the same raw data and target day.
4. Require 24 prediction rows for `CFG05_REAL_READY_24H_LOCAL`.
5. Only after 24H readiness passes, run a small historical evaluation against y_true.
6. If evaluation is run, report metrics honestly, but do not claim source 11.48% reproduced unless the evaluation window and methodology match the source run.

Allowed final statuses:
- `CFG05_REAL_READY_24H_LOCAL` — artifact + 24-row feature input + 24-row prediction + validator pass.
- `CFG05_REAL_READY_INCOMPLETE_23H` — smoke works but missing hour 24; not production-ready.
- `CFG05_HOUR24_FIX_FAILED` — attempted boundary fix but 24 rows not achieved.
- `CFG05_EVAL_READY_NO_METRIC` — 24H smoke passes but evaluation not run.
- `CFG05_EVAL_COMPLETE` — 24H smoke passes and evaluation metrics computed.
- `CFG05_EVAL_FAILED` — 24H smoke passes but metrics/eval failed.

Required files to create or update:
- `scripts/check_cfg05_hour24_completeness.py`
- `scripts/run_p15_cfg05_24h_smoke_and_eval.py`
- `tests/test_p15_hour24_completeness_eval.py`
- `docs/reports/p15_hour24_completeness_cfg05_eval_report.md`

Likely files to update:
- `scripts/train_export_cfg05_local.py`
- `scripts/prepare_cfg05_real_input.py`
- `scripts/run_cfg05_real_smoke_pipeline.py`
- `models/adapters/cfg05_dayahead_lgbm.py`

1. Fix target-day window logic

Canonical day-ahead business day D must include:
- D 01:00
- D 02:00
- ...
- D 23:00
- D+1 00:00 as hour_business 24

Canonical filter can be implemented as either:

```python
start = pd.Timestamp(target_day) + pd.Timedelta(hours=1)
end_exclusive = pd.Timestamp(target_day) + pd.Timedelta(days=1, hours=1)
mask = (ds >= start) & (ds < end_exclusive)
```

or:

```python
start = pd.Timestamp(target_day) + pd.Timedelta(hours=1)
end_inclusive = pd.Timestamp(target_day) + pd.Timedelta(days=1)
mask = (ds >= start) & (ds <= end_inclusive)
```

Use one canonical helper to avoid drift.

Do not use:

```python
(ds >= target_day + 1h) & (ds < target_day + 1d)
```

because it yields only 23 rows.

2. Hour24 completeness checker

File: `scripts/check_cfg05_hour24_completeness.py`

CLI args:
- `--input <csv>`
- `--target-day YYYY-MM-DD`
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Load CSV.
- Require `ds`.
- If `hour_business` exists, validate 1..24 all present exactly once.
- If `hour_business` missing, derive from ds using project business-day logic.
- Validate the 24 expected timestamps for target day D are present.
- Report missing hours, duplicate hours, row count, ds_min, ds_max.
- Strict mode exits nonzero unless complete.

Summary keys:
- `completeness_status`: `COMPLETE_24H` / `INCOMPLETE_23H` / `MISSING_HOURS` / `DUPLICATE_HOURS` / `INVALID`
- `target_day`
- `row_count`
- `expected_hours`
- `present_hours`
- `missing_hours`
- `duplicate_hours`
- `ds_min`
- `ds_max`
- `reason_codes`

3. P15 orchestration

File: `scripts/run_p15_cfg05_24h_smoke_and_eval.py`

CLI args:
- `--raw-data <path>`
- `--source-repo .local_artifacts/source_repos/epf-sota-experiment`
- `--target-day YYYY-MM-DD`
- `--work-dir .local_artifacts/p15_cfg05`
- `--train-window-days 90`
- `--run-eval`
- `--eval-start YYYY-MM-DD`
- `--eval-end YYYY-MM-DD`
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Run P14/P13 pipeline with fixed 24H filtering.
- Validate feature CSV with `check_cfg05_hour24_completeness`.
- Validate prediction output also has 24 rows / hours 1..24.
- If only 23 rows, final status must be `CFG05_REAL_READY_INCOMPLETE_23H`, not 24H-ready.
- If 24 rows and validator passes, final status can be `CFG05_REAL_READY_24H_LOCAL`.
- If `--run-eval`, evaluate predictions against y_true only for days with complete 24 rows and non-null y_true.
- Store generated local artifacts only under `.local_artifacts/p15_cfg05/`.

Summary keys:
- `raw_data_status`
- `model_export_status`
- `feature_export_status`
- `feature_completeness_status`
- `prediction_completeness_status`
- `readiness_label`
- `prediction_rows`
- `validator_passed`
- `eval_attempted`
- `eval_days`
- `eval_rows`
- `metrics`
- `final_status`
- `model_out`
- `features_out`
- `predictions_out`
- `reason_codes`
- `forbidden_files_check`

4. Evaluation rules

If `--run-eval` is supplied:
- Do not use future y_true in features.
- Use only completed historical target days.
- Require 24 prediction rows per evaluated day.
- Compute at least:
  - sMAPE_floor50
  - MAE
  - RMSE
- Report evaluation window, number of days, number of rows.
- Do not say source 11.48% reproduced unless the evaluation window and training scheme match the source report exactly.

5. Tests

Create `tests/test_p15_hour24_completeness_eval.py` using tmp_path only.

Required tests:
1. 23-row CSV returns `INCOMPLETE_23H`.
2. 24-row CSV including D+1 00:00 returns `COMPLETE_24H`.
3. missing hour 24 is reported clearly.
4. duplicate hour is reported clearly.
5. checker derives hour_business from ds if missing.
6. old exclusive-end filter test demonstrates 23 rows.
7. fixed filter test demonstrates 24 rows.
8. P15 orchestration refuses `CFG05_REAL_READY_24H_LOCAL` with 23 rows.
9. P15 orchestration can label 24-row mocked output as `CFG05_REAL_READY_24H_LOCAL` only if validator passes.
10. eval refuses incomplete days.
11. eval computes metrics on mocked complete prediction/y_true data.
12. strict mode exits nonzero on incomplete rows.
13. non-strict mode exits 0 with structured `INCOMPLETE_23H` status.
14. forbidden files check passes.

Run:
- `python -m pytest tests/test_p15_hour24_completeness_eval.py`
- `python -m pytest`

6. Report

Write:
`docs/reports/p15_hour24_completeness_cfg05_eval_report.md`

Report format:
# P15 Hour-24 Completeness + cfg05 Evaluation Report

## 1. Executive status
## 2. P14 milestone recap
## 3. Hour-24 boundary bug / row completeness finding
## 4. Code fixes applied
## 5. 24H cfg05 smoke result
## 6. Evaluation result, if run
## 7. Readiness matrix
## 8. Local artifact files created (ignored only)
## 9. Tests run
## 10. Forbidden files check
## 11. P16 recommendation

Hard wording:
- If still 23 rows, write `CFG05_REAL_READY_INCOMPLETE_23H` and do not call production-ready.
- If 24 rows pass, write `CFG05_REAL_READY_24H_LOCAL`.
- If evaluation metrics are computed, state the exact window and row count.
- Do not say 11.48% reproduced unless methodology matches source.

Final response format:
P15 Hour-24 Completeness + cfg05 Evaluation Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Hour24 fix status:
6. Feature completeness status:
7. Prediction completeness status:
8. 24H REAL smoke status:
9. Evaluation status:
10. Metrics, if any:
11. Local artifact files:
12. Forbidden files check:
13. Commit:
14. Final status:
