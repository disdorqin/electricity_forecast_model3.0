# P16 BIG PROGRESS — Real cfg05 30-Day Backtest + Ledger/Fusion Chain Prompt

You are executing the next major phase for `electricity_forecast_model3.0`.

This is intentionally a larger integrated phase. Do not continue tiny one-file/one-step progress unless a safety blocker appears. The project has already passed structural phases P1–P15 and now has a real 24-hour cfg05 local model path. P16 should convert that milestone into a meaningful end-to-end real day-ahead backtest and ledger-chain handoff.

## Current state you must assume

P15 is complete and pushed to `origin/master`.

Confirmed P15 facts:
- `artifacts/dayahead_window.py` exists and is the canonical 24-hour day-ahead window helper.
- The old 23-hour bug was fixed by changing the day-ahead window from `[D+01:00, D+1 00:00)` to `[D+01:00, D+1+01:00)`.
- `D+1 00:00` is now included as `hour_business=24`.
- `models/adapters/cfg05_dayahead_lgbm.py` now uses `filter_dayahead()`.
- `scripts/train_export_cfg05_local.py` now uses `day_ahead_mask()`.
- `scripts/check_cfg05_hour24_completeness.py` validates `COMPLETE_24H`, `INCOMPLETE_23H`, `MISSING_HOURS`, `DUPLICATE_HOURS`, `INVALID`.
- P15 reports `CFG05_REAL_READY_24H_LOCAL` when 24 prediction rows + validator pass + 24h features are present.
- P15 optional eval computed sample metrics, but it explicitly says these are **not** source 11.48% reproduction because the window/training setup differs.

Existing local artifact/data policy:
- Raw CSV, model artifacts, feature CSVs, prediction CSVs, ledgers, parquet, pkl, joblib, pt/pth/ckpt must never be committed.
- Generated local files must live under `.local_artifacts/` or another ignored path.
- `.local_artifacts/` is gitignored.

Current real data/artifact availability:
- Raw Chinese CSV exists locally in the user's environment, e.g. `electricity_forecast_model2.1/data/shandong_pmos_hourly.csv`, but must not be committed.
- Source repo was cloned locally to `.local_artifacts/source_repos/epf-sota-experiment/`.
- cfg05 can be trained/exported locally.
- A local LightGBM artifact and feature CSV can be created under `.local_artifacts/`.

## Big P16 objective

Move from single-day 24H real smoke to a real multi-day day-ahead experiment:

1. Run cfg05 walk-forward backtest for a configurable date range, default 30 days.
2. Enforce 24-row completeness for every target day.
3. Compute real metrics against y_true: `sMAPE_floor50`, `MAE`, `RMSE`.
4. Write local standard-schema prediction outputs under `.local_artifacts/`, not repo.
5. Append predictions/actuals/fusion/weights into local ledgers using existing P5 ledger utilities, not committed files.
6. Feed cfg05 predictions through existing residual correction no-op/DATA_MISSING, fusion structural chain, and final output path where possible.
7. Produce one comprehensive report that says exactly what is real, what is still structural, and what remains data/artifact missing.

## Hard rules

- Do not commit generated `.csv`, `.xlsx`, `.parquet`, `.pkl`, `.joblib`, `.pt`, `.pth`, `.ckpt`, model text artifacts, or local ledger files.
- Do not claim source 11.48% reproduced unless you implement a methodology alignment check and show the eval window/training scheme matches the source champion report.
- Do not hide incomplete days. A target day with fewer than 24 prediction rows must be excluded from metrics or marked incomplete.
- Do not use future y_true as model features.
- Do not mutate private repos or unrelated repositories.
- Do not bring P5M/RT/ExtremPriceClf into REAL status unless their artifacts exist. They remain `DATA_MISSING` / `STUB` / `RULE_FALLBACK` unless path-verified.

## Required final statuses

P16 final status must be one of:

- `CFG05_30D_BACKTEST_COMPLETE` — 30-day or requested range backtest completed with all evaluated days 24H-complete and metrics computed.
- `CFG05_30D_BACKTEST_PARTIAL` — backtest ran, but some days were incomplete/missing y_true and excluded or marked.
- `CFG05_30D_BACKTEST_FAILED` — raw data/model training/prediction/eval failed.
- `CFG05_CHAIN_HANDOFF_COMPLETE` — predictions were also successfully passed through local ledger/residual/fusion/final chain.
- `CFG05_CHAIN_HANDOFF_PARTIAL` — day-ahead backtest completed, but downstream ledger/fusion/final chain only partially ran.
- `CFG05_SOURCE_METHODOLOGY_MISMATCH` — metrics computed but not comparable to source 11.48% because method/window/train scheme differ.

You may combine statuses, for example:

`CFG05_30D_BACKTEST_COMPLETE + CFG05_CHAIN_HANDOFF_PARTIAL + CFG05_SOURCE_METHODOLOGY_MISMATCH`

## Required files to create or update

Create:
- `scripts/run_p16_cfg05_30d_backtest_chain.py`
- `scripts/evaluate_cfg05_backtest.py`
- `tests/test_p16_cfg05_30d_backtest_chain.py`
- `docs/reports/p16_real_cfg05_30d_backtest_chain_report.md`

Likely update:
- `docs/LOCAL_ARTIFACTS.md` with P16 examples.
- Existing local pipeline utilities only if needed.

Do not update source repo unless explicitly required. Do not commit local artifacts.

---

# 1. P16 backtest + chain orchestration

File: `scripts/run_p16_cfg05_30d_backtest_chain.py`

CLI args:

```bash
python -m scripts.run_p16_cfg05_30d_backtest_chain \
  --raw-data /path/to/shandong_pmos_hourly.csv \
  --source-repo .local_artifacts/source_repos/epf-sota-experiment \
  --start-day 2026-06-01 \
  --end-day 2026-06-30 \
  --train-window-days 90 \
  --work-dir .local_artifacts/p16_cfg05_30d \
  --run-ledger-chain \
  --run-fusion-chain \
  --json --strict
```

Arguments:
- `--raw-data <path>` required unless a local fixture/mock mode is used in tests.
- `--source-repo <path>` default `.local_artifacts/source_repos/epf-sota-experiment`.
- `--start-day YYYY-MM-DD` default configurable.
- `--end-day YYYY-MM-DD` inclusive.
- `--target-days` optional comma-separated override.
- `--train-window-days 90` default.
- `--work-dir .local_artifacts/p16_cfg05_30d` default.
- `--reuse-model-per-day` default false; walk-forward should train per target day unless an explicit mode says otherwise.
- `--run-ledger-chain` optional.
- `--run-fusion-chain` optional.
- `--allow-partial-days` optional, default false for strict metrics.
- `--json`, `--strict`, `--verbose`.

Behavior:
1. Validate raw CSV with P13/P14 contract checker.
2. Validate source repo exists.
3. For each target day:
   - train/export cfg05 using P15-fixed 24H filtering;
   - export local model and feature CSV under `.local_artifacts/p16_cfg05_30d/<target_day>/`;
   - run cfg05 prediction;
   - validate prediction schema;
   - check feature 24H completeness;
   - check prediction 24H completeness;
   - attach y_true for eval only if present and not null;
   - write local prediction CSV under `.local_artifacts/p16_cfg05_30d/predictions/`.
4. Merge all complete-day predictions into a local backtest CSV.
5. Evaluate metrics on complete days only.
6. If `--run-ledger-chain`, append local predictions and actuals to local ledgers under `.local_artifacts/p16_cfg05_30d/ledgers/`.
7. If `--run-fusion-chain`, pass cfg05 predictions through existing residual correction no-op, fusion engine, weight ledger extraction, and final output validator where applicable.
8. Return structured JSON summary.

Summary keys required:
- `raw_data_status`
- `source_repo_status`
- `start_day`
- `end_day`
- `target_days_requested`
- `target_days_completed`
- `target_days_incomplete`
- `prediction_rows_total`
- `eval_rows_total`
- `complete_24h_days`
- `incomplete_days_detail`
- `metrics`
- `methodology_alignment_status`
- `source_11_48_reproduction_claim_allowed`
- `ledger_chain_attempted`
- `ledger_chain_status`
- `fusion_chain_attempted`
- `fusion_chain_status`
- `local_artifact_root`
- `forbidden_files_check`
- `final_status`
- `reason_codes`

## 24H completeness requirements

Each target day must have:
- exactly 24 prediction rows;
- hour_business values 1..24 exactly once;
- `D+1 00:00` present as hour 24;
- no duplicate key rows;
- standard schema fields valid.

Incomplete days must not be silently included in metrics.

---

# 2. Evaluation utility

File: `scripts/evaluate_cfg05_backtest.py`

CLI args:
- `--predictions <path>`
- `--json`
- `--strict`

Input prediction CSV must contain:
- `target_day`
- `business_day`
- `ds`
- `hour_business`
- `y_pred`
- `y_true` for eval only

Behavior:
- Validate each day is 24H complete.
- Exclude rows with null y_true/y_pred.
- Compute:
  - `sMAPE_floor50`
  - `MAE`
  - `RMSE`
  - per-day metrics
  - per-hour metrics
  - period metrics
- Return row count and excluded-day detail.

Do not modify repo files by default.

---

# 3. Ledger/fusion handoff

If `--run-ledger-chain` is supplied:
- Use existing P5 ledger utilities if available.
- Local ledger dir only: `.local_artifacts/p16_cfg05_30d/ledgers/`.
- Append cfg05 predictions in 3.0 standard schema.
- Append actuals from y_true for eval only.
- Deduplicate by existing prediction/actual ledger keys.
- Confirm no leakage: training actuals must use only `business_day < target_day`.

If `--run-fusion-chain` is supplied:
- Feed corrected/no-op residual output into fusion.
- Since only one REAL model may be available, fusion may be `single_model_passthrough` or equal weight 1.0 for cfg05.
- Mark P5M residual as `DATA_MISSING_NO_OP` unless real P5M pack exists.
- Mark realtime/ExtremPriceClf as not in scope / missing unless artifacts exist.
- Final output must preserve 24 rows per day.

Important: downstream chain is partially structural unless real residual/fusion/classifier artifacts exist. Say this plainly.

---

# 4. Methodology alignment with source 11.48%

Add a method-alignment section in the report.

Check whether current P16 evaluation matches source methodology:
- same target date range?
- same training window?
- same cfg05 params?
- same feature builder?
- same rolling/walk-forward rule?
- same metric formula?
- same y_true availability?

Default conclusion should be:
- Metrics are valid for P16 local backtest.
- They are **not** a reproduction claim of source 11.48% unless all alignment checks pass.

Set:
- `source_11_48_reproduction_claim_allowed = false` unless all checks pass.

---

# 5. Tests

Create `tests/test_p16_cfg05_30d_backtest_chain.py` using tmp_path and mocks. Do not require real raw CSV or network in tests.

Required tests:
1. missing raw data returns structured failure, non-strict exit 0.
2. strict mode exits nonzero on missing raw data.
3. unsafe work-dir rejected.
4. one mocked complete day produces 24 rows and metrics.
5. mocked 23-row day is excluded or marks partial.
6. multiple days aggregate correctly.
7. metrics utility computes sMAPE_floor50 / MAE / RMSE.
8. evaluation refuses incomplete days in strict mode.
9. evaluation can allow partial days only when explicitly configured.
10. `source_11_48_reproduction_claim_allowed` defaults false.
11. ledger chain writes only under local ignored path.
12. fusion chain with one model uses passthrough/equal weight and stays 24H complete.
13. no downstream chain can mark P5M/RT/ExtremPriceClf REAL without artifacts.
14. summary JSON contains all required keys.
15. forbidden files check passes.
16. no `.csv`/artifact files tracked or untracked in repo.

Run:
```bash
python -m pytest tests/test_p16_cfg05_30d_backtest_chain.py
python -m pytest
```

---

# 6. Report

Write:
`docs/reports/p16_real_cfg05_30d_backtest_chain_report.md`

Report format:

```md
# P16 Real cfg05 30-Day Backtest + Ledger/Fusion Chain Report

## 1. Executive status
## 2. P15 milestone recap
## 3. Backtest configuration
## 4. 24H completeness results
## 5. Evaluation metrics
## 6. Methodology alignment with source 11.48%
## 7. Ledger chain handoff result
## 8. Fusion/final chain handoff result
## 9. Local artifact files created (ignored only)
## 10. Tests run
## 11. Forbidden files check
## 12. Remaining blockers
## 13. P17 recommendation
```

Report must include:
- date range;
- requested days;
- completed days;
- incomplete days;
- total prediction rows;
- total eval rows;
- per-day/per-hour/period summary;
- final status;
- exact statement whether source 11.48% reproduction claim is allowed.

Hard wording:
- If evaluation window/method differs, write: `Not a source 11.48% reproduction claim`.
- If downstream chain uses no-op residual/fusion passthrough, write that plainly.
- If local files are generated, list paths but confirm they are gitignored and not committed.

---

# 7. Final response format

Reply with:

```md
P16 Real cfg05 30-Day Backtest + Chain Handoff Summary

1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Backtest date range:
6. 24H completeness:
7. Prediction rows:
8. Evaluation status:
9. Metrics:
10. Source 11.48% reproduction claim:
11. Ledger chain status:
12. Fusion/final chain status:
13. Local artifact files:
14. Forbidden files check:
15. Remaining blockers:
16. Commit:
17. Final status:
```

Do not ask follow-up questions unless raw data is truly unavailable. Make best effort with local paths already established by P14/P15.
