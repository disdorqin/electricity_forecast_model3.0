# P10 cfg05 Artifact Export + REAL Adapter Smoke Prompt

You are executing P10 for electricity_forecast_model3.0.

P9 is complete and synchronized on origin/master. The artifact readiness gates exist, but without local real artifacts all six gates are MISSING. P10 should focus on cfg05 because it is the day-ahead champion and the first real-capability gate to unlock.

Important context from source repo:
- Source branch/repo: disdorqin/epf-sota-experiment
- Current champion: cfg05 micro-search LightGBM, sMAPE_floor50 11.48% in the source report.
- Invalid models remain forbidden: lgbm_spike_residual_1127, stage3_old_1164, lightgbm_90d_orig_1197.
- The 3.0 adapter requires cfg05 LightGBM model artifact plus cfg05-compatible input feature CSV.

P10 goals:
1. Locate or export the cfg05 LightGBM artifact from the source repo/local source workspace.
2. Locate or build cfg05-compatible input feature CSV for one target day.
3. Run P9 readiness gates on cfg05 artifact + input.
4. Run cfg05 REAL adapter smoke only if artifact is LOADABLE and input is SCHEMA_READY.
5. If REAL_READY cannot be achieved, produce a precise acquisition/export checklist instead of pretending success.
6. Do not commit artifacts, data, outputs, local reports, CSVs, parquet, pkl/joblib/pt/pth/ckpt, or ledgers.

Hard rule:
REAL_READY requires all of these:
- path-verified cfg05 LightGBM artifact
- adapter load succeeds
- cfg05-compatible non-synthetic input with required feature columns
- prediction produces non-empty standard output
- validate_prediction_dataframe passes

Merely finding a file is not REAL_READY.

Known issue to fix before P10 smoke:
- P9 report text says cfg05 input has 39 feature columns, but `models/adapters/cfg05_dayahead_lgbm.py` defines `CFG05_FEATURE_COLUMNS` directly and the code appears to require 42. P10 must derive the feature count from `len(CFG05_FEATURE_COLUMNS)` at runtime and update any stale doc/test wording. Code should never hard-code 39 or 42 except in tests that import the constant.

Required files to create or update:
- scripts/locate_cfg05_artifact.py
- scripts/prepare_cfg05_real_input.py
- scripts/run_cfg05_real_smoke_pipeline.py
- tests/test_cfg05_artifact_export_readiness.py
- docs/reports/p10_cfg05_artifact_export_real_smoke_report.md
- Optional doc update: docs/reports/p9_real_artifact_readiness_cfg05_smoke_report.md if it contains stale 39-column wording.

Do not modify source repo unless explicitly asked. P10 in this repo should provide locate/prepare/smoke tooling and documentation. If a local source repo path is supplied, read from it. If not supplied, report MISSING with exact next steps.

Suggested local paths to support:
- --source-repo <path-to-epf-sota-experiment>
- --cfg05-model <path-to-cfg05-model-file-or-dir>
- --cfg05-input <path-to-feature-csv>
- --target-day YYYY-MM-DD
- --work-dir <ignored local tmp path>
- --out <optional explicit output path>

Local artifact/output path policy:
- Default work dir should be outside committed repo paths, or an ignored local directory like `.local_artifacts/` only if .gitignore covers it.
- Never write into `data/`, `outputs/`, `ledgers/`, or `reports/local/` unless those paths are ignored and explicitly requested.
- Never commit generated CSV/model/output files.

1. locate cfg05 artifact

File: `scripts/locate_cfg05_artifact.py`

Behavior:
- Accept `--source-repo`, `--model-dir`, `--json`.
- Search for likely model artifact names:
  - cfg05_model.txt
  - model.txt
  - lightgbm_cfg05_dayahead.txt
  - any LightGBM text model under paths containing cfg05/lgbm/lightgbm/champion
- Ignore invalid model names from blacklist.
- For each candidate, run `check_cfg05_artifact(candidate_or_parent)`.
- Return candidates with status PRESENT/LOADABLE/INVALID.
- Do not copy or commit artifacts.

2. prepare cfg05 input

File: `scripts/prepare_cfg05_real_input.py`

Behavior:
- Accept `--input`, `--target-day`, `--out`, `--json`.
- Load a candidate CSV.
- Import `CFG05_FEATURE_COLUMNS` from `models.adapters.cfg05_dayahead_lgbm`.
- Validate that all `CFG05_FEATURE_COLUMNS` and `ds` exist.
- Optionally filter one target day using the adapter's expected rule: ds in `(target_day 00:00, target_day + 1 day 00:00]` as implemented by adapter, but verify 24 rows after filtering.
- Write prepared feature CSV only if `--out` is explicitly supplied.
- Never hard-code feature count; report `len(CFG05_FEATURE_COLUMNS)`.
- Return SCHEMA_READY only if required columns and target-day rows are valid.

3. run cfg05 real smoke pipeline

File: `scripts/run_cfg05_real_smoke_pipeline.py`

Behavior:
- Accept `--cfg05-model`, `--cfg05-input`, `--target-day`, `--out`, `--strict`, `--json`, `--verbose`.
- Call `check_cfg05_artifact` and `check_cfg05_input`.
- If missing/invalid:
  - non-strict: exit 0 with NOT_READY / MISSING / INVALID summary
  - strict: exit nonzero
- If artifact LOADABLE and input SCHEMA_READY:
  - call `scripts/run_cfg05_real_adapter_smoke.py` or import its core function if available
  - verify output passes `validate_prediction_dataframe`
  - verify output row count is 24 unless input intentionally contains fewer rows and this is documented
  - label REAL_READY only if prediction succeeds and validator passes
- Do not write output unless --out is provided.

4. tests

Create `tests/test_cfg05_artifact_export_readiness.py` with tmp_path only.

Required tests:
1. locate script handles missing source repo and returns MISSING/not crash.
2. locate script ignores invalid blacklisted model names.
3. locate script finds candidate names in tmp_path but placeholder artifact does not become REAL_READY.
4. prepare script imports CFG05_FEATURE_COLUMNS and reports dynamic feature count.
5. prepare script rejects CSV missing any required feature column.
6. prepare script rejects missing ds.
7. prepare script can validate a synthetic schema-ready CSV with all required columns + ds.
8. prepare script does not write output unless --out is supplied.
9. real smoke pipeline missing model exits 0 non-strict and nonzero strict.
10. real smoke pipeline missing input exits 0 non-strict and nonzero strict.
11. real smoke pipeline placeholder artifact never returns REAL_READY.
12. no generated files appear in forbidden repo paths.
13. P9 report/docs no longer hard-code stale feature count if updated.

Run all tests:
- python -m pytest tests/test_cfg05_artifact_export_readiness.py
- python -m pytest

5. report

Write:
`docs/reports/p10_cfg05_artifact_export_real_smoke_report.md`

Report format:
# P10 cfg05 Artifact Export + REAL Adapter Smoke Report

## 1. Executive status
## 2. cfg05 source artifact search
## 3. cfg05 input/schema readiness
## 4. cfg05 REAL smoke result
## 5. Feature column count reconciliation
## 6. Artifact readiness matrix
## 7. Tests run
## 8. Forbidden files check
## 9. Known limitations
## 10. P11 recommendation

Report must state one of:
- CFG05_REAL_READY achieved, with artifact path/input path/validator result documented but without committing files.
- CFG05_NOT_READY, with exact missing artifact/input/export steps.

Hard wording rules:
- Do not claim cfg05 REAL unless readiness_label is REAL_READY and validator passed.
- Do not claim source metrics are reproduced in 3.0 unless actual evaluation was run on real y_true.
- Do not commit model files or prepared feature CSVs.
- Do not call placeholder artifacts REAL.

Final response format:
P10 cfg05 Artifact Export + REAL Smoke Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Source repo/artifact search status:
6. cfg05 feature input status:
7. Feature column count:
8. cfg05 REAL smoke status:
9. Artifact readiness matrix:
10. Forbidden files check:
11. Known limitations:
12. Commit:
13. Final status:
