# P11 cfg05 REAL Artifact Export Attempt Prompt

You are executing P11 for electricity_forecast_model3.0.

P10 is complete. It created structural tooling for locating cfg05 artifacts, preparing cfg05 inputs, and running the cfg05 REAL smoke pipeline. P10 did NOT achieve REAL_READY because no real cfg05 artifact or real cfg05 input CSV was present.

P11 is the first actual export/acquisition attempt for cfg05. It should use a local clone/workspace of `disdorqin/epf-sota-experiment` if available. If the source workspace is not available, P11 must produce a precise missing-source/export checklist instead of pretending success.

Hard scope:
- Do not commit model artifacts.
- Do not commit CSV input/output files.
- Do not commit parquet, pkl, joblib, pt, pth, ckpt, or ledger files.
- Do not claim cfg05 REAL unless REAL_READY is achieved by path-verified artifact + real input + prediction + validator pass.
- Do not claim source sMAPE 11.48 is reproduced in 3.0 unless real y_true evaluation is run.

Current cfg05 feature contract:
- Import from `models.adapters.cfg05_dayahead_lgbm`.
- `len(CFG05_FEATURE_COLUMNS) = 42` in current code.
- Never hard-code feature count.

P11 goals:
1. Accept or discover local source repo path for `epf-sota-experiment`.
2. Search for cfg05 LightGBM artifact in source workspace.
3. If no artifact exists, identify exact training/export script or location needed.
4. Search for or generate cfg05 feature input CSV with all `CFG05_FEATURE_COLUMNS` + `ds` for one target day.
5. Run `check_cfg05_artifact`, `check_cfg05_input`, and `run_cfg05_real_smoke_pipeline` when possible.
6. Store any generated artifact/input/output only under ignored local paths, never in git-tracked files.
7. Produce P11 report documenting one of:
   - CFG05_REAL_READY achieved locally, files not committed; or
   - CFG05_EXPORT_BLOCKED with exact blockers and next commands.

Required files to create or update:
- scripts/export_cfg05_from_source.py
- scripts/build_cfg05_feature_input_from_source.py
- tests/test_cfg05_source_export_attempt.py
- docs/reports/p11_cfg05_real_artifact_export_attempt_report.md

Optional updates:
- docs/LOCAL_ARTIFACTS.md — short guide for local ignored artifacts.
- .gitignore only if needed to ensure `.local_artifacts/`, `.local_outputs/`, `.local_ledgers/` are ignored.

Suggested local output policy:
- default local work dir: `.local_artifacts/p11_cfg05/`
- must be ignored by git before writing anything there.
- never write generated files unless explicitly requested by CLI args.

1. export cfg05 from source

File: `scripts/export_cfg05_from_source.py`

CLI args:
- `--source-repo <path>`
- `--target-day YYYY-MM-DD`
- `--work-dir <ignored local dir>`
- `--copy-if-found` optional; only copies into ignored local work dir
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Validate source repo path exists and looks like `epf-sota-experiment`.
- Search files and known scripts for cfg05 references.
- Reuse `scripts/locate_cfg05_artifact.py` logic.
- If a valid LightGBM artifact is found, report path and LOADABLE status.
- If no artifact found, inspect likely scripts/reports:
  - `docs/reports/dayahead_current_champion.md`
  - `docs/reports/dayahead_final_sprint_report.md`
  - `scripts/run_stage3_inline.py`
  - `scripts/run_final_sprint.py`
  - any script containing cfg05 / micro-search / lgbm / save_model
- Output exact next commands or missing training/export step.
- Do not train unless an explicit safe command is present and user supplies `--run-export`; default is analysis only.

2. build cfg05 feature input from source

File: `scripts/build_cfg05_feature_input_from_source.py`

CLI args:
- `--source-repo <path>`
- `--target-day YYYY-MM-DD`
- `--input-csv <path>` optional candidate existing feature CSV
- `--out <ignored local output csv>` optional
- `--json`
- `--strict`
- `--verbose`

Behavior:
- Import `CFG05_FEATURE_COLUMNS` dynamically.
- If `--input-csv` is provided, validate it via `prepare_cfg05_real_input` logic.
- If not provided, search source repo for likely feature CSV or feature generation scripts.
- Identify whether source repo can produce all 42 features.
- If a schema-ready input is found, report SCHEMA_READY.
- If no input is found, produce exact checklist of missing data/command.
- Do not write output unless `--out` is provided and path is ignored/local.

3. optional cfg05 real smoke orchestration

If both artifact LOADABLE and input SCHEMA_READY are found, call:

```bash
python -m scripts.run_cfg05_real_smoke_pipeline \
  --cfg05-model <artifact_path_or_dir> \
  --cfg05-input <input_csv> \
  --target-day <target_day> \
  --strict --json
```

Expected outcomes:
- REAL_READY only if adapter predicts >0 rows and validator passes.
- Otherwise NOT_READY/INVALID with reason codes.

4. tests

Create `tests/test_cfg05_source_export_attempt.py` using tmp_path only.

Required tests:
1. missing source repo returns MISSING / not crash.
2. fake source repo with champion report but no artifact returns CFG05_EXPORT_BLOCKED.
3. fake source repo with blacklisted artifact names ignores them.
4. fake source repo with placeholder cfg05_model.txt finds candidate but marks INVALID, not REAL_READY.
5. export script does not copy unless `--copy-if-found` supplied.
6. export script refuses to write outside ignored/local work dir.
7. build input script missing source repo returns MISSING / not crash.
8. build input script validates dynamic 42-column schema via imported constant.
9. build input script rejects missing ds.
10. build input script rejects missing feature column.
11. build input script does not write unless `--out` supplied.
12. orchestration never labels REAL_READY with placeholder artifact.
13. forbidden files check passes.

Run:
- python -m pytest tests/test_cfg05_source_export_attempt.py
- python -m pytest

5. report

Write:
`docs/reports/p11_cfg05_real_artifact_export_attempt_report.md`

Report format:
# P11 cfg05 REAL Artifact Export Attempt Report

## 1. Executive status
## 2. Source workspace status
## 3. cfg05 artifact search/export result
## 4. cfg05 feature input search/build result
## 5. cfg05 REAL smoke attempt
## 6. Readiness matrix
## 7. Local artifact path policy
## 8. Tests run
## 9. Forbidden files check
## 10. Blockers and exact next commands
## 11. P12 recommendation

Final status must be one of:
- CFG05_REAL_READY_LOCAL — achieved locally, no artifacts committed.
- CFG05_EXPORT_BLOCKED — source repo/artifact/export command missing.
- CFG05_INPUT_BLOCKED — feature input pipeline/data missing.
- CFG05_REAL_SMOKE_FAILED — artifact/input found but adapter/validator failed.

Hard wording rules:
- Do not say cfg05 REAL unless final status is CFG05_REAL_READY_LOCAL and validator passed.
- Do not claim performance metrics.
- Do not commit local files.

Final response format:
P11 cfg05 REAL Artifact Export Attempt Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Source workspace status:
6. Artifact search/export status:
7. Feature input status:
8. cfg05 REAL smoke status:
9. Readiness matrix:
10. Local artifact policy:
11. Forbidden files check:
12. Blockers / next commands:
13. Commit:
14. Final status:
