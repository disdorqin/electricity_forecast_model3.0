# P12 Source Repo Clone + cfg05 REAL Export Run Prompt

You are executing P12 for electricity_forecast_model3.0.

P11 is complete. It proved the local export/readiness tooling works, but the local machine did not have `disdorqin/epf-sota-experiment`, so cfg05 REAL_READY was blocked. P12 is the first real source-repo acquisition run: clone or locate the source repo, then run the P10/P11 tools against the actual source workspace.

P12 is not a new structural phase. It is an operational attempt to obtain cfg05 artifact + cfg05 feature input locally and run cfg05 REAL smoke.

Hard scope:
- Do not commit source repo clone.
- Do not commit model artifacts.
- Do not commit CSV input/output files.
- Do not commit parquet, pkl, joblib, pt, pth, ckpt, or ledger files.
- Do not claim cfg05 REAL unless REAL_READY is achieved by path-verified artifact + real input + prediction + validator pass.
- Do not claim 11.48% reproduced unless real y_true evaluation is run.

Current known status:
- P10: CFG05_REAL_READY NOT ACHIEVED; tools ready.
- P11: CFG05_EXPORT_BLOCKED because source repo not available locally.
- P11 local artifact policy: use `.local_artifacts/`, gitignored.
- Current cfg05 feature count: import `CFG05_FEATURE_COLUMNS`; do not hard-code count.

P12 goals:
1. Clone or locate `disdorqin/epf-sota-experiment` locally.
2. Run `scripts/export_cfg05_from_source.py` against that source repo.
3. If cfg05 artifact exists, verify LOADABLE and optionally copy it to `.local_artifacts/p12_cfg05/`.
4. Run `scripts/build_cfg05_feature_input_from_source.py` against source repo or a discovered input CSV.
5. If feature input exists or can be built, verify SCHEMA_READY.
6. If both artifact LOADABLE and input SCHEMA_READY, run cfg05 REAL smoke pipeline.
7. Produce report with exact outcome and next commands.

Important: P12 may end in one of these final statuses:
- CFG05_REAL_READY_LOCAL — achieved locally, artifact/input/output not committed.
- CFG05_ARTIFACT_FOUND_INPUT_BLOCKED — model artifact found/loadable, but feature input missing.
- CFG05_ARTIFACT_BLOCKED_INPUT_FOUND — input found/schema-ready, but model artifact missing/invalid.
- CFG05_EXPORT_BLOCKED — source repo clone/search failed or no artifact/export route.
- CFG05_INPUT_BLOCKED — feature CSV/pipeline missing.
- CFG05_REAL_SMOKE_FAILED — artifact + input found, but adapter prediction or validator failed.

Required files to create or update:
- scripts/run_p12_cfg05_source_clone_and_smoke.py
- tests/test_p12_cfg05_source_clone_and_smoke.py
- docs/reports/p12_source_repo_clone_cfg05_real_export_run_report.md

Optional updates:
- docs/LOCAL_ARTIFACTS.md if the local workflow needs clarification.

1. P12 orchestration script

File: `scripts/run_p12_cfg05_source_clone_and_smoke.py`

CLI args:
- `--source-repo <path>` optional existing clone
- `--clone-url https://github.com/disdorqin/epf-sota-experiment.git` default
- `--clone-dir <ignored local path>` default `.local_artifacts/source_repos/epf-sota-experiment`
- `--work-dir <ignored local path>` default `.local_artifacts/p12_cfg05`
- `--target-day YYYY-MM-DD`
- `--input-csv <path>` optional known feature CSV
- `--copy-if-found`
- `--run-real-smoke`
- `--json`
- `--strict`
- `--verbose`

Behavior:
- If `--source-repo` exists, use it.
- Else if clone-dir exists, use it.
- Else clone from `--clone-url` into clone-dir.
- Ensure clone-dir and work-dir are ignored/local paths before writing.
- After source repo available:
  1. Run or import `export_cfg05_from_source` logic.
  2. Run or import `build_cfg05_feature_input_from_source` logic.
  3. Run `run_cfg05_real_smoke_pipeline` only when preconditions are met or `--run-real-smoke` is supplied.
- Do not create tracked files outside the required script/test/report.
- Non-strict mode should never crash merely because artifact/input is missing; return structured blocker status.
- Strict mode should fail nonzero if clone/source/artifact/input/smoke required by flags cannot pass.

Summary JSON should include:
- source_repo_status
- source_repo_path
- artifact_status
- artifact_candidates
- copied_artifact_path
- input_status
- input_candidates
- prepared_input_path
- real_smoke_attempted
- real_smoke_status
- prediction_rows
- validator_passed
- readiness_label
- final_status
- reason_codes
- forbidden_files_check

2. Real clone behavior

P12 can actually run:

```bash
git clone https://github.com/disdorqin/epf-sota-experiment.git .local_artifacts/source_repos/epf-sota-experiment
```

But only under ignored `.local_artifacts/` or another ignored local path.

If clone fails, report clone error and final status `CFG05_EXPORT_BLOCKED`.

3. Artifact search behavior

Use existing:

```bash
python -m scripts.export_cfg05_from_source \
  --source-repo .local_artifacts/source_repos/epf-sota-experiment \
  --work-dir .local_artifacts/p12_cfg05 \
  --copy-if-found \
  --json
```

If a loadable cfg05 artifact is found:
- Keep original path in report.
- If copied, copy only into `.local_artifacts/p12_cfg05/`.
- Verify with `check_cfg05_artifact`.

If no artifact found:
- Identify candidate source scripts/reports containing cfg05/lightgbm/save_model/Booster.
- Output exact next command or required export patch.

4. Feature input behavior

Use existing:

```bash
python -m scripts.build_cfg05_feature_input_from_source \
  --source-repo .local_artifacts/source_repos/epf-sota-experiment \
  --target-day YYYY-MM-DD \
  --json
```

If `--input-csv` is provided:
- Validate with dynamic `CFG05_FEATURE_COLUMNS`.
- Require `ds` and 24 target-day rows if target-day provided.

If no input found:
- Identify source data files/scripts that may generate features.
- Output exact next command or required build/export patch.

5. REAL smoke behavior

Only attempt REAL smoke when:
- artifact status is LOADABLE or better
- input status is SCHEMA_READY

Then run:

```bash
python -m scripts.run_cfg05_real_smoke_pipeline \
  --cfg05-model <artifact_path_or_dir> \
  --cfg05-input <feature_csv> \
  --target-day <target-day> \
  --strict --json
```

REAL_READY only if:
- readiness_label == REAL_READY
- prediction_rows > 0, ideally 24
- validator_passed == true

Otherwise final status is `CFG05_REAL_SMOKE_FAILED` with reason codes.

6. Tests

Create `tests/test_p12_cfg05_source_clone_and_smoke.py` using tmp_path and mocked/subprocess-safe behavior. Do not require network in tests.

Required tests:
1. existing source repo path is used without clone.
2. missing source repo with clone disabled/failing returns CFG05_EXPORT_BLOCKED, not crash.
3. clone-dir under forbidden path is rejected.
4. work-dir under forbidden path is rejected.
5. fake source repo with placeholder artifact never reaches REAL_READY.
6. fake source repo with no feature input reports CFG05_INPUT_BLOCKED.
7. supplied schema-ready input is accepted dynamically via CFG05_FEATURE_COLUMNS.
8. real smoke is not attempted unless artifact + input gates pass.
9. non-strict mode returns structured blocker status with exit 0.
10. strict mode exits nonzero on blocker.
11. summary JSON contains all required keys.
12. forbidden files check passes.

Run:
- python -m pytest tests/test_p12_cfg05_source_clone_and_smoke.py
- python -m pytest

7. Report

Write:
`docs/reports/p12_source_repo_clone_cfg05_real_export_run_report.md`

Report format:
# P12 Source Repo Clone + cfg05 REAL Export Run Report

## 1. Executive status
## 2. Source repo clone/location result
## 3. cfg05 artifact search/export result
## 4. cfg05 feature input search/build result
## 5. cfg05 REAL smoke result
## 6. Readiness matrix
## 7. Local artifact files created (ignored only)
## 8. Tests run
## 9. Forbidden files check
## 10. Blockers and exact next commands
## 11. P13 recommendation

Hard wording:
- If no artifact/input/smoke success, write `CFG05_REAL_READY_LOCAL: NOT ACHIEVED`.
- If achieved, write `CFG05_REAL_READY_LOCAL: ACHIEVED`, with artifact path, input path, prediction rows, validator result, and confirmation that files were not committed.
- Never claim metric reproduction unless y_true evaluation is run.

Final response format:
P12 Source Repo Clone + cfg05 REAL Export Run Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Source repo status:
6. Artifact status:
7. Feature input status:
8. REAL smoke status:
9. Readiness matrix:
10. Local artifact files:
11. Forbidden files check:
12. Blockers / next commands:
13. Commit:
14. Final status:
