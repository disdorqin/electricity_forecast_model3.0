# P9 Real Artifact Readiness + cfg05 REAL Adapter Smoke Prompt

You are executing P9 for electricity_forecast_model3.0.

P9 begins the transition from structural/dry-run smoke toward real artifact integration. It must be conservative: do not claim REAL unless artifact paths and input data are explicitly supplied and validated.

Current state:
- P7 single-day structural smoke passed.
- P8 30-day ledger backfill structural smoke passed.
- Current default labels remain: DRY_RUN / STRUCTURAL_ONLY / DATA_MISSING / RULE_FALLBACK.
- No model artifacts, risk packs, actual production ledgers, or classifier artifacts are committed to the repository.

P9 goals:
1. Build artifact readiness checks for all real-capability gates.
2. Add a cfg05 REAL adapter smoke path that runs only when model artifact and compatible input/features are path-verified.
3. Keep all other missing artifacts marked NOT_READY / DATA_MISSING / STUB.
4. Produce a readiness report that clearly separates STRUCTURAL_READY from REAL_READY.
5. Do not commit any artifact, data, prediction, ledger, parquet, CSV, pickle, joblib, model weight, or output file.

Real-capability gates to check:
- cfg05 LightGBM artifact for day-ahead REAL prediction.
- cfg05 feature input file or DataFrame schema sufficient for cfg05 adapter.
- realtime assist pack / residual_model.pkl for safe realtime correction.
- P5M residual risk/canonical pack for real correction.
- actual ledger sufficient for BGEW non-fallback learning.
- ExtremPriceClf artifact and implemented inference path for real negative classifier.

Required files to create or update:
- artifacts/__init__.py
- artifacts/readiness.py
- scripts/check_artifact_readiness.py
- scripts/run_cfg05_real_adapter_smoke.py
- tests/test_artifact_readiness.py
- tests/test_cfg05_real_adapter_smoke.py
- docs/reports/p9_real_artifact_readiness_cfg05_smoke_report.md

Artifact readiness statuses:
- MISSING: path not provided or path does not exist.
- PRESENT: path exists, but not yet loaded or validated.
- LOADABLE: adapter can load artifact.
- SCHEMA_READY: input schema/features are compatible.
- REAL_READY: artifact loaded and real prediction/inference produced valid output.
- NOT_IMPLEMENTED: artifact exists but real inference path is still stub.
- INVALID: path exists but load/validation fails.

Important rule:
REAL_READY requires successful load + successful non-synthetic prediction/inference + validator pass. Merely finding a file is not enough.

P9 scope details:

1. artifacts/readiness.py

Implement functions/classes:

- ArtifactStatus dataclass or typed dict.
- check_path(path, expected_kind=None) -> ArtifactStatus
- check_cfg05_artifact(model_dir_or_file) -> ArtifactStatus
- check_cfg05_input(input_path_or_df) -> ArtifactStatus
- check_rt_assist_pack(pack_dir) -> ArtifactStatus
- check_p5m_pack(pack_dir_or_risk_path) -> ArtifactStatus
- check_actual_ledger(ledger_path_or_df, min_days=7) -> ArtifactStatus
- check_extrempriceclf_artifact(model_dir) -> ArtifactStatus
- run_all_artifact_readiness(...) -> dict

The functions must never require artifacts to exist in tests. Missing paths should return MISSING, not crash.

2. cfg05 REAL adapter smoke

File:
- scripts/run_cfg05_real_adapter_smoke.py

The smoke should support:

--model-dir <path> or --model-file <path>
--input <csv path>
--target-day YYYY-MM-DD
--out <optional output path>
--production
--verbose

Behavior:

- If model artifact path missing: exit gracefully with status NOT_READY unless --strict is supplied.
- If input path missing: exit gracefully with status NOT_READY unless --strict is supplied.
- If both exist: load CFG05DayaheadAdapter, build/validate features if needed, run prediction, validate prediction output.
- Only then label cfg05 as REAL_READY.
- If --out is provided, write only to that explicit path. Do not default-write into repo data/outputs/ledgers.

Output summary JSON should include:
- cfg05_artifact_status
- cfg05_input_status
- cfg05_adapter_loaded
- prediction_rows
- validator_passed
- readiness_label
- reason_codes

3. check_artifact_readiness CLI

File:
- scripts/check_artifact_readiness.py

Suggested args:
--cfg05-model <path>
--cfg05-input <path>
--rt-assist-pack <path>
--p5m-pack <path>
--actual-ledger <path>
--extrempriceclf-dir <path>
--json
--strict

Should print a JSON or readable table with statuses.

4. Tests

Create tests/test_artifact_readiness.py and tests/test_cfg05_real_adapter_smoke.py.

Tests must use synthetic tmp_path files only. Do not include real model artifacts.

Required tests:
1. missing cfg05 model returns MISSING, not crash.
2. missing cfg05 input returns MISSING, not crash.
3. artifact path exists returns PRESENT or LOADABLE depending type.
4. invalid cfg05 artifact returns INVALID or LOADABLE false, not REAL_READY.
5. check_rt_assist_pack missing returns MISSING.
6. check_p5m_pack missing returns MISSING/DATA_MISSING.
7. check_extrempriceclf_artifact missing returns MISSING or NOT_IMPLEMENTED if path exists but inference stub.
8. check_actual_ledger with insufficient rows returns PRESENT but not REAL_READY.
9. run_all_artifact_readiness returns all expected keys.
10. cfg05 smoke missing paths exits 0 in non-strict mode with NOT_READY.
11. cfg05 smoke strict mode exits nonzero if required paths missing.
12. cfg05 smoke never writes output unless --out is supplied.
13. no REAL_READY appears from placeholder/tmp invalid artifacts.
14. forbidden-files check passes.

5. Report

Write:
- docs/reports/p9_real_artifact_readiness_cfg05_smoke_report.md

Report sections:
# P9 Real Artifact Readiness + cfg05 REAL Adapter Smoke Report

## 1. Executive status
## 2. Artifact readiness status matrix
## 3. cfg05 REAL adapter smoke gate
## 4. Realtime assist artifact gate
## 5. P5M residual artifact/risk gate
## 6. Actual ledger/BGEW gate
## 7. ExtremPriceClf artifact gate
## 8. Tests run
## 9. Forbidden files check
## 10. Known limitations
## 11. P10 recommendation

Hard wording rules:
- Do not claim cfg05 REAL unless REAL_READY is achieved.
- Do not claim realtime safe correction is REAL unless pack exists and adapter uses it.
- Do not claim P5M residual improvement unless real risk/canonical pack changes y_pred_corrected.
- Do not claim BGEW production learner unless actual ledger is real and weights are non-fallback.
- Do not claim ExtremPriceClf ML inference unless real inference path is implemented and artifact is validated.

Run all previous tests plus P9 tests.

Final response format:
P9 Real Artifact Readiness + cfg05 Smoke Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Artifact readiness matrix:
6. cfg05 artifact status:
7. cfg05 REAL smoke status:
8. Other artifact gates:
9. Forbidden files check:
10. Known limitations:
11. Commit:
12. Final status:
