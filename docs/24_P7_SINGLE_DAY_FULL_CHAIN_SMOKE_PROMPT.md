# P7 Single-Day Full-Chain Structural Smoke Prompt

You are executing P7 for electricity_forecast_model3.0.

P7 is a single-day full-chain structural/dry-run smoke. It is NOT a production real inference test and must not claim real model performance.

Current P6.5 recommendation:
- GO for P7 single-day full-chain smoke under STRUCTURAL / DRY_RUN conditions.
- NO-GO for REAL production inference claims.

Required label for current repository state:

P7_SINGLE_DAY_FULL_CHAIN_SMOKE = DRY_RUN / STRUCTURAL_ONLY with DATA_MISSING and RULE_FALLBACK paths

Pipeline to smoke:
1. Day-ahead prediction runner: dry-run model zoo or cfg05-only if an external cfg05 artifact is explicitly path-verified.
2. Realtime path: DA_ONLY fallback if used.
3. Prediction validator.
4. Residual correction: DATA_MISSING no-op unless risk/canonical pack exists.
5. Residual validator.
6. Corrected ledger append.
7. Fusion engine: allow_dry_run=True for dry-run models; compute fused output from y_pred_corrected.
8. Fusion validator.
9. Fusion ledger append and weight ledger extraction.
10. Negative classifier: no-artifact fallback + optional rule fallback.
11. Final output validator.
12. Write smoke report only; do not commit output data.

Do not commit:
- data/*
- outputs/*
- reports/local/*
- ledgers/*.csv
- *.csv, *.xlsx, *.xls, *.parquet
- *.pkl, *.joblib, *.pt, *.pth, *.ckpt

Required files to create or update:
- pipelines/full_chain_smoke.py
- scripts/run_full_chain_smoke.py
- tests/test_full_chain_smoke.py
- docs/reports/p7_single_day_full_chain_smoke_report.md

Implementation requirements:

1. Use synthetic tiny data by default.
2. Use tmp_path in tests for any file/ledger writes.
3. The smoke runner must return a structured summary dict with row counts and status labels.
4. Every stage must include a label: DRY_RUN, STRUCTURAL_ONLY, DATA_MISSING, RULE_FALLBACK, or REAL only if artifact path is verified.
5. If any REAL path is requested, require explicit artifact path and verify it exists before using it.
6. If no artifact exists, fall back to dry-run/no-op paths and record reason_codes.
7. Validate outputs after each major stage.
8. Assert final output has 24 rows for one target day unless the test explicitly uses fewer rows.
9. Assert no y_true is required for production smoke.
10. Assert no forbidden output files are committed.

Suggested API:

run_full_chain_smoke(
    target_day: str,
    ledger_dir: Optional[str] = None,
    allow_dry_run: bool = True,
    use_realtime: bool = False,
    classifier_rule_fallback: bool = True,
    cfg05_artifact_path: Optional[str] = None,
    rt_assist_pack_path: Optional[str] = None,
    residual_pack_path: Optional[str] = None,
    classifier_model_dir: Optional[str] = None,
    production: bool = True,
) -> dict

Summary dict should include:
- target_day
- overall_status
- mode_label
- prediction_rows
- corrected_rows
- fusion_rows
- weight_rows
- final_rows
- ledger_dir_used
- stage_labels
- validators_passed
- reason_codes
- forbidden_files_check

CLI:

scripts/run_full_chain_smoke.py

Suggested args:
- --target-day YYYY-MM-DD
- --ledger-dir <path>
- --allow-dry-run
- --use-realtime
- --classifier-rule-fallback / --no-classifier-rule-fallback
- --cfg05-artifact-path <path>
- --rt-assist-pack-path <path>
- --residual-pack-path <path>
- --classifier-model-dir <path>
- --production
- --verbose

Tests:

Create tests/test_full_chain_smoke.py with synthetic tiny tests:
1. full chain smoke returns overall_status PASS.
2. default mode label is DRY_RUN / STRUCTURAL_ONLY with DATA_MISSING and RULE_FALLBACK paths.
3. prediction validator passes.
4. residual validator passes and no-op holds when no residual artifact exists.
5. fusion validator passes and uses y_pred_corrected.
6. ledgers are written only to tmp_path when ledger_dir is provided.
7. weight ledger rows are extracted and weights sum to 1.
8. final output validator passes.
9. final output has 24 rows for single-day dry-run smoke.
10. no REAL label appears unless artifact path is verified.
11. forbidden files are not created in repo paths.
12. CLI dry-run exits 0 and writes only to tmp_path/out path if supplied.

Run all previous tests plus P7 tests.

Report:

Write docs/reports/p7_single_day_full_chain_smoke_report.md

Report sections:
# P7 Single-Day Full-Chain Structural Smoke Report

## 1. Executive status
## 2. Smoke mode and labels
## 3. Pipeline stages exercised
## 4. Row-count summary
## 5. Validator results
## 6. Ledger smoke results
## 7. Fallback paths used
## 8. Forbidden files check
## 9. Known limitations
## 10. P8 recommendation

Hard wording rules:
- Do not claim cfg05 REAL unless artifact is path-verified.
- Do not claim realtime deep model improvement.
- Do not claim P5M residual improvement when DATA_MISSING no-op is used.
- Do not claim BGEW production learner.
- Do not claim ExtremPriceClf ML inference unless real inference path is implemented and artifact is path-verified.

Final response format:
P7 Single-Day Full-Chain Smoke Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Smoke mode label:
6. Pipeline stages passed:
7. Row counts:
8. Validator status:
9. Ledger status:
10. Fallbacks used:
11. Forbidden files check:
12. Known limitations:
13. Commit:
14. Final status:
