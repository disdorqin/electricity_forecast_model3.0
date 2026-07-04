# P8 30-Day Ledger Backfill Structural Smoke Prompt

You are executing P8 for electricity_forecast_model3.0.

P8 is a 30-day ledger backfill structural smoke. It is NOT a production real inference test and must not claim real model performance.

Current state:
- P7 single-day full-chain smoke passed with 494 tests.
- P7 label was DRY_RUN / STRUCTURAL_ONLY with DATA_MISSING and RULE_FALLBACK paths.
- P7 exercised one target day only.

P8 goal:
1. Exercise the P7 full-chain structural smoke across a rolling multi-day window, preferably 30 days.
2. Verify ledger continuity, idempotency, row counts, key uniqueness, and no forbidden file writes.
3. Verify fusion and weight ledgers accumulate correctly across days.
4. Optionally generate synthetic actual ledger to exercise no-leakage training filters and bgew_skeleton fallback/learning behavior, but do not claim real BGEW performance.
5. Produce a P8 report.

Do not do:
- Do not claim production real inference.
- Do not claim real cfg05, P5M, ExtremPriceClf, or realtime deep model performance.
- Do not commit data, outputs, reports/local, ledger CSVs, parquet, model weights, or prediction artifacts.
- Do not train real models.

Required files to create or update:
- pipelines/multi_day_backfill_smoke.py
- scripts/run_multi_day_backfill_smoke.py
- tests/test_multi_day_backfill_smoke.py
- docs/reports/p8_30_day_ledger_backfill_smoke_report.md

Implementation requirements:

1. Use synthetic tiny data by default.
2. Support n_days with default 30.
3. Use tmp_path in tests for all file/ledger writes.
4. Reuse P7 run_full_chain_smoke per day where possible, or reuse its stage functions to avoid duplication.
5. Maintain explicit labels: DRY_RUN, STRUCTURAL_ONLY, DATA_MISSING, RULE_FALLBACK.
6. No REAL label unless artifact path is explicitly provided and verified.
7. Validate ledgers after each day or at the end.
8. Ensure idempotency: running the same 30-day smoke twice with the same run_id or same keys should not duplicate ledger rows.
9. Validate expected row counts.
10. Validate no-leakage actual filtering when synthetic actuals are generated.

Suggested API:

run_multi_day_backfill_smoke(
    start_day: str,
    n_days: int = 30,
    ledger_dir: Optional[str] = None,
    allow_dry_run: bool = True,
    classifier_rule_fallback: bool = True,
    generate_synthetic_actuals: bool = True,
    fusion_method: str = "equal_weight",
    production: bool = True,
) -> dict

Summary dict should include:
- start_day
- end_day
- n_days
- overall_status
- mode_label
- per_day_status
- prediction_rows_total
- corrected_rows_total
- fusion_rows_total
- weight_rows_total
- final_rows_total
- corrected_ledger_rows
- fusion_ledger_rows
- weight_ledger_rows
- actual_ledger_rows
- idempotency_check
- key_uniqueness_check
- validators_passed
- no_leakage_check
- forbidden_files_check
- reason_codes

Expected default row counts for 30 days with 2 dry-run models and 24 hours/day:
- predictions: 30 * 2 * 24 = 1440
- corrected: 1440
- fusion: 30 * 24 = 720
- weights: 30 * 2 * 24 = 1440
- final: 720

If tests use fewer days for speed, still include one 30-day test or one parametrized test that validates the 30-day expected counts.

CLI:

scripts/run_multi_day_backfill_smoke.py

Suggested args:
- --start-day YYYY-MM-DD
- --n-days 30
- --ledger-dir <path>
- --allow-dry-run
- --classifier-rule-fallback / --no-classifier-rule-fallback
- --generate-synthetic-actuals / --no-generate-synthetic-actuals
- --fusion-method equal_weight/prior_weight/bgew_skeleton
- --production
- --verbose

Tests:

Create tests/test_multi_day_backfill_smoke.py with synthetic tests:
1. 3-day smoke returns PASS quickly.
2. 30-day smoke row counts match expected values.
3. corrected/fusion/weight ledgers have no duplicate keys.
4. running the same smoke twice is idempotent or deduped correctly.
5. validators pass for corrected ledger, fusion ledger, weight ledger, and final output.
6. weight rows sum to 1 for every task/target_day/business_day/hour.
7. synthetic actual ledger, if generated, passes validate_actual_ledger.
8. filter_actuals_for_training never returns business_day >= target_day.
9. bgew_skeleton with insufficient actuals falls back safely and records reason codes.
10. no REAL label appears without verified artifacts.
11. forbidden-files check passes.
12. CLI exits 0 with tmp_path ledger_dir.

Report:

Write docs/reports/p8_30_day_ledger_backfill_smoke_report.md

Report sections:
# P8 30-Day Ledger Backfill Structural Smoke Report

## 1. Executive status
## 2. Smoke mode and labels
## 3. Multi-day range and row counts
## 4. Ledger continuity and idempotency
## 5. Validator results
## 6. Weight ledger results
## 7. Actual ledger and no-leakage check
## 8. Fallback paths used
## 9. Forbidden files check
## 10. Known limitations
## 11. P9 recommendation

Hard wording rules:
- This is structural/dry-run multi-day smoke, not production real inference.
- Do not claim cfg05 REAL unless artifact is path-verified.
- Do not claim P5M residual improvement unless y_pred_corrected changes due to real risk data/canonical pack.
- Do not claim BGEW production learner; bgew_skeleton is allowed only as structural/synthetic test.
- Do not claim ExtremPriceClf ML inference unless real inference path and artifact are verified.

Run all previous tests plus P8 tests.

Final response format:
P8 30-Day Ledger Backfill Smoke Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Smoke mode label:
6. Date range:
7. Row counts:
8. Ledger continuity/idempotency:
9. Validator status:
10. Actual/no-leakage status:
11. Fallbacks used:
12. Forbidden files check:
13. Known limitations:
14. Commit:
15. Final status:
