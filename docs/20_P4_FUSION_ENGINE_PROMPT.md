# P4 Fusion Engine Prompt

You are executing P4 for electricity_forecast_model3.0.

P4 is Fusion Core + Weight Learner Skeleton. It must consume corrected prediction output from P3/P3.5. Do not migrate the full 2.5 ledger chain. Do not integrate the negative classifier. Do not claim production performance.

Current component readiness from P3.5:
- cfg05/model zoo: READY_DRY_RUN
- realtime DA-safe assist: READY_DRY_RUN
- SGDFNet: READY_STUB
- P5M residual: DATA_MISSING

P4 goal:
1. Define fusion input and output schema.
2. Build a fusion engine that consumes corrected prediction outputs.
3. Implement safe weight strategies: equal, prior, and optional historical actual-based BGEW skeleton.
4. Add readiness-aware model eligibility: exclude READY_STUB and NOT_READY by default; allow dry-run only with explicit flag.
5. Add validators and synthetic tests.
6. Produce a P4 report.

Required files to create or update:
- data/schema.py
- fusion/__init__.py
- fusion/weights.py
- fusion/learners/__init__.py
- fusion/learners/bgew.py
- fusion/engine.py
- scripts/run_fusion_engine.py
- scripts/validate_fusion_output.py
- tests/test_fusion_schema.py
- tests/test_fusion_weights.py
- tests/test_fusion_engine.py
- tests/test_fusion_readiness_gates.py
- docs/reports/p4_fusion_engine_report.md

Fusion input:
Use CORRECTED_PREDICTION_COLUMNS from data/schema.py. The engine must use y_pred_corrected as model prediction. It must not use y_pred_raw except for diagnostics.

Fusion output schema should include:
- task
- target_day
- business_day
- ds
- hour_business
- period
- fused_price
- weights_json
- included_models
- excluded_models
- fusion_method
- learner_version
- readiness_mode
- reason_codes

Weight strategies:
1. equal_weight: equal weights over included eligible models.
2. prior_weight: optional registry prior weights, normalized.
3. bgew_skeleton: if actuals_df exists, learn weights from past rolling window only. No future target-day actuals. If actuals_df is missing, fall back to equal_weight with reason code ACTUAL_LEDGER_MISSING_EQUAL_WEIGHT.

Eligibility rules:
- READY_REAL: include by default.
- READY_DRY_RUN: include only if allow_dry_run=True.
- READY_STUB: exclude by default.
- DATA_MISSING: exclude from model fusion; residual DATA_MISSING is already represented upstream.
- NOT_READY: exclude always.

For current repo state, synthetic dry-run fusion is allowed only when allow_dry_run=True, because cfg05 and RT assist are READY_DRY_RUN.

Fusion grouping:
Group by task, target_day, business_day, ds, hour_business, period.
Within each group, combine y_pred_corrected from included model_name values.

Validation checks:
- Required columns present.
- fused_price has no NaN.
- hour_business in 1..24.
- no duplicate output key.
- weights_json parses as JSON.
- included_models non-empty unless allow_empty=True.
- weights sum to 1 within tolerance for included models.
- production mode must not require y_true.

Tests must use synthetic tiny DataFrames only.

Run all P1/P2/P3/P3.5 tests plus new P4 tests.

Forbidden:
- Do not commit data, outputs, reports/local, model weights, prediction artifacts, or CSVs.
- Do not fabricate metrics.
- Do not migrate ledger chain.
- Do not integrate negative classifier.

Final response format:
P4 Fusion Engine Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Fusion schema status:
6. Weight learner status:
7. Readiness gate behavior:
8. Synthetic fusion smoke:
9. Known limitations:
10. Forbidden files check:
11. Commit:
12. Final status:
