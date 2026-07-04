# P3 Residual Correction Layer Prompt

You are executing P3 for electricity_forecast_model3.0.

P3 is the residual correction layer. Do not build fusion engine yet. Do not migrate the full 2.5 ledger chain. The architecture order is prediction output -> residual correction -> corrected prediction output -> later fusion.

Inputs:
- P1 completed schema, registry, adapter contracts with 96 tests.
- P2 completed loaders, feature pipelines, prediction runners, and validator with 171 tests total.
- P5M source review says residual_stack, extreme/negative_price, and plugin modules exist, but canonical pack is DATA-MISSING.

P3 scope:
1. Build a standard residual correction schema.
2. Build a residual correction runner that consumes standard prediction output.
3. Wire the existing P5M adapter no-op/DATA-MISSING behavior.
4. Prepare integration points for P5M residual_stack and negative/low-valley residual.
5. Add tests using synthetic tiny data only.
6. Do not claim performance improvement unless real canonical pack exists.

Required files to create or update:
- data/schema.py
- pipelines/residual_correction.py
- scripts/run_residual_correction.py
- scripts/validate_residual_output.py
- tests/test_residual_correction_schema.py
- tests/test_residual_correction_runner.py
- tests/test_residual_output_validator.py
- docs/reports/p3_residual_correction_layer_report.md

Residual output should preserve standard prediction keys and include:
- y_pred_raw
- y_pred_corrected
- residual_delta
- correction_applied
- correction_module
- risk_source
- reason_codes
- correction_version

If no risk data or canonical pack is available, behavior must be no-op:
- y_pred_corrected equals y_pred_raw
- residual_delta equals 0
- correction_applied is false
- risk_source is DATA_MISSING or NONE
- reason_codes includes DATA_MISSING_NO_OP

Support both day-ahead and realtime tasks. Do not require y_true for production correction. Eval-only y_true may be allowed only in validation/eval mode.

Validation checks:
- Required columns present
- No duplicate keys
- y_pred_corrected has no NaN
- hour_business in 1..24
- correction_applied boolean
- residual_delta equals y_pred_corrected - y_pred_raw
- production output does not require y_true

Tests:
Use synthetic tiny DataFrames. Validate no-op behavior, schema, duplicate detection, NaN detection, task support, production no-y_true behavior, and CLI dry-run/synthetic behavior.

Forbidden:
- Do not commit data files.
- Do not commit outputs or reports/local.
- Do not commit csv/xlsx/pkl/joblib/pt/pth/ckpt/parquet artifacts.
- Do not fabricate metrics.
- Do not run large experiments.
- Do not migrate fusion engine.

Run all P1 + P2 tests plus new P3 tests.

Final response format:
P3 Residual Correction Execution Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Residual schema status:
6. Residual runner status:
7. Validator status:
8. DATA-MISSING behavior:
9. Forbidden files check:
10. Commit:
11. Final status:
