# P6 Negative Classifier Integration Prompt

You are executing P6 for electricity_forecast_model3.0.

P6 is Negative Classifier Integration. Do not build gradient-based learner. Do not add confidence intervals. Do not claim real negative-price performance unless real classifier artifacts and evaluation data exist.

Current state:
- P5 completed durable ledgers: prediction, corrected prediction, actual, fusion, weight.
- P4 fusion outputs point forecasts in fusion ledger.
- Source review found the 2.5 negative classifier path: fusion/classifier_bridge.py and ExtremPriceClf.

P6 goal:
1. Define final output schema after fusion + negative classifier.
2. Build a negative classifier adapter with safe no-artifact fallback.
3. Build classifier pipeline consuming fusion ledger output.
4. Produce final delivery output with negative risk fields and reason codes.
5. Add validators and synthetic tiny tests.
6. Keep real classifier optional until artifacts exist.

Required files to create or update:
- data/schema.py
- extreme/__init__.py
- extreme/negative_classifier.py
- pipelines/classifier_pipeline.py
- scripts/run_negative_classifier.py
- scripts/validate_final_output.py
- tests/test_negative_classifier_schema.py
- tests/test_negative_classifier_adapter.py
- tests/test_classifier_pipeline.py
- tests/test_final_output_validator.py
- docs/reports/p6_negative_classifier_integration_report.md

Input:
P4/P5 fusion output or fusion ledger rows. Use fused_price as the base price.

Output schema should include:
- task
- target_day
- business_day
- ds
- hour_business
- period
- fused_price
- final_price
- negative_prob
- negative_flag
- negative_severity
- classifier_applied
- classifier_module
- classifier_version
- risk_source
- reason_codes
- model_lineage_json

Safe fallback behavior when classifier artifacts are missing:
- final_price equals fused_price
- negative_prob is NaN or 0, but document choice consistently
- negative_flag is false
- classifier_applied is false
- risk_source is DATA_MISSING or CLASSIFIER_ARTIFACT_MISSING
- reason_codes includes NEGATIVE_CLASSIFIER_NO_OP

If a lightweight rule fallback is added, it must be explicit and non-ML:
- fused_price < 0 can set negative_flag true
- reason_codes includes RULE_NEGATIVE_PRICE
- Do not call this real classifier performance.

Validation checks:
- Required columns present
- final_price no NaN
- hour_business in 1..24
- negative_prob in [0,1] when not NaN
- negative_flag boolean
- classifier_applied boolean
- no duplicate key: task, target_day, business_day, hour_business
- production mode must not require y_true
- model_lineage_json parses as JSON

Tests must use synthetic tiny DataFrames only. Include tests for no-artifact fallback, optional rule fallback, validator failures, final output ledger compatibility, and CLI dry-run.

Forbidden:
- Do not commit data, outputs, reports/local, ledger CSVs, model weights, parquet, pickle/joblib, or prediction artifacts.
- Do not fabricate metrics.
- Do not claim production classifier is ready without artifacts.
- Do not implement gradient learner or confidence intervals.

Run all previous tests plus P6 tests.

Final response format:
P6 Negative Classifier Integration Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Final output schema:
6. Classifier adapter status:
7. Fallback behavior:
8. Pipeline status:
9. Known limitations:
10. Forbidden files check:
11. Commit:
12. Final status:
