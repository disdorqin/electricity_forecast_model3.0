# P5 Ledger Chain Migration Prompt

You are executing P5 for electricity_forecast_model3.0.

P5 is Ledger Chain Migration. Do not build gradient-based learner yet. Do not add confidence intervals yet. Do not integrate the negative classifier yet. Those require a durable ledger foundation first.

Current state:
- P1 completed schema, registry, adapter contracts.
- P2 completed loaders, feature pipelines, prediction runners, and prediction validator.
- P3 completed corrected prediction schema and residual correction layer.
- P3.5 hardened component readiness and key-based risk merge.
- P4 completed fusion core and weight learner skeleton. It consumes y_pred_corrected and outputs fused predictions.

P5 goal:
1. Create durable prediction ledger, corrected ledger, actual ledger, fusion ledger, and weight ledger schemas.
2. Migrate/adapt the 2.5 ledger chain concepts without copying full chain blindly.
3. Implement idempotent append/update with dedup by canonical keys.
4. Implement ledger backfill and actual merge using synthetic tiny data tests.
5. Persist fusion weights and fusion outputs as ledger tables.
6. Add validators and CLI scripts.
7. Produce a P5 report.

Source reference:
- disdorqin/electricity_forecast_model2.5
- Review paths: pipelines/prediction_ledger.py, pipelines/ledger_predict.py, pipelines/ledger_backfill.py, pipelines/ledger_weight.py, pipelines/ledger_fuse.py, pipelines/ledger_full.py, pipelines/ledger_full_range.py, pipelines/delivery_report.py, fusion/learners/daily_ledger_gef.py.

Required files to create or update:
- data/schema.py
- ledgers/__init__.py
- ledgers/store.py
- ledgers/prediction_ledger.py
- ledgers/actual_ledger.py
- ledgers/fusion_ledger.py
- ledgers/weight_ledger.py
- pipelines/ledger_backfill.py
- pipelines/ledger_fusion.py
- scripts/ledger_append_predictions.py
- scripts/ledger_update_actuals.py
- scripts/run_ledger_fusion.py
- scripts/validate_ledgers.py
- tests/test_ledger_schema.py
- tests/test_ledger_store.py
- tests/test_prediction_ledger.py
- tests/test_actual_ledger.py
- tests/test_fusion_weight_ledgers.py
- tests/test_ledger_pipeline_smoke.py
- docs/reports/p5_ledger_chain_migration_report.md

Ledger storage:
Use CSV/parquet paths only through tmp_path in tests. Do not commit any ledger data files. Implement storage APIs that can write to a user-specified path, but tests must use tmp_path.

Required schemas:
Prediction ledger should store standard prediction output from P2.
Corrected prediction ledger should store P3 corrected output.
Actual ledger should store actual y_true/actual_price by task, target_day, business_day, hour_business, ds.
Fusion ledger should store P4 fusion output.
Weight ledger should store per task/target_day/business_day/hour/period/model weights.

Canonical key rules:
- Prediction ledger key: task, model_name, target_day, business_day, hour_business.
- Corrected ledger key: task, model_name, target_day, business_day, hour_business.
- Actual ledger key: task, target_day, business_day, hour_business.
- Fusion ledger key: task, target_day, business_day, hour_business.
- Weight ledger key: task, target_day, business_day, hour_business, model_name, fusion_method.

Idempotency:
Appending same key twice must keep the latest run_id or updated_at. Do not silently duplicate ledger rows.

No leakage:
Weight learning and fusion may use actual ledger only for business_day < target_day. Never use target_day actuals for its own prediction/fusion weight.

CLI expectations:
- ledger_append_predictions.py: append standard or corrected predictions to ledger.
- ledger_update_actuals.py: merge actuals into actual ledger.
- run_ledger_fusion.py: read corrected ledger, optionally read actual ledger, run P4 fusion engine, write fusion ledger and weight ledger.
- validate_ledgers.py: validate all ledger files.

Tests:
Use synthetic tiny DataFrames only. Test dedup, idempotency, actual merge, no leakage, weight persistence, fusion ledger output, and full tiny smoke: predictions -> correction -> corrected ledger -> fusion -> fusion ledger -> weight ledger.

Run all previous tests plus new P5 tests.

Forbidden:
- Do not commit data, outputs, reports/local, ledger CSVs, model weights, parquet, or prediction artifacts.
- Do not fabricate metrics.
- Do not build gradient learner or confidence intervals yet.
- Do not integrate negative classifier yet.

Final response format:
P5 Ledger Chain Migration Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Ledger schemas:
6. Idempotency/dedup status:
7. Actual merge/no-leakage status:
8. Fusion/weight ledger status:
9. End-to-end synthetic ledger smoke:
10. Known limitations:
11. Forbidden files check:
12. Commit:
13. Final status:
