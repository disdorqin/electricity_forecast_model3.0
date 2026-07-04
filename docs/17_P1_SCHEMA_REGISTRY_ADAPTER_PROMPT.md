# P1 Schema Registry Adapter Prompt

You are executing P1 for electricity_forecast_model3.0.

Do not migrate the full 2.5 chain yet. P1 goal is to create the stable interface layer that later migration will depend on.

Inputs:
- Source review report says day-ahead zoo is NOT_READY because registry, zoo runner, and contract tests are missing.
- Realtime DA-safe assist is READY_FOR_CHAIN_HANDOFF.
- P5M residual stack is READY but canonical pack is DATA-MISSING.
- 2.5 chain is READY but needs adapters for 3.0.

P1 scope:
1. Create unified schema definitions.
2. Create day-ahead model registry.
3. Create realtime model registry.
4. Create adapter contracts.
5. Create contract tests with synthetic tiny data only.
6. Do not copy large outputs, data, or model weights.

Required files to create or update:
- data/schema.py
- data/business_day.py
- src/registry/dayahead_models.py
- src/registry/realtime_models.py
- models/adapters/base.py
- models/adapters/cfg05_dayahead_lgbm.py
- models/adapters/realtime_da_safe_assist.py
- models/adapters/sgdfnet_2_5.py
- models/adapters/p5m_residual_plugin.py
- tests/test_schema_contract.py
- tests/test_dayahead_model_zoo_contract.py
- tests/test_rt_assist_contract.py
- tests/test_p5m_residual_contract.py
- docs/reports/p1_schema_registry_adapter_report.md

Day-ahead registry must include:
- CHAMPION_MODEL_ID = cfg05
- DEFAULT_FUSION_POOL = cfg05, best_two_average, stage3_business_fixed, catboost_spike_residual, catboost_sota
- INVALID_MODELS = lgbm_spike_residual_1127, stage3_old_1164, lightgbm_90d_orig_1197

Realtime registry must include:
- da_safe_realtime_assist
- sgdfnet_2_5

Adapter contract:
Each prediction adapter must expose a predict method that returns a pandas DataFrame with standardized columns. Use stubs or synthetic outputs where real source execution needs data.

Core prediction columns:
- task
- model_name
- target_day
- business_day
- ds
- hour_business
- period
- y_pred
- source_confidence
- model_version

Evaluation columns may include y_true, but production prediction adapters must not require y_true.

Business-day rule:
Timestamp 00:00 belongs to previous business_day as hour_business 24. Timestamp 01:00 to 23:00 belongs to same date with hour_business 1 to 23.

Tests:
Use only synthetic tiny DataFrames. Validate schema, hour_business 1 to 24, no duplicate keys, no NaN y_pred, invalid model blacklist, realtime fallback rt_pred equals da_anchor, and P5M adapter no-op behavior when risk inputs are missing.

Forbidden:
- Do not commit data files.
- Do not commit reports/local.
- Do not commit csv/xlsx/pkl/joblib/pt/pth/ckpt artifacts.
- Do not use invalid models.
- Do not fabricate metrics.

Final response format:
P1 Execution Summary
1. Files created:
2. Tests added:
3. Tests run:
4. Day-ahead registry status:
5. Realtime registry status:
6. Adapter status:
7. Missing items:
8. Forbidden files check:
9. Commit:
10. Final status:
