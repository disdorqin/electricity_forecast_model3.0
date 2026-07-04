# P3.5 Component Hardening Prompt

You are executing P3.5 for electricity_forecast_model3.0.

Do not start fusion yet. P3 completed the residual correction interface, but source reports show important component-level gaps. This phase hardens the prediction-model and residual-module building blocks before learner/fusion.

Goal: make the base prediction outputs and residual correction inputs reliable enough for P4 fusion.

Read first:
- docs/reports/p1_schema_registry_adapter_report.md
- docs/reports/p2_feature_pipeline_prediction_runner_report.md
- docs/reports/p3_residual_correction_layer_report.md
- data/schema.py
- scripts/validate_prediction_output.py
- scripts/validate_residual_output.py
- pipelines/residual_correction.py
- models/adapters/p5m_residual_plugin.py

Scope:
1. Audit and harden corrected prediction unique keys.
2. Replace naive risk_df merge-by-position with key-based merge on task/model_name/business_day/hour_business, and include target_day/ds where available.
3. Add explicit component readiness gates for day-ahead model zoo, realtime assist, SGDFNet, and P5M residual plugin.
4. Add a small end-to-end component smoke using synthetic data: prediction runner -> residual correction -> residual validator.
5. Document which pieces are REAL, DRY_RUN, STUB, or DATA_MISSING.

Required files to create or update:
- data/schema.py
- pipelines/residual_correction.py
- scripts/component_readiness_check.py
- tests/test_component_readiness_check.py
- tests/test_residual_key_merge_contract.py
- tests/test_prediction_to_residual_smoke.py
- docs/reports/p3_5_component_hardening_report.md

Key hardening requirements:
- Corrected unique key must be ledger-safe. If you keep task/model_name/business_day/hour_business, justify why. Prefer including target_day and ds when present.
- risk_df merge must not align by row position. It must merge by canonical keys.
- If risk_df contains unmatched rows, log/report unmatched count.
- If predictions contain rows without matching risk, keep no-op correction and reason code RISK_ROW_MISSING_NO_OP.
- Production mode must not require y_true.
- No performance metrics may be claimed.

Component readiness states:
- READY_REAL: real callable component with required artifacts and non-dry output.
- READY_DRY_RUN: callable only in dry-run/synthetic mode.
- READY_STUB: interface exists but real artifacts/weights missing.
- DATA_MISSING: needs data or canonical pack.
- NOT_READY: contract or import failures.

Expected current classifications unless code proves otherwise:
- day-ahead cfg05/model zoo: READY_DRY_RUN or READY_STUB unless real model artifact path is verified.
- realtime DA-safe assist: READY_REAL or READY_DRY_RUN depending artifact availability; default DA_ONLY is acceptable.
- SGDFNet: READY_STUB unless weights are available.
- P5M residual: DATA_MISSING or READY_STUB unless canonical pack/risk model is available.

Tests must use synthetic tiny DataFrames only.

Run all previous P1/P2/P3 tests plus new P3.5 tests.

Forbidden:
- Do not commit data, outputs, reports/local, model weights, or prediction artifacts.
- Do not fabricate metrics.
- Do not start fusion engine.
- Do not migrate 2.5 ledger chain yet.

Final response format:
P3.5 Component Hardening Summary
1. Files created:
2. Files updated:
3. Tests added:
4. Tests run:
5. Key schema decision:
6. Risk merge status:
7. Component readiness statuses:
8. End-to-end synthetic smoke:
9. Known limitations:
10. Forbidden files check:
11. Commit:
12. Final status:
