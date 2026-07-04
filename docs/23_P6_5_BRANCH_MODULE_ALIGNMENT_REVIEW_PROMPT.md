# P6.5 Branch Module Alignment Review Prompt

You are executing P6.5 for electricity_forecast_model3.0.

P6.5 is not a new implementation phase. It is a branch/module alignment review before P7 single-day full chain smoke.

Goal: verify that every module added in 3.0 corresponds correctly to its source branch or source role, and clearly label what is REAL, DRY_RUN, STUB, DATA_MISSING, or RULE_FALLBACK.

Read first:
- docs/reports/source_review_report.md
- docs/reports/p1_schema_registry_adapter_report.md
- docs/reports/p2_feature_pipeline_prediction_runner_report.md
- docs/reports/p3_residual_correction_layer_report.md
- docs/reports/p3_5_component_hardening_report.md
- docs/reports/p4_fusion_engine_report.md
- docs/reports/p5_ledger_chain_migration_report.md
- docs/reports/p6_negative_classifier_integration_report.md

Review these source mappings:
1. Day-ahead branch: disdorqin/epf-sota-experiment
   - Expected 3.0 modules: src/registry/dayahead_models.py, data/features/dayahead_features.py, scripts/run_dayahead_model_zoo.py, models/adapters/cfg05_dayahead_lgbm.py
   - Check cfg05, DEFAULT_FUSION_POOL, invalid model blacklist, no-leakage/invalid result handling.

2. Realtime branch: disdorqin/electricity_forecast_deep_sgdf_delta
   - Expected 3.0 modules: src/registry/realtime_models.py, data/features/realtime_features.py, models/adapters/realtime_da_safe_assist.py, scripts/run_realtime_assist.py
   - Check DA_ONLY fallback, rt_pred=da_anchor default, no unsupported safe correction claim, no fake deep model superiority claim.

3. Residual branch: disdorqin/electricity_forecast_model2.0_exp
   - Expected 3.0 modules: models/adapters/p5m_residual_plugin.py, pipelines/residual_correction.py, scripts/run_residual_correction.py, validate_residual_output.py
   - Check DATA_MISSING no-op, risk merge by key, no fake P5M metrics, no high-spike/unified completion claim.

4. Chain/fusion branch: disdorqin/electricity_forecast_model2.5
   - Expected 3.0 modules: fusion/, ledgers/, pipelines/ledger_backfill.py, pipelines/ledger_fusion.py
   - Check ledger keys, actual no-leakage filter, fusion consumes y_pred_corrected, weight ledger persists weights.

5. Negative classifier branch: disdorqin/electricity_forecast_model2.5 ExtremPriceClf path
   - Expected 3.0 modules: extreme/negative_classifier.py, pipelines/classifier_pipeline.py, scripts/run_negative_classifier.py, scripts/validate_final_output.py
   - Check no-artifact fallback, rule fallback as rule only, ExtremPriceClf is stub unless artifact and real inference are present.

Tasks:
1. Build a module alignment matrix.
2. For each module, mark one of: REAL, DRY_RUN, STUB, DATA_MISSING, RULE_FALLBACK, STRUCTURAL_ONLY.
3. Identify mismatches between source branch intent and 3.0 implementation.
4. Identify any missing module required before P7 full-chain smoke.
5. Identify any false claim risk, especially metrics, production readiness, or real artifact availability.
6. Recommend whether P7 may proceed.

Do not implement new business logic unless there is a small documentation or test-only fix. If you find code bugs, report them first and label BLOCKER / NON_BLOCKER.

Write report:
- docs/reports/p6_5_branch_module_alignment_review.md

Report format:
# P6.5 Branch Module Alignment Review

## 1. Executive status
## 2. Source-to-3.0 module alignment matrix
## 3. Day-ahead alignment
## 4. Realtime alignment
## 5. Residual alignment
## 6. Ledger/fusion alignment
## 7. Negative classifier alignment
## 8. Readiness labels
## 9. Mismatches and blockers
## 10. False-claim risk check
## 11. P7 go/no-go recommendation

Final response format:
P6.5 Branch Module Alignment Summary
1. Files reviewed:
2. Report path:
3. Day-ahead status:
4. Realtime status:
5. Residual status:
6. Ledger/fusion status:
7. Negative classifier status:
8. Blockers:
9. Non-blockers:
10. P7 recommendation:
11. Commit:
