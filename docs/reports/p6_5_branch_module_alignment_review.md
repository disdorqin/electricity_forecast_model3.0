# P6.5 Branch Module Alignment Review

**Date**: 2026-07-04  
**Phase**: P6.5 — Branch Module Alignment Review  
**Scope**: Review-only alignment pass before P7 single-day full-chain smoke.  
**No new model features, no large experiments, no data/outputs/model artifacts committed.**

---

## 1. Executive status

P6.5 reviewed the 3.0 modules completed through P6 against the intended source branches:

- Day-ahead branch: `disdorqin/epf-sota-experiment`
- Realtime branch: `disdorqin/electricity_forecast_deep_sgdf_delta`
- Residual branch: `disdorqin/electricity_forecast_model2.0_exp`
- 2.5 chain / fusion / ledger branch: `disdorqin/electricity_forecast_model2.5`
- Negative classifier branch: `disdorqin/electricity_forecast_model2.5` + `ExtremPriceClf` path

Overall result:

| Area | P6.5 label | P7 readiness | Notes |
|---|---|---|---|
| Day-ahead model zoo / cfg05 | **DRY_RUN** | Go for dry-run / structural smoke | cfg05 is registered champion, but no real LightGBM artifact is tracked in 3.0. |
| Realtime DA-safe assist | **DRY_RUN** | Go for DA-only dry-run smoke | Default `rt_pred = da_anchor`; safe correction requires external artifact. |
| Residual correction / P5M | **DATA_MISSING** | Go for no-op smoke only | Correct no-op behavior; real P5M risk/canonical pack absent. |
| Fusion / ledger chain | **STRUCTURAL_ONLY** | Go for synthetic / ledger smoke | Schemas, gates, ledgers complete; no real actual ledger / real model artifacts. |
| Negative classifier | **RULE_FALLBACK** | Go for no-artifact / rule fallback smoke | ExtremPriceClf is a stub unless real artifact is supplied. |

**P7 recommendation**: **GO for single-day full-chain structural/dry-run smoke**, with explicit labels in all outputs. **NO-GO for claiming production real inference, production learner performance, residual improvement, deep realtime improvement, or real ML negative-classifier performance.**

---

## 2. Source-to-3.0 module alignment matrix

| Source branch | Source responsibility | 3.0 modules reviewed | Alignment judgment | P6.5 label |
|---|---|---|---|---|
| `epf-sota-experiment` | Day-ahead cfg05 champion, model zoo, invalid model blacklist | `src/registry/dayahead_models.py`, `data/features/dayahead_features.py`, `scripts/run_dayahead_model_zoo.py`, `models/adapters/cfg05_dayahead_lgbm.py`, tests | Correct identity and blacklist. Real adapter exists only for cfg05 and requires external LightGBM artifact. Other zoo models are dry-run / registry candidates. | **DRY_RUN** |
| `electricity_forecast_deep_sgdf_delta` | DA-safe realtime assist, DA-only fallback, optional safe correction | `src/registry/realtime_models.py`, `data/features/realtime_features.py`, `models/adapters/realtime_da_safe_assist.py`, `scripts/run_realtime_assist.py`, tests | Correctly preserves DA-only baseline as default. Does not claim deep model beats DA. Safe correction only works with residual model pack. | **DRY_RUN** |
| `electricity_forecast_model2.0_exp` | P5M residual / negative-low-valley plugin, high-spike/unified data-missing paths | `models/adapters/p5m_residual_plugin.py`, `pipelines/residual_correction.py`, residual scripts and tests | P5M is wired as no-op/DATA_MISSING by default. Risk merge hardened to key-based. No real residual improvement claimed. | **DATA_MISSING** |
| `electricity_forecast_model2.5` | Fusion engine, ledger chain, weight learning skeleton | `fusion/`, `ledgers/`, `pipelines/ledger_backfill.py`, `pipelines/ledger_fusion.py`, ledger/fusion scripts | Corrected-input-only fusion, readiness gate, actual ledger no-leakage filter, weight ledger extraction present. BGEW is skeleton only. | **STRUCTURAL_ONLY** |
| `electricity_forecast_model2.5` / `ExtremPriceClf` | Negative classifier / final output | `extreme/negative_classifier.py`, `pipelines/classifier_pipeline.py`, negative classifier scripts/tests | No-artifact fallback and rule fallback are correct. ExtremPriceClf production inference is stub-only unless artifact exists. | **RULE_FALLBACK** |

---

## 3. Day-ahead alignment

### Files reviewed

- `docs/reports/source_review_report.md`
- `docs/reports/p1_schema_registry_adapter_report.md`
- `docs/reports/p2_feature_pipeline_prediction_runner_report.md`
- `src/registry/dayahead_models.py`
- `data/features/dayahead_features.py`
- `scripts/run_dayahead_model_zoo.py`
- `models/adapters/cfg05_dayahead_lgbm.py`

### Alignment findings

1. **cfg05 remains champion**  
   `src/registry/dayahead_models.py` defines `CHAMPION_MODEL_ID = "cfg05"` and `CHAMPION_SMAPE_FLOOR50 = 11.4838`. The default fusion pool is ordered as cfg05, best_two_average, stage3_business_fixed, catboost_spike_residual, catboost_sota.

2. **DEFAULT_FUSION_POOL is aligned with the day-ahead source branch**  
   The pool contains the expected five valid models:
   - `cfg05` — champion, 11.48
   - `best_two_average` — strong candidate, 11.85
   - `stage3_business_fixed` — strong candidate, 11.86
   - `catboost_spike_residual` — diversity/fallback, 12.47
   - `catboost_sota` — baseline/fallback, 12.58

3. **Invalid blacklist is correct**  
   `INVALID_MODELS` blocks:
   - `lgbm_spike_residual_1127` — target leakage
   - `stage3_old_1164` — natural-day mapping error
   - `lightgbm_90d_orig_1197` — 690 rows / missing hour 24

4. **Feature pipeline has leak prevention**  
   `data/features/dayahead_features.py` defines a deny-list (`y_true`, `residual`, `error`, `abs_error`) and raises if denied columns appear in the produced feature frame. It also preserves business-time columns.

5. **Runner does not pretend every model has a real artifact**  
   `scripts/run_dayahead_model_zoo.py` supports dry-run. In non-dry-run mode, only cfg05 has a real adapter path. Other registry models raise `NotImplementedError` unless `--allow-missing-model-artifacts` is supplied.

6. **cfg05 adapter is not REAL in current repository state**  
   `models/adapters/cfg05_dayahead_lgbm.py` requires a LightGBM model artifact (`cfg05_model.txt`, `model.txt`, or `lightgbm_cfg05_dayahead.txt`) supplied through `model_dir`. No such artifact is tracked in 3.0.

### Day-ahead status

**P6.5 label: DRY_RUN**

Rationale: registry, schema, feature interface, dry-run runner, and adapter contract exist. However, no real cfg05 LightGBM artifact is present in the 3.0 repository, and only cfg05 is wired for real adapter execution. This is appropriate for P7 dry-run smoke, not for production real inference claims.

---

## 4. Realtime alignment

### Files reviewed

- `docs/reports/source_review_report.md`
- `docs/reports/p1_schema_registry_adapter_report.md`
- `docs/reports/p2_feature_pipeline_prediction_runner_report.md`
- `src/registry/realtime_models.py`
- `data/features/realtime_features.py`
- `models/adapters/realtime_da_safe_assist.py`
- `scripts/run_realtime_assist.py`
- `tests/test_rt_assist_contract.py`

### Alignment findings

1. **DA-safe assist default is correct**  
   `src/registry/realtime_models.py` records `default_prediction = "rt_pred = da_anchor"`. The adapter docstring and logic position DA-only as the primary stable baseline.

2. **No false claim that deep model beats DA**  
   Registry risks explicitly state that DA-safe assist does not guarantee beating the DA anchor. The source review recorded the realtime branch verdict: deep model did not beat the DA anchor and DA-only remains safest.

3. **DA_ONLY fallback retained**  
   `DASafeRealtimeAssistAdapter` defaults to `enable_safe_correction=False`. Its prediction path initializes `safe_correction = 0`, therefore `rt_pred = da_anchor` unless a residual model is loaded and correction is explicitly enabled.

4. **Safe correction depends on artifact existence**  
   `load_model_pack()` only loads `residual_model.pkl` from a provided model pack directory. If no residual model is loaded, safe correction is not applied.

5. **Feature naming issue is contained but not fully resolved upstream**  
   The source review noted a prior mismatch around `previous_7d_same_hour_mean` vs `previous_7d_rolling_hourly_mean` in the source branch. In 3.0, `data/features/realtime_features.py` is a normalization/validation layer and does not compute those deep features, so the issue does not block the current DA-only P7 smoke.

### Realtime status

**P6.5 label: DRY_RUN**

Rationale: DA-only path is structurally usable with input data containing `da_anchor`, but no RT assist pack is tracked and safe correction cannot be treated as REAL. P7 may smoke DA-only behavior; no claim of deep-model improvement is allowed.

---

## 5. Residual alignment

### Files reviewed

- `docs/reports/source_review_report.md`
- `docs/reports/p3_residual_correction_layer_report.md`
- `docs/reports/p3_5_component_hardening_report.md`
- `models/adapters/p5m_residual_plugin.py`
- `pipelines/residual_correction.py`
- `scripts/run_residual_correction.py`
- `scripts/validate_residual_output.py`
- `tests/test_p5m_residual_contract.py`
- `tests/test_residual_key_merge_contract.py`

### Alignment findings

1. **P5M no-op / DATA_MISSING behavior is correct**  
   P3 defines residual correction as DATA-MISSING no-op by default: `y_pred_corrected == y_pred_raw`, `residual_delta == 0`, `correction_applied = False`. The P5M adapter likewise states that without canonical pack or risk data it returns a pass-through.

2. **risk_df merge was hardened to key-based merge**  
   P3.5 replaced positional risk merge with `_merge_risk_data()` and `_resolve_risk_merge_key()`, supporting full, partial, degraded, or no merge keys. This directly addresses the P3 known limitation.

3. **high_spike / unified are not falsely marked complete**  
   `models/adapters/p5m_residual_plugin.py` explicitly states high_spike correction is DATA-MISSING until real high_spike_prob is available. P3/P3.5 reports do not claim unified/high_spike production readiness.

4. **No residual improvement is fabricated**  
   P3/P3.5 reports describe schema, no-op behavior, adapter stubs, and contract tests; they do not claim real residual performance gains in 3.0.

5. **Corrected output schema is aligned**  
   The corrected schema uses `y_pred_raw`, `y_pred_corrected`, and `residual_delta`, with `correction_applied`, `correction_module`, `risk_source`, and reason codes.

### Residual status

**P6.5 label: DATA_MISSING**

Rationale: interfaces and no-op behavior are implemented and hardened. Real correction requires canonical pack/risk data/model artifacts, which are absent. This is acceptable for P7 no-op smoke but not for correction-effect claims.

---

## 6. Ledger/fusion alignment

### Files reviewed

- `docs/reports/p4_fusion_engine_report.md`
- `docs/reports/p5_ledger_chain_migration_report.md`
- `fusion/engine.py`
- `fusion/weights.py`
- `fusion/learners/bgew.py`
- `ledgers/actual_ledger.py`
- `ledgers/weight_ledger.py`
- `pipelines/ledger_backfill.py`
- `pipelines/ledger_fusion.py`
- `scripts/run_fusion_engine.py`
- `scripts/run_ledger_fusion.py`

### Alignment findings

1. **Fusion consumes corrected predictions only**  
   P4 explicitly defines fusion input as P3/P3.5 corrected output and the engine computes `fused_price` from `y_pred_corrected`, not raw `y_pred`.

2. **Readiness gate excludes unsafe states**  
   Fusion readiness gate includes `READY_REAL` by default and includes `READY_DRY_RUN` only when `allow_dry_run=True`; `READY_STUB`, `DATA_MISSING`, and `NOT_READY` are excluded.

3. **Actual ledger training filter is future-aware**  
   `filter_actuals_for_training()` returns only rows where `business_day < target_day`. This aligns with the 2.5 ledger-learning intent and prevents training on target-day actuals.

4. **Ledger keys are reasonable and explicit**  
   P5 defines prediction, corrected, actual, fusion, and weight ledger schemas with explicit keys. Weight ledger key includes `model_name` and `fusion_method`, which supports per-model per-hour weight tracking.

5. **Weight ledger expands weights_json**  
   `extract_weight_rows()` parses `weights_json` and emits one row per `(fusion_row, model_name)` with copied fusion metadata.

6. **BGEW remains a skeleton**  
   `bgew_skeleton` is inverse-MAE-based and falls back to equal-weight when actuals are missing or insufficient. It is a skeleton learner, not a completed production learner.

### Ledger/fusion status

**P6.5 label: STRUCTURAL_ONLY**

Rationale: fusion and ledgers are structurally complete and test-covered. No real actual ledger, real model artifacts, or production-grade learner state is present. P7 can run synthetic/dry-run ledger smoke; no production learner claims should be made.

---

## 7. Negative classifier alignment

### Files reviewed

- `docs/reports/p6_negative_classifier_integration_report.md`
- `extreme/negative_classifier.py`
- `pipelines/classifier_pipeline.py`
- `scripts/run_negative_classifier.py`
- `scripts/validate_final_output.py`
- `tests/test_negative_classifier_adapter.py`

### Alignment findings

1. **No-artifact fallback is correct**  
   When no ExtremPriceClf artifact exists, the adapter returns no-op final output with `final_price = fused_price`, `classifier_applied = False`, and `risk_source = CLASSIFIER_ARTIFACT_MISSING`.

2. **Rule fallback is clearly a rule, not ML**  
   With `rule_fallback=True`, rows with `fused_price < 0` are flagged by `negative_classifier_rule`, `negative_prob = 1.0`, and `RULE_NEGATIVE_PRICE` reason code. This does not claim ML inference.

3. **ExtremPriceClf path is stub-only**  
   If an artifact is found, the current `_apply_extremprice_stub()` returns no-op-like output with ExtremPrice metadata. Real production inference is reserved for future deployment.

4. **Final output schema is aligned**  
   P6 final output includes `negative_prob`, `negative_flag`, `negative_severity`, `classifier_applied`, `classifier_module`, `classifier_version`, `risk_source`, `reason_codes`, and `model_lineage_json`.

### Negative classifier status

**P6.5 label: RULE_FALLBACK**

Rationale: no real ExtremPriceClf artifact is present in 3.0; rule fallback and no-artifact fallback are valid. The ML classifier remains a stub / data-artifact-dependent path.

---

## 8. Readiness labels

| Module family | Label | Reason |
|---|---|---|
| Day-ahead cfg05/model zoo | **DRY_RUN** | Registry/runner/adapter exist; no real cfg05 model artifact in repo. |
| Day-ahead feature pipeline | **STRUCTURAL_ONLY** | Feature frame and deny-list exist, but not a full real feature computation/training pack. |
| Realtime DA-safe assist | **DRY_RUN** | DA-only path works structurally; safe correction needs model pack. |
| SGDFNet realtime candidate | **STUB** | Registry entry/adapter skeleton only; no weights. |
| P5M residual correction | **DATA_MISSING** | No canonical pack/risk model/high_spike data; no-op expected. |
| Fusion engine | **STRUCTURAL_ONLY** | Corrected-input fusion and readiness gate exist; real model artifacts absent. |
| Ledger chain | **STRUCTURAL_ONLY** | Ledger schemas/stores/pipelines exist; real ledgers not present. |
| Weight learner / BGEW | **STRUCTURAL_ONLY** | Skeleton inverse-MAE learner only; not a production learner. |
| Negative classifier | **RULE_FALLBACK** | No-artifact + rule fallback valid; ExtremPriceClf ML path stub. |
| Final output validator | **STRUCTURAL_ONLY** | Schema validation complete; not a model capability. |

---

## 9. Mismatches and blockers

### Blockers for REAL P7 / production-style smoke

1. **cfg05 model artifact missing**  
   `CFG05DayaheadAdapter` can load real LightGBM weights, but no `cfg05_model.txt` / `model.txt` / `lightgbm_cfg05_dayahead.txt` artifact is tracked.

2. **Non-cfg05 day-ahead models are not real-wired**  
   `run_dayahead_model_zoo.py` real execution is wired only for cfg05. `best_two_average`, `stage3_business_fixed`, `catboost_spike_residual`, and `catboost_sota` are registry/dry-run candidates unless their adapters/artifacts are added.

3. **Realtime safe correction pack missing**  
   DA-only fallback can run, but `residual_model.pkl` / full rt_assist pack is absent.

4. **P5M residual risk data/canonical pack missing**  
   P5M can no-op, but cannot produce real correction without risk data/canonical pack/model artifact.

5. **ExtremPriceClf artifact missing**  
   Negative classifier ML path is stub/no-op unless real artifact is supplied.

6. **Actual ledger missing**  
   Ledger/fusion BGEW can structurally run but real history-based weighting needs actual ledger rows.

### Non-blockers for P7 structural/dry-run smoke

1. Schema, business-time mapping, adapters, runners, validators exist.
2. DATA_MISSING and no-artifact fallbacks are explicit and audited.
3. Fusion readiness gate prevents stub/data-missing models from silently entering real fusion.
4. P5 ledgers can be exercised with synthetic/tmp-path data.
5. Negative classifier can emit no-op/rule-fallback final output without model artifact.

---

## 10. False-claim risk check

| Claim risk | P6.5 verdict | Required wording |
|---|---|---|
| "cfg05 is production REAL in 3.0" | False | Say: cfg05 champion identity is real, but current 3.0 adapter is DRY_RUN unless external LightGBM artifact is supplied. |
| "model zoo can real-run all five models" | False | Say: default pool exists; only cfg05 has a real adapter path, and even cfg05 needs artifact. |
| "Realtime deep model beats DA" | False | Say: realtime default is DA-only (`rt_pred = da_anchor`); deep assist does not claim beating DA. |
| "Safe realtime correction is active" | False unless artifact exists | Say: safe correction requires loaded residual model pack and explicit enablement. |
| "P5M residual improves forecasts" | False in 3.0 current state | Say: residual layer is DATA_MISSING no-op unless risk data/canonical pack exists. |
| "High-spike/unified correction is complete" | False | Say: high_spike/unified remain DATA_MISSING / not production-complete. |
| "Fusion BGEW is production learner" | False | Say: BGEW is a skeleton inverse-MAE learner with fallback. |
| "Fusion uses future actuals" | Not supported by code | Actual ledger filter and BGEW filter use `business_day < target_day`; keep this invariant. |
| "Negative classifier ML is deployed" | False | Say: no-artifact fallback/rule fallback is deployed; ExtremPriceClf inference is stub unless artifact exists. |
| "Rule fallback is ML" | False | Say: rule fallback is explicitly rule-based. |

---

## 11. P7 go/no-go recommendation

### Recommendation

**GO for P7 single-day full-chain smoke only under STRUCTURAL / DRY_RUN conditions.**

Suggested P7 mode:

1. Day-ahead: run model zoo in dry-run or cfg05-only if an external artifact is explicitly provided and path-verified.
2. Residual: run DATA_MISSING no-op and verify `y_pred_corrected == y_pred_raw`, `residual_delta == 0` when no risk artifact is supplied.
3. Fusion: run with `allow_dry_run=True` for dry-run models, or use explicit readiness overrides for a structural smoke. Do not claim REAL if included models are DRY_RUN.
4. Ledger: append predictions/corrections/fusion/weights to tmp-path or local ignored ledger paths only. Do not commit ledger CSVs.
5. Negative classifier: run no-artifact fallback and optional rule fallback. Verify `final_price = fused_price` unless rule affects only negative flags.
6. Validation: run prediction, residual, fusion, final-output validators.

### No-go conditions for P7 REAL claims

Do not label P7 as REAL unless all of the following are path-verified before execution:

- cfg05 LightGBM artifact exists and is loaded.
- Any non-cfg05 model in fusion has a real adapter/artifact or is explicitly excluded.
- Realtime safe correction pack exists if correction is enabled.
- P5M risk/canonical pack exists if correction is claimed.
- Actual ledger exists and contains only `business_day < target_day` data for weight learning.
- ExtremPriceClf artifact exists if ML negative classifier is claimed.

### P7 target label

For current repository state, the correct P7 label should be:

```text
P7_SINGLE_DAY_FULL_CHAIN_SMOKE = DRY_RUN / STRUCTURAL_ONLY with DATA_MISSING and RULE_FALLBACK paths
```

---

## Final P6.5 summary

P6.5 found that module-to-branch responsibility alignment is mostly correct. The main risk is not code structure; it is **over-claiming readiness**. The 3.0 repository currently contains strong schemas, registries, adapters, dry-run runners, ledgers, fusion skeletons, fallbacks, and validators, but it does **not** contain the real model artifacts, actual ledgers, risk data, or ML negative-classifier artifact required for REAL production inference.

The correct next step is P7 single-day smoke with explicit dry-run / structural labels and no fabricated metrics.
