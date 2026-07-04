# P6.5 Branch Module Alignment Review

**Phase:** P6.5 — Branch Module Alignment Review  
**Date:** 2026-07-04  
**Scope:** Review only. No new feature development, no large experiment, no data/artifact commits.  
**Repository:** `disdorqin/electricity_forecast_model3.0`  
**Base branch:** `master`  
**Review branch:** `agent/p6-5-branch-module-alignment-review`

---

## 1. Executive status

P6.5 reviewed whether the current 3.0 modules correctly align with their source branches before P7 single-day full-chain smoke.

Overall result:

```text
P7 STRUCTURAL SMOKE: GO
P7 REAL PERFORMANCE / PRODUCTION: NO-GO until real artifacts and real input data are available.
```

3.0 has completed the following structural phases:

- P1 schema / registry / adapter
- P2 feature pipeline / prediction runner
- P3 residual correction
- P3.5 component hardening
- P4 fusion engine
- P5 ledger chain
- P6 negative classifier / final output

P6.5 confirms that the branch-to-module mapping is mostly correct and that the system is honest about readiness states. The main remaining requirement for P7 is to run a **single-day full-chain smoke** in structural/dry-run mode, not a real production evaluation.

Current module readiness labels:

| Area | P6.5 label | Rationale |
|---|---:|---|
| Day-ahead branch | `DRY_RUN` | cfg05 champion metadata and adapter exist, but no verified model artifact in 3.0. |
| Realtime branch | `DRY_RUN` | DA_ONLY path works (`rt_pred = da_anchor`), safe correction requires artifact. |
| Residual branch | `DATA_MISSING` + `STRUCTURAL_ONLY` | no-op/default behavior is correct; real P5M risk/canonical data missing. |
| Ledger/fusion branch | `STRUCTURAL_ONLY` | schemas, ledgers, fusion, BGEW skeleton exist; no real production learner/artifact chain. |
| Negative classifier | `RULE_FALLBACK` + `STUB` | rule fallback is real rule behavior; ExtremPriceClf ML path remains stub unless artifact exists. |

---

## 2. Source-to-3.0 module alignment matrix

| Source branch / repo | Source role | 3.0 modules reviewed | Alignment | P6.5 label |
|---|---|---|---|---|
| `disdorqin/epf-sota-experiment` | Day-ahead model zoo and cfg05 champion | `src/registry/dayahead_models.py`, `data/features/dayahead_features.py`, `scripts/run_dayahead_model_zoo.py`, `models/adapters/cfg05_dayahead_lgbm.py`, `tests/test_dayahead_model_zoo_contract.py` | Correct as registry + dry-run/adapter shell. Real cfg05 requires artifact. | `DRY_RUN` |
| `disdorqin/electricity_forecast_deep_sgdf_delta` | Realtime DA-safe assist / deep sidecar | `src/registry/realtime_models.py`, `data/features/realtime_features.py`, `models/adapters/realtime_da_safe_assist.py`, `scripts/run_realtime_assist.py`, `tests/test_rt_assist_contract.py` | Correct DA_ONLY fallback. No claim that deep model beats DA. | `DRY_RUN` |
| `disdorqin/electricity_forecast_model2.0_exp` | P5M residual / negative-low-valley residual stack | `models/adapters/p5m_residual_plugin.py`, `pipelines/residual_correction.py`, `scripts/run_residual_correction.py`, `scripts/validate_residual_output.py`, `tests/test_p5m_residual_contract.py`, `tests/test_residual_key_merge_contract.py` | Correct no-op / DATA_MISSING shell. Risk merge hardened to key-based. Real unified/high-spike not claimed complete in 3.0. | `DATA_MISSING` + `STRUCTURAL_ONLY` |
| `disdorqin/electricity_forecast_model2.5` | Ledger chain / fusion chain | `fusion/`, `ledgers/`, `pipelines/ledger_backfill.py`, `pipelines/ledger_fusion.py`, `scripts/run_fusion_engine.py`, `scripts/run_ledger_fusion.py` | Correct structure. Fusion consumes corrected predictions; readiness gate excludes non-ready models. BGEW remains skeleton. | `STRUCTURAL_ONLY` |
| `disdorqin/electricity_forecast_model2.5` / `ExtremPriceClf` | Negative-price classifier | `extreme/negative_classifier.py`, `pipelines/classifier_pipeline.py`, `scripts/run_negative_classifier.py`, `scripts/validate_final_output.py`, `tests/test_negative_classifier_adapter.py` | Correct no-artifact fallback and rule fallback. ML classifier remains stub unless artifact exists. | `RULE_FALLBACK` + `STUB` |

---

## 3. Day-ahead alignment

### 3.1 Source mapping

Source repo:

```text
disdorqin/epf-sota-experiment
```

3.0 modules:

```text
src/registry/dayahead_models.py
data/features/dayahead_features.py
scripts/run_dayahead_model_zoo.py
models/adapters/cfg05_dayahead_lgbm.py
tests/test_dayahead_model_zoo_contract.py
```

### 3.2 Findings

1. **cfg05 remains the champion.**

   `src/registry/dayahead_models.py` defines:

   ```python
   CHAMPION_MODEL_ID = "cfg05"
   CHAMPION_SMAPE_FLOOR50 = 11.4838
   ```

   This aligns with the source review: cfg05 is the valid day-ahead champion from `epf-sota-experiment`.

2. **DEFAULT_FUSION_POOL is correct and ordered.**

   Current default pool:

   ```text
   cfg05                  11.48
   best_two_average       11.85
   stage3_business_fixed  11.86
   catboost_spike_residual 12.47
   catboost_sota          12.58
   ```

   This matches the source-review valid pool.

3. **Invalid model blacklist is correct.**

   `INVALID_MODELS` includes the required banned models:

   ```text
   lgbm_spike_residual_1127  -> target leakage
   stage3_old_1164           -> natural-day mapping error
   lightgbm_90d_orig_1197    -> missing hour 24 / invalid output shape
   ```

   `get_model_config()` raises `ValueError` if an invalid model is requested.

4. **Feature pipeline has leakage guard.**

   `data/features/dayahead_features.py` has a deny list:

   ```python
   DENY_LIST = {"y_true", "residual", "error", "abs_error"}
   ```

   `build_dayahead_features()` and `validate_dayahead_feature_frame()` reject denied columns. This aligns with the no-leakage requirement.

5. **Runner does not fake real artifact existence.**

   `scripts/run_dayahead_model_zoo.py` supports `--dry-run` and real adapter execution. Real execution is only wired for cfg05; other zoo members raise `NotImplementedError` unless `--allow-missing-model-artifacts` is used. This is correct.

6. **cfg05 adapter requires real model_dir.**

   `models/adapters/cfg05_dayahead_lgbm.py` searches for:

   ```text
   cfg05_model.txt
   model.txt
   lightgbm_cfg05_dayahead.txt
   ```

   If none is present, it raises `FileNotFoundError` / `RuntimeError`. This is correct and does not pretend artifact existence.

### 3.3 Status

```text
Day-ahead status: DRY_RUN
```

Reason:

- registry and adapter are aligned;
- cfg05 champion is correctly frozen;
- invalid models are blocked;
- feature pipeline has leakage guard;
- but no real cfg05 artifact path was verified in 3.0 during P6.5.

---

## 4. Realtime alignment

### 4.1 Source mapping

Source repo:

```text
disdorqin/electricity_forecast_deep_sgdf_delta
```

3.0 modules:

```text
src/registry/realtime_models.py
data/features/realtime_features.py
models/adapters/realtime_da_safe_assist.py
scripts/run_realtime_assist.py
tests/test_rt_assist_contract.py
```

### 4.2 Findings

1. **DA-safe assist default is correct.**

   Registry defines:

   ```text
   default_prediction = "rt_pred = da_anchor"
   positioning = "sidecar_assist"
   ```

   Adapter implementation also sets:

   ```python
   safe_correction = 0
   rt_pred = da_anchor + safe_correction
   ```

   Therefore default behavior is DA_ONLY.

2. **No claim that deep model beats DA.**

   Realtime registry explicitly lists risks:

   ```text
   Does not guarantee beat DA anchor — DA-only is strongest baseline
   Sensitive to missing forecast-side features
   Small correction benefit insufficient for default enablement
   ```

   This is aligned with the source branch result: deep model is not promoted as a champion.

3. **DA_ONLY fallback is preserved.**

   With no residual model loaded and `enable_safe_correction=False`, the adapter returns `y_pred = da_anchor`.

4. **Safe correction only activates with artifact.**

   `load_model_pack()` checks for `manifest.json` and `residual_model.pkl`. If no residual model is present, `_residual_model` remains `None`, so safe correction does not run.

5. **previous_7d naming issue is not active in current 3.0 DA_ONLY path.**

   Source review noted a `previous_7d_same_hour_mean` / `previous_7d_rolling_hourly_mean` naming mismatch in the source repo. 3.0's current realtime feature pipeline only normalises `ds`, `da_anchor`, and optional `rt_actual`. It does not depend on that deep feature list in the DA_ONLY path. Keep this as a future integration warning, not a P7 blocker.

### 4.3 Status

```text
Realtime status: DRY_RUN
```

Reason:

- deterministic DA_ONLY behavior is implemented;
- no ML artifact is required for fallback;
- no deep model performance claim is made;
- safe correction remains artifact-dependent.

---

## 5. Residual alignment

### 5.1 Source mapping

Source repo:

```text
disdorqin/electricity_forecast_model2.0_exp
```

3.0 modules:

```text
models/adapters/p5m_residual_plugin.py
pipelines/residual_correction.py
scripts/run_residual_correction.py
scripts/validate_residual_output.py
tests/test_p5m_residual_contract.py
tests/test_residual_key_merge_contract.py
```

### 5.2 Findings

1. **P5M no-op / DATA_MISSING behavior is correct.**

   `pipelines/residual_correction.py` defaults to:

   ```text
   correction_module = p5m_residual_noop
   correction_version = 0.0.0
   risk_source = DATA_MISSING
   reason_codes = DATA_MISSING_NO_OP
   correction_applied = False
   ```

   This is the right behavior when no risk data or canonical pack is available.

2. **risk_df merge is key-based, not positional.**

   P3.5 replaced positional `.values` merging with `_merge_risk_data()`, resolving keys in this order:

   ```text
   full:     task + model_name + target_day + business_day + ds + hour_business
   partial:  task + model_name + target_day + business_day + hour_business
   partial:  task + target_day + business_day + hour_business
   degraded: business_day + hour_business
   none:     skip merge / no-op
   ```

   This fixes the previous order-dependence risk.

3. **Corrected schema uses explicit raw/corrected/delta fields.**

   Corrected output includes:

   ```text
   y_pred_raw
   y_pred_corrected
   residual_delta
   correction_applied
   correction_module
   risk_source
   reason_codes
   correction_version
   ```

   This is aligned with P5M handoff needs.

4. **High-spike / unified stack is not falsely marked complete in 3.0.**

   The 3.0 P5M adapter says high-spike correction is `DATA-MISSING` until real `high_spike_prob` exists. 3.0 currently does not copy the full 2.0 `residual_stack/` implementation; it wraps no-op / negative-risk shape only.

5. **No residual improvement is claimed without artifact.**

   If no risk data/canonical pack exists, the output is no-op. If the adapter runs and does not change values, pipeline marks `ADAPTER_NO_EFFECT`, not improvement.

### 5.3 Alignment caveat

`models/adapters/p5m_residual_plugin.py` has placeholder text saying correction would use `extreme.negative_price` / `residual_stack.orchestrator`, but the current 3.0 implementation does not actually import the full P5M residual stack. This is acceptable for P6.5 only because it is labeled DATA_MISSING / structural. Do not claim real P5M correction in P7 unless actual risk data and adapter implementation prove changed values.

### 5.4 Status

```text
Residual status: DATA_MISSING + STRUCTURAL_ONLY
```

Reason:

- schema, validator, no-op and key-based risk merge are correct;
- no real P5M artifact or canonical pack is verified;
- high-spike/unified official path remains data-missing.

---

## 6. Ledger/fusion alignment

### 6.1 Source mapping

Source repo:

```text
disdorqin/electricity_forecast_model2.5
```

3.0 modules:

```text
fusion/
ledgers/
pipelines/ledger_backfill.py
pipelines/ledger_fusion.py
scripts/run_fusion_engine.py
scripts/run_ledger_fusion.py
```

### 6.2 Findings

1. **Fusion consumes corrected predictions only.**

   `fusion/engine.py` requires `FUSION_REQUIRED_INPUT_COLUMNS`, then computes:

   ```python
   fused_price = sum(weight * y_pred_corrected)
   ```

   Raw `y_pred` is not used by the fusion engine.

2. **Readiness gate excludes non-production states.**

   `_apply_readiness_gate()` includes:

   ```text
   READY_REAL     -> include
   READY_DRY_RUN  -> include only if allow_dry_run=True
   READY_STUB     -> exclude
   DATA_MISSING   -> exclude
   NOT_READY      -> exclude
   ```

   This correctly prevents stub/data-missing modules from entering default fusion.

3. **Actual ledger filter is no-leakage.**

   `filter_actuals_for_training()` returns only:

   ```text
   business_day < target_day
   ```

   It also applies a rolling window. This aligns with 2.5 chain requirements.

4. **Ledger keys are reasonable.**

   Current key structure:

   ```text
   Prediction ledger: [task, model_name, target_day, business_day, hour_business]
   Corrected ledger: [task, model_name, target_day, business_day, hour_business]
   Actual ledger:    [task, target_day, business_day, hour_business]
   Fusion ledger:    [task, target_day, business_day, hour_business]
   Weight ledger:    [task, target_day, business_day, hour_business, model_name, fusion_method]
   ```

   This is consistent with cross-target-day safety introduced in P3.5.

5. **Weight ledger expands from weights_json.**

   `ledgers/weight_ledger.py` parses `weights_json` and emits one row per `(fusion row, model_name)` with `weight` and `weight_source`.

6. **BGEW is correctly labeled as skeleton.**

   `fusion/weights.py` implements `bgew_skeleton` as rolling inverse-MAE weighting with fallbacks. It is not a trained production learner.

### 6.3 False-claim wording caveat

`p4_fusion_engine_report.md` says current auto-readiness detects `cfg05 -> READY_DRY_RUN (adapter importable, artifact exists)`. In code, cfg05 only becomes `READY_REAL` if actual model files exist under `models/cfg05/model.txt` or `.pkl`; otherwise it is `READY_DRY_RUN`. The phrase “artifact exists” should be read as “adapter exists” unless the file path is actually present. Do not repeat the artifact claim in P7 unless the path is verified.

### 6.4 Status

```text
Ledger/fusion status: STRUCTURAL_ONLY
```

Reason:

- ledger schemas, append/dedup, no-leakage filter, fusion and weight extraction are structurally complete;
- default readiness gate is safe;
- real production fusion requires real corrected predictions and actual ledgers.

---

## 7. Negative classifier alignment

### 7.1 Source mapping

Source repo:

```text
disdorqin/electricity_forecast_model2.5 / ExtremPriceClf
```

3.0 modules:

```text
extreme/negative_classifier.py
pipelines/classifier_pipeline.py
scripts/run_negative_classifier.py
scripts/validate_final_output.py
tests/test_negative_classifier_adapter.py
```

### 7.2 Findings

1. **No-artifact fallback is correct.**

   If no model_dir or no artifact exists:

   ```text
   final_price = fused_price
   classifier_applied = False
   classifier_module = negative_classifier_noop
   risk_source = CLASSIFIER_ARTIFACT_MISSING
   reason_codes contains NEGATIVE_CLASSIFIER_NO_OP
   ```

2. **Rule fallback is explicitly a rule, not ML.**

   If `rule_fallback=True`, rows with `fused_price < 0` receive:

   ```text
   negative_flag = True
   negative_prob = 1.0
   negative_severity = high
   classifier_module = negative_classifier_rule
   risk_source = RULE_FALLBACK
   reason_codes += RULE_NEGATIVE_PRICE
   ```

   This is honest and does not claim ML inference.

3. **ExtremPriceClf path remains stub.**

   If an artifact pattern is found, `_apply_extremprice_stub()` currently returns no-op final prices with ExtremPriceClf metadata. This is a stub path, not real classifier inference.

4. **Final output schema includes required classifier fields.**

   P6 final output has:

   ```text
   negative_prob
   negative_flag
   negative_severity
   classifier_applied
   classifier_module
   classifier_version
   risk_source
   reason_codes
   model_lineage_json
   ```

### 7.3 False-claim wording caveat

`NegativeClassifierAdapter.load()` logs “production inference ready” when an artifact-like file is found, but `_apply_extremprice_stub()` still performs no-op output. Treat any artifact-found path as `STUB` until true ExtremPriceClf inference is implemented.

### 7.4 Status

```text
Negative classifier status: RULE_FALLBACK + STUB
```

Reason:

- rule fallback is valid and honest;
- no-artifact fallback is correct;
- ExtremPriceClf ML inference is not implemented in 3.0.

---

## 8. Readiness labels

| Label | Definition | Modules currently carrying it |
|---|---|---|
| `REAL` | Real artifact + real inference + real input/output validation | None confirmed in P6.5. |
| `DRY_RUN` | Synthetic/dry-run or deterministic fallback path works, no real artifact | Day-ahead runner/registry; realtime DA_ONLY assist. |
| `STUB` | Interface exists but true inference is not connected | SGDFNet 2.5 adapter; ExtremPriceClf artifact path; parts of P5M residual plugin. |
| `DATA_MISSING` | Requires real canonical pack, risk data, actual ledger, or model artifact | P5M residual/high-spike/unified; B/D residual stack evaluation. |
| `RULE_FALLBACK` | Simple rule behavior; not ML | Negative classifier fused_price < 0 rule. |
| `STRUCTURAL_ONLY` | Schema/validator/ledger/pipeline structure complete, not real model capability | Ledger/fusion chain, residual correction shell, BGEW skeleton. |

P6.5 module status summary:

```text
Day-ahead:          DRY_RUN
Realtime:           DRY_RUN
Residual:           DATA_MISSING + STRUCTURAL_ONLY
Ledger/fusion:      STRUCTURAL_ONLY
Negative classifier: RULE_FALLBACK + STUB
```

---

## 9. Mismatches and blockers

### 9.1 P7 blockers

These block a **real** production smoke but not a structural smoke:

1. **No verified cfg05 model artifact in 3.0**
   - Day-ahead real cfg05 execution needs `cfg05_model.txt`, `model.txt`, or `lightgbm_cfg05_dayahead.txt`.
   - Without it, day-ahead is dry-run only.

2. **No verified RT assist model pack**
   - DA_ONLY fallback works.
   - Safe correction requires manifest + residual model artifact.

3. **P5M residual remains DATA_MISSING**
   - No canonical pack / negative risk model / real high_spike_prob verified.
   - Correct behavior is no-op, not claimed improvement.

4. **No real actual ledger for BGEW production learning**
   - Ledger structure and no-leakage filter exist.
   - Real BGEW needs historical actual ledger input.

5. **ExtremPriceClf is stub unless production inference is implemented**
   - Rule fallback is acceptable for structural P7.
   - Do not claim ML classifier performance.

### 9.2 Non-blockers

1. **Day-ahead cfg05 feature count 42 vs source review 44**
   - P2 documents this as a source-review overcount.
   - Current registry/adapter feature list has 42 columns and tests use that.

2. **Realtime `previous_7d_same_hour_mean` naming mismatch from source**
   - Not active in current 3.0 DA_ONLY path.
   - Keep as future deep model integration caution.

3. **P5M full 2.0 residual stack not copied into 3.0**
   - Acceptable for P6.5 because 3.0 currently exposes a structural adapter / no-op shell.
   - Full residual stack should be integrated only when P7+ has real risk data.

4. **BGEW is skeleton**
   - Acceptable as P4/P5 structural learner.
   - Not a production-trained learner.

---

## 10. False-claim risk check

P6.5 found the following phrases/areas where the next agent must be careful:

1. **Do not say cfg05 is REAL in 3.0 unless model file path exists.**
   - cfg05 is champion by source metrics, but 3.0 currently has dry-run/adapter shell unless artifact is present.

2. **Do not say all DEFAULT_FUSION_POOL members have real adapters.**
   - Only cfg05 has a real adapter shell.
   - Non-cfg05 zoo members are registry entries/dry-run only unless adapters are added later.

3. **Do not say realtime deep model beat DA.**
   - Current 3.0 default is DA_ONLY: `rt_pred = da_anchor`.

4. **Do not say RT safe correction is active without a residual model artifact.**
   - It only runs when `_residual_model` is loaded.

5. **Do not say P5M residual improves predictions in 3.0 without real risk data and changed `y_pred_corrected`.**
   - No-op / ADAPTER_NO_EFFECT must be reported honestly.

6. **Do not use synthetic high_spike flags as official high_spike probabilities.**
   - Source P5M policy requires synthetic flag to be dry-run only.

7. **Do not say BGEW is a production learner.**
   - Current implementation is `bgew_skeleton` with inverse-MAE weighting and fallbacks.

8. **Do not say ExtremPriceClf ML inference is active.**
   - Current 3.0 code has no-op/rule/stub paths only.

9. **Do not claim local artifacts/reports exist unless paths are verified.**
   - Especially `reports/local/*`, ledgers CSVs, model weights, prediction CSVs.

---

## 11. P7 go/no-go recommendation

### Recommendation

```text
P7 single-day full-chain structural smoke: GO
P7 real production smoke / performance claim: NO-GO
```

### P7 should run as

A single-day structural smoke with explicit readiness labels:

```text
Day-ahead: dry-run or cfg05 real only if artifact path exists
Realtime: DA_ONLY fallback
Residual: DATA_MISSING no-op unless risk data path exists
Fusion: allow_dry_run=True only for dry-run inputs; otherwise readiness gate must exclude non-real models
Ledger: tmp_path / local smoke only, no committed ledgers
Negative classifier: rule fallback/no-op only, no ML claim
```

### P7 must not

```text
- commit data, outputs, reports/local, ledgers CSV, model weights, prediction CSV
- claim real cfg05 run without verified artifact
- claim realtime deep model improvement
- claim residual improvement from no-op
- claim BGEW production learner completion
- claim ExtremPriceClf ML inference
```

### Final P6.5 verdict

```text
P6.5 PASS as alignment review.
Proceed to P7 structural single-day full-chain smoke.
Keep all status labels explicit in the P7 report.
```
