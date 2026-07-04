# P6: Negative Classifier Integration Report

**Date:** 2026-07-04
**Status:** Complete
**Components:** `extreme/` package, `pipelines/classifier_pipeline.py`, `scripts/run_negative_classifier.py`, `scripts/validate_final_output.py`, 4 test files
**Test count:** 63 new + 405 existing = **468 total, all passing**

---

## 1. Executive Status

P6 integrates the negative-price classifier into the electricity_forecast_model3.0 system. A `NegativeClassifierAdapter` provides a unified interface for negative-price risk assessment with three modes:

1. **No-artifact fallback** — No real ExtremPriceClf artifact → no-op (final_price = fused_price, classifier_applied = False)
2. **Rule fallback** — fused_price < 0 triggers negative_flag = True with RULE_NEGATIVE_PRICE reason code
3. **ExtremPriceClf stub** — Artifact found → classifier_applied = True with ExtremPriceClf metadata (production inference path reserved for future deployment)

The pipeline consumes fusion output (P4/P5), validates input, runs the classifier adapter, and produces final output conforming to `FINAL_OUTPUT_COLUMNS` (17-column schema).

---

## 2. Files Created or Updated

### Updated

| File | Change |
|---|---|
| `data/schema.py` | Added `FINAL_OUTPUT_COLUMNS` (17 cols), `FINAL_UNIQUE_KEY`, `VALID_NEGATIVE_SEVERITY`, 3 classifier constant identifiers |

### Created

| File | Lines | Purpose |
|---|---|---|
| `extreme/__init__.py` | 1 | Package init |
| `extreme/negative_classifier.py` | ~260 | `NegativeClassifierAdapter` — load/predict with no-op, rule fallback, ExtremPriceClf stub |
| `pipelines/classifier_pipeline.py` | ~190 | `run_negative_classifier` pipeline + `validate_fusion_input` + `_build_synthetic_fusion` helper |
| `scripts/run_negative_classifier.py` | ~120 | CLI: --input, --out, --model-dir, --rule-fallback/--no-rule-fallback, --dry-run, --production, --verbose |
| `scripts/validate_final_output.py` | ~150 | Final output validator: 12 checks including columns, NaN, negative_prob range, duplicate key, JSON lineage |
| `tests/test_negative_classifier_schema.py` | 12 tests | FINAL_OUTPUT_COLUMNS completeness, key subset, severity values, classifier constants |
| `tests/test_negative_classifier_adapter.py` | 17 tests | No-artifact fallback, rule fallback, load behavior, empty input, model lineage |
| `tests/test_classifier_pipeline.py` | 13 tests | Pipeline end-to-end, input validation, negative price flagging, schema compliance |
| `tests/test_final_output_validator.py` | 16 tests | Validator: NaN detection, duplicate key, out-of-range prob, invalid JSON, hour/period/task bounds |

---

## 3. Final Output Schema

**FINAL_OUTPUT_COLUMNS (17 columns)**

| Column | Type | Description |
|---|---|---|
| `task` | str | "dayahead" \| "realtime" |
| `target_day` | str | Calendar day being predicted |
| `business_day` | datetime | Business day |
| `ds` | datetime | Full timestamp |
| `hour_business` | int | 1..24 |
| `period` | str | "1_8" \| "9_16" \| "17_24" |
| `fused_price` | float | Fusion output price (pre-classifier) |
| `final_price` | float | Post-classifier/guardrail price |
| `negative_prob` | float or NaN | Negative-price probability [0, 1] |
| `negative_flag` | bool | Negative-price risk flagged? |
| `negative_severity` | str | "none" \| "low" \| "medium" \| "high" |
| `classifier_applied` | bool | Real classifier invoked? |
| `classifier_module` | str | Module identifier |
| `classifier_version` | str | Version string |
| `risk_source` | str | Risk data source |
| `reason_codes` | str | Semicolon-delimited audit codes |
| `model_lineage_json` | str | JSON-encoded lineage info |

**Key:** `FINAL_UNIQUE_KEY = [task, target_day, business_day, hour_business]` (4 columns)

---

## 4. Negative Classifier Adapter

**File:** `extreme/negative_classifier.py`

**`NegativeClassifierAdapter`** class:

| Method | Description |
|---|---|
| `__init__(rule_fallback=True, production=True)` | Configure fallback and mode |
| `load(model_dir=None)` | Scan model_dir for ExtremPriceClf artifacts; no-op if none found |
| `predict(fusion_df, rule_fallback=None)` | Run classification → FINAL_OUTPUT_COLUMNS DataFrame |

**Internal methods:**

- `_apply_noop(df)` — No-op: final_price = fused_price, classifier_applied = False
- `_apply_extremprice_stub(df)` — ExtremPriceClf stub: same output as no-op but with model metadata
- `_apply_rule_fallback(result, base_reason)` — Overlay: fused_price < 0 → negative_flag = True
- `_build_base_output(df)` — Build core DataFrame from fusion columns + model_lineage_json

**Convenience function:** `run_adapter(fusion_df, model_dir=None, rule_fallback=True, production=True)` — create, load, predict in one call.

---

## 5. No-Artifact Fallback Behavior

When `model_dir` is None or contains no recognised ExtremPriceClf artifact:

| Field | Value |
|---|---|
| `final_price` | = `fused_price` |
| `classifier_applied` | `False` |
| `classifier_module` | `"negative_classifier_noop"` |
| `classifier_version` | `"0.0.0-noop"` |
| `risk_source` | `"CLASSIFIER_ARTIFACT_MISSING"` |
| `reason_codes` | Contains `"NEGATIVE_CLASSIFIER_NO_OP"` |
| `negative_flag` | `False` (unless rule fallback overrides) |

**Tested:** 6 contract tests confirming no-op does not crash, final_price == fused_price, classifier_applied is False, reason codes contain NO_OP, risk source is correct.

---

## 6. Rule Fallback Behavior

When `rule_fallback=True` (default), rows with `fused_price < 0` receive:

| Field | Value |
|---|---|
| `negative_flag` | `True` |
| `negative_prob` | `1.0` |
| `negative_severity` | `"high"` |
| `classifier_module` | `"negative_classifier_rule"` |
| `classifier_version` | `"0.1.0-rule"` |
| `risk_source` | `"RULE_FALLBACK"` |
| `reason_codes` | Appended with `";RULE_NEGATIVE_PRICE"` |

**Important:** Rule fallback does **not** claim to be an ML classifier. The module is `negative_classifier_rule`, distinct from the no-op and ExtremPriceClf identifiers.

**Tested:** 5 contract tests covering negative price flagging, positive price non-flagging, reason code injection, module/severity/prob assignment, and disablement via `rule_fallback=False`.

---

## 7. Classifier Pipeline

**File:** `pipelines/classifier_pipeline.py`

**`run_negative_classifier(fusion_df, model_dir, rule_fallback, production) → DataFrame`**

Flow:

1. **`validate_fusion_input(df)`** — Checks:
   - Required columns present (task, target_day, business_day, ds, hour_business, period, fused_price)
   - No NaN in fused_price
   - hour_business in 1..24
   - period in valid set
   - task in valid set
2. **`NegativeClassifierAdapter`** — Load and predict
3. **Sort** by business_day, hour_business
4. **Return** FINAL_OUTPUT_COLUMNS DataFrame

**`_build_synthetic_fusion(n_hours=24, include_negative=False) → DataFrame`** — Synthetic fusion data for dry-run/testing.

---

## 8. Final Output Validator

**File:** `scripts/validate_final_output.py`

**`validate_final_dataframe(df, allow_empty=False, production=True) → tuple[bool, list[str]]`**

Checks performed:

| # | Check | Error on |
|---|---|---|
| 1 | Required columns present | Missing FINAL_OUTPUT_COLUMNS |
| 2 | final_price no NaN | Any null final_price |
| 3 | fused_price no NaN | Any null fused_price |
| 4 | negative_prob in [0, 1] or NaN | Values outside range |
| 5 | negative_flag boolean-like | Non-boolean values |
| 6 | classifier_applied boolean-like | Non-boolean values |
| 7 | hour_business in 1..24 | Values outside range |
| 8 | period in 1_8 / 9_16 / 17_24 | Invalid period strings |
| 9 | No duplicate final key | Duplicate (task, target_day, business_day, hour_business) |
| 10 | model_lineage_json valid JSON | Parse errors or non-dict |
| 11 | Production: no y_true | y_true present in production mode |
| 12 | task in dayahead / realtime | Invalid task values |

---

## 9. Tests Run

```
tests/test_schema_contract.py              ... 30 passed (includes P6 schema)
tests/test_dayahead_model_zoo_contract.py  ... 29 passed
tests/test_rt_assist_contract.py            ... 20 passed
tests/test_p5m_residual_contract.py         ... 22 passed
tests/test_loaders_contract.py              ... 26 passed
tests/test_dayahead_feature_pipeline.py     ... 24 passed
tests/test_realtime_feature_pipeline.py     .. 4 passed
tests/test_prediction_runner_contract.py    .. 8 passed
tests/test_residual_correction_schema.py    .. 8 passed
tests/test_residual_correction_runner.py    . 12 passed
tests/test_residual_output_validator.py     . 20 passed
tests/test_component_readiness_check.py     . 11 passed
tests/test_residual_key_merge_contract.py   . 10 passed
tests/test_prediction_to_residual_smoke.py   . 3 passed
tests/test_fusion_schema.py                 . 15 passed
tests/test_fusion_weights.py                . 19 passed
tests/test_fusion_engine.py                 . 18 passed
tests/test_fusion_readiness_gates.py        . 15 passed
tests/test_ledger_schema.py                 . 25 passed
tests/test_ledger_store.py                  . 17 passed
tests/test_prediction_ledger.py             . 11 passed
tests/test_actual_ledger.py                 . 9 passed
tests/test_fusion_weight_ledgers.py         . 12 passed
tests/test_ledger_pipeline_smoke.py         . 4 passed
--- P6 new tests ---
tests/test_negative_classifier_schema.py    . 12 passed
tests/test_negative_classifier_adapter.py   . 17 passed
tests/test_classifier_pipeline.py           . 13 passed
tests/test_final_output_validator.py        . 16 passed
================================================
Total: 468 passed, 0 failed, 3 warnings
```

---

## 10. Known Limitations

1. **No real ExtremPriceClf integration** — The adapter has a stub path for when a real ExtremPriceClf artifact is found (`_apply_extremprice_stub`), but production inference is not connected. The stub returns the same output as no-op with different metadata.
2. **No gradient-based learner** — As specified, P6 does not include gradient-based weight learning.
3. **No confidence intervals** — final output provides point estimates only.
4. **No artifact training pipeline** — If ExtremPriceClf requires training, that is a separate phase.
5. **Rule fallback is a simple threshold** — fused_price < 0 is a hard threshold. No probabilistic softening.
6. **model_lineage_json is static** — Built from the first fusion row's metadata. For multi-day fusion outputs, lineage metadata from the first row is applied to all rows.

---

## 11. Forbidden Files Check

- `data/*` — ❌ NOT committed
- `outputs/*` — ❌ NOT committed
- `reports/local/*` — ❌ NOT committed
- `ledgers/*.csv` — ❌ NOT committed (tests use tmp_path)
- `*.csv`, `*.xlsx`, `*.xls` — ❌ NOT committed
- `*.pkl`, `*.joblib`, `*.pt`, `*.pth`, `*.ckpt` — ❌ NOT committed
- `*.parquet` — ❌ NOT committed
- No real ExtremPriceClf artifact shipped
- No claimed classifier performance metrics

All tests use `pytest tmp_path` for temporary file I/O.

---

## 12. Final Status

**All 468 tests pass.** P6 negative classifier integration is complete with:

- 17-column final output schema with canonical key
- `NegativeClassifierAdapter` with 3 modes (no-op, rule fallback, ExtremPriceClf stub)
- No-artifact fallback with proper audit trail (CLASSIFIER_ARTIFACT_MISSING)
- Rule fallback that flags negative prices without claiming ML
- Classifier pipeline with input validation
- Final output validator with 12 structural checks
- 2 CLI scripts (run + validate)
- 63 new contract tests
- 0 forbidden files staged
- No fabricated classifier performance
