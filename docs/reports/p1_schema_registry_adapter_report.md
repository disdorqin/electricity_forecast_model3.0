# P1 Execution Report: Schema, Registry & Adapter Layer

## 1. Overview

**Phase:** P1 — Stable Interface Layer
**Date:** 2026-07-04
**Status:** COMPLETE

P1 establishes the 3.0 stable interface layer that all downstream migration (P2–P5) depends on. It defines the canonical schema, business day mapping rules, model registries, adapter base contract, and four concrete adapters — all validated by contract tests.

**Scope boundary:** No 2.5 full chain migration. No bulk business logic migration. No large-scale experiments. No data/outputs/reports/local/model weights/prediction CSV commits.

---

## 2. Schema Layer (`data/`)

### Files created

| File | Description |
|---|---|
| `data/__init__.py` | Package init (empty) |
| `data/schema.py` | Canonical column definitions, ledger schemas, validators |
| `data/business_day.py` | Business day / hour mapping utilities |

### Key design decisions

- **PREDICTION_OUTPUT_COLUMNS** (10 columns): `task`, `model_name`, `target_day`, `business_day`, `ds`, `hour_business`, `period`, `y_pred`, `source_confidence`, `model_version`
- **EVAL_ONLY_COLUMNS**: `y_true` — production predict path MUST NOT contain it
- **Business day rule**: timestamp D 00:00 → business_day D-1, hour 24; timestamp D HH:00 → business_day D, hour HH
- **Period mapping**: 1-8→"1_8", 9-16→"9_16", 17-24→"17_24"
- **Prediction ledger unique key**: (task, model_name, forecast_date, target_day, business_day, hour_business)
- **Actual ledger unique key**: (task, target_day, business_day, hour_business)
- **Production guard**: `ensure_output_schema(production=True)` raises if y_true is present

### Validated by test_schema_contract.py (31 tests)

- Business day round-trip mapping for all 24 hours
- Period mapping boundaries (1-8, 9-16, 17-24)
- `add_business_time_columns` correctness
- `standardize_business_columns` validation
- `validate_daily_predictions` detects missing/duplicate hours and NaN y_pred
- Schema completeness and column presence

---

## 3. Registry Layer (`src/registry/`)

### Files created

| File | Description |
|---|---|
| `src/registry/__init__.py` | Package init (empty) |
| `src/registry/dayahead_models.py` | Day-ahead model registry |
| `src/registry/realtime_models.py` | Realtime model registry |

### Day-ahead registry (`dayahead_models.py`)

- **Champion**: `cfg05` (LightGBM, 90d window, MAE objective, num_leaves=191, lr=0.015, n_estimators=2000, sMAPE_floor50=11.48%)
- **DEFAULT_FUSION_POOL** (5 models, sorted by rank): cfg05 (11.48), best_two_average (11.85), stage3_business_fixed (11.86), catboost_spike_residual (12.47), catboost_sota (12.58)
- **INVALID_MODELS** (3, permanently banned):
  - `lgbm_spike_residual_1127` — target leakage (y_true as feature)
  - `stage3_old_1164` — natural-day business_day mapping error
  - `lightgbm_90d_orig_1197` — 690 rows only (missing hour 24)
- **MODEL_CONFIGS**: Full hyperparameter definitions for all valid models
- **Helpers**: `is_valid_model()`, `is_invalid_model()`, `get_model_config()`, `list_valid_models()`

### Realtime registry (`realtime_models.py`)

- **`da_safe_realtime_assist`** — DA-Safe Assist Model. Status: `READY_FOR_CHAIN_HANDOFF`. Positioning: sidecar_assist. Default prediction: `rt_pred = da_anchor` (DA-only, safe correction disabled by default)
- **`sgdfnet_2_5`** — SGDFNet delta regressor candidate. Status: `CANDIDATE` — needs adapter wiring + model weights
- **Helpers**: `get_realtime_model()`, `list_realtime_models()`, `is_ready_for_chain_handoff()`

### Validated by test_dayahead_model_zoo_contract.py (23 tests)

- Invalid models raise ValueError (all 3)
- Unknown models raise KeyError
- Champion identity and sMAPE
- Fusion pool has 5 entries, sorted by sMAPE, all valid
- cfg05 params (MAE objective, num_leaves=191, etc.)
- Helper function correctness

---

## 4. Adapter Layer (`models/adapters/`)

### Files created/updated

| File | Description |
|---|---|
| `models/adapters/__init__.py` | Package init (docstring) |
| `models/adapters/base.py` | `BasePredictionAdapter` ABC |
| `models/adapters/cfg05_dayahead_lgbm.py` | cfg05 LightGBM champion adapter |
| `models/adapters/realtime_da_safe_assist.py` | DA-Safe Realtime Assist adapter |
| `models/adapters/sgdfnet_2_5.py` | SGDFNet candidate adapter (stub) |
| `models/adapters/p5m_residual_plugin.py` | P5M negative/low-valley residual plugin |

### Base contract (`base.py`)

- `BasePredictionAdapter(ABC)` with:
  - `model_id`, `model_version` constructor params
  - `@property task` → "dayahead" | "realtime"
  - `load()` abstract method
  - `predict(**kwargs)` abstract method → returns standard-schema DataFrame
  - `validate_output(df)` — checks required columns, no eval columns, no NaN in y_pred, hour_business in [1,24]

### CFG05 Day-ahead adapter (`cfg05_dayahead_lgbm.py`)

- **task**: `dayahead`
- **Features**: 44 feature columns with graceful degradation for missing columns
- **Model loading**: Searches `cfg05_model.txt`, `model.txt`, `lightgbm_cfg05_dayahead.txt`
- **Predict**: Builds features, calls LightGBM predict, returns standard schema
- **Guard**: Raises RuntimeError if model not loaded

### DA-Safe Realtime Assist adapter (`realtime_da_safe_assist.py`)

- **task**: `realtime`
- **Default**: `rt_pred = da_anchor` (DA-only, most stable)
- **Optional**: `enable_safe_correction=True` adds `alpha * residual_pred`
- **Model pack**: Loads `manifest.json` + `residual_model.pkl` from model directory
- **Column normalization**: Accepts `times`→`ds`, `da_price`→`da_anchor`, `rt_price`→`rt_actual`
- **Date filtering**: `start=` and `end=` parameters

### SGDFNet 2.5 adapter (`sgdfnet_2_5.py`)

- **task**: `realtime`
- **Status**: CANDIDATE stub — returns empty DataFrame with warning if weights not loaded
- **Dependency**: External model weight files not tracked in git

### P5M Residual Plugin adapter (`p5m_residual_plugin.py`)

- **task**: `dayahead` (operates on fused predictions)
- **Behavior**: No-op (output == input) when risk data is absent (DATA-MISSING)
- **Profiles**: `conservative` (default), `moderate`, `aggressive`
- **Risk model**: `load_negative_risk_model()` loads `negative_risk_model.pkl` or `risk_model.pkl`
- **High_spike**: DATA-MISSING until real high_spike_prob is available

### Validated by adapter contract tests

**test_rt_assist_contract.py** (21 tests):
- Default rt_pred == da_anchor
- Standard schema output, no NaN in y_pred
- CSV path loading
- Date filtering (start/end)
- Column name normalization
- ValueError on missing da_anchor

**test_p5m_residual_contract.py** (21 tests):
- No-op when DATA-MISSING (no risk data)
- No crash with minimal input
- Standard schema output
- Profile validation (conservative/moderate/aggressive)
- Missing y_pred fills with 0 (no-crash)
- Missing required column raises ValueError

---

## 5. Test Summary

| Test suite | Tests | Passed | Failed |
|---|---|---|---|
| `test_schema_contract.py` | 31 | 31 | 0 |
| `test_dayahead_model_zoo_contract.py` | 23 | 23 | 0 |
| `test_rt_assist_contract.py` | 21 | 21 | 0 |
| `test_p5m_residual_contract.py` | 21 | 21 | 0 |
| **Total** | **96** | **96** | **0** |

All tests use synthetic tiny DataFrames. No dependency on real data files.

---

## 6. P1 Checklist

| Item | Status |
|---|---|
| Schema definitions (`data/schema.py`) | Done |
| Business day utilities (`data/business_day.py`) | Done |
| Day-ahead model registry (`src/registry/dayahead_models.py`) | Done |
| Realtime model registry (`src/registry/realtime_models.py`) | Done |
| Adapter base contract (`models/adapters/base.py`) | Done |
| cfg05 day-ahead adapter | Done |
| DA-Safe Realtime Assist adapter | Done |
| SGDFNet 2.5 candidate adapter | Done |
| P5M Residual Plugin adapter | Done |
| Schema contract tests | 31/31 pass |
| Day-ahead contract tests | 23/23 pass |
| RT Assist contract tests | 21/21 pass |
| P5M Residual contract tests | 21/21 pass |
| P1 execution report | Done |

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| SGDFNet adapter is a stub — no real model weights | Blocks P5 realtime chain | Adapter skeleton ready for weight wiring in P5 |
| P5M high_spike correction is DATA-MISSING | No spike correction until real data available | Documented in adapter docstring; no-op guard in place |
| cfg05 feature pipeline not yet built | Cannot run cfg05 end-to-end | Feature column list defined; adapter accepts raw df with available columns |
| No data pipeline for feature generation | P2–P4 blocked on this | Out of scope for P1 — next phase |

---

## 8. File Inventory (P1-created)

```
data/
  __init__.py
  schema.py
  business_day.py
src/registry/
  __init__.py
  dayahead_models.py
  realtime_models.py
models/adapters/
  __init__.py
  base.py
  cfg05_dayahead_lgbm.py
  realtime_da_safe_assist.py
  sgdfnet_2_5.py
  p5m_residual_plugin.py
tests/
  test_schema_contract.py
  test_dayahead_model_zoo_contract.py
  test_rt_assist_contract.py
  test_p5m_residual_contract.py
docs/reports/
  p1_schema_registry_adapter_report.md
```

Total: 16 P1 files (8 foundation + 4 adapters + 4 tests + 1 report)

---

## 9. Next Phase Recommendations

1. **P2: Feature Pipeline** — Build the feature engineering pipeline for cfg05 (44 features from raw market data). This unlocks end-to-end day-ahead prediction.
2. **P3: Fusion Engine** — Implement prediction fusion over the DEFAULT_FUSION_POOL using the weight learner schema defined in the ledger.
3. **P4: Evaluation Framework** — Build eval pipeline with sMAPE_floor50, ledger accumulation, and backtesting.
4. **P5: Realtime Chain** — Wire SGDFNet weights and complete the realtime prediction chain with DA-safe fallback.

---

## 10. Commands to reproduce

```bash
# Install dependencies
pip install pandas numpy pytest

# Run all P1 contract tests
python -m pytest tests/test_schema_contract.py -v
python -m pytest tests/test_dayahead_model_zoo_contract.py -v
python -m pytest tests/test_rt_assist_contract.py -v
python -m pytest tests/test_p5m_residual_contract.py -v
```
