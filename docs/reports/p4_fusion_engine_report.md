# P4: Fusion Core + Weight Learner Skeleton

**Date:** 2026-07-04
**Status:** Complete
**Components:** `fusion/` package, `scripts/run_fusion_engine.py`, `scripts/validate_fusion_output.py`
**Test count:** 67 new + 262 existing = **329 total, all passing**

---

## 1. Executive Summary

P4 implements the Fusion Core engine and Weight Learner Skeleton for the
electricity_forecast_model3.0 system.  It consumes corrected predictions
(P3/P3.5 output) and produces fused (ensemble) dayahead price forecasts
through configurable weight strategies.

Three weight strategies are supported:
- **equal_weight** — each contributing model receives 1/N.
- **prior_weight** — user-supplied weight dict, normalised to sum 1.
- **bgew_skeleton** — rolling window inverse-mean-absolute-error weighting
  using historical actuals, with future-awareness guard.

A readiness-aware eligibility gate prevents models in non-production states
(READY_STUB, DATA_MISSING, NOT_READY) from entering fusion.  Dry-run models
(READY_DRY_RUN) are included only with an explicit `allow_dry_run=True` flag.

---

## 2. New / Changed Files

### Schema (`data/schema.py`)
- **FUSION_OUTPUT_COLUMNS** — 14-column output schema:
  `task`, `target_day`, `business_day`, `ds`, `hour_business`, `period`,
  `fused_price`, `weights_json`, `included_models`, `excluded_models`,
  `fusion_method`, `learner_version`, `readiness_mode`, `reason_codes`
- **FUSION_UNIQUE_KEY** — 5-column key: `task, target_day, business_day, ds, hour_business`
- **FUSION_GROUPING_KEY** — 6-column group: adds `period` to unique key
- **FUSION_REQUIRED_INPUT_COLUMNS** — 8 columns validated on input
- **VALID_FUSION_METHODS** — `equal_weight`, `prior_weight`, `bgew_skeleton`
- **FUSION_READINESS_STATES** — `READY_REAL`, `READY_DRY_RUN`, `READY_STUB`,
  `DATA_MISSING`, `NOT_READY`

### Fusion Package (`fusion/`)

| File | Lines | Purpose |
|---|---|---|
| `__init__.py` | 1 | Package init |
| `weights.py` | ~280 | equal_weight, prior_weight, bgew_skeleton, compute_weights dispatch |
| `engine.py` | ~230 | run_fusion, _apply_readiness_gate, _auto_readiness, validators |
| `learners/__init__.py` | 1 | Learners subpackage init |
| `learners/bgew.py` | ~60 | BGEWLearner skeleton wrapping bgew_skeleton |

### CLI & Validation (`scripts/`)

| File | Purpose |
|---|---|
| `run_fusion_engine.py` | CLI runner with --dry-run, --method, --prior-weights-json, --actuals, --allow-dry-run |
| `validate_fusion_output.py` | 13-check validator for fusion output DataFrames |

### Tests (`tests/`)

| File | Tests |
|---|---|
| `test_fusion_schema.py` | 15 — column lists, uniqueness, subset relations, DataFrame construction |
| `test_fusion_weights.py` | 19 — equal/prior/bgew weight correctness, fallbacks, edge cases |
| `test_fusion_engine.py` | 18 — run_fusion end-to-end, readiness gate, dispatch, validator integration, CLI dry-run |
| `test_fusion_readiness_gates.py` | 15 — gate logic, auto-readiness, schema constants |

---

## 3. Weight Strategy Details

### equal_weight

Every model in `model_names` receives `1/N`.  Empty input returns `({}, ["EQUAL_WEIGHT_NO_MODELS"])`.

### prior_weight

User supplies `{model: raw_weight}` dict.  Raw weights are normalised to
sum 1.  Models in `model_names` but absent from the prior get
`FALLBACK_WEIGHT = 1e-6`.  Models in the prior but not in `model_names` are
silently ignored.  None/empty prior falls back to equal_weight with reason
`"PRIOR_NOT_PROVIDED_FALLBACK_EQUAL"`.

### bgew_skeleton

Rolling window inverse-MAE weighting:

1. Determine earliest `target_day` in corrected data.
2. Filter actuals to `business_day < target_day` (future-awareness).
3. Require >= `min_history` (default 7) business days of pre-cut actuals.
4. Limit to rolling `window` (default 30) business days.
5. Per model, merge predictions with actuals on `(business_day, hour_business)`.
6. Compute MAE per model; inverse-MAE = `1 / max(mae, 1e-10)`.
7. Normalise inverse-MAE scores to sum 1.

Falls back to equal_weight when actuals are unavailable or insufficient.

---

## 4. Readiness Gate Logic

The readiness-aware eligibility gate (`_apply_readiness_gate`) filters models
by their readiness state:

| State | Default | `allow_dry_run=True` |
|---|---|---|
| READY_REAL | Included | Included |
| READY_DRY_RUN | Excluded | Included |
| READY_STUB | Always excluded | Always excluded |
| DATA_MISSING | Always excluded | Always excluded |
| NOT_READY | Always excluded | Always excluded |

The gate returns `(included, excluded, readiness_mode)` where `readiness_mode`
is `"REAL"` if all included models are READY_REAL, `"DRY_RUN"` otherwise.

Auto-readiness (`_auto_readiness`) detects component states by scanning
adapter importability and artifact existence.  Current states:
- cfg05 → READY_DRY_RUN (adapter importable, artifact exists)
- RT assist → READY_DRY_RUN
- SGDFNet → READY_STUB (registered but no real adapter)
- P5M → DATA_MISSING (depends on external 2.5 pipeline)

---

## 5. Fusion Engine (`run_fusion`)

The main entry point:

```python
result = run_fusion(
    corrected_df,           # P3 corrected predictions
    method="equal_weight",  # weight strategy
    actuals_df=None,        # for bgew_skeleton
    prior_weights=None,     # for prior_weight
    allow_dry_run=False,    # include dry-run models?
    readiness_status=None,  # override auto-detection
    production=True,        # strip y_true from output
    learner_version="0.1.0-skeleton",
)
```

**Flow:**
1. Validate input has all required columns.
2. Detect duplicate `(grouping_key, model_name)` rows → ValueError.
3. Apply readiness gate.
4. Group by `FUSION_GROUPING_KEY`.
5. Per group: compute weights, compute `fused_price = Σ(w_i * y_pred_corrected_i)`.
6. Serialise weights to JSON, join included/excluded model lists.
7. Return DataFrame with FUSION_OUTPUT_COLUMNS.

**Duplicate handling:** same `(grouping_key + model_name)` raises ValueError.
Fusion never silently averages duplicates.

---

## 6. CLI Runner (`run_fusion_engine.py`)

```
python -m scripts.run_fusion_engine [--dry-run] [--method equal_weight]
    [--prior-weights-json '{"cfg05":0.7}'] [--actuals PATH]
    [--allow-dry-run] [--production] [--verbose]
```

Dry-run mode generates synthetic corrected predictions for 2 models (cfg05,
best_two_average) over 24 hours.

---

## 7. Fusion Output Validator (`validate_fusion_output.py`)

13 checks:

| # | Check |
|---|---|
| 1 | Required columns present |
| 2 | No NaN in fused_price |
| 3 | hour_business in 1…24 |
| 4 | period in valid values |
| 5 | weights_json is valid JSON |
| 6 | Weights sum to 1 (tolerance 1e-4) |
| 7 | included_models non-empty (configurable) |
| 8 | task in valid values |
| 9 | No duplicate fusion keys |
| 10 | No NaN in key columns |
| 11 | readiness_mode in valid values |
| 12 | No y_true in production mode |
| 13 | Empty DataFrame handling |

---

## 8. Test Results

```
tests/test_fusion_schema.py ...............                             15
tests/test_fusion_weights.py ...................                        19
tests/test_fusion_engine.py ..................                           18
tests/test_fusion_readiness_gates.py ...............                     15
                                                                       ---
Total P4 tests                                                          67
Total full suite                                                       329
Passed                                                                 329
Failed                                                                   0
```

---

## 9. Key Design Decisions

1. **Corrected-input-only**: Fusion consumes corrected predictions only
   (`y_pred_corrected`).  Raw predictions never enter the fusion engine.
2. **Future-aware BGEW**: Training actuals are filtered to
   `business_day < earliest_target_day`, preventing any future leakage.
3. **Reasons merging**: Weight strategy reasons are merged with
   equal-weight fallback reasons so the caller sees the full chain.
4. **Explicit readiness gate**: Models must pass a readiness check to
   participate.  Stub models (valid registry entry, no real artifact) are
   always excluded.
5. **Group-level weighting**: Weights are re-computed per fusion group
   (every hour), supporting time-varying ensemble compositions.

---

## 10. Limitations & Next Steps

- **BGEW learner**: BGEWLearner weights are computed fresh per group using
  `bgew_skeleton`.  A persistent learner that caches weights across runs
  would reduce recomputation.
- **No online weight store**: Weights are not persisted between runs; each
  `run_fusion` call re-computes from scratch.
- **No negative-risk integration**: The negative classifier path is not
  connected; fusion assumes all predictions are valid.
- **No production metric logging**: No dashboard metrics or weight drifts
  are tracked.
- **P5 candidate**: Performance weighting (gradient-based learner),
  per-period weight specialisation, and confidence-interval fusion.

---

## 11. File Inventory

```
data/schema.py                         # +6 fusion constants
fusion/__init__.py                     # package init
fusion/engine.py                       # run_fusion, readiness gate, auto-readiness
fusion/learners/__init__.py            # learners subpackage init
fusion/learners/bgew.py               # BGEWLearner skeleton
fusion/weights.py                     # weight strategies
scripts/run_fusion_engine.py          # CLI runner
scripts/validate_fusion_output.py     # fusion output validator
tests/test_fusion_engine.py           # 18 engine tests
tests/test_fusion_readiness_gates.py  # 15 readiness gate tests
tests/test_fusion_schema.py           # 15 schema tests
tests/test_fusion_weights.py          # 19 weight strategy tests
```
