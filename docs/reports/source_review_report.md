# Source Review Report

## Status

```text
PARTIAL_SOURCE_REVIEW_UPDATED
```

## Confirmed GitHub repositories

### Day-ahead source

```text
repo: disdorqin/epf-sota-experiment
default_branch: main
visibility: public
status: CONFIRMED
```

Role in 3.0:

```text
Day-ahead model zoo source repository.
```

Confirmed from repository reports:

```text
cfg05 = trusted champion, 11.48% reported
best_two_average = 11.85%
stage3 business-fixed baseline = 11.86%
catboost_spike_residual = 12.47%
catboost_sota = 12.58%
```

Confirmed invalid results:

```text
lgbm_spike_residual_corrected = 11.27% INVALID due to target leakage
Stage3 old = 11.64% INVALID due to natural-day mapping
lightgbm_90d_orig = INVALID / limited because 690 rows, missing hour 24
```

Current code-path check:

```text
src/registry/dayahead_models.py = NOT_FOUND_ON_MAIN
scripts/run_dayahead_model_zoo.py = NOT_CONFIRMED
scripts/validate_dayahead_model_zoo.py = NOT_CONFIRMED
tests/test_dayahead_model_zoo_contract.py = NOT_CONFIRMED
docs/reports/dayahead_model_zoo.md = NOT_FOUND_ON_MAIN
```

Decision:

```text
3.0 should design around the 5-model day-ahead pool from the handoff, but execution AI must first confirm or create the registry/CLI wrapper in the day-ahead source or in 3.0 adapters.
```

### Realtime SOTA source

```text
repo: disdorqin/electricity_forecast_deep_sgdf_delta
default_branch: main
visibility: public
status: CONFIRMED
```

Role in 3.0:

```text
Realtime DeepRT / DA-Safe Realtime Assist source repository.
```

Confirmed source facts:

```text
business_time.py exists and defines the Shandong business-day mapping.
metrics.py exists and states metrics operate on realtime price, not delta/residual.
docs/DEEP_RT_SOTA_2B_RESULTS.md exists and records NO_GO: failed to beat DA anchor.
```

Command decision:

```text
Use this branch as realtime assist sidecar, not as an unconditional price-overwrite model.
Default rt_pred = da_anchor.
Expose assist scores to learner/fusion.
```

Hardening required before chain handoff:

```text
hourly output schema
predict_rt_assist_pack.py
exported_models/rt_assist_pack/
DA-only fallback manifest
no-leakage tests
dataset interface test repair
hourly mode disabled or implemented
MLP input dimension risk resolved
formal mode must not fill target NaN with 0
same-hour feature naming fixed or renamed
```

### Residual source

```text
repo: disdorqin/electricity_forecast_model2.0_exp
default_branch: tune-timemixer
visibility: public
status: CONFIRMED
```

Role in 3.0:

```text
P5M residual / negative residual stack source.
```

Need deeper review:

```text
plugin/
extreme/negative_price/
residual_stack/
P5M tests
calibration scripts
monitor scripts
canonical pack availability
```

### 2.5 chain source

```text
repo: disdorqin/electricity_forecast_model2.5
default_branch: main
visibility: public
status: CONFIRMED
```

Role in 3.0:

```text
Ledger production chain, learner/fusion, negative price classifier, and SGDFNet source.
```

Need deeper review:

```text
ledger_full
ledger_full_range
ledger_weight
ledger_fuse
ledger_classifier
negative classifier
SGDFNet location
prediction ledger schema
actual ledger schema
```

## Updated execution decision

Do not migrate business code until the next source review pass confirms file paths for:

```text
day-ahead registry/CLI
realtime assist export/predict pack
P5M plugin directories
2.5 ledger learner and negative classifier
2.5 SGDFNet
```

No metrics were newly computed in this report. Existing metrics are only restated from source reports or user-provided handoff summaries.
