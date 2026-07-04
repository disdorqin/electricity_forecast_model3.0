# Model and Module Registry

## Day-ahead candidates

Source:

```text
disdorqin/epf-sota-experiment
```

Registry status:

```text
DESIGN_CONFIRMED_FROM_HANDOFF_AND_REPORTS
CODE_REGISTRY_PATH_NOT_FOUND_ON_MAIN
```

Expected default fusion pool:

```python
DEFAULT_FUSION_POOL = [
    "cfg05",
    "best_two_average",
    "stage3_business_fixed",
    "catboost_spike_residual",
    "catboost_sota",
]
```

### cfg05 / lightgbm_cfg05_dayahead

Role:

```text
champion
```

Metric:

```text
sMAPE_floor50 = 11.4838% / reported 11.48%
```

Config:

```text
model = LightGBM
window = 90d
objective = mae
num_leaves = 191
min_data_in_leaf = 30
learning_rate = 0.015
lambda_l1 = 0.1
lambda_l2 = 5.0
feature_fraction = 0.85
bagging_fraction = 0.95
bagging_freq = 5
n_estimators = 2000
```

Required contract:

```text
720 rows
task = dayahead
hour_business = 1..24
business_day mapping correct
hour 24 = D+1 00:00
no target leakage
sMAPE_floor50 unified
```

### best_two_average

Role:

```text
strong_candidate / fusion input
```

Metric:

```text
approximately 11.85%
```

### stage3_business_fixed

Role:

```text
strong_candidate / fusion input
```

Metric:

```text
approximately 11.86%
```

### catboost_spike_residual

Role:

```text
diversity_fallback
```

Metric:

```text
approximately 12.47%
```

Condition:

```text
May enter fusion only if schema, no-leakage, and business-day checks pass.
```

### catboost_sota

Role:

```text
baseline_fallback
```

Metric:

```text
approximately 12.58%
```

## Invalid day-ahead models

### lgbm_spike_residual_1127

Status:

```text
INVALID_DO_NOT_USE
```

Reason:

```text
Target leakage: prediction-day y_true used as feature.
```

### stage3_old_1164

Status:

```text
INVALID_DO_NOT_USE
```

Reason:

```text
Natural-day mapping error; violates business_day rule.
```

### lightgbm_90d_orig_1197

Status:

```text
INVALID_DO_NOT_USE_AS_STANDARD_CANDIDATE
```

Reason:

```text
690 rows only; missing hour_business = 24.
```

## Realtime candidates

### da_safe_realtime_assist

Source:

```text
disdorqin/electricity_forecast_deep_sgdf_delta
```

Status:

```text
SIDE_CAR_ASSIST_CANDIDATE
NEEDS_FINAL_HARDENING_BEFORE_CHAIN_HANDOFF
```

Default behavior:

```text
rt_pred = da_anchor
```

Required output:

```text
business_day
hour_business
ds
da_anchor
rt_pred
safe_correction
final_pred_source
da_error_prob_50
da_error_prob_100
da_error_prob_150
da_error_prob_200
prob_residual_up
prob_residual_down
prob_residual_neutral
expected_abs_residual
residual_magnitude_bucket
uncertainty_score
correction_permission
reason_codes
model_version
```

Important verdict:

```text
DeepRT experiments are NO_GO as direct DA replacement.
Use as assist/risk sidecar, not unconditional price correction.
```

Hardening requirements:

```text
exported_models/rt_assist_pack/
scripts/export_rt_assist_pack.py
scripts/predict_rt_assist_pack.py
hourly output schema
DA-only fallback manifest
no-leakage tests
fix dataset interface mismatch
disable or implement hourly mode
fix MLP input dimension risk
formal mode must not fill target NaN with 0
rename or fix same-hour rolling feature
```

### sgdfnet_2_5

Source:

```text
disdorqin/electricity_forecast_model2.5
```

Status:

```text
NEEDS_SOURCE_REVIEW
```

Use:

```text
Realtime model candidate for learner/fusion.
```

## Residual modules

### p5m_negative_low_valley_residual

Source:

```text
disdorqin/electricity_forecast_model2.0_exp
branch: tune-timemixer
```

Status:

```text
PLUGIN_CANDIDATE
```

Known official:

```text
C negative-only GO
negative_MAE_improvement +3.32%
low_valley_MAE_improvement +3.42%
overall_sMAPE_improvement +0.01
high_spike_MAE_improvement -0.54%
```

Limitation:

```text
B/D high_spike/unified DATA-MISSING because high_spike_prob missing.
```

## Fusion / learner

### 2.5 ledger learner

Source:

```text
disdorqin/electricity_forecast_model2.5
```

Status:

```text
TO_BE_MIGRATED
```

Responsibility:

```text
Use past 30-day prediction ledger + actual ledger to learn weights.
```

### Regime-Ledger-GEF

Source:

```text
3.0 new design
```

Status:

```text
ROADMAP_AFTER_2.5_CHAIN_STABLE
```

Responsibility:

```text
Add regime / spike / negative / uncertainty gating to ledger learner.
```

## Final classifier

### 2.5 negative price classifier

Source:

```text
disdorqin/electricity_forecast_model2.5
```

Status:

```text
TO_BE_MIGRATED
```

Position:

```text
after fusion, before final output
```
