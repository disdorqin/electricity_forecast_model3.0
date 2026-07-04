# Model and Module Registry

## Day-ahead candidates

### cfg05_dayahead_lgbm

来源：

```text
disdorqin/epf-sota-experiment
```

状态：

```text
TRUSTED_CHAMPION_FROM_BRANCH_SUMMARY
```

指标：

```text
sMAPE_floor50 = 11.4838%
```

条件：

```text
720 rows
task = dayahead
hour_business = 1..24
business_day mapping correct
no target leakage
contract tests pass
```

### catboost_sota / catboost_spike_residual

来源：

```text
disdorqin/epf-sota-experiment
```

状态：

```text
NEEDS_SOURCE_REVIEW
```

已知：

```text
catboost_sota = 12.58%
catboost_spike_residual_corrected = 12.47%
```

是否进入 fusion candidates：

```text
MISSING - 需要审阅真实输出是否无泄漏、schema 是否合格。
```

### invalid lgbm_spike_residual_corrected

状态：

```text
INVALID_DO_NOT_USE
```

原因：

```text
11.27% 结果存在 target leakage。
```

## Realtime candidates

### da_safe_realtime_assist

来源：

```text
realtime SOTA repo/path: MISSING
```

状态：

```text
SIDE_CAR_ASSIST_CANDIDATE
```

默认行为：

```text
rt_pred = da_anchor
```

输出：

```text
da_error_prob
residual_direction_prob
uncertainty_score
correction_permission
reason_codes
model_version
```

### sgdfnet_2_5

来源：

```text
electricity_forecast_model2.5
```

状态：

```text
NEEDS_SOURCE_REVIEW
```

用途：

```text
Realtime model candidate for learner/fusion.
```

## Residual modules

### p5m_negative_low_valley_residual

来源：

```text
disdorqin/electricity_forecast_model2.0_exp
branch: tune-timemixer
```

状态：

```text
PLUGIN_CANDIDATE
```

已知 official：

```text
C negative-only GO
negative_MAE_improvement +3.32%
low_valley_MAE_improvement +3.42%
overall_sMAPE_improvement +0.01
high_spike_MAE_improvement -0.54%
```

限制：

```text
B/D high_spike/unified DATA-MISSING because high_spike_prob missing.
```

## Fusion / learner

### 2.5 ledger learner

来源：

```text
electricity_forecast_model2.5
```

状态：

```text
TO_BE_MIGRATED
```

职责：

```text
Use past 30-day prediction ledger + actual ledger to learn weights.
```

### Regime-Ledger-GEF

来源：

```text
3.0 new design
```

状态：

```text
ROADMAP
```

职责：

```text
Add regime / spike / negative / uncertainty gating to ledger learner.
```

## Final classifier

### 2.5 negative price classifier

来源：

```text
electricity_forecast_model2.5
```

状态：

```text
TO_BE_MIGRATED
```

位置：

```text
after fusion, before final output
```
