# Source Repository Map

## 3.0 主仓库

```text
repo: disdorqin/electricity_forecast_model3.0
url: https://github.com/disdorqin/electricity_forecast_model3.0
```

## Day-ahead 来源

```text
repo: disdorqin/epf-sota-experiment
default_branch: main
status: CONFIRMED
```

用户口径：

```text
日前用 catboost/day-ahead model zoo 仓库。
```

当前确认：

```text
cfg05 LightGBM = 11.4838% trusted champion
best_two_average = 约 11.85%
stage3_business_fixed = 约 11.86%
catboost_spike_residual = 约 12.47%
catboost_sota = 约 12.58%
```

待确认：

```text
src/registry/dayahead_models.py 在 main 上未找到
scripts/run_dayahead_model_zoo.py 待确认
tests/test_dayahead_model_zoo_contract.py 待确认
```

禁用：

```text
lgbm_spike_residual_1127: target leakage
stage3_old_1164: natural-day mapping error
lightgbm_90d_orig_1197: 690 rows / missing hour 24
```

## Realtime 来源

```text
repo: disdorqin/electricity_forecast_deep_sgdf_delta
default_branch: main
status: CONFIRMED
```

用户口径：

```text
实时用 SOTA 仓库。
当前最稳定位是 DA-Safe Realtime Assist Model。
```

当前确认：

```text
models/deep_sgdf_delta/business_time.py exists
models/deep_sgdf_delta/metrics.py exists
docs/DEEP_RT_SOTA_2B_RESULTS.md exists
```

3.0 定位：

```text
realtime assist sidecar
not direct DA replacement
default rt_pred = da_anchor
assist scores feed learner/fusion
```

需要 final hardening：

```text
exported_models/rt_assist_pack/
scripts/export_rt_assist_pack.py
scripts/predict_rt_assist_pack.py
hourly prediction schema
DA-only fallback manifest
no-leakage tests
fix dataset interface mismatch
disable or implement hourly production mode
fix MLP input dimension path
formal mode must not fill target NaN with 0
same-hour feature naming fix or rename
```

额外 realtime 候选：

```text
2.5 的 SGDFNet 模型
```

## Residual 来源

```text
repo: disdorqin/electricity_forecast_model2.0_exp
branch/default_branch: tune-timemixer
status: CONFIRMED
```

模块：

```text
P5M residual / negative residual stack
plugin/interface
negative risk calibration
monitor
unified residual stack
```

需要确认目录：

```text
plugin/
extreme/negative_price/
residual_stack/
scripts/calibrate_p5m_negative_risk.py
scripts/monitor_p5m_residual_health.py
scripts/evaluate_p5m_residual_stack.py
tests/test_p5m_plugin_interface.py
tests/test_p5m_negative_residual_module.py
tests/test_p5m_negative_risk_calibration.py
tests/test_p5m_residual_stack.py
```

## Chain / Learner 来源

```text
repo: disdorqin/electricity_forecast_model2.5
default_branch: main
status: CONFIRMED
```

需要迁移：

```text
ledger production chain
ledger_full
ledger_full_range
learner / fusion
negative classifier
SGDFNet
```

执行 AI 必须先定位真实文件路径，不能猜。
