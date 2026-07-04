# Source Repository Map

## 3.0 主仓库

```text
repo: disdorqin/electricity_forecast_model3.0
url: https://github.com/disdorqin/electricity_forecast_model3.0
```

## Day-ahead 来源

用户称：

```text
日前用那个 catboost 那个仓库
```

当前理解：

```text
repo: disdorqin/epf-sota-experiment
```

需要审阅确认：

```text
1. 仓库中实际有哪些 day-ahead 模型。
2. 哪些模型是有效候选。
3. cfg05 LightGBM 是否仍为最终可信 champion。
4. CatBoost 相关输出是否只是 baseline / spike residual，还是可作为候选。
```

已知信息：

```text
cfg05 LightGBM = 11.4838% trusted champion
CatBoost baseline = 12.58%
CatBoost spike residual = 12.47%
best_two_average = 11.85%
```

禁用：

```text
lgbm_spike_residual_corrected = 11.27% target leakage
Stage3 old = 11.64% natural-day mapping error
```

## Realtime 来源

用户称：

```text
实时用那个 SOTA 仓库
```

当前状态：

```text
repo/path: MISSING
```

已知资料：

```text
DA-Safe Realtime Assist Model
主预测使用 da_anchor
深度/机器学习模块作为 sidecar
不默认覆盖 DA
```

额外候选：

```text
2.5 的 SGDFNet 模型
```

需要审阅确认：

```text
1. realtime SOTA 仓库真实路径 / GitHub repo。
2. 其 exported_models/rt_assist_pack 是否存在。
3. scripts/predict_rt_assist_pack.py 是否可用。
4. 输出 schema 是否包含 assist scores。
5. 2.5 SGDFNet 如何迁移为 adapter。
```

## Residual 来源

```text
repo: disdorqin/electricity_forecast_model2.0_exp
branch: tune-timemixer
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
repo/path: electricity_forecast_model2.5
```

需要迁移：

```text
ledger production chain
ledger_full
ledger_full_range
learner / fusion
negative classifier
```

当前状态：

```text
local path or repo url: MISSING
```

执行 AI 必须先定位真实来源，不能猜。
