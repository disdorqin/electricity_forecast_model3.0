# 3.0 总控索引

## 目标

本仓库 `electricity_forecast_model3.0` 用于承接并融合以下来源：

- 2.0_exp 实验经验
- 2.5 ledger production 思路
- day-ahead cfg05 champion
- realtime DA-Safe assist branch
- P5M residual / negative residual stack
- 后续尖峰 / 负电价 / 极端风险模块

## 总架构层

```text
base prediction layer
  -> ledger production layer
  -> risk feature / regime layer
  -> fusion learner layer
  -> assist sidecar layer
  -> residual plugin layer
  -> guardrail fallback layer
  -> final delivery/report layer
```

## 冻结决策

### Day-ahead

当前可信 champion：

```text
cfg05_dayahead_lgbm
sMAPE_floor50 = 11.4838%
```

只作为冻结候选接入总融合，不在当前汇合阶段继续调参。

### Realtime

最终定位：

```text
DA-Safe Realtime Assist Model
```

默认：

```text
rt_pred = da_anchor
```

辅助输出给融合层使用，不默认覆盖 DA。

### P5M

当前定位：

```text
negative / low-valley residual plugin
```

官方结果：

```text
C negative-only GO
negative_MAE_improvement +3.32%
low_valley_MAE_improvement +3.42%
overall_sMAPE_improvement +0.01
high_spike_MAE_improvement -0.54%
```

B/D high_spike/unified 仍为 DATA-MISSING，因为缺真实 high_spike_prob。

## 绝对禁止

- 不再以 validation tap 作为生产主线。
- 不每天为了学习权重重跑 D-30 ~ D-1。
- 不使用 11.27% leakage 结果。
- 不使用自然日 Stage3 11.64% 结果。
- 不伪造任何实验指标。
- 不提交本地大文件、模型权重、CSV、Excel。
