# Realtime DA-Safe Assist Branch 记录

## 来源

```text
repository/local_path: MISSING - 需要在用户本地确认
target doc requested: docs/REALTIME_BRANCH_PROJECT_SUMMARY.md
```

## 支线定位

本支线负责 2.0 系统中的独立实时电价预测分支，与：

```text
日前电价分支
盘差模块
尖峰模块
负电价模块
```

并行。

## 最终模型定位

```text
DA-Safe Realtime Assist Model
```

核心结论：

```text
主预测使用 da_anchor
深度/机器学习模块作为辅助 sidecar
不默认覆盖 DA
```

## 设计原因

需要在最终文档中说明：

```text
全局 residual deep model 失败
residual 自相关弱
DA anchor 与 RT 高相关
条件专家没有安全门会造成 2026-02 灾难
DA-safe enhancer 稳定但提升很小
最终选择稳定优先
```

## 已完成实验

```text
DeepRT-SOTA v2
Reproduce Group 4
Baseline-safe shrink/gate
Residual history features
Feature signal hunt
Regime specialist
DA-safe enhancer
RT assist pack
```

## 关键结论

```text
DA-only 是当前最强稳定主预测
不建议默认开启价格修正
分类 / 不确定性 / permission gate 可作为后续融合链路输入
```

## 最终产物

```text
exported_models/rt_assist_pack/
scripts/predict_rt_assist_pack.py
docs/DEEP_RT_FINAL_MODEL_CARD.md
docs/CHAIN_HANDOFF_REALTIME_BRANCH.md
```

## 输入 schema

### 必需字段

```text
business_day
hour_business
ds
da_anchor
```

### 可能必需但需仓库确认字段

```text
load_forecast: MISSING
wind_forecast: MISSING
solar_forecast: MISSING
net_load: MISSING
recent_price_features: MISSING
calendar_features: MISSING
```

### 可选字段

```text
weather features
holiday features
recent residual history
model disagreement features
regime features
negative/spike risk features
```

实际字段必须以仓库现有脚本和报告为准，缺失则写 MISSING，不得补编。

## 输出 schema

```text
business_day
hour_business
ds
da_anchor
rt_pred
da_error_prob
residual_direction_prob
uncertainty_score
correction_permission
reason_codes
model_version
```

如果没有训练分类器：

```text
rt_pred = da_anchor
```

## 3.0 接入建议

实时分支输出：

```text
rt_pred
assist scores
reason_codes
model_version
```

后续融合层可以使用：

```text
da_error_prob
residual_direction_prob
uncertainty_score
correction_permission
```

不建议直接使用未验证 correction 覆盖价格。

## 风险与限制

```text
当前模型不保证 beat DA
对缺失 forecast-side 特征敏感
小幅修正收益不足
需要后续主系统融合模块评估
```

## 最终状态

```text
READY_FOR_CHAIN_HANDOFF or NOT_READY: MISSING
```

必须在审阅真实仓库后决定，不能凭空判断。
