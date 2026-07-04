# electricity_forecast_model3.0

3.0 是新的总融合仓库，不是在旧 2.0_exp 上继续修补。

本仓库目标：

> 构建一个面向电力现货极端价格风险的多模型预测、ledger 动态融合、尖峰识别、负电价修正、残差插件和安全兜底系统。

## 当前总控决策

1. 主链路仿照 2.5 ledger production。
2. 后端分为 day-ahead 和 realtime 两条预测支线。
3. day-ahead 使用 `disdorqin/epf-sota-experiment` 支线输出，接入前必须以该仓库真实报告和 contract tests 为准。
4. realtime 使用实时 SOTA / DA-safe assist 支线，同时可吸收 2.5 的 SGDFNet 模型作为实时候选之一。
5. 两条预测支线先输出标准 prediction ledger。
6. 预测结果进入 P5M residual / negative residual stack 做安全修正。
7. 修正后进入 2.5 风格 learning/fusion learner，基于过去 30 天真实 ledger 学习权重。
8. 融合后进入 2.5 负电价分类器。
9. 最终输出 final delivery + report。

## 快速阅读顺序

1. `docs/00_MASTER_INDEX.md`
2. `docs/01_3.0_EXECUTION_PLAN.md`
3. `docs/02_ARCHITECTURE_DECISION_RECORD.md`
4. `docs/03_SOURCE_REPOSITORY_MAP.md`
5. `docs/04_MODEL_AND_MODULE_REGISTRY.md`
6. `docs/05_LEDGER_CHAIN_SPEC.md`
7. `docs/06_AI_EXECUTION_PROMPT.md`

## 核心原则

- 不再使用 2.0 validation tap 作为生产主线。
- 不为了学习权重每天重跑 D-30 ~ D-1。
- 不使用任何 target leakage 结果。
- 不使用自然日口径错误结果。
- 不伪造指标；缺失就写 `MISSING` 或 `DATA-MISSING`。
- 所有模型必须统一输出 `business_day + hour_business` 口径。

## 当前状态

```text
COMMAND_CENTER_INITIALIZED
READY_FOR_SOURCE_REVIEW_AND_MIGRATION
```
