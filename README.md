# electricity_forecast_model3.0

3.0 是新的总融合仓库，不是在旧 2.0_exp 上继续修补。

本仓库目标：

> 构建一个面向电力现货极端价格风险的多模型预测、ledger 动态融合、尖峰识别、负电价修正、残差插件和安全兜底系统。

## 总原则

1. 主链路参考 2.5 ledger production，不再以 2.0 validation tap 作为生产主线。
2. Day-ahead 当前冻结候选为 cfg05 LightGBM champion。
3. Realtime 当前采用 DA-Safe Realtime Assist Model：主预测使用 da_anchor，深度/机器学习模块只作为 sidecar assist。
4. P5M residual / negative residual stack 作为可插拔残差风险插件。
5. 尖峰、负电价、极端聚类、guardrail 不再只是 final 后处理，而是进入 fusion / risk / report 全链路。
6. 所有模型输出必须遵守 business_day + hour_business 口径。
7. 禁止使用任何 target leakage、自然日口径错误、缺 hour 24 或伪造指标的结果。

## 快速阅读顺序

1. `docs/00_MASTER_INDEX.md`
2. `docs/07_SOURCE_PATHS_AND_MIGRATION_MAP.md`
3. `docs/handoff/AI_EXECUTION_HANDOFF_PROMPT.md`
4. `docs/memory/*.md`
5. `docs/09_DO_NOT_USE_INVALID_RESULTS.md`
