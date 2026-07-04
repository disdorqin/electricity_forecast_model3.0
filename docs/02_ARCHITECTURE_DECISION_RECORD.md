# Architecture Decision Record: electricity_forecast_model3.0

## ADR-001：3.0 是新仓库，不是 2.0_exp 继续修补

决定：

```text
3.0 建立独立仓库。
2.0_exp 只作为经验和模块来源。
```

原因：

```text
2.0_exp 的 validation tap / staged pipeline 复杂度过高，不适合作为生产主线。
```

## ADR-002：主链路采用 2.5 ledger production

决定：

```text
3.0 主链路仿照 2.5 ledger production。
```

链路：

```text
真实预测 -> prediction ledger -> 过去30天 ledger 学权重 -> fusion -> negative classifier -> final
```

不采用：

```text
每天为 D 构造 D-30 ~ D-1 validation tap。
```

## ADR-003：后端分为 day-ahead 和 realtime 两条预测分支

决定：

```text
dayahead branch 使用 day-ahead 支线仓库候选模型。
realtime branch 使用 realtime SOTA/DA-safe assist，并考虑 2.5 SGDFNet。
```

两条分支都必须输出统一 schema。

## ADR-004：预测结果先进入 residual，再进入 learner/fusion

决定：

```text
base predictions -> residual correction -> learner/fusion
```

理由：

```text
P5M residual / negative residual stack 本质是对模型输出做安全修正，修正后的结果才应进入权重学习和融合。
```

注意：

```text
如果 residual 模块输出 DATA-MISSING 或风险源不足，则必须 dry-run 或 no-op，不得伪造修正。
```

## ADR-005：fusion 后再过 2.5 negative classifier

决定：

```text
fused output -> 2.5 negative price classifier -> final
```

理由：

```text
负电价分类器更适合作为最终风险校正层，尤其针对 realtime final。
```

## ADR-006：模型数量不够时，优先引入安全候选而不是乱凑

问题：

```text
学习器需要多个候选模型才能学权重。
```

策略：

```text
Day-ahead：从 epf-sota-experiment 中审阅有效候选。
Realtime：使用 realtime SOTA/DA-safe assist + 2.5 SGDFNet。
```

禁止：

```text
把已作废或泄漏模型当 candidate。
```

## ADR-007：所有模型通过 adapter 接入

决定：

```text
旧仓库模型不能直接污染主链路。
每个模型必须封装为 adapter。
```

adapter 只负责：

```text
输入标准数据
输出标准 prediction ledger rows
```
