# 3.0 总控索引

## 目标

`electricity_forecast_model3.0` 是新的总融合仓库。它要把散落在本地和历史仓库里的内容统一收口为一条生产式链路。

需要吸收：

```text
2.0_exp 实验经验
2.5 ledger production 思路
day-ahead cfg05 / CatBoost 支线仓库
realtime SOTA / DA-safe assist 支线
P5M residual / negative residual stack
后续尖峰 / 负电价 / 极端风险模块
```

## 最新用户口径

用户明确表示：

```text
日前：使用“catboost那个仓库”/ day-ahead 支线仓库
实时：使用“sota仓库”，并考虑 2.5 的 SGDFNet 模型
残差模块：来自 2.0 实验仓库
链路：仿照 2.5 仓库，包括完整链路和学习器
后端：分为实时预测和日前预测
预测结果：进入残差模块修正
修正后：进入学习器学习权重并融合
融合后：经过 2.5 负电价分类器
至此链路结束
```

## 总链路草图

```text
Day-ahead model candidates
Realtime model candidates
        ↓
standard prediction ledger
        ↓
P5M residual / negative residual correction
        ↓
2.5-style ledger learner / fusion learner
        ↓
2.5 negative price classifier
        ↓
final output + delivery report
```

## 当前优先级

### P0：先确认真实源仓库和模型数量

必须审阅：

```text
disdorqin/epf-sota-experiment
disdorqin/electricity_forecast_model2.0_exp
electricity_forecast_model2.5
realtime SOTA 仓库：MISSING
```

### P1：冻结标准接口

所有模型必须输出标准 long table / ledger schema。

### P2：搬运模块

只通过 adapter 搬运，不允许把旧仓库混乱目录直接复制成主链路。

### P3：跑通 single-day ledger_full

先跑单日，再跑 range。

### P4：接入风险模块和报告

尖峰、负价、极端风险、guardrail 逐步接入。
