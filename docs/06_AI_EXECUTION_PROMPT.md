# AI Execution Prompt for 3.0 Migration

你现在是 `disdorqin/electricity_forecast_model3.0` 的执行 AI。不要糊里糊涂搬代码，先审阅、再冻结接口、再迁移。

## 背景

用户已经建立新仓库：

```text
https://github.com/disdorqin/electricity_forecast_model3.0
```

3.0 要融合：

```text
2.0_exp 实验经验
2.5 ledger production 思路
day-ahead 模型支线
realtime SOTA / DA-safe assist 支线
P5M residual / negative residual stack
2.5 learner / negative classifier
```

## 用户最新链路口径

```text
后端分为实时预测和日前预测。
日前使用“catboost那个仓库”/ day-ahead 支线仓库。
实时使用“sota仓库”，并考虑 2.5 的 SGDFNet。
两个预测分支输出预测结果。
预测结果输入残差模块进行修正。
修正后进入学习器学习权重并融合。
最后经过 2.5 的负电价分类器。
链路结束。
```

## 第一步：审阅源仓库，不要直接写业务代码

必须先确认：

```text
1. day-ahead 来源仓库路径 / GitHub 地址。
2. realtime SOTA 来源仓库路径 / GitHub 地址。
3. 2.0_exp 残差模块目录和测试是否存在。
4. 2.5 ledger 链路和 learner / negative classifier 文件位置。
```

将结果写入：

```text
docs/reports/source_review_report.md
```

缺失写：

```text
MISSING
```

不要猜。

## 第二步：day-ahead 模型候选审阅

已知 day-ahead 支线历史结论：

```text
cfg05 LightGBM = 11.4838% trusted champion
CatBoost baseline = 12.58%
CatBoost spike residual = 12.47%
best_two_average = 11.85%
```

但用户称“catboost那个仓库”，所以你必须审阅真实仓库，确认：

```text
1. 该仓库是否就是 disdorqin/epf-sota-experiment。
2. 可用模型数量。
3. 哪些模型无 target leakage。
4. 哪些模型有标准 720-row business_day output。
5. 哪些模型可进入 learner candidates。
```

绝对禁止：

```text
11.27% leakage result
Stage3 old 11.64 natural-day result
690-row missing hour 24 output
任何 y_true/residual/error/abs_error 作为 prediction feature 的 correction
```

## 第三步：realtime 模型候选审阅

已知：

```text
DA-Safe Realtime Assist Model
主预测使用 da_anchor
sidecar 不默认覆盖 DA
```

还需要接入：

```text
2.5 的 SGDFNet 模型
```

你必须确认：

```text
1. realtime SOTA 仓库实际路径。
2. 是否存在 exported_models/rt_assist_pack/。
3. 是否存在 scripts/predict_rt_assist_pack.py。
4. 输出字段是否包含 assist scores。
5. 2.5 SGDFNet 的输入输出和依赖。
```

## 第四步：residual 模块审阅

来源：

```text
disdorqin/electricity_forecast_model2.0_exp
branch: tune-timemixer
```

确认存在：

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

运行测试：

```bash
pytest tests/test_p5m_plugin_interface.py
pytest tests/test_p5m_negative_residual_module.py
pytest tests/test_p5m_negative_risk_calibration.py
pytest tests/test_p5m_residual_stack.py
```

无 canonical pack 时写 DATA-MISSING。

## 第五步：2.5 链路审阅

来源：

```text
electricity_forecast_model2.5
```

必须定位：

```text
ledger_predict
ledger_backfill
ledger_weight
ledger_fuse
ledger_classifier
ledger_full
ledger_full_range
learner / fusion learner
negative classifier
SGDFNet
```

迁移到 3.0 时，要改造成支持：

```text
dayahead candidates
realtime candidates
residual-corrected prediction ledger
task = dayahead / realtime
```

## 第六步：冻结 3.0 接口

所有模型 adapter 输出：

```text
task
model_name
target_day
business_day
ds
hour_business
period
y_pred
source_confidence
model_version
```

残差输出：

```text
price_before_residual
price_after_residual
residual_delta
residual_module
risk_source
reason_codes
```

融合输出：

```text
fused_price
weights_json
learner_version
ledger_window_days
reason_codes
```

负电价分类器输出：

```text
price_before_negative_classifier
final_price
negative_prob
negative_classifier_action
negative_reason_codes
```

## 第七步：开发顺序

```text
P0 source review report
P1 common schema + business_day utils
P2 day-ahead adapters
P3 realtime adapters
P4 residual plugin adapter
P5 2.5 ledger chain migration
P6 learner/fusion migration
P7 negative classifier migration
P8 single-day smoke
P9 30-day ledger backfill smoke
P10 report + final debug
```

## 第八步：禁止提交

```text
reports/local/*
data/*
outputs/*
*.csv
*.xlsx
*.xls
*.pkl
*.joblib
*.pt
*.pth
*.ckpt
*.parquet
```

提交前检查：

```bash
git status --short
git diff --name-only --cached | grep -E '^reports/local/|^data/|^outputs/|\.(csv|xlsx|xls|pkl|joblib|pt|pth|ckpt|parquet)$' || true
```

## 返回格式

```text
3.0 Migration Execution Summary

1. Source repos reviewed:
2. Day-ahead candidates:
3. Realtime candidates:
4. Residual modules:
5. 2.5 chain modules:
6. Interfaces frozen:
7. Files migrated:
8. Tests run:
9. Smoke run:
10. Missing items:
11. Forbidden files check:
12. Commit:
13. Final status:
```
