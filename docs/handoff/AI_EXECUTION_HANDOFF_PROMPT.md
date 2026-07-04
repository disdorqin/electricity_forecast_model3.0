# AI Execution Handoff Prompt for electricity_forecast_model3.0

你现在接手新的 3.0 总融合仓库。目标路径：

```text
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model3.0
```

## 总任务

建立并推进 `electricity_forecast_model3.0`，它不是旧 2.0_exp 的继续修补，而是一个新的总融合仓库。

需要吸收：

```text
2.0_exp 实验经验
2.5 ledger production 思路
day-ahead cfg05 champion
realtime DA-Safe assist branch
P5M residual / negative residual stack
后续尖峰 / 负电价 / 极端风险模块
```

## 第一步：初始化仓库

在目标路径创建仓库：

```bash
cd "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料"
mkdir electricity_forecast_model3.0
cd electricity_forecast_model3.0
git init
```

复制本初始化包中的所有文件到该目录。

然后：

```bash
git add README.md docs config pipelines models extreme fusion runtime scripts archive .gitignore
git commit -m "Initialize 3.0 command center docs"
```

## 第二步：确认来源路径

必须定位并记录以下本地路径：

```text
2.0_exp: disdorqin/electricity_forecast_model2.0_exp, branch tune-timemixer
2.5: electricity_forecast_model2.5
dayahead: disdorqin/epf-sota-experiment
realtime branch: MISSING
P5M: disdorqin/electricity_forecast_model2.0_exp, branch tune-timemixer
```

将确认后的路径写入：

```text
docs/07_SOURCE_PATHS_AND_MIGRATION_MAP.md
```

未知项写 MISSING，不得猜。

## 第三步：冻结接口

所有模型 adapter 输出必须统一为标准 long table：

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

如果带真值评估，则另行 merge actual ledger，不能依赖模型输出自带 y_true。

## 第四步：接入 day-ahead cfg05

来源：

```text
disdorqin/epf-sota-experiment
```

只接入冻结 champion：

```text
cfg05_dayahead_lgbm
sMAPE_floor50 = 11.4838%
```

不要重新调参。

必须保留检查：

```bash
python -m pytest tests/test_no_target_leakage.py
python scripts/check_stage3_business_day_mapping.py
python -m pytest tests/test_cfg05_champion_contract.py
```

禁止使用：

```text
11.27 leakage result
Stage3 old 11.64 natural-day result
690-row / missing hour 24 output
```

## 第五步：接入 realtime DA-safe assist

定位：

```text
DA-Safe Realtime Assist Model
```

默认：

```text
rt_pred = da_anchor
```

输出给融合层：

```text
da_error_prob
residual_direction_prob
uncertainty_score
correction_permission
reason_codes
```

不要默认让未验证 correction 覆盖价格。

需要生成或迁移文档：

```text
docs/REALTIME_BRANCH_PROJECT_SUMMARY.md
```

不得伪造指标；缺失写 MISSING。

## 第六步：接入 P5M residual / negative stack

来源：

```text
disdorqin/electricity_forecast_model2.0_exp
branch: tune-timemixer
```

确认：

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

运行：

```bash
pytest tests/test_p5m_plugin_interface.py
pytest tests/test_p5m_negative_residual_module.py
pytest tests/test_p5m_negative_risk_calibration.py
pytest tests/test_p5m_residual_stack.py
```

若无 canonical pack，写 DATA-MISSING。

新增最终文档：

```text
docs/reports/P5M_final_release_report.md
```

## 第七步：仿照 2.5 建立 ledger 主链路

主链路模块：

```text
pipelines/ledger_predict.py
pipelines/ledger_backfill.py
pipelines/ledger_weight.py
pipelines/ledger_fuse.py
pipelines/ledger_classifier.py
pipelines/ledger_full.py
pipelines/ledger_full_range.py
pipelines/delivery_quality.py
pipelines/delivery_report.py
```

原则：

```text
prediction ledger 是模型预测事实来源
actual ledger 是真值事实来源
过去30天真实 ledger 驱动权重学习
不再每天重跑 D-30~D-1 validation tap
```

## 第八步：建立极端风险层

新增：

```text
extreme/risk_features.py
extreme/spike_predictor.py
extreme/price_clusterer.py
extreme/negative_risk.py
extreme/guardrail.py
fusion/learners/regime_ledger_gef.py
```

第一版先做 rule-based + classical ML 骨架，不一开始上复杂深度模型。

## 第九步：禁止提交

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

必须无 forbidden file 输出。

## 第十步：返回格式

```text
3.0 Repository Initialization Summary

1. Repo path:
2. Git status:
3. Source paths confirmed:
4. Docs created:
5. Day-ahead cfg05 status:
6. Realtime DA-safe assist status:
7. P5M residual status:
8. Ledger production scaffold:
9. Extreme risk scaffold:
10. Missing items:
11. Forbidden files check:
12. Commit:
13. Final status:
```
