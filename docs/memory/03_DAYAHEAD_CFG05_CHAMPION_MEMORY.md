# Day-Ahead cfg05 Champion 记录

## 来源

```text
repository: disdorqin/epf-sota-experiment
branch/local_path: MISSING - 需要在用户本地确认
scope: day-ahead 日前电价预测模型调优支线
```

## 支线定位

本支线只做：

```text
day-ahead 日前预测模型调优
```

不做：

```text
realtime
主生产链路改造
Chronos / TimesFM / TiRex
无泄漏体系之外的 correction
```

## 当前最终可信 champion

```text
cfg05 = 11.4838%
报告四舍五入 cfg05 = 11.48%
```

## cfg05 配置

```text
model = LightGBM
window = 90d
objective = mae
num_leaves = 191
min_data_in_leaf = 30
learning_rate = 0.015
lambda_l1 = 0.1
lambda_l2 = 5.0
feature_fraction = 0.85
bagging_fraction = 0.95
bagging_freq = 5
n_estimators = 2000
```

## 合格条件

```text
720 rows
task = dayahead
hour_business = 1..24
business_day mapping 正确
hour 24 = D+1 00:00
无 y_true 泄漏
sMAPE_floor50 统一口径
cfg05 contract test 通过
no-target-leakage test 通过
business-day mapping check 通过
```

## 最终排名

```text
1. cfg05 micro-search LGBM: 11.48%
2. best_two_average: 11.85%
3. stage3 business-fixed baseline: 11.86%
4. catboost_spike_residual_corrected: 12.47%
5. catboost_sota: 12.58%
```

## 已作废结果，禁止使用

```text
lgbm_spike_residual_corrected = 11.27%
```

原因：

```text
旧 correction 代码在预测阶段使用了 y_true 作为特征，属于 target leakage。
```

另一个作废结果：

```text
Stage3 old = 11.64%
```

原因：

```text
使用自然日 ds.date() 作为 target_day，违反业务日规则。
修复 business_day mapping 后结果为 11.86%。
```

## 已封装交付物

```text
scripts/run_champion_cfg05.py
tests/test_cfg05_champion_contract.py
tests/test_no_target_leakage.py
scripts/check_stage3_business_day_mapping.py
docs/reports/dayahead_current_champion.md
docs/reports/dayahead_cfg05_champion_freeze_report.md
docs/reports/dayahead_final_sprint_report.md
```

## 3.0 接入方式

3.0 中模型名建议：

```text
cfg05_dayahead_lgbm
```

或：

```text
lightgbm_cfg05_dayahead
```

标准 long table 字段：

```text
task
model_name
target_day
business_day
ds
hour_business
period
y_true
y_pred
```

汇合前必须跑：

```bash
python -m pytest tests/test_no_target_leakage.py
python scripts/check_stage3_business_day_mapping.py
python -m pytest tests/test_cfg05_champion_contract.py
```

复现 cfg05：

```bash
python scripts/run_champion_cfg05.py
```

预期：

```text
full_30d sMAPE_floor50 ≈ 11.4838%
720 rows
hour_business 1-24
无 NaN
无 target leakage
```

## 3.0 决策

当前汇合阶段不重新调 cfg05 参数。cfg05 是冻结候选，只作为总融合输入。
