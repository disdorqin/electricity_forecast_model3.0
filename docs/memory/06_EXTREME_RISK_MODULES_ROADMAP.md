# 后续尖峰 / 负电价 / 极端风险模块路线图

## 总目标

3.0 不是普通多模型融合系统，而是：

```text
电力现货极端价格风险预测与安全交付系统
```

需要处理：

```text
高价尖峰
超高价尖峰
负电价
高波动日
模型极端区间失准
```

## 模块一：spike predictor

输出：

```text
spike_prob
super_spike_prob
spike_level_pred
spike_correction
spike_warning_flag
risk_source
reason_codes
```

建议第一版先使用 rule-based + classical ML，不直接上复杂深度模型。

## 模块二：price clusterer

输出：

```text
regime_cluster_id
regime_name
regime_confidence
extreme_type
extreme_risk_level
```

支持：

```text
day-level clustering
hour-level clustering
```

候选方法：

```text
KMeans
GaussianMixture
HDBSCAN
```

## 模块三：negative risk module

从旧 negative classifier 升级为 risk module，输出：

```text
negative_prob
negative_hours
negative_severity
negative_correction
negative_confidence
reason_codes
```

需要：

```text
概率校准
多阈值策略
与聚类联动
与模型分歧联动
低置信度兜底
```

## 模块四：guardrail fallback

触发信号：

```text
模型分歧过大
spike_prob 高
negative_prob 高
regime 属于 extreme cluster
预测与历史同类 regime 不一致
关键 period = 9_16 或 17_24
```

输出字段：

```text
fallback_triggered
fallback_reason
fallback_method
price_before_fallback
price_after_fallback
risk_level
```

## 模块五：Regime-Ledger-GEF

输入：

```text
过去30天 prediction ledger
actual ledger
regime_cluster_id
spike_prob
negative_prob
period
model predictions
source_confidence
```

学习逻辑：

```text
按 task + period 学基础权重
按 regime_cluster_id 学 regime-specific adjustment
对 extreme regime 单独计算模型表现
对尖峰/负价样本提高样本权重
输出 base_weight + regime_weight + risk_adjusted_weight
```

样本权重：

```text
sample_weight = day_gate * extreme_gate * source_confidence
```
