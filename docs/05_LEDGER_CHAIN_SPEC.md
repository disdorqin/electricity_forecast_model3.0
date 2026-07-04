# 3.0 Ledger Chain Spec

## 总链路

```text
[Day-ahead adapters]       [Realtime adapters]
          ↓                         ↓
        prediction ledger rows by task/model
          ↓
        residual correction layer
          ↓
        corrected prediction ledger
          ↓
        30-day ledger learner
          ↓
        task-aware fusion
          ↓
        2.5 negative price classifier
          ↓
        final output / delivery report
```

## Task 分流

```text
task = dayahead
task = realtime
```

两类 task 独立预测、独立 ledger、独立评估，但可以共享：

```text
business_day utilities
loaders
metrics
ledger learner base classes
risk feature generator
reporting framework
```

## Business-day 合约

所有模块必须遵守：

```text
hour_business = 24 of business_day D => ds = D+1 00:00:00
```

标准字段：

```text
business_day
ds
hour_business = 1..24
period = 1_8 / 9_16 / 17_24
```

禁止：

```text
只用 ds.date()
只用 ds.normalize()
自然日 00:00-23:00 错误切分
```

## Prediction ledger schema

```text
task
model_name
business_day
target_day
ds
hour_business
period
y_pred
source_confidence
model_version
created_at
```

## Residual correction schema

```text
task
model_name
business_day
target_day
ds
hour_business
period
price_before_residual
price_after_residual
residual_delta
residual_module
risk_source
reason_codes
model_version
created_at
```

## Corrected prediction ledger schema

```text
task
model_name
business_day
target_day
ds
hour_business
period
y_pred_raw
y_pred_corrected
source_confidence
correction_applied
correction_reason_codes
model_version
created_at
```

## Actual ledger schema

```text
task
business_day
target_day
ds
hour_business
period
y_true
actual_source
created_at
```

## Learner input

```text
past 30 days corrected prediction ledger
past 30 days actual ledger
optional risk signals
optional assist scores
optional residual debug
```

## Fusion output schema

```text
task
business_day
target_day
ds
hour_business
period
fused_price
weights_json
learner_version
ledger_window_days
reason_codes
created_at
```

## Negative classifier final schema

```text
task
business_day
target_day
ds
hour_business
period
price_before_negative_classifier
final_price
negative_prob
negative_classifier_action
negative_reason_codes
model_version
created_at
```

## Debug report required fields

```text
model_candidates_available
model_candidates_used
residual_applied
learner_weights
negative_classifier_triggered
missing_inputs
forbidden_inputs_detected
```
