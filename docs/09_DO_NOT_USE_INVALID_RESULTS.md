# Do Not Use Invalid Results

## 永久禁止使用

### 1. lgbm_spike_residual_corrected = 11.27%

状态：

```text
INVALID
```

原因：

```text
旧 correction 代码在预测阶段使用 y_true 作为特征，属于 target leakage。
```

禁止作为：

```text
champion
baseline
融合输入
主链路候选
报告结论
```

### 2. Stage3 old = 11.64%

状态：

```text
INVALID
```

原因：

```text
使用自然日 ds.date() 作为 target_day，违反 business_day 规则。
```

修复后：

```text
business-fixed Stage3 = 11.86%
```

### 3. 缺 hour 24 / 690-row LightGBM 输出

状态：

```text
INVALID
```

原因：

```text
不满足 720 rows 和 hour_business 1..24 业务日合约。
```

### 4. 任何把 y_true / residual / error / abs_error 当 prediction feature 的 correction

状态：

```text
INVALID
```

原因：

```text
target leakage 或后验信息泄漏。
```

## 总原则

宁可写：

```text
MISSING
DATA-MISSING
NOT_READY
```

也不能补编或伪造指标。
