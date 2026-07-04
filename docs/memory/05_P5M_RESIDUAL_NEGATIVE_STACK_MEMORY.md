# P5M Residual / Negative Residual Stack 记录

## 来源

```text
repository: disdorqin/electricity_forecast_model2.0_exp
branch: tune-timemixer
local_path: MISSING - 需要在用户本地确认
```

## 当前状态

P5M 已完成并合并：

```text
PR #17: plugin/interface + negative residual base
PR #19: negative risk calibration + monitor
PR #18: unified residual stack
```

当前任务不是继续重构，不是继续调模型，而是最终 release 文档和 sanity check。

## 核心目录

需要确认：

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

## 必跑测试

```bash
pytest tests/test_p5m_plugin_interface.py
pytest tests/test_p5m_negative_residual_module.py
pytest tests/test_p5m_negative_risk_calibration.py
pytest tests/test_p5m_residual_stack.py
```

## canonical pack smoke

如果本地有 canonical pack：

```bash
python scripts/calibrate_p5m_negative_risk.py   --canonical-pack reports/local/p4_canonical/canonical_prediction_pack.csv   --out-dir reports/local/p5m_calibration

python scripts/evaluate_p5m_residual_stack.py   --canonical-pack reports/local/p4_canonical/canonical_prediction_pack.csv   --negative-risk-path reports/local/p5m_calibration/negative_risk_predictions.csv   --out-dir reports/local/p5m_residual_stack   --negative-profile aggressive

python scripts/monitor_p5m_residual_health.py   --canonical-pack reports/local/p4_canonical/canonical_prediction_pack.csv   --risk-path reports/local/p5m_calibration/negative_risk_predictions.csv   --out-dir reports/local/p5m_monitor
```

如果没有 canonical pack：

```text
DATA-MISSING
```

不得伪造结果。

## 最终 release 文档路径

```text
docs/reports/P5M_final_release_report.md
```

必须包含：

```text
P5M v1.0 已完成
当前 Phase2 champion 不变
plugin/interface 已可接外部模型 prediction CSV
negative / low-valley residual 已完成
negative risk calibration 已完成
residual_stack 已完成
official / dry-run / data-missing 风险源策略已完成
```

## 当前 official 结果

```text
C negative-only GO
negative_MAE_improvement +3.32%
low_valley_MAE_improvement +3.42%
overall_sMAPE_improvement +0.01
high_spike_MAE_improvement -0.54%
```

## 仍然 DATA-MISSING

```text
B/D high_spike/unified
```

原因：

```text
缺真实 high_spike_prob
```

后续高价支线必须输出：

```text
high_spike_prob
risk_source
```

## 禁止提交

```text
reports/local/*
data/*
*.csv
*.xlsx
*.pkl
*.joblib
*.pt
*.pth
*.ckpt
```

检查命令：

```bash
git diff --name-only origin/tune-timemixer...HEAD | grep -E '^reports/local/|^data/|\.(csv|xlsx|pkl|joblib|pt|pth|ckpt)$' || true
```

必须无输出。

## 3.0 接入决策

P5M 作为：

```text
negative / low-valley residual plugin
```

当前不把 high_spike/unified 当已完成能力。
