# Source Paths and Migration Map

## 3.0 新仓库目标路径

```text
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model3.0
```

## 来源登记

| 来源 | 仓库 / 路径 | 当前状态 | 3.0 迁移方式 |
|---|---|---|---|
| 2.0_exp 实验经验 | `disdorqin/electricity_forecast_model2.0_exp`, branch `tune-timemixer` | 已有总结；真实本地路径 MISSING | 只继承经验与可复用模块，legacy tap 进 archive |
| 2.5 ledger production | `electricity_forecast_model2.5` | 本地路径 MISSING | 主链路仿照 ledger production |
| day-ahead cfg05 | `disdorqin/epf-sota-experiment` | 支线封装完成 | 作为 `cfg05_dayahead_lgbm` adapter 接入 |
| realtime DA-safe assist | repository/local_path MISSING | 支线资料已给，真实代码待审阅 | 作为 realtime assist sidecar 接入 |
| P5M residual / negative stack | `disdorqin/electricity_forecast_model2.0_exp`, branch `tune-timemixer` | PR #17/#19/#18 已完成 | 作为 negative / low-valley residual plugin 接入 |
| spike / negative / extreme risk | 未来 3.0 内新增 | 待开发 | 放入 `extreme/` 与 `fusion/` |

## 建议 3.0 目录结构

```text
electricity_forecast_model3.0/
  README.md
  config/
    model_sets.yaml
    thresholds.yaml
    runtime_profiles.yaml

  data/
    schema.py
    loaders.py
    business_day.py

  pipelines/
    ledger_predict.py
    ledger_backfill.py
    ledger_weight.py
    ledger_fuse.py
    ledger_classifier.py
    ledger_full.py
    ledger_full_range.py
    delivery_quality.py
    delivery_report.py

  models/
    adapters/
      cfg05_dayahead_lgbm.py
      realtime_da_safe_assist.py
      p5m_residual_plugin.py

  extreme/
    risk_features.py
    spike_predictor.py
    price_clusterer.py
    negative_risk.py
    guardrail.py

  fusion/
    learners/
      daily_ledger_gef.py
      regime_ledger_gef.py
    apply_weights.py
    metrics.py

  runtime/
    resource_scheduler.py
    gpu_memory.py
    reproducibility.py

  docs/
    memory/
    reports/
    handoff/

  archive/
    legacy_validation_tap/
    legacy_r3d_tap/
    legacy_tf/
    legacy_scripts/
```

## 迁移原则

1. 先文档和接口冻结，再搬模块。
2. 模型代码只通过 adapter 接入。
3. 主链路只能读统一 ledger 和标准 long table。
4. `archive/` 中代码不得被 production pipeline import。
5. outputs/data/model artifacts 不提交。
6. 所有 MISSING 项必须由本地 AI 审阅真实文件后补齐。
