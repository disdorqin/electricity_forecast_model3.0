# P40 Multi-Model Fusion Delivery Report

> **Phase**: P31–P40 Sprint — cfg05 single-model to multi-model BGEW fusion
> **Generated**: 2026-07-05

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Trained models | 5 (cfg05 + 4 candidates) |
| REAL_24H_READY models | 5/5 |
| 30-day full coverage models | 5/5 |
| BGEW fusion sMAPE (overall) | **2.97%** |
| cfg05-alone sMAPE (overall) | **9.90%** |
| sMAPE improvement | **+69.96%** |
| Negative hours analyzed | 57 (8.2%) |
| Low-price hours analyzed | 76 (10.9%) |
| Fusion status | **P36_FUSION_IMPROVED** |
| Full chain status | **P38_FULL_CHAIN_COMPLETE** |

## 2. Model Pool

| Model | Readiness | Features | 30-Day Coverage |
|-------|-----------|----------|-----------------|
| cfg05_dayahead_lgbm | REAL_24H_READY | 54 (v3) | 720/720 rows |
| best_two_average | REAL_24H_READY | 54 (v3) | 720/720 rows |
| stage3_business_fixed | REAL_24H_READY | 42 (source) | 720/720 rows |
| catboost_sota | REAL_24H_READY | 24 (base) | 720/720 rows |
| catboost_spike_residual | REAL_24H_READY | 54 (v3) | 720/720 rows |

## 3. Per-Period BGEW Weights

Weights learned via `score_m = exp(-0.5 * sMAPE_m)`, then constrained (min=0.05, max=0.75, cfg05_min_prior=0.30):

| Model | Period 1_8 | Period 9_16 | Period 17_24 |
|-------|-----------|-------------|--------------|
| cfg05 | 0.224 | 0.219 | 0.230 |
| best_two_average | 0.088 | 0.036 | 0.111 |
| stage3_business_fixed | **0.545** | **0.671** | **0.451** |
| catboost_sota | 0.106 | 0.038 | 0.170 |
| catboost_spike_residual | 0.037 | 0.036 | 0.038 |

**Note**: stage3_business_fixed dominates due to very low sMAPE (0.26–0.60%). This may indicate a data leakage issue — investigation recommended before production deployment.

## 4. Fusion vs cfg05-Alone

| Metric | cfg05 Alone | BGEW Fusion | Improvement |
|--------|------------|-------------|-------------|
| sMAPE_floor50 (overall) | 9.90% | 2.97% | **+69.96%** |
| MAE | 27.63 CNY | 9.53 CNY | **-65.5%** |
| RMSE | 39.33 CNY | 13.17 CNY | **-66.5%** |

### Per-Period sMAPE

| Period | cfg05 | Fusion | Improvement |
|--------|-------|--------|-------------|
| 1_8 (01:00–08:00) | 9.75% | 2.85% | +70.8% |
| 9_16 (09:00–16:00) | 14.47% | 3.88% | +73.2% |
| 17_24 (17:00–24:00) | 5.49% | 2.19% | +60.1% |

## 5. Regime Analysis

| Regime | Hours | cfg05 MAE | Fusion MAE |
|--------|-------|-----------|------------|
| Negative (< 0 CNY) | 57 | 7.31 | **3.52** |
| Low (0–100 CNY) | 76 | 19.79 | **6.61** |
| Normal (≥ 100 CNY) | 563 | 30.74 | **10.53** |

**Key finding**: 55 of 57 negative hours occur in period 9_16 (mid-day), likely due to solar over-generation during June. Fusion reduces MAE by 51.8% in negative regimes.

## 6. A/B/C Success Criteria

| Tier | Criteria | Status |
|------|----------|--------|
| **A** | ≥2 REAL_24H_READY models | ✅ 5/5 |
| **B** | ≥1 REAL_24H_READY + fusion improves over cfg05 | ✅ fusion +70% |
| **C** | Full chain produces 24 fused rows | ✅ P38_FULL_CHAIN_COMPLETE |

## 7. Files Created/Changed

| File | Action |
|------|--------|
| `models/adapters/multimodel_pool.py` | **CREATED** — 4 model adapters + factory |
| `scripts/run_p31_train_dayahead_model_pool.py` | **CREATED** — multi-model training |
| `scripts/run_p32_multimodel_30d_backtest.py` | **CREATED** — 30-day backtest |
| `scripts/run_p33_multimodel_prediction_ledger.py` | **CREATED** — prediction ledger |
| `scripts/run_p34_actual_ledger_alignment.py` | **CREATED** — actual ledger alignment |
| `scripts/train_p35_period_bgew_multimodel.py` | **CREATED** — BGEW weight learner |
| `scripts/run_p36_fusion_backtest.py` | **CREATED** — fusion backtest |
| `scripts/analyze_p37_negative_low_price_regime.py` | **CREATED** — regime analysis |
| `scripts/run_p38_fused_full_chain.py` | **CREATED** — full chain fusion |
| `docs/RUNBOOK_REAL_LOCAL_CHAIN.md` | **CREATED** — production runbook |
| `docs/reports/p40_multimodel_fusion_delivery_report.md` | **CREATED** — this report |

## 8. Known Issues

1. **Stage3 sMAPE seems too low** (0.26–0.60%). **Confirmed source-repo training leakage** in P41-P45 investigation (sMAPE=0.39%, 82.5% within 1%, corr=0.9999, MAE=1.20). `stage3_business_fixed` is excluded from delivery and labeled SUSPECT_LEAKAGE.

2. **cfg05 model name mismatch**: In the 3.0 adapter, model_name is `lightgbm_cfg05_dayahead`, not `cfg05_dayahead_lgbm`. The P32 backtest script uses `cfg05_dayahead_lgbm` as the directory name, but the actual model_name column in prediction outputs uses the adapter's internal name. This was handled in P35 by the _ALL_MODELS list.

3. **v3 feature gap**: The source feature builder produces 42 columns, but cfg05 3.0 adapter expects 54 (including 14 v3 features). P31 fills these with approximations. For production, the full 54-column pipeline should be implemented.

4. **Test counts**: P31-P38 test scripts created (114 tests). P41-P45 test file created (39 tests). Total: ~1414 tests passing.

## 9. P41-P45 Follow-up: Trusted Delivery Fix

After the P40 report, a dedicated P41-P45 sprint investigated the stage3 leakage and created a production-ready delivery pipeline:

**Findings**:
- **Stage3 confirmed leaked**: Source repo training data issue (not fixable in 3.0). sMAPE=0.39% is artifactual.
- **best_two_average** and **catboost_sota** flagged SUSPECT_LEAKAGE (corr > 0.995) — conservative threshold
- **Trusted pool**: cfg05 (9.90%) + catboost_spike_residual (11.35%) — only 2 models
- **Trusted fusion**: 9.23% sMAPE (+6.79% vs cfg05) — holds out-of-sample (split/rolling validated)

**Delivery recommendation**: Use `trusted_no_stage3` profile with `trusted_bgew_fusion` as default. The 69.96% / 2.97% research result is **not** delivery-claimable.

See [P45 delivery report](p45_trusted_delivery_report.md) for full details.

---

*End of P40 delivery report. Fusion improves cfg05-alone by +70% sMAPE (research). **Delivery result: +6.79% improvement with trusted_no_stage3 profile.***
