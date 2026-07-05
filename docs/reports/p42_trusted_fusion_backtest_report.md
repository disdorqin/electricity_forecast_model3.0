# P42 Trusted Fusion Backtest Report

> **Generated**: 2026-07-05
> **Profile**: trusted_no_stage3 (cfg05 + catboost_spike_residual)
> **Status**: TRUSTED_FUSION_IMPROVED

---

## 1. Model Pool

Models used in trusted fusion (P41 gate output):

| Model | sMAPE_floor50 | MAE | RMSE |
|-------|--------------|-----|------|
| lightgbm_cfg05_dayahead | 9.904% | 27.63 | 39.33 |
| catboost_spike_residual | 11.351% | 40.06 | 46.77 |

**Best single model**: cfg05 (9.904% sMAPE)

## 2. Fusion Results

| Method | sMAPE_floor50 | MAE | RMSE | vs cfg05 |
|--------|--------------|-----|------|----------|
| cfg05 alone | 9.904% | 27.63 | 39.33 | — |
| Best single (cfg05) | 9.904% | 27.63 | 39.33 | 0.00% |
| Equal-weight fusion | 9.941% | 27.86 | 39.29 | -0.37% |
| **BGEW fusion** | **9.231%** | **26.27** | **37.79** | **+6.79%** |

BGEW fusion achieves a 6.79% improvement over cfg05 alone. The improvement is modest (compared to the 49% from the 4-model pool) because the pool is limited to 2 models.

## 3. Per-Period Performance

| Period | cfg05 sMAPE | Fusion sMAPE | Improvement |
|--------|-------------|-------------|-------------|
| 1_8 (hours 1-8) | 9.75% | 9.14% | +6.3% |
| 9_16 (hours 9-16) | 14.47% | 12.90% | +10.9% |
| 17_24 (hours 17-24) | 5.49% | 5.65% | -2.9% |

Fusion helps most during hours 9-16 (highest volatility period). Slight degradation in hours 17-24 where cfg05 already performs well.

## 4. Fusion Weights

| Period | cfg05 | catboost_spike_residual |
|--------|-------|-------------------------|
| 1_8 | 0.724 | 0.276 |
| 9_16 | 0.740 | 0.260 |
| 17_24 | 0.679 | 0.321 |

cfg05 dominates weights (68-74%) due to its substantially lower sMAPE. Spike residual contributes ~26-32%.
