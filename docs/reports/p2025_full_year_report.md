# 2025 Full-Year Prediction Report (陪跑模式)

> **Date**: 2026-07-05
> **Run**: main.py with cfg05 day-ahead model + DA-Safe Realtime Baseline
> **Target**: 2025-01-01 → 2025-12-31 (365 days)
> **Mode**: Practice/陪跑 (not strict full production)
> **Verdict**: `FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS`

## 1. Summary

The full 2025 year prediction completed successfully with **zero failures**. The system ran 365 consecutive daily predictions through the complete pipeline: data loading, feature engineering, model inference, ledger building, fusion, classification, residual correction, safety supervision, and final output.

## 2. Day-Ahead Prediction Results

| Metric | Value |
|---|---|
| sMAPE_floor50 (mean) | **20.22%** |
| sMAPE_floor50 (median) | 18.24% |
| sMAPE_floor50 (std) | 10.23% |
| Best single day | 3.43% |
| Worst single day | 59.72% |
| MAE (mean) | 65.35 |
| RMSE (mean) | 82.32 |

### Distribution

| Range | Days | Percentage |
|---|---|---|
| < 5% | 6 | 1.6% |
| 5-10% | 46 | 12.6% |
| 10-15% | 81 | 22.2% |
| 15-20% | 73 | 20.0% |
| 20-25% | 56 | 15.3% |
| 25-30% | 40 | 11.0% |
| ≥ 30% | 63 | 17.3% |

### Quarterly Breakdown

| Quarter | Days | sMAPE |
|---|---|---|
| Q1 (Jan-Mar) | 90 | 20.77% |
| Q2 (Apr-Jun) | 91 | 18.30% |
| Q3 (Jul-Sep) | 92 | 18.07% |
| Q4 (Oct-Dec) | 92 | 23.74% |

### Monthly Breakdown

| Month | Days | sMAPE |
|---|---|---|
| January | 31 | 19.91% |
| February | 28 | 22.48% |
| March | 31 | 20.08% |
| April | 30 | 20.06% |
| May | 31 | 17.83% |
| June | 30 | 17.04% |
| July | 31 | 20.77% |
| August | 31 | 16.82% |
| September | 30 | 16.56% |
| October | 31 | 22.17% |
| November | 30 | 20.83% |
| December | 31 | 28.14% |

## 3. Realtime Prediction Results

| Metric | Value |
|---|---|
| sMAPE_floor50 (mean) | **33.03%** |
| MAE (mean) | 77.31 |
| RMSE (mean) | 110.05 |

Realtime uses DA-Safe Baseline strategy: `rt_pred = da_anchor`. The 33% sMAPE reflects the inherent spread between day-ahead and real-time electricity prices in the Shandong market.

## 4. Pipeline Status

| Step | Status |
|---|---|
| Profile load | ✅ PASSED |
| Raw data check | ✅ PASSED |
| Day-ahead prediction | ✅ COMPLETE (365 days) |
| Realtime prediction | ✅ COMPLETE (rt_da_anchor) |
| Prediction ledgers | ✅ PASSED (8760 rows each) |
| Adaptive training days | ✅ COMPLETE_30D |
| Residual correction | ✅ COMPLETE |
| Unified weight learner | ⚠️ DEGRADED |
| Classifier | ⚠️ RULE_FALLBACK |
| Fallback ladder | ✅ PASSED |
| Postflight | ✅ PASSED |
| Safety supervisor | ✅ FULL_CHAIN_SAFETY_PASS |
| Claim guard | ✅ PASSED |

## 5. Caveats

- REALTIME_DA_SAFE_BASELINE — rt_pred = da_anchor, no SGDFNet assist
- RESIDUAL_NO_OP_FALLBACK — no full P5M residual stack
- CLASSIFIER_RULE_FALLBACK — no ML classifier in path
- ADAPTIVE_LEARNER_DEGRADED — limited training days for weight learning

## 6. Raw Output

- Prediction ledger: 8760 rows (365 days × 24 hours)
- Final output: 24 rows (last target day)
- No y_true leakage in production output
- All hours 1..24 complete with no NaN in price columns
