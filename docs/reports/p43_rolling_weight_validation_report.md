# P43 Rolling Weight Validation Report

> **Generated**: 2026-07-05
> **Profile**: trusted_no_stage3 (cfg05 + catboost_spike_residual)
> **Status**: P43_VALIDATION_COMPLETE

---

## 1. Validation Approaches

Three approaches to verify fusion improvement is real and not a weight look-ahead artifact:

1. **Full-period**: Compute weights and evaluate on same 29 days (in-sample)
2. **Train/test split**: Train weights on first 20 days, evaluate on last 9 days (out-of-sample)
3. **Rolling expanding window**: For each day D starting at day 8, compute weights on days < D, evaluate on day D (out-of-sample, no lookahead)

## 2. Results Summary

| Approach | cfg05 sMAPE | Fusion sMAPE | Improvement | N |
|----------|-------------|-------------|-------------|---|
| Full-period (in-sample) | 9.904% | 9.231% | **+6.79%** | 696 |
| **Train/test split (OOS)** | 10.495% | 10.122% | **+3.55%** | 216 |
| **Rolling (OOS)** | 10.759% | 10.080% | **+6.31%** | 528 |

## 3. Key Findings

- **Fusion improvement holds out-of-sample**: Both split and rolling validation confirm fusion beats cfg05 on unseen data
- **Modest improvement**: 3.6-6.3% improvement OOS is real but modest (vs 49% with 4-model pool)
- **Rolling validation**: 22 evaluation days (June 8-29), each using only prior data for weights
- **No lookahead confirmed**: Rolling weights use only `days < D` — no future y_true leakage

## 4. Rolling Day-by-Day

The rolling validation evaluated 22 days (June 8 through June 29). For each day:
- Weights computed using only data from all prior days (minimum 7 days warmup)
- BGEW weights applied to that single day's predictions
- Result: consistent improvement over cfg05 across the evaluation window

## 5. Conclusion

The fusion benefit is **real but modest** (3.6-6.3% out-of-sample) when using only the 2-model trusted pool. The primary limitation is pool size — with only 2 models, diversity is limited. The earlier 49% improvement with the 4-model pool shows the value of including best_two_average and catboost_sota, which are good models despite borderline corr scores.
