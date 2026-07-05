# P140: Realtime Performance Unblock Report

**Phase**: P140  
**Status**: Framework ready; execution depends on raw data availability.  
**Date**: 2025

---

## 1. Objective

Unblock realtime price prediction performance by implementing delta models that improve upon the DA-Safe baseline (rt_da_anchor = using dayahead prediction as realtime), which currently achieves 33.03% sMAPE_floor50.

**Target**: Beat 33.03%, aim for 24%.

## 2. Baseline

| Strategy | sMAPE_floor50 | Description |
|----------|--------------|-------------|
| rt_da_anchor | 33.03% | Uses dayahead prediction as realtime (no delta) |

## 3. Delta Models

### Model 1: simple_delta_rolling_median

For each target_day D, hour h:
```
delta = median(realtime_price - dayahead_price) over previous 30 complete days, same hour_business
rt_pred = da_anchor + delta
```

Properties:
- Non-parametric, robust to outliers
- Uses only past data (no lookahead)
- Hour-specific captures diurnal delta patterns

### Model 2: simple_delta_lgbm_safe

```
target = realtime_price - dayahead_price (delta)
features: hour_business, period, da_anchor, lagged deltas (1-7d), past-day stats (mean/std 7d, mean 14d)
rt_pred = da_anchor + delta_pred
```

Properties:
- LightGBM regressor trained on pre-target data
- Walk-forward validation (train on days < target_day)
- Features available at D-1 15:00 cutoff
- Cannot use realtime_price as feature

## 4. Strict Rules

1. **No lookahead**: Delta models can only use past days (no target-day actuals)
2. **No realtime features**: Cannot use realtime_price as a feature
3. **D-1 15:00 cutoff**: All features must be available before the day-ahead market closes

## 5. BGEW Fusion

Delta models are fused using the canonical BGEW weighting:

```python
def compute_bgew_weights(smape_values, alpha=0.05, min_weight=0.05, max_weight=0.75):
    scores = {k: np.exp(-alpha * v) for k, v in smape_values.items()}
    total = sum(scores.values())
    weights = {k: v / total for k, v in scores.items()}
    weights = {k: np.clip(v, min_weight, max_weight) for k, v in weights.items()}
    total2 = sum(weights.values())
    weights = {k: v / total2 for k, v in weights.items()}
    return weights
```

## 6. Canonical sMAPE_floor50

```python
def compute_smape_floor50(y_true, y_pred, floor=50.0):
    y_true_f = np.maximum(y_true, floor)
    y_pred_f = np.maximum(y_pred, floor)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    return float(200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask]))
```

## 7. Status Codes

| Status | Meaning |
|--------|---------|
| `RT_DELTA_IMPROVED` | At least one delta model beats 33.03% |
| `RT_DELTA_NOT_IMPROVED` | Neither delta model beats baseline |
| `RT_DELTA_BLOCKED` | Missing inputs prevented evaluation |

## 8. Output Artifacts

All outputs written to `.local_artifacts/p140_realtime_unblock/`:

| File | Description |
|------|-------------|
| `rt_da_anchor_metrics.json` | Baseline metrics (da_anchor) |
| `rolling_delta_metrics.json` | Rolling median delta model metrics |
| `lgbm_delta_metrics.json` | LightGBM delta model metrics |
| `pooled_realtime_bgew_metrics.json` | BGEW fusion of delta models |
| `daily_realtime_metrics.csv` | Per-day per-model metrics |

## 9. Test Coverage

12+ tests covering:
- Rolling median delta computation (2 tests)
- No lookahead verification (2 tests)
- rt_pred = da_anchor + delta formula (2 tests)
- LightGBM delta past-only training (1 test)
- Output metrics format (2 tests)
- BGEW fusion (2 tests)
- Baseline comparison (2 tests)
- Blocked status (2 tests)
- sMAPE_floor50 canonical (2 tests)
- Output file generation (2 tests)
- Eval frame preparation (2 tests)
- Not-improved status (1 test)

## 10. Dependencies

- Raw data: `data/shandong_pmos_hourly.csv` (GBK encoding)
- Business day utilities: `data/business_day.py`
- LightGBM (optional; falls back to rolling median if unavailable)
