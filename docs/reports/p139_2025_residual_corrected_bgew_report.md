# P139: 2025 Residual-Corrected BGEW Report

**Phase**: P139  
**Status**: Framework ready; execution depends on P138 BGEW artifacts and residual model availability.  
**Date**: 2025

---

## 1. Objective

Evaluate whether applying the P5M residual correction layer to the P138 rolling-BGEW fused day-ahead predictions improves the 2025 full-year sMAPE_floor50 metric.

## 2. Pipeline

```
P138 BGEW fused predictions (or re-derived from P137 trusted ledger)
        |
        v
  Evaluate raw metrics (sMAPE_floor50, MAE, RMSE)
        |
        v
  ResidualP5MAdapter: search .local_artifacts/ for correction model
        |
   +---------+-----------+
   |                     |
 Model found         No model
   |                     |
   v                     v
Apply correction    RESIDUAL_NO_OP
   |                 (no claim)
   v
Evaluate corrected metrics
   |
   v
Compare: delta_sMAPE < 0?
   |           |
  Yes          No
   |           |
   v           v
RESIDUAL_    RESIDUAL_CORRECTED_
CORRECTED_   NOT_IMPROVED
IMPROVED
```

## 3. Canonical Metrics

All metrics use the **sMAPE_floor50** formula:

```python
def compute_smape_floor50(y_true, y_pred, floor=50.0):
    y_true_f = np.maximum(y_true, floor)
    y_pred_f = np.maximum(y_pred, floor)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    return float(200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask]))
```

## 4. BGEW Weights

Weights are computed using the canonical exponential-inverse formula:

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

## 5. Status Codes

| Status | Meaning |
|--------|---------|
| `RESIDUAL_CORRECTED_IMPROVED` | Real correction applied and sMAPE improved |
| `RESIDUAL_NO_OP` | No residual model found; no improvement claim possible |
| `RESIDUAL_BLOCKED` | Missing inputs prevented evaluation |
| `RESIDUAL_CORRECTED_NOT_IMPROVED` | Correction applied but did not improve sMAPE |

## 6. Output Artifacts

All outputs are written to `.local_artifacts/p139_residual_corrected/`:

| File | Description |
|------|-------------|
| `bgew_raw_metrics.json` | sMAPE/MAE/RMSE before correction |
| `bgew_residual_corrected_metrics.json` | sMAPE/MAE/RMSE after correction |
| `residual_delta_summary.json` | Delta between before/after |
| `period_metrics.json` | Per-period (1_8, 9_16, 17_24) breakdown |

## 7. Critical Invariant

> A no-op residual correction **CANNOT** claim improvement.  
> If the correction is identity (delta = 0), status = `RESIDUAL_NO_OP`.

## 8. Test Coverage

12 tests covering:
- Canonical sMAPE_floor50 formula (4 tests)
- BGEW weight computation (3 tests)
- Residual correction behaviour (5 tests)
- Delta summary format (1 test)
- Output file generation (2 tests)
- Blocked status on missing data (1 test)
- BGEW re-derivation from ledger (2 tests)

## 9. Dependencies

- P138 rolling BGEW output (`.local_artifacts/p138_rolling_bgew/`)
- P137 trusted ledger (`.local_artifacts/p137_trusted_2025/ledger/dayahead_prediction_ledger_2025_trusted.csv`)
- ResidualP5MAdapter (`adapters/residual_p5m_adapter.py`)
- Raw data (`data/shandong_pmos_hourly.csv`)
