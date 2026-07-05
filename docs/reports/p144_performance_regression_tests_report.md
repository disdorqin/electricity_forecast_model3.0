# P144: Performance Regression Tests Report

**Generated:** Phase placeholder — will be populated by running `pytest tests/test_p144_performance_regression_claims.py`

## Purpose

P144 is a TEST-ONLY phase. No script is needed — only tests that validate the
integrity of performance claims.

## Test Categories

### 1. BGEW Model Count Guard
BGEW cannot be claimed unless the ledger contains predictions from >= 2 models.
A single-model "BGEW" is just the model itself — no fusion occurred.

### 2. Realtime Assist Candidate Guard
Realtime assist cannot be claimed unless candidate count >= 2.
DA-Safe baseline (rt_pred = da_anchor) is a single-candidate fallback, not assist.

### 3. Residual No-Op Guard
Residual improvement cannot be claimed if the residual correction is a no-op.
A no-op residual produces the same output as its input.

### 4. Same-Window Comparison Guard
A "3.0 beats 2.5" claim requires comparison on the SAME time window.
Comparing 2025 full-year to 2026 local window is invalid.

### 5. sMAPE_floor50 Formula Canonical
The sMAPE_floor50 formula uses floor=50 and produces values in [0, 200]:
```
sMAPE = mean( 2*|y_true - y_pred| / max(|y_true| + |y_pred|, 50) ) * 100
```

### 6. No Lookahead in Prediction Ledger
Prediction ledger must NOT contain y_true. Including actuals in the prediction
ledger creates lookahead leakage.

### 7. Artifact Support for Claims
Every claimed metric must have a supporting artifact file that can be inspected
and reproduced independently.

### 8. Improvement Percentage Consistency
Improvement percentages must sum correctly across stages:
cfg05 -> BGEW -> residual improvements must chain consistently.

### 9. No Lookahead in Rolling BGEW
Rolling BGEW weights at day T must use only data from days < T.
Using day T's actuals to compute day T's weights is lookahead.

### 10. Performance Target Classification
Targets are correctly classified:
- Minimum: < 20.22% (below cfg05-only baseline)
- Reasonable: <= 15%
- Strong: <= 12%
- Stretch: <= 10%

## Running the Tests

```bash
pytest tests/test_p144_performance_regression_claims.py -v
```

## Status

All tests are defined and ready to run. Results will be populated when executed.
