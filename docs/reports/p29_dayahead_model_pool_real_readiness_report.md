## P29 Day-Ahead Model Pool Real-Readiness Audit Report

### Status: P29_MODEL_POOL_SINGLE_MODEL_ONLY

### Objective

Audit each candidate day-ahead model for real-readiness status.

### Results

| Model | Type | Artifact | Training Script | 24H Predict | Readiness | Blocker |
|-------|------|----------|-----------------|-------------|-----------|---------|
| cfg05 | LightGBM | YES | YES | YES | REAL_24H_READY | — |
| best_two_average | ensemble_average | NO | YES | NO | TRAINABLE_LOCAL | needs training + 24H capability |
| stage3_business_fixed | LightGBM | NO | YES | NO | TRAINABLE_LOCAL | needs training + 24H capability |
| catboost_spike_residual | CatBoost | NO | YES | YES | TRAINABLE_LOCAL | needs local training |
| catboost_sota | CatBoost | NO | YES | YES | TRAINABLE_LOCAL | needs local training |

### Banned Models (correctly excluded)

- lgbm_spike_residual_1127 — target leakage
- stage3_old_1164 — business_day mapping error
- lightgbm_90d_orig_1197 — missing hour 24

### Summary

- **Real-ready**: 1 (cfg05 only)
- **Trainable locally**: 4 (best_two_average, stage3_business_fixed, catboost_spike_residual, catboost_sota)
- **Can form fusion pool**: NO — need at least 2 real-ready models

### Implication for P30

The fusion learner is **blocked** — only cfg05 is available as a real model.
P30 status: P30_LEARNER_BLOCKED_SINGLE_MODEL.

To unblock multi-model fusion, at least one of the TRAINABLE_LOCAL models needs to be:
1. Trained locally with the correct feature set
2. Validated for 24H prediction completeness
3. Backtested to confirm sMAPE_floor50 < 20%

### Tests

10 tests in `tests/test_p29_dayahead_model_pool_real_readiness.py` — all passing.

### Files

- `scripts/audit_p29_dayahead_model_pool_real_readiness.py`
- `tests/test_p29_dayahead_model_pool_real_readiness.py`
