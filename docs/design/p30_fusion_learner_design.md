## P30 Fusion Learner Design Document

### Overview

The period-based BGEW (Bayesian-Gamma-Exponential-Weight) learner assigns per-period
weights to each model in the fusion pool based on rolling sMAPE_floor50 performance.

### Weight Formula

```
score_m = exp(-alpha * smape_m)
weight_m = score_m / sum(score)
```

### Constraints

```
min_weight = 0.05
max_weight = 0.75
cfg05_min_prior = 0.30 (until more models prove better)
renormalize after clipping
```

### Periods

```
period 1_8   (hours 1-8,   overnight/early morning)
period 9_16  (hours 9-16,  business hours)
period 17_24 (hours 17-24, evening/night)
```

### Rolling Window

```
rolling_days = 30
```

For each period, compute rolling sMAPE_floor50 per model over the last 30 days of
actual data.

### Model Eligibility

Allowed into learner:
- REAL_READY
- REAL_24H_READY
- BACKTESTED
- TRAINABLE_LOCAL (with predictions generated)

Blocked from learner:
- DRY_RUN
- STUB
- DATA_MISSING
- INVALID_BANNED

### Single-Model Guard

If only 1 model is available, the learner MUST refuse to train:
```
final_status = P30_LEARNER_BLOCKED_SINGLE_MODEL
```

Do NOT produce fake multi-model fusion weights.

### Output

```
.local_artifacts/p26_p30_fusion/ledgers/weight_ledger.csv
.local_artifacts/p26_p30_fusion/ledgers/fusion_ledger.csv
```

### Report Questions

1. How many real models? → 1 (cfg05 only)
2. Can we train a real learner? → NO
3. Why single model? → Other candidates need local training first
4. Weights? → N/A (blocked)
5. Fusion better than cfg05? → Cannot evaluate yet
6. Negative period analysis? → Available in actual ledger

### Path to Unblocking

To enable real multi-model fusion:
1. Train catboost_sota locally using source repo adapter
2. Validate 24H completeness
3. Backtest and confirm sMAPE < 20%
4. Re-run P30 with >=2 models
