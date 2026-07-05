## P30 Period BGEW Learner Report

### Status: P30_LEARNER_BLOCKED_SINGLE_MODEL

### Objective

Train a period-based BGEW weight learner for multi-model fusion.

### Results

- **Models available**: 1 (lightgbm_cfg05_dayahead)
- **Models in pool**: 1
- **Final status**: P30_LEARNER_BLOCKED_SINGLE_MODEL
- **Reason**: Cannot fake multi-model fusion with only 1 model

### Why Blocked

P29 audit found only cfg05 as REAL_24H_READY. The other 4 candidates
(best_two_average, stage3_business_fixed, catboost_spike_residual, catboost_sota)
are all TRAINABLE_LOCAL — they have training scripts but no trained artifacts locally.

The learner correctly refuses to produce fake fusion weights when only 1 model exists.

### Design (for when >=2 models are available)

See `docs/design/p30_fusion_learner_design.md` for the full design specification.

Key parameters:
- Periods: 1_8, 9_16, 17_24
- Rolling window: 30 days
- Alpha: 0.05 (smoothing for exp weighting)
- Min weight: 0.05, Max weight: 0.75
- cfg05 min prior: 0.30

### Tests

15 tests in `tests/test_p30_period_bgew_learner.py` — all passing.

### Files

- `scripts/train_p30_period_bgew_learner.py`
- `tests/test_p30_period_bgew_learner.py`
