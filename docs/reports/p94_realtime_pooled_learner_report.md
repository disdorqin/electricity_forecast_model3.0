# P94: Realtime 30D Pooled Learner Report

> **Date**: 2026-07-05
> **Status**: COMPLETE

## 1. Summary

The realtime weight learner has been updated to use **pooled_30d_bgew** instead
of the previous period/regime dimensional approach.

## 2. Learner Policy

```python
learner_policy = {
    "dayahead": "period_regime_bgew",   # task x period x regime (unchanged)
    "realtime": "pooled_30d_bgew",       # task-level pooled 24H
}
```

## 3. Pooled 30D BGEW Algorithm

For target_day D:
1. Use complete days D-30 ... D-1 (no-lookahead)
2. Use all hours 1..24 (pooled, no period/regime split)
3. Rows ~ 720 (30 days x 24 hours)
4. Compute sMAPE_floor50 for each candidate model
5. BGEW over model dimension only

## 4. Single Model Case

If only rt_da_anchor is available:
- `model_name = rt_da_anchor`
- `weight = 1.0`
- `learner_method = realtime_single_model_safe_baseline`
- `reason_codes = SGDFNET_ASSIST_DISABLED`

## 5. Hard Reject Bad Assist (Optional)

If `hard_reject_bad_assist=True`, an SGDFNet model whose weight drops below
`min_weight` is completely excluded (weight set to 0) instead of keeping
a minimal weight. This is optional and disabled by default.

## 6. Module Changes

- `fusion/unified_weight_learner.py` — Added `train_pooled_30d_bgew()`,
  `LEARNER_POLICY` dict, updated `train_unified_weights()` with policy support
- `scripts/run_p94_realtime_pooled_learner.py` — CLI runner
- `tests/test_p94_realtime_pooled_learner.py` — Test suite
