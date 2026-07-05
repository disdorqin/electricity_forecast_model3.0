# P56 Trust-Gated Adaptive Regime BGEW — Design Document

> **Date**: 2026-07-05
> **Status**: P56 design complete

---

## 1. Why Regime-Aware Fusion for 3.0?

The existing P30/P35 period-based BGEW assigns weights at the period level (1_8, 9_16,
17_24) using rolling sMAPE.  This works well when market conditions are stationary, but
electricity markets exhibit **distinct regimes**:

| Regime | Characteristics | Impact on Fusion |
|--------|----------------|------------------|
| Normal | Stable prices, typical volatility | Period-level weights are adequate |
| Low price | Prices < 100 CNY, often off-peak | Models trained on normal data may over-predict |
| Negative risk | Prices < 0 CNY, rare but critical | Standard sMAPE breaks down; models with negative-aware training win |
| High spike | Prices > recent p90, sudden jumps | Fast-following models should be upweighted |

A single set of period-level weights cannot adapt to these regime shifts.  The **regime
dimension** allows the fusion engine to dynamically reweight models based on the detected
regime of the target day, giving more influence to models that perform well in the current
conditions and less to those that do not.

## 2. Architecture Overview

```
                    ┌─────────────────────────────┐
                    │      Trust Gate (P41)        │
                    │  TRUSTED → pass              │
                    │  SUSPECT_LEAKAGE → block     │
                    │  CONSERVATIVE_QUARANTINE →   │
                    │    block/allowed by profile  │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Ledger Loader              │
                    │   prediction_ledger.csv      │
                    │   actual_ledger.csv          │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Training Data Preparation  │
                    │   Merge on target_day + hour │
                    │   No future leakage (BD < D) │
                    │   30-day rolling window      │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────┐
          │         Regime Classification             │
          │  classify_regime(ensemble_median,         │
          │                   recent_p90,             │
          │                   historical_median)      │
          └─────────────┬────────────────────────────┘
                        │
                        ▼
     ┌──────────────────────────────────────────────────┐
     │        Weight Computation (per period P)          │
     │                                                   │
     │  base_score[m] = exp(-alpha * period_smape[m,P])  │
     │  stability[m]  = exp(-beta * rmse_vol[m,P])       │
     │  regime_score[m]= exp(-alpha*regime_smape[m,P,R]) │
     │                   or 1.0 if insufficient data     │
     │                                                   │
     │  score[m] = base * stability * regime_score       │
     │  weight[m] = normalize(score, constraints)        │
     └─────────────┬────────────────────────────────────┘
                   │
                   ▼
     ┌──────────────────────────────────────────────────┐
     │         Fallback Chain                           │
     │  1. regime_bgew  (>= 10 training days)           │
     │  2. period_bgew  (>= 5 training days)            │
     │  3. equal_weight                                 │
     │  4. failed (return None, caller uses cfg05)      │
     └─────────────┬────────────────────────────────────┘
                   │
                   ▼
     ┌──────────────────────────────────────────────────┐
     │         Output Builder                           │
     │  24-row DataFrame: business_day, ds,             │
     │  hour_business, period, dayahead_price,          │
     │  realtime_price                                  │
     └──────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **File-path based ledger loading** — the function accepts paths; callers can also pass
   pre-loaded DataFrames for testing.
2. **Regime is per-day** — one regime label for the entire target day.  This is a
   pragmatic simplification: extreme regimes (negative risk, spike) typically persist for
   a full day.
3. **Regime score is a multiplier** — it modulates the base period score, not replaces it.
   This preserves the period-level sMAPE signal.
4. **Neutral regime score when data is scarce** — rather than producing noisy weights,
   regime_score = 1.0 when fewer than `min_regime_hours` training hours are available for
   the detected regime.

## 3. Weight Formula Derivation

### Core Formula

```python
score_m = base_score_m * stability_score_m * trust_score_m * regime_score_m
weight_m = score_m / sum(score)
```

### base_score_m

```python
base_score_m = exp(-alpha * sMAPE_m)
```

where `sMAPE_m` is the period-level sMAPE of model *m* over the training window.

- **alpha = 5.0** is the default exponential scaling.
- A model with sMAPE = 10% gets `exp(-5 * 0.10) = exp(-0.5) = 0.607`.
- A model with sMAPE = 20% gets `exp(-5 * 0.20) = exp(-1.0) = 0.368`.
- The ratio 0.607 / 0.368 = 1.65 means the better model gets ~65% more raw score.

### stability_score_m

```python
stability_score_m = exp(-beta * rmse_volatility_m)
```

where `rmse_volatility_m` is the coefficient of variation (std/mean) of daily RMSE for
model *m* within the period.

- **beta = 3.0** is the default exponential scaling.
- A stable model with CV = 0.1 gets `exp(-3 * 0.1) = exp(-0.3) = 0.741`.
- An unstable model with CV = 0.5 gets `exp(-3 * 0.5) = exp(-1.5) = 0.223`.
- This penalises models with erratic day-to-day error patterns.

### trust_score_m

```python
trust_score_m = 1.0  # for TRUSTED (all models that pass the gate)
```

Since the trust gate already blocks non-trusted models, trust_score is always 1.0 for
eligible models.  The gate itself is enforced earlier in the pipeline.

### regime_score_m

```python
if n_regime_training_hours >= min_regime_hours:
    regime_score_m = exp(-alpha * regime_smape_m)
else:
    regime_score_m = 1.0  # neutral — no regime adjustment
```

`regime_smape_m` is the sMAPE of model *m* on training days that were classified as the
same regime as the target day.

- When the regime signal is strong (enough training data in that regime), models that
  historically performed well in this regime get a boost.
- When regime data is scarce, regime_score = 1.0 is neutral — the product is governed by
  base and stability scores alone.

### Combined Example

For a "high_spike" day with sufficient regime data:

| Model | sMAPE (base) | base_score | stability | regime_smape | regime_score | score | weight |
|-------|-------------|------------|-----------|-------------|-------------|-------|--------|
| cfg05 | 10% | 0.607 | 0.741 | 12% | 0.549 | 0.247 | 0.55 |
| catboost_spike | 11% | 0.576 | 0.619 | 8% | 0.670 | 0.239 | 0.53 |
| *Total* | | | | | | 0.486 | 1.00 |

Normalize: cfg05 = 0.247/0.486 = 0.508, catboost_spike = 0.239/0.486 = 0.492.

After constraints (min 0.05, max 0.75, cfg05_floor 0.30): cfg05 = 0.51, catboost_spike = 0.49.

## 4. Regime Classification Logic

```python
def classify_regime(
    ensemble_median: float,      # median of all trusted model predictions (target day)
    recent_p90: float,           # p90 of daily ensemble medians (training window)
    historical_same_hour_median: float,  # median of actuals (training window)
) -> str:
```

| Condition | Regime |
|-----------|--------|
| historical_actual_median < 0 OR ensemble_median < 0 | `negative_risk` |
| ensemble_median < 100 (CNY) | `low_price` |
| ensemble_median > recent_p90 | `high_spike` |
| everything else | `normal` |

### Rationale

- **negative_risk first**: If prices are negative, no other regime matters.  This is the
  highest-priority safety check.
- **low_price**: 100 CNY is approximately the 20th percentile of Shandong PMOS prices.
  This catches off-peak / low-demand days.
- **high_spike**: Above the recent p90 threshold.  This catches price spikes relative to
  the recent market context (rather than an absolute threshold).
- **normal**: Default — most days fall here.

## 5. Safety Constraints

### cfg05_floor (0.30)

For the `trusted_delivery` profile, cfg05 always gets at least 30% weight.  This is a
conservative safety measure:

- cfg05 is the champion baseline with 12+ months of production history.
- Fusion should never produce a result worse than cfg05 alone.
- The floor ensures cfg05 retains meaningful influence even when other models have
  stronger recent metrics.

### min_weight (0.05)

Every eligible model gets at least 5% weight.  This prevents any single model from being
completely ignored, preserving diversity.

### max_weight (0.75)

No model can receive more than 75% weight.  This prevents over-concentration on a single
model, which would defeat the purpose of fusion.

### Enforcement Order

```
1. Normalize raw scores to sum 1
2. Clip to [min_w, max_w]
3. Apply cfg05_floor (trusted_delivery profile only)
4. Renormalize to sum 1
5. Re-clip (renormalization may push some above max_w)
6. Final renormalize
```

## 6. Fallback Chain Rationale

| Level | Method | Threshold | Rationale |
|-------|--------|-----------|-----------|
| 1 | `regime_bgew` | >= 10 training days | Full power: regime + period weights. 10 days ensures ~2+ occurrences of each regime. |
| 2 | `period_bgew` | >= 5 training days | Period-only weights. 5 days is the minimum for meaningful sMAPE computation. |
| 3 | `equal_weight` | always | Simple average.  Works when data is too scarce for BGEW. |
| 4 | `failed` | — | Return None; caller falls back to cfg05 single-model output. |

The chain ensures graceful degradation: the system never produces a worse output than the
champion baseline.

## 7. Integration Points

### Existing Fusion Engine (`fusion/engine.py`)

The P56 module is a **specialized weight computation strategy**, not a replacement for
the existing fusion engine.  Integration can be done by adding a new fusion method:

```python
# In fusion/weights.py or as a new method in engine.py
def trust_gated_regime_bgew(
    target_date, trusted_models, prediction_ledger_path, actual_ledger_path, ...
) -> dict:
    ...
```

### Trust Gate (P41)

P56 consumes the trust states produced by P41.  The `trusted_models` parameter
corresponds to the TRUSTED state from P41.  The `model_trust_states` parameter can carry
CONSERVATIVE_QUARANTINE states for the `balanced_candidate` profile.

### Prediction Ledger (P34/P35)

The function reads the prediction ledger to obtain:
- Historical predictions for training (business_day < target_date)
- Target day predictions for fusion output

### Actual Ledger (P34)

The function reads the actual ledger to obtain:
- Historical actuals for sMAPE computation
- Historical actuals for regime classification thresholds

## 8. Files

| File | Purpose |
|------|---------|
| `fusion/trust_gated_regime_bgew.py` | Main implementation |
| `tests/test_p56_trust_gated_regime_bgew.py` | 15+ contract tests |
| `docs/reports/p56_trust_gated_regime_bgew_report.md` | Implementation report |

## 9. Open Questions

1. **Regime persistence**: Should the regime be computed as a rolling label (last N days)
   rather than per-day?  For production, per-day is simpler and sufficient.
2. **Cold start**: How quickly can the regime classifier converge?  10 days is a
   reasonable minimum, but fewer days may still produce useful regime signals.
3. **Profile expansion**: The `balanced_candidate` profile is a starting point.  Future
   profiles may allow/block different sets of trust states.
