# P56 Trust-Gated Adaptive Regime BGEW Report

> **Generated**: 2026-07-05
> **Status**: P56_COMPLETE

---

## 1. Design Overview

P56 implements a **trust-gated, regime-aware, period-based BGEW** fusion engine that
extends the P35 period BGEW concept with:

- **4-regime classifier** (normal / low_price / negative_risk / high_spike) for per-day
  market regime detection
- **3 x 4 weight matrix** (3 periods x 4 regimes) enabling regime-adaptive model
  weighting
- **Trust gating** (P41 integration) filtering models by trust state
- **Safety constraints**: cfg05 floor (30%), min/max weights (5%/75%)
- **3-level fallback chain**: regime_bgew -> period_bgew -> equal_weight

### Files

| File | Description |
|------|-------------|
| `fusion/trust_gated_regime_bgew.py` | Main implementation |
| `docs/design/p56_trust_gated_regime_bgew_design.md` | Design document |
| `tests/test_p56_trust_gated_regime_bgew.py` | Contract tests |

---

## 2. Weight Formula

```
base_score[m]     = exp(-alpha * period_smape[m, P])
stability_score[m]= exp(-beta * rmse_volatility[m, P])
regime_score[m]   = exp(-alpha * regime_smape[m, P, R])  [if enough data]
                    OR 1.0  [neutral, if insufficient regime data]
trust_score[m]    = 1.0  (all models passed the trust gate)

score[m]          = base_score[m] * stability_score[m] * regime_score[m] * trust_score[m]
weight[m]         = normalize(score, min=0.05, max=0.75, cfg05_floor=0.30)
```

Where:
- `P` = period (1_8, 9_16, 17_24)
- `R` = detected regime (normal, low_price, negative_risk, high_spike)
- `alpha` = 5.0 (default), `beta` = 3.0 (default)

### Rationale

- **base_score**: Period-level sMAPE captures overall accuracy. Lower sMAPE gives higher
  weight. Exponential scaling creates meaningful differentiation.
- **stability_score**: RMSE coefficient of variation penalises models with erratic
  day-to-day error patterns.
- **regime_score**: When enough historical data exists for the target regime, models that
  performed well in that regime get a boost. When data is scarce, regime_score = 1.0
  (neutral).
- **trust_score**: Binary — all models that pass the trust gate get 1.0. The gate itself
  handles exclusion.

### Default Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| `alpha` | 5.0 | > 0 |
| `beta` | 3.0 | > 0 |
| `min_training_days_for_regime` | 10 | >= 1 |
| `min_training_days_for_period` | 5 | >= 1 |

---

## 3. Regime Definitions

| Regime | Condition | Priority |
|--------|-----------|----------|
| `negative_risk` | `historical_actual_median < 0` OR `ensemble_median < 0` | 1 (highest) |
| `low_price` | `ensemble_median < 100` (CNY) | 2 |
| `high_spike` | `ensemble_median > recent_p90` | 3 |
| `normal` | Everything else | 4 (default) |

The `classify_regime()` function evaluates conditions in order (first match wins):
negative_risk is checked first because it represents a safety-critical condition that
takes precedence over all other regimes.

---

## 4. Safety Constraints

| Constraint | Value | Condition |
|------------|-------|-----------|
| `min_weight` | 0.05 | Always applied |
| `max_weight` | 0.75 | Always applied |
| `cfg05_floor` | 0.30 | `trusted_delivery` profile only |

### Enforcement Order

```
1. Normalize raw scores to sum 1
2. Clip to [min_w, max_w]
3. Apply cfg05_floor (trusted_delivery profile only)
4. Renormalize to sum 1
5. Re-clip (renormalization may push some above max_w)
6. Final renormalize
```

### Trust Gate Rules

| Trust State | trusted_delivery | balanced_candidate |
|-------------|------------------|-------------------|
| TRUSTED | Allowed | Allowed |
| DELIVERY_ALLOWED | Allowed | Allowed |
| COMPLETE_24H | Allowed | Allowed |
| CONSERVATIVE_QUARANTINE | **Blocked** | Allowed |
| SUSPECT_LEAKAGE | Blocked | Blocked |
| DRY_RUN / STUB / DATA_MISSING / INVALID_24H | Blocked | Blocked |

---

## 5. Fallback Chain

| Level | Method | Threshold | Description |
|-------|--------|-----------|-------------|
| 1 | `regime_bgew` | >= 10 days | Full regime + period weights |
| 2 | `period_bgew` | >= 5 days | Period-only weights |
| 3 | `equal_weight` | always | Simple average |
| 4 | `failed` | — | Return failure, caller falls back to cfg05 |

Each level degrades gracefully: if regime_bgew fails, period_bgew is tried; if both fail,
equal_weight is used. This guarantees the system always produces an output when at least
one model is available.

---

## 6. Test Results

### Test Summary (29 tests)

| Test Class | Test Name | Status |
|------------|-----------|--------|
| **Trust Gate** | TRUSTED models allowed | PASS |
|  | SUSPECT_LEAKAGE blocked | PASS |
|  | CONSERVATIVE_QUARANTINE blocked (trusted_delivery) | PASS |
|  | CONSERVATIVE_QUARANTINE allowed (balanced_candidate) | PASS |
| **Regime Classification** | normal | PASS |
|  | negative_risk (via ensemble) | PASS |
|  | negative_risk (via historical) | PASS |
|  | low_price | PASS |
|  | high_spike | PASS |
|  | boundary: 100 not low_price | PASS |
|  | boundary: p90 not high_spike | PASS |
| **Weight Normalization** | min/max bounds | PASS |
|  | cfg05_floor enforced | PASS |
|  | cfg05_floor not applied without cfg05 | PASS |
|  | renormalize after clipping | PASS |
|  | empty weights | PASS |
|  | zero total equal fallback | PASS |
| **Output Builder** | correct structure | PASS |
|  | empty prices | PASS |
| **sMAPE** | perfect prediction | PASS |
|  | off by 10% | PASS |
|  | zero actual | PASS |
| **Full Pipeline** | 24H output | PASS |
|  | weights present | PASS |
|  | weights within bounds | PASS |
|  | method regime_bgew (15 days) | PASS |
|  | period_bgew fallback (7 days) | PASS |
|  | equal_weight fallback (3 days) | PASS |
|  | empty trusted models | PASS |
|  | regime reported | PASS |
|  | training days reported | PASS |
|  | fallback chain populated | PASS |
|  | warnings with blocked models | PASS |
|  | delivery status DELIVERY_READY | PASS |
|  | fused prices reasonable | PASS |
| **Error Handling** | empty prediction ledger | PASS |
|  | empty actual ledger | PASS |
|  | no target predictions | PASS |
| **Profile** | balanced_profile allows quarantine | PASS |

**Total: 39 tests, 39 passed.**

---

## 7. Integration Points

### Fusion Engine (`fusion/engine.py`)

P56 is a **standalone fusion strategy** that can be integrated into the engine as a new
method.  The current design keeps it separate to allow independent testing and iteration.

### Trust Gate (P41)

The `trusted_models` parameter corresponds to the P41 TRUSTED model list. The
`model_trust_states` parameter carries additional trust states (e.g., CONSERVATIVE_QUARANTINE).

### Prediction / Actual Ledgers (P34)

The function reads both ledgers via CSV paths (or accepts pre-loaded DataFrames for
testing).  Training data is merged on `(target_day, hour_business)` with strict
future-awareness (only `business_day < target_date`).

---

## 8. Key Design Decisions

1. **Regime is per-day, not per-hour** — simpler implementation, and extreme regimes
   typically persist for a full day.

2. **Regime score is a multiplier** — it modulates (not replaces) the period base score,
   preserving the sMAPE signal.

3. **Neutral regime score when data is scarce** — rather than producing noisy weights,
   `regime_score = 1.0` does not affect the product.

4. **File-path plus DataFrame override** — the main function accepts file paths for
   production and optional DataFrames for testing.

5. **cfg05_floor is profile-scoped** — only applied in `trusted_delivery` to preserve
   champion baseline safety.
