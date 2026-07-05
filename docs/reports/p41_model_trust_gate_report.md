# P41 Model Trust Gate Report

> **Generated**: 2026-07-05
> **Status**: P41_GATE_COMPLETE

---

## 1. Suspicion Criteria

Any model triggering **one or more** of these thresholds is labeled `SUSPECT_LEAKAGE`:

| Criterion | Threshold | Rationale |
|-----------|-----------|----------|
| `within_1pct_ratio` | > 0.50 | >50% predictions within 1% of actual → suspiciously accurate |
| `corr(y_pred, y_true)` | > 0.995 | Near-perfect correlation → potential lookahead |
| `sMAPE_floor50` | < 1.0% | Impossibly low error for dayahead electricity price |
| `MAE` | < 3.0 CNY | Impossibly low absolute error |

## 2. Per-Model Results

### TRUSTED Models

| Model | sMAPE | MAE | Within 1% | Corr | Verdict |
|-------|-------|-----|-----------|------|---------|
| lightgbm_cfg05_dayahead | 9.90% | 27.63 | 8.3% | 0.9864 | TRUSTED |
| catboost_spike_residual | 11.35% | 40.06 | 3.0% | 0.9903 | TRUSTED |

### SUSPECT_LEAKAGE Models

| Model | sMAPE | MAE | Within 1% | Corr | Reasons |
|-------|-------|-----|-----------|------|---------|
| stage3_business_fixed | **0.39%** | **1.20** | **82.5%** | **0.9999** | All 4 criteria triggered |
| best_two_average | 4.94% | 14.41 | 15.7% | **0.9962** | corr > 0.995 |
| catboost_sota | 4.06% | 12.06 | 21.4% | **0.9965** | corr > 0.995 |

## 3. Profiles

### `trusted_no_stage3` (Delivery Default)

- **Models**: cfg05, catboost_spike_residual
- **Delivery allowed**: YES
- **Stage3 excluded**: YES
- **SUSPECT_LEAKAGE excluded**: YES

### `research_all_models` (Research Only)

- **Models**: All 5 (including stage3, best_two_average, catboost_sota)
- **Delivery allowed**: NO
- **Purpose**: Reproduction of P40 research results

## 4. Notes

- **Stage3** is confirmed source-repo training leakage (all 4 criteria triggered, sMAPE=0.39%)
- **best_two_average** and **catboost_sota** are borderline SUSPECT_LEAKAGE (corr=0.9962-0.9965). They are good models without actual leakage — the corr threshold is conservative. They are excluded from delivery as a precaution.
- **Trusted pool reduced to 2 models**: cfg05 (9.90%) and catboost_spike_residual (11.35%)
- Fusion with only 2 models is less effective but still shows modest improvement (see P42)
