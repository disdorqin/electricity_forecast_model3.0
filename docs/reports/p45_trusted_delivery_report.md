# P45 Trusted Fusion Delivery Report

> **Generated**: 2026-07-05
> **Sprint**: P41-P45 Trusted Fusion Delivery Fix
> **Status**: DELIVERY_READY (trusted_no_stage3 profile)

---

## 1. Executive Summary

The P31-P40 research results showed a 69.96% sMAPE improvement (9.90% → 2.97%) using BGEW fusion across 5 models. However, the `stage3_business_fixed` model was identified as having **source-repo training data leakage** (sMAPE=0.39%, MAE=1.20, 82.5% of predictions within 1% of actual, corr=0.9999).

This sprint quarantined stage3 and re-ran the entire fusion pipeline on a **trusted model pool**:

### Delivery Result

| Metric | cfg05-alone | Trusted Fusion | Improvement |
|--------|------------|---------------|-------------|
| **Full-period (in-sample)** | 9.90% | 9.23% | **+6.79%** |
| **Train/test split (OOS)** | 10.49% | 10.12% | **+3.55%** |
| **Rolling validation (OOS)** | 10.76% | 10.08% | **+6.31%** |

**Recommended default**: `trusted_bgew_fusion` (BGEW fusion on trusted_no_stage3 pool)

### Research Results (for reference only)

| Method | sMAPE | Improvement | Status |
|--------|-------|-------------|--------|
| cfg05 alone | 9.90% | — | Delivery baseline |
| Best single (catboost_sota) | 4.06% | 59.0% | Research only (suspect leakage) |
| Eual-weight fusion (5 models) | 5.19% | 47.6% | Research only |
| BGEW fusion (5 models, with stage3) | 2.97% | 69.96% | **Research only (stage3 leakage)** |
| BGEW fusion (trusted pool, 2 models) | 9.23% | 6.79% | **Delivery ✅** |

## 2. Phase Summary

### P41 - Model Trust Gate ✅

- Evaluated all 5 models against 4 suspicion criteria
- **2 TRUSTED**: cfg05, catboost_spike_residual
- **3 SUSPECT_LEAKAGE**: stage3_business_fixed (all 4 criteria), best_two_average and catboost_sota (corr > 0.995)
- Two profiles created: `research_all_models` and `trusted_no_stage3`
- [Full report](p41_model_trust_gate_report.md)

### P42 - Trusted Fusion Backtest ✅

- Ran BGEW fusion on trusted_no_stage3 pool (cfg05 + catboost_spike_residual)
- Best single: cfg05 (9.90%)
- BGEW fusion: 9.23% (+6.79% vs cfg05)
- Status: `TRUSTED_FUSION_IMPROVED`
- [Full report](p42_trusted_fusion_backtest_report.md)

### P43 - Rolling Weight Validation ✅

- Three validation approaches all confirm fusion > cfg05
- No lookahead in weight computation (rolling uses only `days < D`)
- Split fusion 10.12% < split cfg05 10.49%
- Rolling fusion 10.08% < rolling cfg05 10.76%
- Status: `P43_VALIDATION_COMPLETE`
- [Full report](p43_rolling_weight_validation_report.md)

### P44 - Delivery Readiness Packager ✅

- Assembles P41-P43 into delivery-ready summary
- Generates: trusted pool, quarantined models, delivery metrics, caveats, forbidden claims
- Recommended default: `trusted_bgew_fusion`
- Status: `P44_DELIVERY_READINESS_PACKAGED`

### P45 - This Report ✅

## 3. Forbidden Claims

The following claims are **NOT** permitted in delivery context:

| Claim | Reason |
|-------|--------|
| "2.97% production sMAPE" | Research result with leakage-suspect stage3 |
| "69.96% production improvement" | Same — stage3 dominated fusion |
| "Source 11.48% reproduction" | Not verified on trusted pool |
| "Stage3 production readiness" | Confirmed source-repo training leakage |

## 4. Permitted Delivery Claims

| Claim | Evidence |
|-------|----------|
| "cfg05 baseline: 9.90% sMAPE (30-day backtest)" | P32, P42 confirmed |
| "Trusted BGEW fusion: 9.23% sMAPE (+6.79% vs cfg05)" | P42 confirmed |
| "Improvement holds out-of-sample (split/rolling)" | P43 confirmed |
| "Delivery profile: trusted_no_stage3 (2 trusted models)" | P41-P44 gate passed |

## 5. Caveats

1. **Small trusted pool**: Only 2 models passed the trust gate. best_two_average and catboost_sota are good models (4-5% sMAPE) but flagged on corr > 0.995
2. **Modest improvement**: 6.79% in-sample, 3.6-6.3% out-of-sample
3. **Stage3 retraining needed**: The source repo must retrain stage3 with proper temporal CV
4. **30-day window only**: All validation is on June 2026 data
5. **Corr threshold is conservative**: best_two_average and catboost_sota are likely not leaking — the 0.995 threshold is a deliberate safety margin

## 6. Delivery Commands

```bash
# Default: trusted_no_stage3 profile (P41-P43 gate passed)
python -m scripts.run_p41_model_trust_gate --json
python -m scripts.run_p42_trusted_fusion_backtest --json
python -m scripts.run_p43_rolling_weight_fusion_validation --json

# Research only (not delivery): full pool including stage3
python -m scripts.run_p36_fusion_backtest --json
```

## 7. Files Created/Modified

| File | Action |
|------|--------|
| `scripts/run_p41_model_trust_gate.py` | Created |
| `scripts/run_p42_trusted_fusion_backtest.py` | Created (+ trusted_models param) |
| `scripts/run_p43_rolling_weight_fusion_validation.py` | Created (+ trusted_models param) |
| `scripts/run_p44_delivery_readiness_packager.py` | Created (+ P41 pool pass-through) |
| `tests/test_p41_p45_trusted_fusion_delivery.py` | Created (39 tests) |
| `docs/reports/p41_model_trust_gate_report.md` | Created |
| `docs/reports/p42_trusted_fusion_backtest_report.md` | Created |
| `docs/reports/p43_rolling_weight_validation_report.md` | Created |
| `docs/reports/p45_trusted_delivery_report.md` | Created |
| `docs/reports/p40_multimodel_fusion_delivery_report.md` | Updated with stage3 caveat |
| `docs/RUNBOOK_REAL_LOCAL_CHAIN.md` | Updated with trusted_no_stage3 default |

## 8. Test Results

```
39 passed in 12.52s  (test_p41_p45_trusted_fusion_delivery.py)
```

Full test suite: ~1414 tests passing (no regressions).
