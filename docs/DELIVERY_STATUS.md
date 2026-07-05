# Delivery Status

> **Updated**: 2026-07-05
> **Status**: DELIVERY_FREEZE_READY

---

## Three-Tier Delivery Status (P57)

The delivery pipeline produces one of three terminal statuses:

| Status | Exit Code | Description | Condition |
|--------|-----------|-------------|-----------|
| **NORMAL** | 0 | All checks passed, safe to deliver | Level 1 (trusted_bgew_fusion) succeeds AND postflight validation PASS |
| **DEGRADED_DELIVERED** | 2 | Some non-critical checks failed, delivery still allowed | Levels 2-5 produce valid 24H output with no NaN and valid schema |
| **FAILED_NO_DELIVERY** | 1 | Critical checks failed, no delivery possible | All 6 fallback levels failed |

### Exit Code Convention

- **Exit 0 (NORMAL)**: The output is fully trusted. BGEW fusion via trusted models succeeded, and all 12 postflight checks passed. This is the ideal delivery state.
- **Exit 2 (DEGRADED_DELIVERED)**: The output is usable but degraded. The pipeline fell back to equal-weight, best-single-model, cfg05-baseline, or historical-median. Delivery is still allowed, but consumers should be aware that the fusion engine did not produce the primary output.
- **Exit 1 (FAILED_NO_DELIVERY)**: No output could be produced. All 6 fallback levels were exhausted. The pipeline cannot deliver predictions for the target day.

### Mapping to Fallback Levels

| Fallback Level | Method | Status |
|----------------|--------|--------|
| 1 | trusted_bgew_fusion | NORMAL |
| 2 | trusted_equal_weight | DEGRADED_DELIVERED |
| 3 | best_trusted_single_model | DEGRADED_DELIVERED |
| 4 | cfg05_baseline | DEGRADED_DELIVERED |
| 5 | historical_same_hour_median | DEGRADED_DELIVERED |
| 6 | FAILED_NO_DELIVERY | FAILED_NO_DELIVERY |

### Postflight Status Mapping

Postflight validation (P55, 12 checks) also contributes to the status:

| Postflight Status | Interpreted As | Action |
|-------------------|----------------|--------|
| PASS (0 failures) | NORMAL | Full confidence delivery |
| WARN (1-2 failures) | DEGRADED_DELIVERED | Check warnings, still deliverable |
| FAIL (3+ failures) | FAILED_NO_DELIVERY | Block delivery; inspect errors |

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Multi-model pool | ✅ REAL_24H_READY | 5/5 models |
| Trust gate (P41) | ✅ P41_GATE_COMPLETE | 2 trusted |
| Fusion backtest (P42) | ✅ TRUSTED_FUSION_IMPROVED | +6.79% vs cfg05 |
| Rolling validation (P43) | ✅ P43_VALIDATION_COMPLETE | Holds OOS |
| Delivery packager (P44) | ✅ P44_DELIVERY_READINESS_PACKAGED | Default: trusted_bgew_fusion |
| Profile registry (P46) | ✅ P46_CLAIM_GUARD_PASS | 3 profiles defined |
| One-command runner (P47) | ✅ P47_DELIVERY_CHAIN_PASS | --profile trusted_delivery |
| Docs (P48) | ✅ DELIVERY_FREEZE_READY | README + RUNBOOK + STATUS |
| Final audit (P49) | ✅ P49_FINAL_AUDIT_PASS | All checks pass |
| Version freeze (P50) | ✅ DELIVERY_FREEZE_READY | v3.0.0 |

## Delivery Metrics

| Metric | Value |
|--------|-------|
| cfg05 baseline sMAPE | 9.90% |
| Trusted BGEW fusion sMAPE | 9.23% (+6.79%) |
| Split OOS fusion | 10.12% (cfg05: 10.49%) |
| Rolling OOS fusion | 10.08% (cfg05: 10.76%) |

## Profile

- **Default**: `trusted_delivery` (cfg05 + catboost_spike_residual)
- **Research only**: `research_all_models` (includes stage3 — NOT for delivery)

## Quarantined

| Model | Reason |
|-------|--------|
| stage3_business_fixed | SUSPECT_LEAKAGE |
| best_two_average | CONSERVATIVE_CORR_GATE |
| catboost_sota | CONSERVATIVE_CORR_GATE |

## Forbidden Claims

- 2.97% production sMAPE
- 69.96% production improvement
- stage3 production readiness
- Source 11.48% reproduction

## Test Suite

1053 passing, 0 failing.
