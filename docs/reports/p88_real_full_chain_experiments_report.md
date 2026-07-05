# P88: Real Full-Chain Experiments Report

> **Generated**: 2026-07-07
> **Status**: P88_COMPLETE
> **Phase**: P88 — Real Full-Chain Experiment Battery

---

## 1. Overview

P88 executes 8 experiment scenarios against the full delivery chain to verify
correct behavior under realistic conditions, covering normal operation, missing
data paths, injection attacks, and safety enforcement. Each experiment exercises
a distinct failure mode or operational configuration and records the resulting
delivery status.

### Files

| File | Description |
|------|-------------|
| `scripts/run_p88_real_full_chain_experiments.py` | 8 experiments + orchestrator |
| `tests/test_p88_real_full_chain_experiments.py` | 24 tests for experiment harness |
| `docs/reports/p88_real_full_chain_experiments_report.md` | This report |

---

## 2. Experiment Results Summary

| Exp | Scenario | Verdict | Key Finding |
|-----|----------|---------|-------------|
| 1 | fast-dev-run | GO_WITH_CAVEATS | Fallbacks present in chain |
| 2 | missing realtime strict | NO_GO | Realtime data required but unavailable |
| 3 | da_anchor fallback non-strict | GO_WITH_CAVEATS | da_anchor fallback engaged |
| 4 | residual missing | GO_WITH_CAVEATS | CatBoost spike residual found as fallback |
| 5 | classifier missing | GO_WITH_CAVEATS | Rule fallback engaged |
| 6 | stage3 injection | NO_GO | Safety supervisor blocks |
| 7 | y_true injection | NO_GO | Safety supervisor blocks |
| 8 | current-day actual in weights | NO_GO | No-lookahead guard triggers |

**Overall**: 3 GO_WITH_CAVEATS, 0 clean GO, 5 NO_GO. The chain correctly
degrades or blocks under every adverse condition.

---

## 3. Experiment Details

### Exp1: Fast-Dev-Run

**Objective**: Verify the full chain runs end-to-end in fast-dev mode with
all components present but some operating in fallback configuration.

**Configuration**:
- `--profile trusted_delivery`
- `--fusion-engine period_bgew`
- `--allow-degraded`
- Real data: `data/shandong_pmos_hourly.csv` (39408 rows)
- Target date: 2026-06-30

**Result**: `GO_WITH_CAVEATS`

| Step | Status | Detail |
|------|--------|--------|
| raw_data_check | PASSED | 39408 rows loaded |
| trust_gate | OVERRIDDEN | 2 trusted, 3 quarantined |
| actual_ledger | EXISTING | 720 rows |
| prediction_ledger | EXISTING | 3600 rows |
| safety_preflight | PASSED | 0 blocked models |
| adaptive_training_days | DEGRADED | 29 days (min 30 required) |
| trusted_fusion | TRUSTED_FUSION_IMPROVED | 9.23% sMAPE dayahead |
| fallback_ladder | PASSED | DEGRADED_DELIVERED |
| postflight | WARNING | realtime_price NaN (dayahead-only) |
| claim_guard | PASSED | 0 violations |

**Caveats**:
- Realtime deep model is da_anchor fallback (not real realtime prediction)
- Residual model is CatBoost spike residual only (not full P5M stack)
- Classifier is rule-based fallback (not ML classifier)
- Adaptive training days degraded (29/30)

---

### Exp2: Missing Realtime Strict

**Objective**: Verify chain behavior when realtime data is completely absent
and strict mode is enabled (no `--allow-degraded`).

**Configuration**:
- `--strict` (no degraded mode allowed)
- Realtime columns removed from raw data
- `--strict-no-leakage`

**Result**: `NO_GO`

The chain correctly identifies that realtime data is missing and refuses to
produce output in strict mode. Exit code 1 (FAILED).

| Check | Status | Detail |
|-------|--------|--------|
| raw_data_check | FAILED | realtime_price column missing |
| postflight | BLOCKED | Cannot validate realtime fields |
| delivery_status | FAILED_NO_DELIVERY | Strict mode requires complete data |

**Analysis**: Strict mode correctly enforces data completeness. The chain
does not silently produce partial output when realtime data is absent.

---

### Exp3: DA Anchor Fallback Non-Strict

**Objective**: Verify that when the realtime deep model is unavailable, the
da_anchor fallback engages correctly in non-strict mode.

**Configuration**:
- `--allow-degraded`
- Realtime model predictions removed from prediction ledger
- da_anchor (dayahead-as-realtime proxy) fallback enabled

**Result**: `GO_WITH_CAVEATS`

| Check | Status | Detail |
|-------|--------|--------|
| realtime_model_check | FALLBACK | da_anchor engaged |
| fallback_ladder | DEGRADED_DELIVERED | Level 4 (cfg05_baseline) |
| output_validation | PASSED | 24 rows, no NaN in dayahead |
| postflight | WARNING | realtime derived from dayahead anchor |

**Caveats**:
- da_anchor uses dayahead predictions as realtime proxy
- This is explicitly NOT a real realtime deep model
- Accuracy degrades for realtime-specific patterns

---

### Exp4: Residual Missing

**Objective**: Verify chain behavior when the full P5M residual stack is
unavailable but CatBoost spike residual exists as a fallback.

**Configuration**:
- Full P5M residual models removed
- CatBoost spike residual retained in trusted pool
- `--allow-degraded`

**Result**: `GO_WITH_CAVEATS`

| Check | Status | Detail |
|-------|--------|--------|
| residual_check | PARTIAL | CatBoost spike residual found |
| P5M_stack_check | FALLBACK | Full P5M stack not available |
| fallback_ladder | DEGRADED_DELIVERED | Single-model residual |
| output_validation | PASSED | 24 rows valid |

**Caveats**:
- Only CatBoost spike residual is available (not the full P5M residual stack)
- Spike detection works for extreme price events but lacks multi-model residual
  diversity
- Full P5M residual stack requires additional model training

---

### Exp5: Classifier Missing

**Objective**: Verify chain behavior when the ML-based negative classifier is
unavailable and the rule-based fallback engages.

**Configuration**:
- ML classifier models (XGBoost, RandomForest) excluded from pipeline
- Rule-based classifier fallback enabled
- `--allow-degraded`

**Result**: `GO_WITH_CAVEATS`

| Check | Status | Detail |
|-------|--------|--------|
| classifier_check | FALLBACK | Rule-based classifier engaged |
| ML_classifier_check | NOT_LOADED | ML models exist but not in production path |
| output_validation | PASSED | Classification applied via rules |
| postflight | WARNING | Rule-based accuracy < ML target |

**Caveats**:
- Rule-based classifier uses threshold heuristics (price < median * 0.5)
- ML classifier models exist in `classifiers/` directory but are not loaded
  in the production delivery path
- Rule fallback has lower precision/recall than target ML classifier

---

### Exp6: Stage3 Injection

**Objective**: Verify that the safety supervisor correctly blocks stage3
(same-day business fixed) data when injected into the prediction ledger.

**Configuration**:
- stage3_business_fixed model added to prediction ledger
- `--strict-no-leakage`
- All other data valid

**Result**: `NO_GO`

| Check | Status | Detail |
|-------|--------|--------|
| leakage_sentinel | DETECTED | stage3_business_fixed flagged as SUSPECT_LEAKAGE |
| safety_preflight | BLOCKED | stage3 in blocked_models list |
| strict_no_leakage | FAILED | Blocked models present, strict mode active |
| delivery_status | FAILED_NO_DELIVERY | Safety override |

**Analysis**: The leakage sentinel correctly identifies stage3 as
SUSPECT_LEAKAGE via model name pattern matching. The safety supervisor
blocks the entire chain when `--strict-no-leakage` is active and blocked
models are detected. This is the intended behavior — stage3 contains
same-day information that would constitute lookahead leakage.

---

### Exp7: Y_True Injection

**Objective**: Verify that the safety supervisor correctly blocks when
y_true (actual prices) for the target day is found in the output weights
or prediction data.

**Configuration**:
- y_true values for target date injected into prediction ledger
- `--strict-no-leakage`
- All other data valid

**Result**: `NO_GO`

| Check | Status | Detail |
|-------|--------|--------|
| leakage_sentinel | DETECTED | y_true pattern found in prediction data |
| safety_preflight | BLOCKED | Leakage indicators present |
| strict_no_leakage | FAILED | Actuals leakage detected |
| delivery_status | FAILED_NO_DELIVERY | Safety override |

**Analysis**: The safety supervisor detects y_true contamination in the
prediction path and blocks the chain. This prevents the most dangerous
form of leakage — using actual outcomes as predictions. The no-lookahead
guard correctly propagates through all pipeline stages.

---

### Exp8: Current-Day Actual in Weights

**Objective**: Verify that the no-lookahead guard blocks when current-day
actual prices appear in the BGEW weight training data.

**Configuration**:
- Current-day actual prices included in weight training window
- `--strict-no-leakage`
- No-lookahead sentinel enabled

**Result**: `NO_GO`

| Check | Status | Detail |
|-------|--------|--------|
| no_lookahead_guard | TRIGGERED | Current-day actuals in weight window |
| safety_preflight | BLOCKED | Lookahead contamination detected |
| strict_no_leakage | FAILED | Weight training uses future data |
| delivery_status | FAILED_NO_DELIVERY | Safety override |

**Analysis**: The no-lookahead guard correctly identifies that BGEW weight
training must only use historical data strictly before the target date.
When current-day actuals are present in the weight training window, the
chain refuses to produce output. This ensures that fusion weights are
computed from genuinely out-of-sample data only.

---

## 4. Verdict Distribution

```
GO_WITH_CAVEATS:  3  (Exp1, Exp3, Exp4, Exp5)
NO_GO:            4  (Exp2, Exp6, Exp7, Exp8)
Clean GO:         0
```

The absence of a clean GO is expected: the current system has known fallback
components (realtime da_anchor, partial residual, rule classifier) that
prevent a clean production-ready verdict. The 4 NO_GO results confirm that
safety mechanisms correctly block leakage and data contamination.

---

## 5. Safety Mechanism Effectiveness

| Safety Mechanism | Tested By | Result |
|-----------------|-----------|--------|
| Leakage sentinel (model name) | Exp6 (stage3) | DETECTED |
| Leakage sentinel (y_true pattern) | Exp7 (y_true injection) | DETECTED |
| No-lookahead guard | Exp8 (current-day actual) | TRIGGERED |
| Strict mode enforcement | Exp2 (missing realtime) | BLOCKED |
| Claim guard | Exp1 (full run) | 0 violations |

All 5 safety mechanisms performed correctly across 8 experiments.

---

## 6. Test Summary

| Test File | Count | Status |
|-----------|-------|--------|
| `test_p88_real_full_chain_experiments.py` | 24 | ALL PASS |
| Full suite cumulative | 1836 | ALL PASS |

---

## 7. Conclusions

1. **Safety is solid**: All 4 NO_GO experiments confirm that safety mechanisms
   correctly block leakage, contamination, and missing-data scenarios.
2. **Fallbacks work but are degraded**: The 3 GO_WITH_CAVEATS experiments
   confirm that the chain produces output under degraded conditions, but
   with explicit caveats logged in the manifest.
3. **No clean GO yet**: The system cannot claim full production readiness
   due to da_anchor realtime fallback, partial residual stack, and
   rule-based classifier.
4. **Strict mode is trustworthy**: The `--strict` and `--strict-no-leakage`
   flags correctly enforce data completeness and leakage prevention.

---

## 8. Final Verdict

```
P88 OVERALL: GO_WITH_CAVEATS (safety PASS, production caveats remain)
```
