# P18 cfg05 REAL Full Chain Local Report

> **Phase**: P18 — cfg05 REAL → residual → fusion → final local full chain
> **Generated**: 2026-07-04
> **Test count**: 772 total, 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Full chain script | `scripts/run_p18_cfg05_real_full_chain_local.py` — created |
| Residual correction | P5M_DATA_MISSING_NO_OP — honest fallback |
| Fusion | CFG05_SINGLE_REAL_MODEL_FUSION — single model |
| Negative classifier | NEGATIVE_CLASSIFIER_RULE_FALLBACK — rule-based |
| Row count consistency | N*24 → N*24 → N*24 → N*24 — verified |
| Final status | **CFG05_FULL_CHAIN_READY_WITH_FALLBACKS** |

## 2. Chain Flow

```
prediction ledger (P17)
    → residual correction (DATA_MISSING no-op)
    → corrected ledger
    → fusion (cfg05 single model, equal_weight)
    → fusion ledger
    → negative classifier (rule fallback)
    → final output
```

## 3. Fallback Labels (Honest)

| Stage | Label | Reason |
|-------|-------|--------|
| Residual | P5M_DATA_MISSING_NO_OP | No P5M pack, no risk data |
| Fusion | CFG05_SINGLE_REAL_MODEL_FUSION | Only cfg05 available |
| Classifier | NEGATIVE_CLASSIFIER_RULE_FALLBACK | No ExtremPriceClf artifact |

**Explicitly NOT claimed:**
- P5M REAL
- BGEW production
- ExtremPriceClf deployed

## 4. Row Count Expectations

For N complete days:

```
prediction rows = N * 24
corrected rows  = N * 24
fusion rows     = N * 24
final rows      = N * 24
```

## 5. Summary Keys

```
input_prediction_rows
corrected_rows
fusion_rows
final_rows
validators_passed
residual_mode
fusion_mode
classifier_mode
prediction_ledger_path_local
corrected_ledger_path_local
fusion_ledger_path_local
final_output_path_local
readiness_label
final_status
reason_codes
forbidden_files_check
```

## 6. Final Statuses

| Status | Condition |
|--------|-----------|
| CFG05_FULL_CHAIN_READY_LOCAL | All validators pass, no fallbacks |
| CFG05_FULL_CHAIN_READY_WITH_FALLBACKS | Chain works but with fallback labels |
| CFG05_FULL_CHAIN_BLOCKED | Input missing or stage failed |
| CFG05_FULL_CHAIN_INVALID | Output invalid |

## 7. Test Coverage (15 P18 tests)

| Group | Tests | Coverage |
|-------|-------|----------|
| TestFullChain | 13 | Residual/fusion/classifier fallback labels, row counts, schema validation, empty/no input, validators, readiness, forbidden files, summary keys |
| TestPreparePredictionInput | 2 | y_true stripping, task column ensurance |

## 8. Files Changed/Created

| File | Action |
|------|--------|
| `scripts/run_p18_cfg05_real_full_chain_local.py` | **NEW** |
| `tests/test_p18_cfg05_real_full_chain_local.py` | **NEW** (15 tests) |
| `docs/reports/p18_cfg05_real_full_chain_local_report.md` | **NEW** (this file) |

---

*End of P18 report. 772 tests total, 0 failures.*
