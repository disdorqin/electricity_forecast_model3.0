# P17 cfg05 REAL Prediction Ledger Report

> **Phase**: P17 — cfg05 REAL predictions → prediction ledger
> **Generated**: 2026-07-04
> **Test count**: 772 total, 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Ledger conversion script | `scripts/run_p17_cfg05_predictions_to_ledger.py` — created |
| Canonical schema | `data/schema.py` PREDICTION_LEDGER_COLUMNS — reused |
| Business day logic | `data/business_day.py` — reused |
| Final status | **CFG05_PREDICTION_LEDGER_READY_LOCAL** (structural) |

## 2. Implementation

### Input

P16 local predictions (`all_predictions.csv`) or any DataFrame with standard prediction columns.

### Output

Canonical prediction ledger at:
```
.local_artifacts/p16_p20_cfg05_chain/ledgers/prediction_ledger.csv
```

### Production columns

```
task, model_name, target_day, business_day, ds, hour_business,
period, y_pred, source_confidence, model_version, run_id, created_at, updated_at
```

### Eval-only columns

```
y_true (only in eval mode, stripped in production mode)
```

### Validation rules

1. Each target_day must have 24 rows
2. hour_business must be 1..24
3. business_day must follow P15 business-day logic
4. No duplicate keys (task, model_name, target_day, business_day, hour_business)
5. Production mode: no y_true allowed
6. Eval mode: y_true permitted, report must label as eval-only

## 3. Summary Keys

```
input_rows
ledger_rows
target_days
complete_days
duplicate_keys
schema_valid
completeness_status
ledger_path_local
final_status
reason_codes
```

## 4. Final Statuses

| Status | Condition |
|--------|-----------|
| CFG05_PREDICTION_LEDGER_READY_LOCAL | Schema valid, all days complete |
| CFG05_PREDICTION_LEDGER_INCOMPLETE | Some days incomplete |
| CFG05_PREDICTION_LEDGER_INVALID | Schema invalid or empty |

## 5. Test Coverage (12 P17 tests)

| Group | Tests | Coverage |
|-------|-------|----------|
| TestLedgerConversion | 12 | Complete/incomplete days, missing hour 24, duplicate hours, canonical schema, production no y_true, eval with y_true, hour range, empty input, row counts, summary keys |

## 6. Files Changed/Created

| File | Action |
|------|--------|
| `scripts/run_p17_cfg05_predictions_to_ledger.py` | **NEW** |
| `tests/test_p17_cfg05_predictions_to_ledger.py` | **NEW** (12 tests) |
| `docs/reports/p17_cfg05_real_prediction_ledger_report.md` | **NEW** (this file) |

---

*End of P17 report. 772 tests total, 0 failures.*
