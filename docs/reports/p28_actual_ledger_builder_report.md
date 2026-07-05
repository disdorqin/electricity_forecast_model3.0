## P28 Actual Ledger Builder Report

### Status: P28_ACTUAL_LEDGER_READY

### Objective

Build actual ledger from raw CSV for BGEW / fusion learner training.

### Results

| Metric | Value |
|--------|-------|
| Total rows | 720 |
| Target days | 30 |
| Complete days (24H) | 30 |
| Duplicate keys | 0 |
| Null y_true rows | 24 |
| Null y_true days | 2026-06-30 |
| Hour business range | [1, 24] |
| Schema valid | true |

### Key Findings

1. **720 rows** generated for June 2026 (30 days x 24 hours).
2. **24 null y_true rows** on June 30 — the last day's actual prices are not yet available in the raw CSV.
3. **Zero duplicate keys** — each (task, target_day, business_day, hour_business) is unique.
4. **Schema valid** — all ACTUAL_LEDGER_COLUMNS present.
5. Output saved to `.local_artifacts/p26_p30_fusion/ledgers/actual_ledger.csv` (gitignored).

### Tests

10 tests in `tests/test_p28_actual_ledger_builder.py` — all passing.

### Files

- `scripts/build_actual_ledger_from_raw_csv.py`
- `tests/test_p28_actual_ledger_builder.py`
