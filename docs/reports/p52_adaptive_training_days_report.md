## P52 Adaptive Complete Training Day Selector Report

### Status: P52_COMPLETE

### Objective

Implement the adaptive complete training day selector (ported from 2.5's
`ledger_weight.py`) for 3.0's model pool and profile system.  The selector
scans backwards from `target_date - 1` and collects the most recent N
complete training days for weight learning.

### What Was Implemented

- **`fusion/adaptive_training_days.py`** (core module)
  - `select_complete_training_days()` — the main function
  - Adaptive scanning with configurable `required_days` (default 30),
    `max_lookback_days` (default 180), `min_days_for_degraded` (default 7)

- **`tests/test_p52_adaptive_training_days.py`** (30+ tests)
  - Full contract coverage including COMPLETE_30D, DEGRADED_MIN_DAYS,
    INSUFFICIENT_DAYS, NO_VALID_DAYS, and all data quality checks

- **`fusion/__init__.py`** — updated to export `select_complete_training_days`

### Adaptive Training Day Logic for 3.0

For a target day D, scan from D-1 backwards:

1. For each calendar day, check **prediction ledger**:
   - Every **trusted model** (models with TRUSTED or DELIVERY_ALLOWED status)
     has non-zero rows
   - Each trusted model has hour_business 1..24 (complete day, no gaps)
   - No NaN in `y_pred` column
   - No duplicate keys on `(task, model_name, target_day, business_day, hour_business)`

2. Then check **actual ledger**:
   - Has 24 hour_business values (1..24)
   - No NaN in `y_true` column
   - No duplicate keys on `(task, target_day, business_day, hour_business)`

3. If both pass, the day is **complete** and added to `selected_days`.

#### Status levels

| Status               | Condition                                    |
|----------------------|----------------------------------------------|
| COMPLETE_30D         | >= required_days (default 30) found          |
| DEGRADED_MIN_DAYS    | >= min_days_for_degraded (default 7) found   |
| INSUFFICIENT_DAYS    | > 0 but < min_days_for_degraded found        |
| NO_VALID_DAYS        | 0 days, or ledger files not found / empty    |

#### Output dict

```python
{
    "status": "COMPLETE_30D",
    "selected_days": ["2026-06-30", "2026-06-29", ...],  # newest first
    "selected_count": 30,
    "skipped_days": [("2026-05-01", "models_with_nan_y_pred=['model_b']"), ...],
    "errors": [],
    "warnings": [],
    "latest_selected_day": "2026-06-30",
    "oldest_selected_day": "2026-06-01",
    "training_rows": 1440,   # 30 * n_models * 24
    "actual_rows": 720,       # 30 * 24
    "required_days": 30,
    "max_lookback_days": 180,
    "min_days_for_degraded": 7,
}
```

### Differences from 2.5

| Aspect               | 2.5                                        | 3.0                                                    |
|----------------------|---------------------------------------------|--------------------------------------------------------|
| **Input models**     | `expected_models` (hardcoded list)          | `trusted_models` (from trust gate / profile system)    |
| **Ledger format**    | CSV (read via pipeline helper)              | Parquet (direct `pd.read_parquet`)                     |
| **Status levels**    | PASS / FAIL (binary)                        | 4-level: COMPLETE_30D / DEGRADED_MIN_DAYS / INSUFFICIENT_DAYS / NO_VALID_DAYS |
| **Output richness**  | Basic (selected_days, errors)               | Full: skipped_days with reasons, training_rows/actual_rows, warnings, latest/oldest day |
| **Duplicate check**  | Not explicit (dedup handled by caller)      | Explicit check on prediction + actual key columns      |
| **Lookback default** | 90 days                                     | 180 days (wider scan)                                  |
| **Degraded mode**    | Not present                                 | DEGRADED_MIN_DAYS (>= 7 days but < required)           |
| **Realtime**         | Full support (separate model list)          | DA_ONLY/DRY_RUN — filtered to "dayahead" task only     |

### Test Results Summary

**Total tests: 27** — all passing.

| Test Class                    | # Tests | Coverage                                       |
|-------------------------------|---------|------------------------------------------------|
| TestComplete30D               | 3       | COMPLETE_30D status, row counts, clean output  |
| TestDegradedMinDays           | 3       | DEGRADED_MIN_DAYS with 7, 15 days, row counts |
| TestInsufficientDays          | 2       | INSUFFICIENT_DAYS with 3 and 1 day             |
| TestNoValidDays               | 4       | Missing prediction, actual, both, empty ledger |
| TestDataQualityDetection      | 5       | Missing hours, NaN y_pred, duplicates, missing models, skipped days |
| TestActualLedgerQuality       | 2       | NaN y_true, missing actual hours                |
| TestCustomParameters          | 5       | Custom required_days, max_lookback, min_days_for_degraded |
| TestTrustedModelsEdgeCases    | 3       | Empty trusted_models, single, three models     |

### Files

- `fusion/adaptive_training_days.py`
- `tests/test_p52_adaptive_training_days.py`
- `fusion/__init__.py` (updated)
