# P54: Fallback Ladder Report

## Implementation Overview

The fallback ladder (`delivery/fallback_ladder.py`) implements the 3.0 delivery
fallback mechanism. It provides a six-level degradation path so that the delivery
pipeline always produces a valid 24-hour output under any failure scenario, down
to the last-resort historical median.

The module exposes one public function:

```python
def run_fallback_ladder(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger_path: str,
    actual_ledger_path: str,
    raw_data_path: str | None = None,
) -> dict:
```

It reads from:
- **Prediction ledger** (CSV) — raw or corrected model predictions per hour
- **Actual ledger** (CSV) — historical actual prices for BGEW weight training
- **Raw data file** (CSV/Excel) — historical raw dayahead prices for the
  median fallback (optional)

Five private helper functions implement each level:

| Level | Function |
|-------|----------|
| 1 | `_try_trusted_bgew_fusion()` |
| 2 | `_try_trusted_equal_weight()` |
| 3 | `_try_best_trusted_single()` |
| 4 | `_try_cfg05_baseline()` |
| 5 | `_try_historical_median()` |

Each returns a dict with `level`, `method`, `success`, `reason`, and optionally
`output` (a 24-row DataFrame). The main function orchestrates them in order,
running postflight validation after each successful attempt.

---

## Fallback Ladder Levels

| Level | Method | Description | Delivery Status |
|-------|--------|-------------|-----------------|
| 1 | `trusted_bgew_fusion` | BGEW inverse-error weighting over trusted models | NORMAL |
| 2 | `trusted_equal_weight` | Simple average across all trusted models | DEGRADED_DELIVERED |
| 3 | `best_trusted_single_model` | Best single model by recent MAE against actuals | DEGRADED_DELIVERED |
| 4 | `cfg05_baseline` | cfg05 champion model output (always available) | DEGRADED_DELIVERED |
| 5 | `historical_same_hour_median` | Per-hour median of historical raw prices | DEGRADED_DELIVERED |
| 6 | `FAILED_NO_DELIVERY` | All levels failed | FAILED_NO_DELIVERY |

---

## Delivery Status Rules

| Condition | Status |
|-----------|--------|
| Level 1 succeeds AND postflight PASS | **NORMAL** |
| Levels 2-5 produce valid 24H output with no NaN and valid schema | **DEGRADED_DELIVERED** |
| All fallback levels fail | **FAILED_NO_DELIVERY** |

Postflight validation checks that the output:
- Has exactly 24 rows
- Contains all required schema columns (`business_day`, `ds`, `hour_business`,
  `period`, `dayahead_price`, `realtime_price`)
- Has all `hour_business` values 1..24 with no duplicates
- Has no NaN in `dayahead_price`

---

## Output Format

Each attempt produces a DataFrame with 24 rows:

| Column | Type | Description |
|--------|------|-------------|
| `business_day` | datetime | Business day |
| `ds` | datetime | Wall-clock timestamp |
| `hour_business` | int | Hour 1..24 |
| `period` | str | Period bucket (`1_8`, `9_16`, `17_24`) |
| `dayahead_price` | float | Predicted day-ahead price |
| `realtime_price` | float or None | Predicted real-time price (None for dayahead tasks) |

---

## Key Design Decisions

1. **BGEW weight reuse** — Level 1 delegates to `fusion.weights.bgew_skeleton`
   for inverse-error weighting, consistent with the P4 fusion engine.

2. **Best-model selection** — Level 3 computes MAE over recent historical data
   shared between the prediction and actual ledgers, selecting the model with
   the lowest error. Falls back to the first trusted model when actuals are
   unavailable.

3. **Leakage-aware historical median** — Level 5 explicitly filters to rows
   where `business_day < target_date` before computing per-hour medians,
   preventing future-data leakage.

4. **cfg05 champion guarantee** — Level 4 always looks for a model whose name
   contains "cfg05" (case-insensitive) in the prediction ledger, ensuring
   the champion baseline is always available.

5. **Empty-tolerance** — All helper functions return `None` or a failure dict
   rather than raising exceptions, allowing the main ladder to gracefully
   cascade through all levels.

---

## Test Summary

| Test | File | Count |
|------|------|-------|
| `TestValidateFallbackOutput` | `test_p54_fallback_ladder.py` | 6 |
| `TestRunPostflight` | `test_p54_fallback_ladder.py` | 2 |
| `TestBuildOutputFromPredictions` | `test_p54_fallback_ladder.py` | 2 |
| `TestTryTrustedBgewFusion` | `test_p54_fallback_ladder.py` | 3 |
| `TestTryTrustedEqualWeight` | `test_p54_fallback_ladder.py` | 2 |
| `TestTryBestTrustedSingle` | `test_p54_fallback_ladder.py` | 2 |
| `TestTryCfg05Baseline` | `test_p54_fallback_ladder.py` | 2 |
| `TestTryHistoricalMedian` | `test_p54_fallback_ladder.py` | 3 |
| `TestRunFallbackLadder` | `test_p54_fallback_ladder.py` | 8 |
| `TestFallbackLevelNames` | `test_p54_fallback_ladder.py` | 1 |
| **Total** | | **31** |

All tests pass across:
- Validation edge cases (23 rows, NaN, duplicates, wrong schema)
- Per-level success paths (BGEW, equal weight, best single, cfg05, historical
  median)
- Per-level failure paths (missing data, empty ledgers, no price column)
- Full ladder end-to-end scenarios (NORMAL, DEGRADED_DELIVERED,
  FAILED_NO_DELIVERY)
- State tracking (attempt records, warning accumulation)
- Edge cases (empty input, single-model trusted pool)

---

## Files

| File | Description |
|------|-------------|
| `delivery/fallback_ladder.py` | Fallback ladder implementation |
| `delivery/__init__.py` | Package init; exports `run_fallback_ladder` |
| `tests/test_p54_fallback_ladder.py` | 31 tests covering all levels and validation |
