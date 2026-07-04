# P16 cfg05 30-day Walk-Forward Backtest Report

> **Phase**: P16 — cfg05 30-day 24H walk-forward backtest
> **Generated**: 2026-07-04
> **Test count**: 772 total (713 prior + 16 P16 new + 43 P17-P19), 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Walk-forward script | `scripts/run_p16_cfg05_30d_walkforward_backtest.py` — created |
| Shared day-ahead window | `artifacts/dayahead_window.py` — reused (P15) |
| Hour-24 completeness | `scripts/check_cfg05_hour24_completeness.py` — reused (P15) |
| Raw data contract | `scripts/check_cfg05_raw_data_contract.py` — reused (P13) |
| Metric computation | sMAPE_floor50 / MAE / RMSE — implemented |
| Final status | **CFG05_BACKTEST_BLOCKED** (no real data in CI) |

## 2. Implementation

### Walk-forward logic

For each target day D in [start_day, end_day]:

1. Train window: [D - train_window_days, D) — strictly before D (no leakage)
2. Prediction window: [D 01:00, D+1 01:00) — canonical 24H day-ahead
3. Uses `filter_dayahead()` from P15 shared helper (no reimplementation)
4. Hour-24 completeness check on each day's predictions
5. Incomplete days excluded from metrics

### Two modes

- **reuse_model** (default): Train once on pre-start data, predict all days
- **no_reuse_model** (--no-reuse-model): Retrain per day (true walk-forward)

### Metric computation

- sMAPE_floor50: `200 * mean(|y_f - yp_f| / (|y_f| + |yp_f|))` with floor=50
- MAE: `mean(|y - yp|)`
- RMSE: `sqrt(mean((y - yp)^2))`
- Only complete days (24H) and valid y_true rows enter metrics

## 3. Summary Keys

```
raw_data_status
eval_start
eval_end
attempted_days
complete_days
metric_days
eval_rows
missing_y_true_rows
incomplete_days
metrics
per_day_metrics_path_local
per_hour_metrics_path_local
predictions_path_local
final_status
source_reproduction_claim
reason_codes
forbidden_files_check
```

## 4. Source Reproduction Claim

```
source 11.48% reproduction not claimed
```

Reason: evaluation window, walk-forward method, training strategy, and feature builder are not verified to match source methodology. P19 will perform a formal audit.

## 5. Final Statuses

| Status | Condition |
|--------|-----------|
| CFG05_BACKTEST_COMPLETE | All days complete, metrics computed |
| CFG05_BACKTEST_INCOMPLETE | Some days incomplete or errors |
| CFG05_BACKTEST_BLOCKED | Raw data missing or contract failed |
| CFG05_BACKTEST_NO_VALID_YTRUE | No valid y_true rows after merge |

## 6. Test Coverage (16 P16 tests)

| Group | Tests | Coverage |
|-------|-------|----------|
| TestMetrics | 6 | sMAPE_floor50, MAE, RMSE, floor effect, empty arrays |
| TestPathSafety | 4 | Safe/unsafe path validation |
| TestRawDataLoading | 1 | Raw CSV loading with y_true extraction |
| TestBacktestCore | 5 | Missing data blocked, claim present, eval range, forbidden files, summary keys |

## 7. Files Changed/Created

| File | Action |
|------|--------|
| `scripts/run_p16_cfg05_30d_walkforward_backtest.py` | **NEW** |
| `tests/test_p16_cfg05_30d_walkforward_backtest.py` | **NEW** (16 tests) |
| `docs/reports/p16_cfg05_30d_walkforward_backtest_report.md` | **NEW** (this file) |

---

*End of P16 report. 772 tests total, 0 failures.*
