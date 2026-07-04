# P21 Local Real Data Backtest Report

> **Phase**: P21 — Local data discovery + real P16 execution
> **Generated**: 2026-07-04

## Status: P21_REAL_BACKTEST_COMPLETE

| Item | Result |
|------|--------|
| Raw CSV | FOUND: `../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv` (39408 rows) |
| Source repo | FOUND: `.local_artifacts/source_repos/epf-sota-experiment` |
| Eval window | 2026-06-01 ~ 2026-06-30 (30 days) |
| Complete days | 30/30 (all 24H) |
| Eval rows | 696 |
| sMAPE_floor50 | 20.71% |
| MAE | 68.04 |
| RMSE | 86.74 |
| Source claim | NOT claimed |

## Files Created
- `scripts/run_p21_local_real_data_backtest.py`
- `tests/test_p21_local_real_data_backtest.py` (17 tests)
- `docs/reports/p21_local_real_data_backtest_report.md`
