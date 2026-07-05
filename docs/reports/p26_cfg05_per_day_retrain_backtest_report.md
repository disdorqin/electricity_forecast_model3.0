## P26 cfg05 Per-Day Retrain Walk-Forward Backtest Report

### Status: P26_PER_DAY_RETRAIN_COMPLETE

### Objective

Convert P21's train-once-reuse strategy into a true per-day retrain walk-forward:
for each target_day D, train on [D-90d, D), predict 24H for D.

### Method

Delegates to P16 engine with `reuse_model=False`. Every day gets a fresh LightGBM model
trained strictly on data before D. No leakage.

### CLI

```bash
python -m scripts.run_p26_cfg05_per_day_retrain_backtest \
  --raw-data ../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv \
  --source-repo .local_artifacts/source_repos/epf-sota-experiment \
  --start-day 2026-06-01 --end-day 2026-06-30 \
  --train-window-days 90 \
  --work-dir .local_artifacts/p26_p30_fusion \
  --json --strict
```

### Output Schema

- attempted_days, complete_days, metric_days, eval_rows
- sMAPE_floor50, MAE, RMSE
- per_day_metrics, per_hour_metrics
- training_time_seconds, failed_days, reason_codes
- improvement_vs_p21 (delta_pp, direction)

### Comparison with P21

| Metric | P21 (train-once) | P26 (per-day retrain) |
|--------|-------------------|------------------------|
| Strategy | Train once before June, reuse all June | Retrain each day on [D-90d, D) |
| sMAPE_floor50 | 20.71% | **17.06%** |
| MAE | 68.04 | **53.89** |
| RMSE | 86.74 | **64.41** |
| Metric days | 30 | 29 (Jun 30 null y_true) |
| Eval rows | 696 | 696 |
| Direction | — | IMPROVED (-3.65pp) |

### Tests

15 tests in `tests/test_p26_cfg05_per_day_retrain_backtest.py` — all passing.

### Files

- `scripts/run_p26_cfg05_per_day_retrain_backtest.py`
- `tests/test_p26_cfg05_per_day_retrain_backtest.py`
