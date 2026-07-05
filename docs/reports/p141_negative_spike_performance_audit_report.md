# P141 - Negative / Spike Specialized Performance Audit Report

## Purpose

Identify **where forecast error comes from** by classifying every hour of the
2025 cfg05 dayahead predictions into price-regime categories and computing
dedicated error metrics for each.

## Categories

| Category   | Condition             | Description                        |
|------------|-----------------------|------------------------------------|
| negative   | y_true < 0            | Negative price hours               |
| spike      | y_true > 500          | Price spike hours                  |
| low_price  | 0 <= y_true < 50      | Very low price hours               |
| high_price | y_true >= 300         | High price hours (non-spike range) |
| normal     | 50 <= y_true < 300    | Normal operating range             |

## Method

1. Load `.local_artifacts/p2025_full/dayahead/all_predictions.csv` (8760 rows, y_true merged).
2. Classify each hour using the rules above (priority: negative > spike > low_price > high_price > normal).
3. Compute **sMAPE_floor50**, **MAE**, **RMSE**, and **count** per category.
4. Compute per-period (1_8, 9_16, 17_24) and per-month breakdowns.
5. Extract top-50 worst error hours by absolute error.
6. Build 24-row hourly heatmap (avg error per hour-of-day).

## Metric: sMAPE_floor50

Both y_true and y_pred are floored at 50 before computing sMAPE. This prevents
extreme percentage errors on near-zero prices from dominating the metric while
still penalising large absolute deviations.

```
sMAPE_floor50 = 200 * mean(|max(y_true,50) - max(y_pred,50)| / (|max(y_true,50)| + |max(y_pred,50)|))
```

## Outputs

All artefacts are written to `.local_artifacts/p141_negative_spike/`:

| File                          | Description                                    |
|-------------------------------|------------------------------------------------|
| `normal_hours_metrics.json`   | sMAPE_floor50, MAE, RMSE, count for normal     |
| `negative_hours_metrics.json` | Same for negative-price hours                  |
| `spike_hours_metrics.json`    | Same for spike-price hours                     |
| `low_price_metrics.json`      | Same for low-price hours                       |
| `high_price_metrics.json`     | Same for high-price hours                      |
| `hourly_heatmap.json`         | 24 entries: hour, avg_error, count             |
| `top_50_error_hours.csv`      | Top-50 worst hours: ds, y_true, y_pred, etc.   |
| `audit_summary.json`          | Full summary answering "where does error come from?" |

## Key Questions Answered

1. **Where does error come from?** -- `category_error_share_pct` shows the
   percentage of total absolute error attributable to each category.
2. **Which category is worst?** -- `worst_error_category` names the single
   largest contributor.
3. **Which hours of the day are hardest?** -- `hourly_heatmap.json` ranks the
   24 hours by average absolute error.
4. **Which months are hardest?** -- `month_breakdown` in the audit summary.
5. **What are the single worst predictions?** -- `top_50_error_hours.csv`.

## Running

```bash
python scripts/run_p141_negative_spike_performance_audit.py
```

Or with custom paths:

```bash
python scripts/run_p141_negative_spike_performance_audit.py \
    <predictions_csv> <raw_data_csv> <output_dir>
```

## Tests

```bash
pytest tests/test_p141_negative_spike_performance_audit.py -v
```

Minimum 10 tests covering classification logic, metric computation, output
format validation, and integration-level audit summary checks.
