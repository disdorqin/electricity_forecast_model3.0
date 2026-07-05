# P142 - 2.5 Fair Comparison Import Report

## Purpose

Perform a **fair, same-window comparison** between model 2.5 and model 3.0
(cfg05 + BGEW) on the 2025 dayahead prediction task. If 2.5 artefacts are
not available, report the status honestly without fabricating data.

## Design Principles

1. **Same window**: Both 2.5 and 3.0 are evaluated on exactly the same date
   range (default: 2025-01-01 to 2025-12-31).
2. **Same metric**: Canonical sMAPE_floor50 (both y_true and y_pred floored
   at 50 before computing sMAPE).
3. **No fabrication**: If 2.5 predictions cannot be found, the status is
   `2.5_COMPARISON_UNAVAILABLE` and `comparison` is `null`.

## Search Strategy

The script checks the following paths (relative to repo root):

| # | Path | Description |
|---|------|-------------|
| 1 | `../electricity_forecast_model2.5` | Sibling repo for model 2.5 |
| 2 | `../electricity_forecast_model2.0_exp` | Sibling experimental repo |
| 3 | `.local_artifacts/source_repos/electricity_forecast_model2.5` | Cloned 2.5 source |
| 4+ | `.local_artifacts/*` | Any sub-directory whose name contains "2.5", "2_5", "v2.5", etc. |

For each directory, the script looks for a CSV file containing at minimum
`y_pred` and `y_true` columns (with a date/ds column for window filtering).

## Outputs

All artefacts are written to `.local_artifacts/p142_fair_comparison/`:

| File | Description |
|------|-------------|
| `search_log.json` | Every path checked, whether it exists, and what was found |
| `comparison_metrics.json` | Full comparison result (or unavailability report) |

### When Available

```json
{
  "status": "2.5_COMPARISON_AVAILABLE",
  "comparison": {
    "model_25": { "smape_floor50": ..., "mae": ..., "rmse": ..., "count": ... },
    "model_30_cfg05": { "smape_floor50": ..., "mae": ..., "rmse": ..., "count": ... },
    "bgew": { ... },
    "window": { "start": "2025-01-01", "end": "2025-12-31" },
    "delta_smape_floor50": ...,
    "better_model": "2.5" | "3.0_cfg05"
  }
}
```

### When Unavailable

```json
{
  "status": "2.5_COMPARISON_UNAVAILABLE",
  "paths_checked": 4,
  "paths_with_predictions": 0,
  "comparison": null,
  "reason": "No model-2.5 prediction artefacts were found..."
}
```

## Running

```bash
python scripts/run_p142_25_fair_comparison_import.py
```

Or with custom parameters:

```bash
python scripts/run_p142_25_fair_comparison_import.py <output_dir> [day_start] [day_end]
```

## Tests

```bash
pytest tests/test_p142_25_fair_comparison_import.py -v
```

Minimum 8 tests covering search logic, unavailable path, no-fabrication
guarantee, available-path comparison format, same-window evaluation, and
search log completeness.
