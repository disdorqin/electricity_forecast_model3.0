# P137: Trusted 2025 Prediction Ledger V2 — Report

## Phase

P137 — Trusted 2025 Prediction Ledger V2

## Objective

Merge cfg05 and catboost_spike_residual predictions into a single,
prediction-only trusted ledger for the full year 2025.  The ledger is the
authoritative input for P138 rolling BGEW fusion.

## Inputs

| Source | Path | Rows |
|--------|------|------|
| cfg05 predictions | `.local_artifacts/p2025_full/dayahead/all_predictions.csv` | 8760 |
| catboost_spike predictions | `.local_artifacts/p136_catboost_spike_2025/catboost_spike_2025_predictions.csv` | 8737 |

## Outputs

| Artifact | Path |
|----------|------|
| Trusted ledger CSV | `.local_artifacts/p137_trusted_2025/ledger/dayahead_prediction_ledger_2025_trusted.csv` |
| Manifest JSON | `.local_artifacts/p137_trusted_2025/ledger/manifest.json` |

## Ledger Schema

| Column | Type | Description |
|--------|------|-------------|
| task | str | Always `dayahead` |
| model_name | str | `lightgbm_cfg05_dayahead` or `catboost_spike_residual` |
| business_day | str | Business day (YYYY-MM-DD) |
| ds | str | Wall-clock timestamp |
| hour_business | int | Business hour (1..24) |
| period | str | `1_8`, `9_16`, or `17_24` |
| y_pred | float | Model prediction |
| source_confidence | float | Confidence score |
| model_version | str | Model version identifier |

## Safety Checks

1. **model_count >= 2**: Ledger must contain predictions from at least 2 models.
   If only 1 model is available, status = `TRUSTED_LEDGER_BLOCKED_SINGLE_MODEL`.

2. **No y_true leakage**: The `y_true` column (and other forbidden columns) must
   NOT appear in the ledger.  cfg05 predictions include `y_true` which is
   explicitly stripped before merging.

3. **No NaN in y_pred**: All prediction values must be finite.

4. **business_day / hour_business present**: Required merge keys must exist and
   hour_business must be in range [1, 24].

## Status Codes

| Status | Meaning |
|--------|---------|
| `TRUSTED_LEDGER_V2_READY` | Ledger built successfully with >= 2 models |
| `TRUSTED_LEDGER_BLOCKED_SINGLE_MODEL` | Fewer than 2 models available |
| `TRUSTED_LEDGER_BLOCKED_NAN_YPRED` | NaN detected in y_pred |
| `TRUSTED_LEDGER_BLOCKED_MISSING_KEYS` | business_day / hour_business missing |
| `TRUSTED_LEDGER_BLOCKED_YTRUE_LEAK` | y_true or forbidden column detected |

## Key Design Decisions

- **y_true stripped**: The prediction ledger is a pure prediction artifact.
  Actuals are loaded separately from raw data in P138.
- **target_day derived for catboost_spike**: The catboost_spike predictions do
  not include a `target_day` column.  It is derived from `ds` using the
  convention: `target_day = date(ds) + 1 day`.
- **Column standardization**: Both sources are standardized to the same 9-column
  schema before merging.

## Running

```bash
python scripts/run_p137_trusted_2025_prediction_ledger_v2.py --json
```

## Tests

```bash
pytest tests/test_p137_trusted_2025_prediction_ledger_v2.py -v
```

12+ tests covering: model combination, y_true stripping, NaN checks,
single-model blocking, file existence, manifest structure, column schema,
business_day/hour_business correctness, forbidden column detection, date range,
rows-per-model accounting, and integration with real artifacts.
