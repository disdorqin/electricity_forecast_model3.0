# Client Delivery Note — Electricity Price Forecasting System v3.0.0-rc1

## What This Is

A production-oriented multi-model fusion system for day-ahead and real-time electricity price forecasting in the Shandong PMOS market.

## What It Does

- Combines multiple ML models (LightGBM, CatBoost) via BGEW fusion
- Produces 24-hour day-ahead price predictions
- Produces DA-Safe real-time price predictions
- Supports optional SGDFNet neural network assist
- Includes residual correction, risk classification, and safety supervision

## How to Run

```bash
python main.py \
    --raw-data data/shandong_pmos_hourly.csv \
    --dayahead-source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --fusion-engine period_bgew \
    --work-dir .local_artifacts/production_run \
    --strict --strict-no-leakage \
    --json
```

## Key Metrics

| Component | sMAPE |
|-----------|-------|
| cfg05 baseline | 9.90% |
| Trusted BGEW fusion | 9.23% |
| DA-Safe Realtime | Metric pending / evaluated separately |

## Current Status

`FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS`

## Important Caveats

See `CLIENT_CAVEATS.md` for full details.
