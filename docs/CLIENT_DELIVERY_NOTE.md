# Client Delivery Note — Electricity Price Forecasting System v3.0.0-rc1-p143

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

## Key Metrics (Verified — P143)

| Component | sMAPE | Period | Notes |
|-----------|-------|--------|-------|
| cfg05-only day-ahead sMAPE (2025 full year) | 20.22% | 2025-01-01 to 2025-12-31 |  |
| Realtime DA-Safe Baseline sMAPE (2025 full year) | 33.03% | 2025-01-01 to 2025-12-31 | DA-Safe Baseline only (rt_pred = da_anchor), no SGDFNet assi |
| Trusted BGEW fusion sMAPE (June 2026 local window) | 9.23% | 2026-06 local window (NOT | LOCAL WINDOW ONLY — not comparable to 2025 full-year cfg05-o |
| Residual-corrected BGEW sMAPE (2025) | 19.3475% | 2025-01-01 to 2025-12-31 |  |
| Improved realtime sMAPE (2025) | 17.3472% | 2025-01-01 to 2025-12-31 |  |

## Current Status

`PERFORMANCE_IMPROVED_WITH_CAVEATS`

## Important Caveats

See `CLIENT_CAVEATS.md` for full details.
