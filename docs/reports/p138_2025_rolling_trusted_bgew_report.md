# P138: 2025 Rolling Trusted BGEW — Report

## Phase

P138 — 2025 Rolling Trusted BGEW (Bayesian-Guided Exponential Weighting)

## Objective

Perform walk-forward (rolling) BGEW fusion over the full 2025 year using the
trusted prediction ledger from P137 and actuals from the raw Shandong PMOS data.
For each target day, the learner looks back 30 days, computes per-model
sMAPE_floor50, derives BGEW weights, and fuses predictions — all without
lookahead.

## Inputs

| Source | Path | Description |
|--------|------|-------------|
| Trusted ledger | `.local_artifacts/p137_trusted_2025/ledger/dayahead_prediction_ledger_2025_trusted.csv` | P137 output |
| Raw data | `data/shandong_pmos_hourly.csv` | GBK-encoded Shandong PMOS data |

## Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| Daily metrics | `.local_artifacts/p138_rolling_bgew/daily_metrics.csv` | Per-day sMAPE, MAE, weights |
| Weights | `.local_artifacts/p138_rolling_bgew/weights.csv` | Per-day per-model weights |
| Weight summary | `.local_artifacts/p138_rolling_bgew/model_weight_summary.json` | Mean weights per model |
| Overall metrics | `.local_artifacts/p138_rolling_bgew/bgew_2025_metrics.json` | Full-year aggregate metrics |

## Algorithm

For each target_day D in 2025:

1. **History window**: Gather predictions + actuals for business_days D-30 .. D-1.
   No-lookahead invariant: only days strictly before D are used.

2. **Per-model sMAPE_floor50**: Compute the canonical sMAPE over the 30-day window.

   ```
   sMAPE_floor50 = 200 * mean(|y_f - yp_f| / (|y_f| + |yp_f|))
   where y_f = max(y, 50), yp_f = max(yp, 50)
   ```

3. **BGEW weight learning**: Apply `compute_bgew_weights()`:
   - score = exp(-alpha * smape) for each model
   - weight = score / sum(scores)
   - Clip to [0.05, 0.75]
   - Renormalize

4. **Fuse**: Weighted average of model predictions for day D.

5. **Evaluate**: Compare fused prediction against actuals.

### Fallback

If fewer than 14 history days are available, use equal weights (1/N per model).

## daily_metrics.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| target_day | str | Business day |
| cfg05_smape | float | cfg05 sMAPE_floor50 for this day |
| catboost_smape | float | catboost_spike sMAPE_floor50 for this day |
| bgew_smape | float | Fused BGEW sMAPE_floor50 for this day |
| cfg05_mae | float | cfg05 MAE |
| catboost_mae | float | catboost_spike MAE |
| bgew_mae | float | Fused BGEW MAE |
| cfg05_weight | float | BGEW weight for cfg05 |
| catboost_weight | float | BGEW weight for catboost_spike |

## Status Codes

| Status | Meaning |
|--------|---------|
| `BGEW_2025_IMPROVED` | BGEW sMAPE < cfg05 sMAPE (fusion helps) |
| `BGEW_2025_NOT_IMPROVED` | BGEW sMAPE >= cfg05 sMAPE |
| `BGEW_2025_BLOCKED` | Cannot run (single model, missing data) |

## Key Design Decisions

- **Canonical sMAPE_floor50**: Uses floor=50, NOT the P129 formula.  This
  prevents artificially low sMAPE from near-zero prices.
- **Merge on business_day + hour_business**: The raw data's `日前电价` maps to
  y_true for dayahead.  Merge keys are business_day + hour_business (not ds).
- **30-day lookback**: Matches the P94 realtime pooled learner convention.
- **14-day minimum**: If fewer than 14 complete days are available, falls back
  to equal weights to avoid overfitting on tiny samples.

## Running

```bash
python scripts/run_p138_2025_rolling_trusted_bgew.py --json
```

## Tests

```bash
pytest tests/test_p138_2025_rolling_trusted_bgew.py -v
```

12+ tests covering: sMAPE_floor50 formula correctness, BGEW weight validity,
no-lookahead invariant, equal-weight fallback, daily_metrics format,
improvement calculation, single-model blocking, output file existence,
weight summary structure, fuse_predictions helper, compute_model_smape helper,
and integration with real artifacts.
