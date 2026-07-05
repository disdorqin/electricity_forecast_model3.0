# P93: Realtime Two-Candidate Prediction Ledger Report

> **Date**: 2026-07-05
> **Status**: COMPLETE

## 1. Summary

The realtime prediction ledger now supports two candidate models:
1. **rt_da_anchor** — DA-Safe Baseline (always available)
2. **sgdfnet_rt_assist** — SGDFNet Assist (optional)

## 2. Schema

| Column | Always | Description |
|---|---|---|
| `task` | Yes | "realtime" |
| `model_name` | Yes | "rt_da_anchor" or "sgdfnet_rt_assist" |
| `target_day` | Yes | Target day |
| `business_day` | Yes | Business day |
| `ds` | Yes | Timestamp |
| `hour_business` | Yes | 1-24 |
| `period` | Yes | 1_8, 9_16, 17_24 |
| `y_pred` | Yes | Prediction value |
| `source_confidence` | Yes | 0.0-1.0 |
| `model_version` | Yes | Version string |
| `da_error_prob` | Yes | Error probability |
| `residual_direction_prob` | Yes | Direction probability |
| `uncertainty_score` | Yes | Uncertainty score |
| `correction_permission` | Yes | Correction allowed |
| `reason_codes` | Yes | Audit codes |
| `run_id` | Meta | Run identifier |
| `created_at` | Meta | Creation timestamp |
| `updated_at` | Meta | Update timestamp |

## 3. Rules

- **rt_da_anchor** must always be present in a non-empty ledger.
- **sgdfnet_rt_assist** is optional.
- If SGDFNet is unavailable, only rt_da_anchor entries exist.
- If both models exist, both entries are created for the realtime learner.
- **y_true** is forbidden in the prediction ledger.
- Dedup key: `(task, model_name, target_day, business_day, hour_business)`.

## 4. Module Structure

- `ledgers/realtime_prediction_ledger.py` — Build, append, validate, extract
- `scripts/run_p93_realtime_two_candidate_ledger.py` — CLI runner
- `tests/test_p93_realtime_two_candidate_ledger.py` — Test suite
