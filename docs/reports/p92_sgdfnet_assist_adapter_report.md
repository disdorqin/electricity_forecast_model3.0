# P92: SGDFNet Assist Adapter Report

> **Date**: 2026-07-05
> **Status**: CODE_ONLY (SGDFNet import requires 2.0 exp repo)
> **Adapter**: SGDFNetAssistAdapter

## 1. Summary

The SGDFNet assist adapter wraps the SGDFNet model from the 2.0 experiment
repository into the 3.0 adapter contract. It is an **optional** realtime
candidate — the system delivers fully without it.

## 2. Adapter Status

| Status | Meaning |
|---|---|
| `SGDFNET_ASSIST_READY` | SGDFNet imported and executable |
| `SGDFNET_ASSIST_CODE_ONLY` | Adapter code exists but 2.0 repo not found |
| `SGDFNET_ASSIST_BLOCKED` | Import or runtime error |

**Current status**: `SGDFNET_ASSIST_CODE_ONLY` — the 2.0 experiment repo is not
available at the expected path. The adapter code is complete and ready for when
the repo becomes available.

## 3. Output Schema

The adapter produces two files:

### `sgdfnet_realtime_assist_pack.csv`

| Column | Type | Description |
|---|---|---|
| `business_day` | str | Business day |
| `ds` | datetime | Timestamp |
| `hour_business` | int | 1-24 |
| `period` | str | 1_8, 9_16, 17_24 |
| `model_name` | str | sgdfnet_rt_assist |
| `rt_pred` | float | Realtime prediction |
| `sgdfnet_pred` | float | SGDFNet raw prediction |
| `da_anchor` | float | Day-ahead anchor |
| `assist_available` | bool | Whether SGDFNet was available |
| `source_confidence` | float | 0.0-1.0 |
| `da_error_prob` | float | Error probability |
| `residual_direction_prob` | float | Residual direction |
| `uncertainty_score` | float | Uncertainty |
| `correction_permission` | bool | Correction allowed |
| `reason_codes` | str | Audit codes |

### `sgdfnet_realtime_assist_manifest.json`

Contains model name, version, status, row count, column list.

## 4. Behavior When SGDFNet Unavailable

- `sgdfnet_pred` = NaN
- `rt_pred` = da_anchor (safe baseline)
- `assist_available` = False
- `correction_permission` = False
- `reason_codes` = SGDFNET_ASSIST_DISABLED

## 5. No y_true Contract

The assist pack never contains y_true, in accordance with the 3.0 production
contract.
