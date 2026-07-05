# P91: Realtime Design Reclassification Report

> **Date**: 2026-07-05
> **Status**: COMPLETE
> **Verdict**: REALTIME_DA_SAFE_BASELINE

## 1. Summary

The realtime prediction design has been reclassified from a "da_anchor fallback
caveat" to an **official DA-Safe Realtime Baseline**. This is a naming and
positioning change only — no model weights were changed, no deep realtime
training was performed.

## 2. Old vs New State Naming

| Old (P65-P90) | New (P91) | Meaning |
|---|---|---|
| `REALTIME_DA_ANCHOR_FALLBACK` | `REALTIME_DA_SAFE_BASELINE` | rt_pred = da_anchor is the **official default**, not a fallback |
| `REALTIME_DEEP_READY_FAST_DEV` | `REALTIME_ASSIST_SGDFNET_AVAILABLE` | SGDFNet is available as an optional assist |
| `FAST_DEV_ONLY` | `REALTIME_ASSIST_DISABLED` | Only DA-Safe Baseline active; **not** NO_GO |
| *(new)* | `REALTIME_HYBRID_READY` | Both candidates available + ledger + learner PASS |

## 3. Realtime Status Rules

```
If rt_da_anchor is available AND final_output realtime_price has no NaN
AND safety checks PASS:
    realtime core = READY

If SGDFNet is unavailable:
    status = REALTIME_READY_DA_SAFE_ONLY

If SGDFNet is available AND passes ledger + learner:
    status = REALTIME_HYBRID_READY

If realtime_price is NaN:
    NO_GO
```

## 4. Enhanced Online Pack Schema

The online pack now includes assist/risk fields:

| Field | Source | Description |
|---|---|---|
| `trend_pred` | = rt_pred | Primary realtime prediction |
| `deep_rt_pred` | = da_anchor | Day-ahead anchor value |
| `sgdfnet_pred` | NaN (or SGDFNet) | SGDFNet assist prediction |
| `blend_pred` | = da_anchor | Blended realtime prediction |
| `da_anchor` | Day-ahead | Original day-ahead prediction |
| `da_error_prob` | computed | Estimated day-ahead error probability |
| `residual_direction_prob` | computed | Residual direction probability |
| `uncertainty_score` | computed | Prediction uncertainty (0-1) |
| `correction_permission` | computed | Whether correction is permitted |
| `reason_codes` | computed | Audit codes |

## 5. Files Modified

- `config/model_sets.yaml` — Added rt_da_anchor as official_default; added sgdfnet_rt_assist
- `models/realtime_state.py` — New module with state constants
- `models/adapters/realtime_da_safe_assist.py` — Enhanced output schema with assist fields
- `docs/reports/p91_realtime_design_reclassification_report.md` — This report

## 6. Key Design Decision

**The realtime design is now:**

```
rt_pred = da_anchor  (DA-Safe Baseline)
         +
SGDFNet Assist / sidecar (optional enhancement)
da_error_prob, residual_direction_prob, uncertainty_score,
correction_permission, reason_codes
```

SGDFNet is treated as an **optional enhancement**, not a replacement for the
DA-Safe Baseline. If SGDFNet is unavailable, the system still delivers with
full confidence via the DA-Safe Baseline.
