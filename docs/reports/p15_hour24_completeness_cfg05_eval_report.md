# P15 Hour-24 Completeness & cfg05 Evaluation Report

> **Phase**: P15 — Fix 23→24 hour filter, hour-24 completeness checker, optional historical evaluation
> **Generated**: 2026-07-04
> **Test count**: 713 total (675 prior + 38 P15 new), 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Shared day-ahead window helper | **artifacts/dayahead_window.py** — created |
| Hour-24 completeness checker | **scripts/check_cfg05_hour24_completeness.py** — created |
| P15 orchestration | **scripts/run_p15_cfg05_24h_smoke_and_eval.py** — created |
| cfg05 adapter 24h filter | **Fixed** — uses `filter_dayahead()` shared helper |
| Train/export 24h filter | **Fixed** — uses `day_ahead_mask()` shared helper |
| Hour-24 fix verification | **COMPLETE_24H** — all 24 rows present |
| Historical evaluation | **Optional** — sMAPE_floor50 / MAE / RMSE |
| Final status | **CFG05_REAL_READY_24H_LOCAL** (when 24 rows + validator pass) |

## 2. The 23-Hour Bug

### Root Cause

The original day-ahead filter used an **exclusive end** of `D+1 00:00`:

```python
# OLD (broken) — excludes D+1 00:00 (hour 24)
start = target_dt + pd.Timedelta(hours=1)       # D 01:00
end   = target_dt + pd.Timedelta(days=1)          # D+1 00:00
mask = (ds >= start) & (ds < end)                 # only 23 hours!
```

Because `D+1 00:00` is strictly less than the end boundary, it gets excluded. This drops hour 24 (`hour_business=24`), producing 23 rows instead of 24.

### Fix

Changed the exclusive end to `D+1 + 1h`:

```python
# NEW (fixed) — includes D+1 00:00 as hour 24
start = target_dt + pd.Timedelta(hours=1)         # D 01:00
end   = target_dt + pd.Timedelta(days=1, hours=1) # D+1 01:00
mask = (ds >= start) & (ds < end)                  # all 24 hours!
```

This ensures the canonical 24-hour window `[D+01:00, D+1+01:00)` covers all 24 business hours.

### `get_business_day_info` Bug Fix

The original `hour_business` calculation used `ds.hour + 1` directly, but this didn't align with the 1-hour back-shift used for `business_day`. For example:

- Timestamp `D+1 00:00`: `ds.hour = 0` → `hour_business = 1` **(wrong, should be 24)**
- Timestamp `D 01:00`: `ds.hour = 1` → `hour_business = 2` **(wrong, should be 1)**

**Fix**: Apply the same 1-hour shift to `hour_business`:

```python
shifted = ds - pd.Timedelta(hours=1)
hour_business = shifted.dt.hour + 1  # 0→1, 23→24
```

Now:
- `D+1 00:00`: `shifted = D 23:00` → `shifted.hour = 23` → `hour_business = 24` ✓
- `D 01:00`: `shifted = D 00:00` → `shifted.hour = 0` → `hour_business = 1` ✓

## 3. Shared Helper: `artifacts/dayahead_window.py`

A new shared module that all filter paths must use to prevent filter drift:

| Function | Purpose |
|----------|---------|
| `get_dayahead_window(target_day)` | Returns `(start, end_exclusive)` = `(D+01:00, D+1+01:00)` |
| `day_ahead_mask(df, target_day)` | Boolean mask for the 24-hour window |
| `filter_dayahead(df, target_day)` | Filtered + sorted copy |
| `get_business_day_info(ds_series)` | `business_day`, `hour_business` (1..24), `period` |

**Updated consumers:**
- `models/adapters/cfg05_dayahead_lgbm.py` — `predict()` uses `filter_dayahead()`
- `scripts/train_export_cfg05_local.py` — feature input uses `day_ahead_mask()`

## 4. Hour-24 Completeness Checker

**Script**: `scripts/check_cfg05_hour24_completeness.py`

Verifies that a day-ahead prediction or feature CSV contains exactly 24 rows for the target day.

### Status Codes

| Status | Description |
|--------|-------------|
| `COMPLETE_24H` | All 24 hours present (1..24), no duplicates |
| `INCOMPLETE_23H` | Only hour 24 missing — characteristic of old exclusive-end filter |
| `MISSING_HOURS` | One or more hours missing (not just hour 24) |
| `DUPLICATE_HOURS` | Duplicate hour entries found |
| `INVALID` | File missing, no ds column, read failure, etc. |

### Usage

```bash
python -m scripts.check_cfg05_hour24_completeness \
    --input predictions.csv --target-day 2026-06-30

python -m scripts.check_cfg05_hour24_completeness \
    --input features.csv --target-day 2026-06-30 --json --strict
```

## 5. P15 Orchestration

**Script**: `scripts/run_p15_cfg05_24h_smoke_and_eval.py`

One-command pipeline:

```
raw CSV → contract check → train/export → REAL smoke
→ hour24 completeness (features + predictions)
→ optional historical eval → final status
```

### Final Statuses

| Status | Condition |
|--------|-----------|
| `CFG05_REAL_READY_24H_LOCAL` | 24 prediction rows + validator pass + 24h features |
| `CFG05_REAL_READY_INCOMPLETE_23H` | 23 rows only (filter not fully fixed) |
| `CFG05_HOUR24_FIX_FAILED` | Smoke passed but hour-24 check failed |
| `CFG05_EVAL_COMPLETE` | Ready 24h + historical eval completed |
| `CFG05_EVAL_READY_NO_METRIC` | Ready but eval not attempted |
| `CFG05_EVAL_FAILED` | Evaluation attempted but failed |

### Metric Computation

| Metric | Formula |
|--------|---------|
| **sMAPE_floor50** | `200 * mean(|y_true_f - y_pred_f| / (|y_true_f| + |y_pred_f|))` where values below 50 are floored to 50 |
| **MAE** | `mean(|y_true - y_pred|)` |
| **RMSE** | `sqrt(mean((y_true - y_pred)^2))` |

**Important**: sMAPE_floor50 applies a floor of 50 to both y_pred and y_true, preventing extreme low-price periods from distorting the metric. The reported champion value of 11.48% was computed under the source methodology; P15 evaluation metrics use the same formula but are computed over a different time window and model training setup, so direct comparison is not valid.

## 6. Test Coverage (38 P15 tests)

| Group | Tests | Coverage |
|-------|-------|----------|
| **DayaheadWindowHelper** | 8 | `get_dayahead_window`, `day_ahead_mask`, `filter_dayahead`, `get_business_day_info` |
| **Hour24CompletenessChecker** | 10 | COMPLETE_24H, INCOMPLETE_23H, MISSING_HOURS, DUPLICATE_HOURS, INVALID, strict/non-strict CLI |
| **Metrics** | 6 | sMAPE_floor50, MAE, RMSE, empty arrays |
| **P15Pipeline** | 14 | Missing data, unsafe paths, forbidden files, summary keys, structural checks |

## 7. Files Changed/Created

| File | Action |
|------|--------|
| `artifacts/dayahead_window.py` | **NEW** — shared day-ahead window helper |
| `scripts/check_cfg05_hour24_completeness.py` | **NEW** — hour-24 completeness checker |
| `scripts/run_p15_cfg05_24h_smoke_and_eval.py` | **NEW** — P15 orchestration with eval |
| `tests/test_p15_hour24_completeness_eval.py` | **NEW** — 38 P15 tests |
| `docs/reports/p15_hour24_completeness_cfg05_eval_report.md` | **NEW** — this report |
| `models/adapters/cfg05_dayahead_lgbm.py` | **UPDATED** — uses `filter_dayahead()` |
| `scripts/train_export_cfg05_local.py` | **UPDATED** — uses `day_ahead_mask()` |

---

*End of P15 report. 713 tests total, 0 failures.*
