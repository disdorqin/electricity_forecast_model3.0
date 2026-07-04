# P8: 30-Day Ledger Backfill Structural Smoke Report

**Date:** 2026-07-04
**Status:** Complete
**Components:** `pipelines/multi_day_backfill_smoke.py`, `scripts/run_multi_day_backfill_smoke.py`, `tests/test_multi_day_backfill_smoke.py`
**Test count:** 23 new + 494 existing = **517 total, all passing**

---

## 1. Executive Status

P8 extends the P7 single-day full-chain structural smoke to N days (default 30), exercising:

- Per-day prediction → residual correction → fusion → classifier → final output
- Corrected / fusion / weight ledger accumulation across N days
- Synthetic actual ledger generation with no-leakage validation
- Key uniqueness verification across all 4 ledgers
- Idempotency verification (re-append does not duplicate rows)

**This is structural/dry-run multi-day smoke, not production real inference.**

---

## 2. Smoke Mode and Labels

Default mode_label set: `[DATA_MISSING, DRY_RUN, RULE_FALLBACK, STRUCTURAL_ONLY]`

| Stage | Label | Condition |
|---|---|---|
| Day-ahead prediction | `DRY_RUN` | Always (no real artifact) |
| Residual correction | `DATA_MISSING` | Always (no risk data) |
| Fusion | `STRUCTURAL_ONLY` | Always |
| Corrected ledger | `STRUCTURAL_ONLY` | Always |
| Fusion ledger | `STRUCTURAL_ONLY` | Always |
| Weight ledger | `STRUCTURAL_ONLY` | Always |
| Negative classifier | `RULE_FALLBACK` | rule_fallback=True (default) |
| Negative classifier | `CLASSIFIER_ARTIFACT_MISSING` | rule_fallback=False |
| Final output | `STRUCTURAL_ONLY` | Always |

No `REAL` label appears without path-verified artifacts.

---

## 3. Multi-Day Range and Row Counts

**Default: 30 days, 2 models, 24 hours/day**

| Artifact | Formula | Expected | Actual |
|---|---|---|---|
| Predictions | 30 × 2 × 24 | 1,440 | 1,440 |
| Corrected | 30 × 2 × 24 | 1,440 | 1,440 |
| Fusion | 30 × 24 | 720 | 720 |
| Weights | 30 × 2 × 24 | 1,440 | 1,440 |
| Final output | 30 × 24 | 720 | 720 |
| Corrected ledger | 30 × 2 × 24 | 1,440 | 1,440 |
| Fusion ledger | 30 × 24 | 720 | 720 |
| Weight ledger | 30 × 2 × 24 | 1,440 | 1,440 |
| Actual ledger | 30 × 24 | 720 | 720 |

Per-day success tracked via `per_day_status` dict (all 30 days PASS).

---

## 4. Ledger Continuity and Idempotency

### Continuity

All 4 ledgers accumulate correctly across the full 30-day range:
- Corrected ledger: 1,440 rows, no duplicate keys
- Fusion ledger: 720 rows, no duplicate keys
- Weight ledger: 1,440 rows, no duplicate keys
- Actual ledger: 720 rows (when generated), no duplicate keys

### Idempotency

Re-appending the same day's data to the corrected ledger produces no new rows — the `keep="latest"` dedup strategy correctly matches existing keys via `groupby(key_cols).last()`. Idempotency check: **PASS**.

### Key uniqueness

| Ledger | Key columns | Status |
|---|---|---|
| Corrected | `[task, model_name, target_day, business_day, hour_business]` | PASS |
| Fusion | `[task, target_day, business_day, hour_business]` | PASS |
| Weight | `[task, target_day, business_day, hour_business, model_name, fusion_method]` | PASS |
| Actual | `[task, target_day, business_day, hour_business]` | PASS |

---

## 5. Validator Results

| Validator | Status |
|---|---|
| `validate_prediction_dataframe` | PASS |
| `validate_residual_dataframe` | PASS |
| `validate_fusion_dataframe` | PASS |
| `validate_final_dataframe` | PASS |
| `validate_actual_ledger` | PASS |

All 5 validators pass.

---

## 6. Weight Ledger Results

Weights extracted from fusion output via `extract_weight_rows` are verified:
- 1,440 weight rows for 30 days × 24 hours × 2 models
- Each `(task, target_day, hour_business)` group sums to 1.0 (within 1e-4 tolerance)
- No duplicate `WEIGHT_LEDGER_KEY` rows

---

## 7. Actual Ledger and No-Leakage Check

### Synthetic actuals
- 720 rows (30 days × 24 hours)
- `y_true` values: uniform random in [80, 200]
- `actual_source`: `"synthetic_smoke"`
- Validated via `validate_actual_ledger`: PASS

### No-leakage verification
For each of the 30 target days:
1. `filter_actuals_for_training(actual_ledger, target_day=day, window=30)` called
2. Result checked: **no row with `business_day >= target_day`**
3. All 30 days pass

**No-leakage check: PASS**

---

## 8. Fallback Paths Used

| Component | Fallback | Reason |
|---|---|---|
| Day-ahead | Synthetic data (DRY_RUN) | No real model artifact |
| Residual correction | DATA_MISSING no-op | No risk data / canonical pack |
| Fusion | equal_weight (default) | No prior weights supplied |
| Fusion (bgew_skeleton) | equal_weight fallback | Insufficient historical actuals for BGEW learner |
| Negative classifier | No-artifact + rule fallback | No ExtremPriceClf artifact |

No REAL correction, REAL fusion, or REAL classifier inference is claimed.

---

## 9. Forbidden Files Check

- `data/*` — ❌ NOT written
- `outputs/*` — ❌ NOT written
- `reports/local/*` — ❌ NOT written
- `ledgers/*.csv` — ❌ NOT written (test `tmp_path` only)
- `*.csv`, `*.xlsx`, `*.xls` — ❌ NOT written
- `*.pkl`, `*.joblib`, `*.pt`, `*.pth`, `*.ckpt` — ❌ NOT written
- `*.parquet` — ❌ NOT written

**Status: PASS** — all ledger I/O uses `tmp_path` or user-specified paths with forbidden-path detection.

---

## 10. Known Limitations

1. **Synthetic predictions only** — No real model adapters or features.
2. **DATA-MISSING residual** — No risk data; correction is always no-op.
3. **No realtime path** — DA-only throughout.
4. **equal_weight fusion** — BGEW integration is structural: `bgew_skeleton` with synthetic actuals produces fallback equal_weight output. No real BGEW learner.
5. **No ExtremPriceClf inference** — Classifier runs in no-op + rule fallback mode.
6. **Contiguous dates only** — The smoke assumes `start_day + N` consecutive days. Gap handling is not tested.
7. **No ledger compaction** — Ledgers grow linearly with days. No archival of old entries.
8. **No multi-model fusion** — Always 2 models (`cfg05`, `best_two_average`). Variable model counts not tested.
9. **No performance metrics** — Structural smoke only; no MAE, RMSE, or profit simulation.

---

## 11. P9 Recommendation

**GO_FOR_STRUCTURAL_MULTI_DAY_SMOKE** — P8 confirms that the full chain works across 30 days with ledger continuity, idempotency, key uniqueness, and no-leakage. Recommended for P9:

1. **P9.1: Negative classifier integration with ExtremPriceClf** — If ExtremPriceClf artifact is available, wire real inference into the classifier adapter and verify `classifier_applied=True` outputs.
2. **P9.2: BGEW learner with real actuals** — If actual ledger has sufficient history, test BGEW weight learning with non-uniform weights.
3. **P9.3: Production-grade ledger storage** — Evaluate Parquet format, partitioning, and compression for ledger persistence.
4. **P9.4: Real adapter integration** — Wire cfg05 adapter with real features and verify `validate_prediction_dataframe` on non-synthetic data.

**NO_GO_FOR_REAL_FULL_CHAIN_CLAIMS** until `cfg05`, `P5M`, and `ExtremPriceClf` are all artifact-verified and all validators pass on real (non-synthetic) data.
