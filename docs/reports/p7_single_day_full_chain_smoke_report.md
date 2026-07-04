# P7: Single-Day Full-Chain Structural Smoke Report

**Date:** 2026-07-04
**Status:** Complete
**Components:** `pipelines/full_chain_smoke.py`, `scripts/run_full_chain_smoke.py`, `tests/test_full_chain_smoke.py`
**Test count:** 26 new + 468 existing = **494 total, all passing**

---

## 1. Executive Status

P7 implements a single-day full-chain structural / dry-run smoke that exercises all P1–P6 components in sequence:

```
synthetic predictions (DRY_RUN)
    → residual correction (DATA_MISSING no-op)
    → corrected output validator
    → corrected ledger append (STRUCTURAL_ONLY)
    → fusion (STRUCTURAL_ONLY, equal_weight)
    → fusion output validator
    → fusion ledger append (STRUCTURAL_ONLY)
    → weight extraction → weight ledger append
    → negative classifier (RULE_FALLBACK / no-artifact)
    → final output validator
    → summary dict
```

**This is structural/dry-run smoke, not production real inference.**
No REAL labels appear unless artifacts are path-verified.

---

## 2. Smoke Mode and Labels

Each pipeline stage receives a label that reflects its operational mode:

| Stage | Label | Condition |
|---|---|---|
| Day-ahead prediction | `DRY_RUN` | Default (no cfg05 artifact) |
| Day-ahead prediction | `REAL` | cfg05_artifact_path verified |
| Residual correction | `DATA_MISSING` | Default (no risk data / canonical pack) |
| Corrected ledger | `STRUCTURAL_ONLY` | Always |
| Fusion | `STRUCTURAL_ONLY` | Always |
| Fusion ledger | `STRUCTURAL_ONLY` | Always |
| Weight ledger | `STRUCTURAL_ONLY` | Always |
| Negative classifier | `RULE_FALLBACK` | rule_fallback=True (default) |
| Negative classifier | `CLASSIFIER_ARTIFACT_MISSING` | rule_fallback=False |
| Negative classifier | `REAL` | classifier_model_dir verified |
| Final output | `STRUCTURAL_ONLY` | Always |

Default mode_label set: `[DATA_MISSING, DRY_RUN, RULE_FALLBACK, STRUCTURAL_ONLY]`

---

## 3. Pipeline Stages Exercised

### Stage 1: Day-ahead prediction
- `_build_synthetic_predictions()` generates 24 rows per model × 2 models = 48 rows
- Model names: `cfg05`, `best_two_average`
- Includes negative prices for rule fallback testing
- Validated via `validate_prediction_dataframe()`
- Label: `DRY_RUN` (or `REAL` if cfg05_artifact_path exists)

### Stage 2: Residual correction
- `apply_residual_correction()` with `risk_df=None`
- Guaranteed DATA-MISSING no-op: `y_pred_corrected == y_pred_raw`
- Validated via `validate_residual_dataframe()`
- Label: `DATA_MISSING`

### Stage 3: Corrected ledger append
- `append_corrected_predictions_to_ledger()` with run_id
- Written to `tmp_path` CSV when ledger_dir provided
- Validated via `validate_corrected_ledger()`
- Label: `STRUCTURAL_ONLY`

### Stage 4: Fusion
- `run_fusion()` with `method="equal_weight"`, `allow_dry_run=True`
- Explicit `readiness_status={"cfg05": READY_DRY_RUN, "best_two_average": READY_DRY_RUN}`
- Produces 24 fused rows (one per hour)
- Validated via `validate_fusion_dataframe()`
- Label: `STRUCTURAL_ONLY`

### Stage 5: Fusion ledger + weight ledger
- `append_fusion_to_ledger()` with run_id
- `extract_weight_rows()` expands weights_json → 48 rows (24h × 2 models)
- `append_weights_to_ledger()` with run_id
- Validated via `validate_fusion_ledger()`, `validate_weight_ledger()`
- Label: `STRUCTURAL_ONLY`

### Stage 6: Negative classifier + final output
- `run_negative_classifier()` with `rule_fallback=True`
- No-artifact fallback: `final_price == fused_price`
- Rule fallback: `fused_price < 0` → `negative_flag = True`
- Validated via `validate_final_dataframe()`
- Label: `RULE_FALLBACK` (or `CLASSIFIER_ARTIFACT_MISSING`)

---

## 4. Row-Count Summary

| Stage | Input rows | Output rows |
|---|---|---|
| Predictions | — | 48 (24h × 2 models) |
| Residual correction | 48 | 48 |
| Fusion | 48 | 24 (1 per hour) |
| Weight ledger | 24 | 48 (24h × 2 models) |
| Negative classifier | 24 | 24 |
| **Final output** | — | **24** |

---

## 5. Validator Results

| Validator | Status |
|---|---|
| `validate_prediction_dataframe` | PASS |
| `validate_residual_dataframe` | PASS |
| `validate_corrected_ledger` | PASS |
| `validate_fusion_dataframe` | PASS |
| `validate_fusion_ledger` | PASS |
| `validate_weight_ledger` | PASS |
| `validate_final_dataframe` | PASS |

All 7 validators pass in the default smoke run.

---

## 6. Ledger Smoke Results

With `ledger_dir` set to a `tmp_path`:

| Ledger | File created | Validator |
|---|---|---|
| Corrected ledger | `corrected_ledger.csv` | PASS |
| Fusion ledger | `fusion_ledger.csv` | PASS |
| Weight ledger | `weight_ledger.csv` | PASS |

No ledgers are written to the repository — only to user-specified paths (enforced by forbidden-files check).

---

## 7. Fallback Paths Used

| Component | Fallback | Reason |
|---|---|---|
| Day-ahead | Synthetic data (DRY_RUN) | No cfg05 artifact available |
| Residual correction | DATA_MISSING no-op | No risk data / canonical pack |
| Negative classifier | No-artifact + rule fallback | No ExtremPriceClf artifact; rule_fallback=True |

No REAL correction, REAL fusion, or REAL classifier inference is claimed.

---

## 8. Forbidden Files Check

- `data/*` — ❌ NOT written
- `outputs/*` — ❌ NOT written
- `reports/local/*` — ❌ NOT written
- `ledgers/*.csv` — ❌ NOT written (test tmp_path only)
- `*.csv`, `*.xlsx`, `*.xls` — ❌ NOT written
- `*.pkl`, `*.joblib`, `*.pt`, `*.pth`, `*.ckpt` — ❌ NOT written
- `*.parquet` — ❌ NOT written

The smoke pipeline checks `forbidden_files_check` in its summary and will FAIL if ledger_dir is inside repo data paths.

---

## 9. Known Limitations

1. **Synthetic predictions only** — No real model adapters invoked. Predictions are random numbers seeded for reproducibility.
2. **DATA-MISSING residual** — No risk data or canonical pack; correction is always no-op.
3. **No realtime path** — `use_realtime=True` is accepted but not structurally wired; defaults to DA-only.
4. **Fusion uses equal_weight** — No BGEW learner or prior weights in default smoke.
5. **No ExtremPriceClf inference** — Classifier adapter runs in no-op + rule fallback mode.
6. **No metric computation** — This is structural smoke only; no error metrics or performance stats.
7. **No ledger compaction test** — Append-only; dedup tested in P5 but not in full-chain context.
8. **Single-day only** — The smoke operates on one target day; multi-day ledger accumulation is not tested here.

---

## 10. P8 Recommendation

**GO_FOR_DRY_RUN_STRUCTURAL_SMOKE** — P7 confirms that all structural paths are connected and validators pass. The following are recommended for P8:

1. **P8.1: Real adapter smoke** — If cfg05 adapter artifact exists (`model.txt` or `model.pkl`), verify that `cfg05_dayahead_lgbm` adapter produces valid predictions with real features.
2. **P8.2: Risk data integration** — Wire real assist ledger data into residual correction and verify correction-applied path.
3. **P8.3: BGEW integration** — If actuals ledger has data, run fusion with `bgew_skeleton` method.
4. **P8.4: Multi-day ledger accumulation** — Run smoke across 3+ days with backfill to verify ledger continuity.
5. **P8.5: Real classifier integration** — Deploy ExtremPriceClf artifact and wire into classifier adapter.

**NO_GO_FOR_REAL_FULL_CHAIN_CLAIMS** until:
- cfg05 REAL verified via `validate_prediction_dataframe` + real features
- P5M residual REAL verified via `apply_residual_correction` with risk data producing `y_pred_corrected != y_pred_raw`
- ExtremPriceClf REAL verified via classifier model_dir with artifact
- All validators pass on real (non-synthetic) data
