# P10 — cfg05 Artifact Export + REAL Adapter Smoke Report

- **Date:** 2026-07-04
- **Status:** Structural gates verified; REAL_READY not achieved
- **Test count:** 570 → 596 (+26 P10 tests)

---

## 1. Executive Status

```
cfg05 REAL smoke pipeline: NOT_READY / DATA_MISSING
CFG05_REAL_READY: NOT ACHIEVED
```

P10 implements the structural machinery for locating, preparing, and validating
cfg05 artifacts and input data. All three new CLI tools execute correctly and
all gates report correctly. However, **no real cfg05 LightGBM champion artifact
or compatible input CSV is present in this repository**, so REAL_READY cannot
be claimed.

**Current status: STRUCTURAL_READY / ARTIFACT_GATE_READY — NO_GO_FOR_REAL_CFG05_CLAIM**

---

## 2. cfg05 Source Artifact Search

**Script:** `scripts/locate_cfg05_artifact.py`

Searches a source repository (e.g. `disdorqin/epf-sota-experiment`) for
cfg05-compatible LightGBM text model files.

**Search strategy:**
1. Exact name match: `cfg05_model.txt`, `model.txt`, `lightgbm_cfg05_dayahead.txt`
2. Keyword match: `.txt` files with `cfg05`/`lgbm`/`lightgbm`/`champion` in path
3. Blacklist filter: ignores `lgbm_spike_residual_1127`, `stage3_old_1164`,
   `lightgbm_90d_orig_1197`

**Result without source repo:** `MISSING` — no candidates, no crash.
**Result with placeholder file:** `INVALID` (not a valid LightGBM model).

**To export:** clone `disdorqin/epf-sota-experiment` and run:
```
python -m scripts.locate_cfg05_artifact \
    --source-repo /path/to/epf-sota-experiment --json
```

---

## 3. cfg05 Input / Schema Readiness

**Script:** `scripts/prepare_cfg05_real_input.py`

Validates candidate input CSV against the dynamically resolved
`CFG05_FEATURE_COLUMNS`.

**Feature column count:** `len(CFG05_FEATURE_COLUMNS) = 42` (dynamically resolved,
not hardcoded).

**Checks performed:**
1. File exists and is readable
2. All 42 feature columns present
3. `ds` timestamp column present and parsable
4. Target-day 24-row filter (if `--target-day` provided)

**Result without input:** `MISSING` — no crash.
**Result with incomplete CSV:** `INVALID` — missing columns listed.
**Result with full columns + ds:** `SCHEMA_READY`.

**Write behavior:** Only writes prepared CSV when `--out` is explicitly provided.
Never writes to `data/`, `outputs/`, `ledgers/`.

---

## 4. cfg05 REAL Smoke Result

**Script:** `scripts/run_cfg05_real_smoke_pipeline.py`

**Full pipeline:** artifact gate → input gate → adapter load → predict → validate.

**Result (no real artifacts):**

| Field | Value |
|-------|-------|
| cfg05_artifact_status | `MISSING` |
| cfg05_input_status | `MISSING` |
| cfg05_adapter_loaded | `False` |
| prediction_rows | `0` |
| validator_passed | `False` |
| readiness_label | `DATA_MISSING` |
| overall_status | `PASS_STRUCTURAL` |

**REAL_READY requirements (NOT met):**

- [ ] cfg05 artifact path exists on disk
- [ ] adapter `load()` succeeds
- [ ] cfg05 input is non-synthetic with schema-compatible features
- [ ] prediction produces > 0 rows
- [ ] `validate_output()` passes

**To attempt REAL_READY:**
1. Export cfg05 champion from `epf-sota-experiment` repo
2. Prepare compatible input CSV covering target-day feature columns
3. Run:
```
python -m scripts.run_cfg05_real_smoke_pipeline \
    --cfg05-model /path/to/model.txt \
    --cfg05-input /path/to/data.csv \
    --target-day YYYY-MM-DD \
    --strict
```

---

## 5. Feature Column Count Reconciliation

**P9 issue:** Report hardcoded "39 CFG05_FEATURE_COLUMNS".

**Actual count (P10):**

```python
from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
feature_count = len(CFG05_FEATURE_COLUMNS)  # 42
```

**Fixed:** P9 report updated to reference dynamic feature count. All P10 code
and tests dynamically import `CFG05_FEATURE_COLUMNS` — no hardcoded counts.

---

## 6. Artifact Readiness Matrix

| Gate | Status | Condition |
|------|--------|-----------|
| cfg05_artifact | `MISSING` | No real artifact in repo |
| cfg05_input | `MISSING` | No real input CSV in repo |
| rt_assist_pack | `MISSING` | No RT assist pack in repo |
| p5m_pack | `MISSING` | No P5M pack in repo |
| actual_ledger | `MISSING` | No actual ledger in repo |
| extrempriceclf_artifact | `MISSING` | No ExtremPriceClf in repo |

---

## 7. Tests Run

**P10 test file:** `tests/test_cfg05_artifact_export_readiness.py` (26 tests)

| Section | Tests | Description |
|---------|-------|-------------|
| TestLocateCfg05Artifact | 6 | Missing source repo, blacklist filter, placeholder not REAL_READY, CLI |
| TestPrepareCfg05RealInput | 8 | Dynamic feature count, missing cols, missing ds, schema-ready, file write |
| TestRunCfg05RealSmokePipeline | 9 | Missing model (strict/non-strict), missing input, placeholder not REAL_READY, summary keys, reason codes, out file |
| TestForbiddenFilesCheck | 2 | No CSV/pkl/joblib in untracked or tracked files |

**Before P10:** 570 tests passing
**After P10:** 596 tests passing (+26 new)

---

## 8. Forbidden Files Check

- `.gitignore` blocks: `*.csv`, `*.xlsx`, `*.pkl`, `*.joblib`, `*.pt`, `*.pth`,
  `*.ckpt`, `*.parquet`, `*.feather`, `data/`, `outputs/`, `reports/local/`
- No forbidden files in `git ls-files` or untracked
- Test suite confirms no leaked extensions

**Status: PASS**

---

## 9. Known Limitations

1. **No real cfg05 artifact in repo** — The cfg05 LightGBM champion resides in
   `disdorqin/epf-sota-experiment`. It has not been exported to this repository.
   The locate script can find it if pointed at the source repo, but the artifact
   itself is not committed.

2. **No real input CSV in repo** — cfg05 requires 42 feature columns
   (weather, lag, calendar, etc.) from a full feature pipeline. No such CSV
   exists in this repo. A synthetic CSV was used for structural testing.

3. **REAL_READY not achieved** — Without real artifact + real input, the
   adapter `predict()` cannot produce non-synthetic output. The readiness
   gate correctly reports `DATA_MISSING`.

4. **Realtime safe correction NOT REAL** — RT assist pack is MISSING.
   No REAL claim can be made for realtime correction.

5. **P5M residual not improved** — No real risk/canonical pack present.
   `y_pred_corrected` has not been verified against real correction data.

6. **BGEW not production** — Weight learner is structural skeleton only.
   No production BGEW training with real actuals has been performed.

7. **ExtremPriceClf not deployed** — Classifier adapter loads structurally
   but no real ExtremPriceClf artifact is present.

---

## 10. P11 Recommendation

1. **Export cfg05 LightGBM champion** from `disdorqin/epf-sota-experiment`:
   - Locate the champion model file (LightGBM text format)
   - Place it at an appropriate path (e.g. `artifacts/models/cfg05_model.txt`)
   - Run `locate_cfg05_artifact` to verify LOADABLE status

2. **Prepare real feature CSV** from the source repo's data pipeline:
   - Extract 24 hours of feature rows for a target day
   - Verify all 42 CFG05_FEATURE_COLUMNS + ds are present
   - Run `prepare_cfg05_real_input` to confirm SCHEMA_READY

3. **Run full cfg05 REAL smoke pipeline**:
   ```
   python -m scripts.run_cfg05_real_smoke_pipeline \
       --cfg05-model artifacts/models/cfg05_model.txt \
       --cfg05-input data/cfg05_features.csv \
       --target-day 2026-07-01 --strict
   ```
   Target: `readiness_label: REAL_READY`

4. **Do NOT commit** any artifact, CSV, or output file to the repository.

5. If REAL_READY is achieved, P12 can proceed to full-chain validation with
   real actuals and weight learner training.
