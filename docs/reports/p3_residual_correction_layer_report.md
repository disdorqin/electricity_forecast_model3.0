# P3 Residual Correction Layer Report

## 1. Executive status

P3 establishes the residual correction layer for the 3.0 system: corrected prediction schema, the residual correction pipeline with DATA-MISSING no-op default, P5M adapter integration for real correction, residual output validation, and CLI runner. All 53 new contract tests pass alongside the 171 P1+P2 tests (224 total, 0 failures). No actual risk model integration, no training, no production P5M weights.

| Component | Status |
|-----------|--------|
| Corrected prediction schema (`data/schema.py`) | DONE — 11 tests |
| Residual correction pipeline (`pipelines/residual_correction.py`) | DONE — 24 tests |
| Residual output validator (`scripts/validate_residual_output.py`) | DONE — 18 tests |
| CLI runner (`scripts/run_residual_correction.py`) | DONE — covered by pipeline tests |

---

## 2. Files created or updated

| File | Action | Description |
|------|--------|-------------|
| `data/schema.py` | **UPDATED** | Added `CORRECTED_PREDICTION_COLUMNS` (17 cols), `CORRECTED_UNIQUE_KEY`, `CORRECTED_REQUIRED_KEYS` |
| `pipelines/residual_correction.py` | **NEW** | Residual correction pipeline with no-op and P5M correction paths |
| `scripts/run_residual_correction.py` | **NEW** | CLI runner for residual correction |
| `scripts/validate_residual_output.py` | **NEW** | Corrected prediction output validator (13 checks) |
| `tests/test_residual_correction_schema.py` | **NEW** | 11 schema contract tests |
| `tests/test_residual_correction_runner.py` | **NEW** | 24 pipeline contract tests |
| `tests/test_residual_output_validator.py` | **NEW** | 18 validator contract tests |
| `docs/reports/p3_residual_correction_layer_report.md` | **NEW** | This report |

---

## 3. Corrected prediction schema (`data/schema.py`)

### New schema constants

```python
CORRECTED_PREDICTION_COLUMNS: Final[list[str]] = [
    "task", "model_name", "target_day", "business_day", "ds",
    "hour_business", "period",
    "y_pred_raw", "y_pred_corrected", "residual_delta",
    "correction_applied", "correction_module", "risk_source",
    "reason_codes", "correction_version",
    "source_confidence", "model_version",
]

CORRECTED_UNIQUE_KEY: Final[list[str]] = [
    "task", "model_name", "business_day", "hour_business",
]

CORRECTED_REQUIRED_KEYS: Final[list[str]] = [
    "task", "model_name", "target_day", "business_day", "ds",
    "hour_business", "period",
]
```

### Column semantics

| Column | Type | Description |
|--------|------|-------------|
| `y_pred_raw` | float | Original prediction before correction |
| `y_pred_corrected` | float | Prediction after correction (= y_pred_raw for no-op) |
| `residual_delta` | float | y_pred_corrected - y_pred_raw |
| `correction_applied` | bool | True if any correction was applied |
| `correction_module` | str | Module identifier (`p5m_residual_noop` or `p5m_residual_plugin`) |
| `risk_source` | str | `DATA_MISSING`, `NEGATIVE_RISK`, `ADAPTER_NO_EFFECT`, or `ADAPTER_ERROR` |
| `reason_codes` | str | Semicolon-delimited audit trail (e.g. `DATA_MISSING_NO_OP`) |
| `correction_version` | str | Version of correction module used |

### Unique key

The composite key `(task, model_name, business_day, hour_business)` uniquely identifies a corrected prediction row, matching the standard prediction ledger key.

### Validated (11 tests)

- Schema has exactly 17 columns
- All prediction key columns preserved (task, model_name, target_day, business_day, ds, hour_business, period)
- All P3-specific fields present (y_pred_raw, y_pred_corrected, residual_delta, correction_applied, correction_module, risk_source, reason_codes, correction_version)
- CORRECTED_UNIQUE_KEY has 4 columns
- No-op row: y_pred_corrected == y_pred_raw, residual_delta == 0, correction_applied == False
- Corrected row: delta != 0, correction_applied == True
- DataFrame sortable by business_day + hour_business

---

## 4. Residual correction pipeline (`pipelines/residual_correction.py`)

### API

```python
def apply_residual_correction(
    predictions_df: pd.DataFrame,
    correction_profile: str = "conservative",
    risk_df: Optional[pd.DataFrame] = None,
    canonical_pack_path: Optional[str] = None,
    production: bool = True,
    **kwargs: Any,
) -> pd.DataFrame
```

### Flow

```
prediction output (standard schema)
  │
  ├─ Profile validation (conservative / moderate / aggressive)
  ├─ Production check (reject y_true)
  ├─ Input validation (require ds + y_pred)
  ├─ Normalise y_pred → y_pred_raw
  ├─ Add business-time columns (business_day, hour_business, period)
  ├─ Set defaults for task / model_name / target_day / period
  │
  ├─ Decision: has risk data OR has canonical pack?
  │   ├─ YES → P5MResidualPluginAdapter.predict()
  │   │         ├─ Values changed → correction_applied = True
  │   │         ├─ Values unchanged → ADAPTER_NO_EFFECT
  │   │         └─ Error → ADAPTER_ERROR fallback
  │   └─ NO  → DATA-MISSING no-op (default)
  │
  └─ Return corrected DataFrame (17 columns, sorted)
```

### No-op defaults

| Field | Default value |
|-------|---------------|
| correction_module | `p5m_residual_noop` |
| correction_version | `0.0.0` |
| risk_source | `DATA_MISSING` |
| reason_codes | `DATA_MISSING_NO_OP` |
| correction_applied | `False` |

### Business-time column derivation

Business-time columns are derived from `ds` when absent:
- `ds` at HH:00 → `business_day` = ds.date, `hour_business` = HH
- `ds` at 00:00 → `business_day` = ds.date - 1 day, `hour_business` = 24
- `period` derived from hour_business: 1-8 → `1_8`, 9-16 → `9_16`, 17-24 → `17_24`

### Helper functions

```python
def get_corrected_schema_columns() -> list[str]
    # Returns list of 17 corrected prediction column names

def is_data_missing_noop(corrected_df: pd.DataFrame) -> bool
    # Returns True if all rows are DATA-MISSING no-ops
    # (correction_applied == False AND risk_source == "DATA_MISSING")
```

### Validated (24 tests)

**No-op behavior (7 tests)**:
- y_pred_corrected == y_pred_raw
- residual_delta == 0
- correction_applied == False
- reason_codes contains DATA_MISSING_NO_OP
- is_data_missing_noop returns True
- risk_source == DATA_MISSING
- correction_module == p5m_residual_noop

**Task support (3 tests)**:
- Dayahead task supported
- Realtime task supported
- Task column preserved from input

**Schema compliance (6 tests)**:
- Output has all 17 corrected columns
- No NaN in y_pred_raw, y_pred_corrected, or residual_delta
- hour_business in 1..24
- No y_true in production output

**Input validation (3 tests)**:
- Invalid profile raises ValueError
- Missing required column raises ValueError
- Empty input returns empty DataFrame with correct columns

**Risk data integration (2 tests)**:
- risk_df present does not crash (falls back to no-op)
- Non-existent canonical pack silently ignored

**CLI dry-run (2 tests)**:
- Exit code 0
- No-op behavior confirmed

---

## 5. Residual correction CLI (`scripts/run_residual_correction.py`)

### CLI

```
python scripts/run_residual_correction.py --dry-run --out corrected.csv
python scripts/run_residual_correction.py --input predictions.csv --out corrected.csv
python scripts/run_residual_correction.py --input predictions.csv \
    --risk-path risk_data.csv --profile aggressive --out corrected.csv
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--input` | Input CSV path (standard prediction schema) |
| `--out` | Output CSV path for corrected predictions |
| `--risk-path` | Path to risk data CSV (negative_prob, risk_source columns) |
| `--canonical-pack` | Path to canonical prediction pack (for P5M adapter) |
| `--profile` | Correction aggressiveness: conservative (default), moderate, aggressive |
| `--dry-run` | Synthetic 24h prediction data (no input needed) |
| `--production` | Production mode: y_true forbidden (default) |
| `--no-production` | Eval mode: allow y_true column |
| `--verbose` | Debug logging |

### Dry-run behavior

- Generates 24 hourly synthetic predictions using cfg05 model name
- Uses fixed random seed (42) for reproducibility
- Applies DATA-MISSING no-op by default (no risk data)
- Writes corrected CSV when `--out` specified
- Prints summary to stdout when `--out` omitted

### Design

- Loads predictions from CSV or synthetic generation
- Loads risk data from CSV if `--risk-path` provided
- Routes to `apply_residual_correction()` pipeline
- Logs correction summary: module, row count, correction count, risk source

---

## 6. Residual output validator (`scripts/validate_residual_output.py`)

### CLI

```
python scripts/validate_residual_output.py corrected.csv
python scripts/validate_residual_output.py corrected.csv --verbose
python scripts/validate_residual_output.py corrected.csv --no-production
```

### Validation checks (13 total)

| # | Check | Error if |
|---|-------|----------|
| 1 | Input not empty | Empty DataFrame |
| 2 | Required columns | Missing any of CORRECTED_PREDICTION_COLUMNS |
| 3 | Eval-only columns | y_true present in production mode |
| 4 | hour_business range | Outside [1, 24] |
| 5 | period values | Not in 1_8 / 9_16 / 17_24 |
| 6 | y_pred_raw NaN | Any NaN in y_pred_raw |
| 7 | y_pred_corrected NaN | Any NaN in y_pred_corrected |
| 8 | residual_delta NaN | Any NaN in residual_delta |
| 9 | residual_delta arithmetic | residual_delta != y_pred_corrected - y_pred_raw (tolerance 1e-6) |
| 10 | correction_applied boolean | Non-boolean values detected |
| 11 | Key column NaN | Null in any CORRECTED_UNIQUE_KEY column |
| 12 | Duplicate keys | Same (task, model_name, business_day, hour_business) |
| 13 | Task values | Not dayahead or realtime |

### Programmatic API

```python
validate_residual_dataframe(df, production=True, verbose=False) -> (bool, list[str])
validate_residual_file(path, production=True, verbose=False) -> (bool, list[str])
```

### Validated (18 tests)

- Valid DataFrame passes all checks
- Corrected DataFrame with non-zero deltas passes
- Duplicate keys detected
- NaN in y_pred_raw, y_pred_corrected, residual_delta all detected
- residual_delta arithmetic mismatch detected
- Correct residual_delta passes
- Non-boolean correction_applied detected
- Integer 0/1 for correction_applied passes
- hour_business out of range detected
- y_true rejected in production mode, allowed in eval mode
- Missing required column detected
- Empty DataFrame detected
- Invalid period detected
- Missing file returns error (not crash)
- Valid CSV file passes

---

## 7. Tests run

| Test suite | Tests | Passed | Failed |
|---|---|---|---|
| `test_schema_contract.py` (P1) | 31 | 31 | 0 |
| `test_dayahead_model_zoo_contract.py` (P1) | 23 | 23 | 0 |
| `test_rt_assist_contract.py` (P1) | 21 | 21 | 0 |
| `test_p5m_residual_contract.py` (P1) | 21 | 21 | 0 |
| `test_loaders_contract.py` (P2) | 23 | 23 | 0 |
| `test_dayahead_feature_pipeline.py` (P2) | 15 | 15 | 0 |
| `test_realtime_feature_pipeline.py` (P2) | 15 | 15 | 0 |
| `test_prediction_runner_contract.py` (P2) | 22 | 22 | 0 |
| `test_residual_correction_schema.py` (P3) | 11 | 11 | 0 |
| `test_residual_correction_runner.py` (P3) | 24 | 24 | 0 |
| `test_residual_output_validator.py` (P3) | 18 | 18 | 0 |
| **Total** | **224** | **224** | **0** |

---

## 8. Known limitations

1. **No real P5M risk model integrated**: The P5M residual plugin adapter is a contract stub. Real correction requires a trained residual model with `negative_prob` and `spike_prob` outputs. Without risk data or a canonical pack, the pipeline always returns DATA-MISSING no-op.

2. **No actual P5M model weights**: The adapter has `load()` and `predict()` methods but returns synthetic predictions (uniform noise scaled by profile). Production deployment requires trained model artifacts.

3. **P5M correction profiles are template defaults**: The conservative/moderate/aggressive profile parameters (`-10%/-20%/-30%`) are initial template defaults. Real profiles require calibration against validation data.

4. **No low-valley correction path**: The current P5M adapter only has a stub for low-valley correction. Only negative risk correction is modelled.

5. **No correction versioning strategy**: `correction_version` defaults to `0.0.0` for no-op and reads `adapter.model_version` for real corrections. No formal versioning or migration strategy exists yet.

6. **Risk data merge is naive**: When `risk_df` is provided, the pipeline merges risk columns by position (`.values`). A proper merge on business_day + hour_business is needed for production.

7. **No spike correction**: The current residual correction layer covers negative/low-valley residuals but does not model high-spike correction. Unified spike correction requires a separate module.

8. **Validator does not check correction continuity**: The validator checks individual row correctness but does not validate that corrections are temporally consistent (e.g., smooth transitions between consecutive hours).

---

## 9. Forbidden files check

All 224 tests use synthetic tiny DataFrames. No real data files, model weights, CSVs, Excel files, or pickle files are tracked. Test fixtures using CSV use `pytest tmp_path` exclusively.

```
data/*       → NOT committed
outputs/*    → NOT committed
reports/local/* → NOT committed
*.csv        → NOT committed (except test fixtures via tmp_path)
*.xlsx       → NOT committed
*.pkl        → NOT committed
*.joblib     → NOT committed
*.pt / *.pth → NOT committed
*.ckpt       → NOT committed
*.parquet    → NOT committed
```

**Result**: PASS

---

## 10. Final status

```
P3 Execution Summary

1. Files created:      6 (pipelines/residual_correction.py, 2 scripts, 3 test files, 1 report)
2. Files updated:      1 (data/schema.py — added corrected schema constants)
3. Tests added:        53 (11 schema + 24 pipeline + 18 validator)
4. Tests run:          224 total (96 P1 + 75 P2 + 53 P3), all pass
5. Corrected schema:   DONE — 17 columns, 4-column unique key, no-op defaults
6. Correction pipeline: DONE — DATA-MISSING no-op default, P5M adapter path, profile validation
7. Output validator:   DONE — 13 checks, CLI and programmatic API
8. Known limitations:  See §8 — 8 items documented
9. Forbidden files:    PASS
10. Commit:            Pending (P3 commit)
11. Final status:      COMPLETE — Ready for P4 (Fusion Engine)
```
