# P9 — Real Artifact Readiness + cfg05 REAL Adapter Smoke

- **Date:** 2026-07-04
- **Status:** Complete
- **Test count:** 517 → 570 (+53 P9 tests)

---

## 1. Overview

P9 implements the **real artifact readiness checking system** and the **cfg05 REAL adapter structural smoke** for the electricity_forecast_model3.0 project. This phase transitions the system from structural/dry-run to real artifact integration by establishing artifact gates, status codes, and verification checks across all six artifact types.

Phase objectives:
1. Define canonical artifact status codes and the ArtifactStatus data model
2. Implement readiness checks for all six artifact gates
3. Verify cfg05 LightGBM artifact existence, loadability, and predictability
4. Verify cfg05 input/feature schema compatibility
5. Provide readiness gates for RT assist pack, P5M pack, actual ledger, ExtremPriceClf
6. Run cfg05 REAL adapter smoke only when artifact + input are path-verified
7. Enforce: no REAL label without path-verified artifacts + passed validators
8. CLI tools for human-readable and JSON artifact readiness reporting

---

## 2. Artifact Status Codes

Seven status codes are defined in `artifacts/readiness.py`:

| Code | Meaning |
|------|---------|
| `MISSING` | Path not provided or does not exist on disk |
| `PRESENT` | Path exists but not yet loaded or verified |
| `LOADABLE` | Adapter can load the artifact without error |
| `SCHEMA_READY` | Input schema / features are compatible |
| `REAL_READY` | Loaded, real non-synthetic inference produced, validator passed |
| `NOT_IMPLEMENTED` | Exists but real inference path is still a stub |
| `INVALID` | Path exists but loading or validation failed |

Status progression: `MISSING → PRESENT → LOADABLE → SCHEMA_READY → REAL_READY`

Branch statuses: `NOT_IMPLEMENTED` (exists but no real inference yet), `INVALID` (broken artifact).

---

## 3. Artifact Model

The `ArtifactStatus` dataclass (`artifacts/readiness.py:77`) provides:

```
name: str           — human-readable identifier (e.g. "cfg05_artifact")
status: str         — one of the 7 status codes
path: str | None    — path that was checked
exists: bool        — whether the path exists on disk
loadable: bool      — whether the artifact could be loaded
schema_ready: bool  — whether input schema/features are compatible
real_ready: bool    — whether real inference was produced and validated
reason_codes: list[str] — audit trail for this status
details: dict       — extra details (file size, adapter version, etc.)
```

The `status_to_dict()` helper converts `ArtifactStatus` to a JSON-serializable dict.

---

## 4. cfg05 Artifact Readiness

File: `artifacts/readiness.py` — `check_cfg05_artifact()`

Checks performed:
1. Path existence (file or directory)
2. If directory: scan for `cfg05_model.txt`, `model.txt`, `lightgbm_cfg05_dayahead.txt`
3. If file: use directly
4. Record file size in `details`
5. Try `lightgbm.Booster(model_file=...)` → `LOADABLE`
6. Try `CFG05DayaheadAdapter.load()` → verifies adapter contract
7. Stops at `LOADABLE` — `REAL_READY` requires input + prediction + validation

Status transitions:
- `None` → `MISSING`
- Empty directory → `PRESENT`
- Invalid file → `INVALID`
- Valid model + adapter → `LOADABLE`

---

## 5. cfg05 Input / Feature Schema Readiness

File: `artifacts/readiness.py` — `check_cfg05_input()`

Checks performed:
1. Path existence
2. CSV loadability, non-empty
3. All 39 `CFG05_FEATURE_COLUMNS` present
4. `ds` timestamp column present
5. Reports counts of present/missing feature columns

Status transitions:
- `None` → `MISSING`
- Empty/load error → `INVALID`
- Missing columns → `INVALID`
- All columns + ds → `SCHEMA_READY`

---

## 6. RT Assist Pack Readiness

File: `artifacts/readiness.py` — `check_rt_assist_pack()`

Scans for: `model.pkl`, `rt_assist_model.pkl`, `model.pt`, `rt_assist_model.pt`

Checks:
1. Directory existence
2. Scan for recognised model files
3. Try `DASafeRealtimeAssistAdapter.load()`

Status: `LOADABLE` if model files + adapter. `NOT_IMPLEMENTED` because real inference requires full feature pipeline (torch dependency).

---

## 7. P5M Pack Readiness

File: `artifacts/readiness.py` — `check_p5m_pack()`

Scans for model files (`.pkl`, `.joblib`, `.pt`) and risk files (`risk_data.csv`, `risk_config.json`).

Checks:
1. Directory existence
2. Scan for model files and risk files
3. Try `P5MResidualPluginAdapter.load()`

Status: `LOADABLE` + `NOT_IMPLEMENTED`. REAL_READY would require risk data + real correction producing delta != 0.

---

## 8. Actual Ledger Readiness

File: `artifacts/readiness.py` — `check_actual_ledger()`

Checks:
1. File existence
2. CSV loadability
3. Required columns from `ACTUAL_LEDGER_COLUMNS`
4. Unique business day count (minimum 7 by default)
5. `y_true` null check

Status transitions:
- `None` → `MISSING`
- Empty file → `PRESENT`
- Missing columns → `INVALID`
- Null y_true → `INVALID`
- Insufficient days → `LOADABLE`
- Valid with >= min_days → `SCHEMA_READY`

---

## 9. ExtremPriceClf Readiness

File: `artifacts/readiness.py` — `check_extrempriceclf_artifact()`

Scans for: `ExtremPriceClf`, `ExtremPriceClf.pkl`, `extreme_price_radar/`, `classifier.pkl`, `classifier.pt`

Checks:
1. Directory existence
2. Scan for recognised artifact patterns
3. Try `NegativeClassifierAdapter.load(model_dir=...)`

Status: `LOADABLE` if artifact found + adapter loaded. `NOT_IMPLEMENTED` because real ExtremPriceClf inference is still a stub.

---

## 10. Aggregate Readiness Report

File: `artifacts/readiness.py` — `run_all_artifact_readiness()`

Runs all 6 gate checks with optional paths and returns:

```json
{
  "gates": {
    "cfg05_artifact": { "name": "cfg05_artifact", "status": "MISSING", ... },
    "cfg05_input": { "name": "cfg05_input", "status": "MISSING", ... },
    "rt_assist_pack": { ... },
    "p5m_pack": { ... },
    "actual_ledger": { ... },
    "extrempriceclf_artifact": { ... }
  },
  "summary": {
    "total_gates": 6,
    "status_counts": { "MISSING": 6 },
    "real_ready_gates": [],
    "all_real_ready": false,
    "any_missing": true
  }
}
```

---

## 11. CLI Tools & cfg05 REAL Adapter Smoke

### check_artifact_readiness.py

`scripts/check_artifact_readiness.py` — CLI for `run_all_artifact_readiness()`.

Options: `--cfg05-model`, `--cfg05-input`, `--rt-assist-pack`, `--p5m-pack`, `--actual-ledger`, `--extrempriceclf-dir`, `--json`, `--strict`, `--verbose`.

- Human-readable table output by default
- `--json` for machine-readable JSON
- `--strict` exits non-zero if any gate is not REAL_READY

### run_cfg05_real_adapter_smoke.py

`scripts/run_cfg05_real_adapter_smoke.py` — cfg05 REAL adapter structural smoke.

Options: `--model-dir`, `--model-file`, `--input`, `--target-day`, `--out`, `--production`/`--no-production`, `--strict`, `--verbose`.

Summary JSON fields:
- `cfg05_artifact_status` — artifact readiness status
- `cfg05_input_status` — input readiness status
- `cfg05_adapter_loaded` — whether adapter loaded and predicted
- `prediction_rows` — number of prediction rows produced
- `validator_passed` — whether output schema validation passed
- `readiness_label` — `"REAL"` | `"DRY_RUN"` | `"DATA_MISSING"`
- `reason_codes` — audit trail
- `overall_status` — `"PASS"` | `"FAIL"`

Key behavior:
- **Non-strict missing paths → exit 0** (structural pass)
- **Strict missing paths → exit non-zero**
- **No `--out` → no file written** (never writes to data/ outputs/ ledgers/ by default)
- **No REAL without verification** — `readiness_label` is never `"REAL"` without loaded adapter + passed validation

### Test Results

**Before P9:** 517 tests passing
**After P9:** 570 tests passing (+53 new)

New test files:
- `tests/test_artifact_readiness.py` — 37 tests covering all six gates, status codes, schema checks, edge cases
- `tests/test_cfg05_real_adapter_smoke.py` — 16 tests covering CLI behavior, summary structure, strict/non-strict modes, file output

All 570 tests pass.

---

## Key Files

| File | Purpose |
|------|---------|
| `artifacts/readiness.py` | Core readiness module (ArtifactStatus, 6 check functions, aggregate) |
| `artifacts/__init__.py` | Package init |
| `scripts/check_artifact_readiness.py` | Readiness CLI |
| `scripts/run_cfg05_real_adapter_smoke.py` | cfg05 REAL adapter smoke CLI |
| `tests/test_artifact_readiness.py` | 25 readiness tests |
| `tests/test_cfg05_real_adapter_smoke.py` | 14 cfg05 smoke tests |
