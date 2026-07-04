# P11 — cfg05 REAL Artifact Export Attempt Report

- **Date:** 2026-07-04
- **Status:** Source repo not found; REAL_READY not achieved
- **Test count:** 596 → 617 (+21 P11 tests)

---

## 1. Executive Status

```
CFG05_REAL_READY_LOCAL: NOT ACHIEVED
```

P11 attempted to locate and export the cfg05 LightGBM champion artifact from
the `disdorqin/epf-sota-experiment` source repository. The source repository
is **not available on this machine**, so artifact export could not proceed.

**Current status:** CFG05_EXPORT_BLOCKED — source repo required.

---

## 2. Source Workspace Status

| Check | Result |
|-------|--------|
| Source repo path provided | `None` (no `--source-repo` given) |
| Source repo exists on disk | Not checked (no path provided) |
| Training scripts in 3.0 repo | None found containing cfg05/export logic |
| Champion reports in docs/ | Not found (`docs/reports/` has no `dayahead_current_champion.md`) |

**Blocker:** The `epf-sota-experiment` repository is not cloned or available
at any nearby path. This repository contains the cfg05 micro-search training
runs and the champion LightGBM model file.

---

## 3. cfg05 Artifact Search / Export Result

**Script:** `scripts/export_cfg05_from_source.py`

| Attempt | Result |
|---------|--------|
| `--source-repo /nonexistent` | `CFG05_EXPORT_BLOCKED` — source not found |
| No `--source-repo` | `CFG05_EXPORT_BLOCKED` — no path provided |
| Fake repo with no artifact | `CFG05_EXPORT_BLOCKED` — no candidates found |
| Fake repo with `cfg05_model.txt` | artifact found → `INVALID` (not a real model) |

**No real cfg05 artifact was located.**

---

## 4. cfg05 Feature Input Search / Build Result

**Script:** `scripts/build_cfg05_feature_input_from_source.py`

| Attempt | Result |
|---------|--------|
| No `--input-csv` or `--source-repo` | `MISSING` |
| Full column CSV + ds | `SCHEMA_READY` |
| Missing feature columns | `INVALID` |
| Missing ds | `INVALID` |
| Source repo search | Not performed (no source repo available) |

**No real cfg05 feature input CSV was located.**

---

## 5. cfg05 REAL Smoke Attempt

The full smoke pipeline (`run_cfg05_real_smoke_pipeline`) was **not executed
with real artifacts** — both artifact and input are missing.

Using placeholder artifacts, the pipeline correctly reports:
- `readiness_label: DATA_MISSING`
- `overall_status: PASS_STRUCTURAL`
- `cfg05_adapter_loaded: False`
- `validator_passed: False`

**REAL_READY not attempted because preconditions are unmet.**

---

## 6. Readiness Matrix

| Gate | Status | Condition |
|------|--------|-----------|
| cfg05_artifact | `MISSING` | No source repo available; no artifact exported |
| cfg05_input | `MISSING` | No input CSV provided or found |
| rt_assist_pack | `MISSING` | Not in scope for P11 |
| p5m_pack | `MISSING` | Not in scope for P11 |
| actual_ledger | `MISSING` | Not in scope for P11 |
| extrempriceclf_artifact | `MISSING` | Not in scope for P11 |

---

## 7. Local Artifact Path Policy

A local artifact policy has been established:

**Policy document:** `docs/LOCAL_ARTIFACTS.md`

**Ignored path:** `.local_artifacts/` (added to `.gitignore`)

All artifact export, CSV preparation, and smoke output must use paths under
`.local_artifacts/`. These files are never committed.

---

## 8. Tests Run

**P11 test file:** `tests/test_cfg05_source_export_attempt.py` (21 tests)

| Section | Tests | Description |
|---------|-------|-------------|
| TestExportCfg05FromSource | 9 | Missing source, blacklist, placeholder, copy flag, strict mode |
| TestBuildCfg05FeatureInputFromSource | 9 | Dynamic feature count, missing ds/cols, schema-ready, file write |
| TestCfg05RealSmokeAttempt | 1 | Placeholder never REAL_READY |
| TestForbiddenFilesCheck | 2 | No forbidden extensions in tracked/untracked files |

**Before P11:** 596 tests passing
**After P11:** 617 tests passing (+21 new)

---

## 9. Forbidden Files Check

**Status: PASS**

- `.gitignore` updated to include `.local_artifacts/`
- No CSV, pkl, joblib, parquet, feather, pt in tracked or untracked files
- `docs/LOCAL_ARTIFACTS.md` documents the policy
- Test suite verifies no leaks

---

## 10. Blockers and Exact Next Commands

### Blocker 1: Source repo not available

```
git clone https://github.com/disdorqin/epf-sota-experiment.git
```

Then re-run:

```
python -m scripts.export_cfg05_from_source \
    --source-repo /path/to/epf-sota-experiment --json --strict
```

### Blocker 2: No artifact found (after cloning)

If clone exists but no artifact is found:

```
python -m scripts.export_cfg05_from_source \
    --source-repo /path/to/epf-sota-experiment --verbose
```

This will search for training scripts containing `cfg05`/`lgbm`/`LightGBM`/
`save_model`/`Booster` patterns.

### Blocker 3: No feature input CSV

After artifact obtained:

```
python -m scripts.build_cfg05_feature_input_from_source \
    --source-repo /path/to/epf-sota-experiment
```

Or with an existing CSV:

```
python -m scripts.build_cfg05_feature_input_from_source \
    --input-csv /path/to/features.csv --target-day YYYY-MM-DD --strict
```

### To attempt REAL_READY

Once artifact + input are available:

```
python -m scripts.run_cfg05_real_smoke_pipeline \
    --cfg05-model /path/to/model.txt \
    --cfg05-input /path/to/features.csv \
    --target-day YYYY-MM-DD --strict --json
```

Target output: `readiness_label: REAL_READY`

---

## 11. P12 Recommendation

1. **Clone epf-sota-experiment** and locate the cfg05 champion model file
   (expected: LightGBM text model, filename `cfg05_model.txt` or similar).

2. **Export artifact** to `.local_artifacts/p12_cfg05/` using
   `export_cfg05_from_source.py --copy-if-found`.

3. **Prepare feature input** by running the source repo's feature pipeline,
   or by locating an existing CSV with all 42 `CFG05_FEATURE_COLUMNS` + `ds`.

4. **Run full REAL smoke pipeline** to achieve `REAL_READY`.

5. If both artifact and input are obtained but the smoke pipeline fails,
   diagnose the adapter or validator issue in P13.

6. **Never commit** artifact files, CSVs, or outputs to this repository.
