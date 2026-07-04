# Local Artifacts Policy

## Purpose

This document defines the policy for handling local model artifacts, CSV data
files, and other non-source files that are needed for REAL_READY verification
but must **never** be committed to the repository.

## Ignored Path

```
.local_artifacts/
```

All local artifact export, copy, and generation operations must use paths
under `.local_artifacts/`. This path is `.gitignore`d and will never be
committed.

## What May Go in .local_artifacts/

- Model weight files (`.txt`, `.pkl`, `.joblib`, `.pt`)
- Prepared input CSVs with feature columns
- Prediction output JSON summaries
- Smoke test output files
- Any generated data needed for REAL_READY verification

## What Must Never Be Committed

| File type | Reason |
|-----------|--------|
| `*.csv` | Data files — blocked by `.gitignore` |
| `*.pkl`, `*.joblib` | Model serialization — blocked by `.gitignore` |
| `*.parquet`, `*.feather` | Columnar data — blocked by `.gitignore` |
| `*.pt`, `*.pth`, `*.ckpt` | Torch checkpoints — blocked by `.gitignore` |
| `data/` contents | Raw/processed data |
| `outputs/` contents | Generated outputs |
| `reports/local/` contents | Local reports |

## Export Workflow

1. **Locate artifact** — `scripts/locate_cfg05_artifact.py`
2. **Export to local dir** — `scripts/export_cfg05_from_source.py --copy-if-found`
3. **Build input** — `scripts/build_cfg05_feature_input_from_source.py --out .local_artifacts/...`
4. **Run smoke** — `scripts/run_cfg05_real_smoke_pipeline.py --strict`

All output paths should use `.local_artifacts/p11_cfg05/` or similar.

## Enforcement

- `.gitignore` blocks all forbidden extensions and paths
- Test suite (`test_forbidden_files_check`) verifies no leaks
- CI should reject any PR containing forbidden file types
