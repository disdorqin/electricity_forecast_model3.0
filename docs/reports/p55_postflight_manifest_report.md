# P55 Postflight + Manifest + Delivery Report

> **Generated**: 2026-07-05
> **Phase**: P55
> **Status**: IMPLEMENTED

---

## 1. Implementation Overview

P55 adds three modules to the `delivery/` package that together form the final validation and reporting layer for the delivery pipeline:

| Module | File | Purpose |
|--------|------|---------|
| **Postflight** | `delivery/postflight.py` | 12 quality/safety checks on the final output CSV |
| **Manifest** | `delivery/manifest.py` | Create, write, read, and validate delivery run manifests (JSON) |
| **Report** | `delivery/report.py` | Generate markdown + JSON delivery reports and terminal output |

These modules are designed to be called at the end of the delivery chain (after P47 runner completes) to validate the output, record metadata, and produce human-readable and machine-readable reports.

## 2. Postflight Check List

The `run_postflight()` function executes 12 checks against the delivery output CSV:

| # | Check Name | What It Validates |
|---|------------|-------------------|
| 1 | `file_exists_readable` | final_output exists and is a readable CSV |
| 2 | `twenty_four_rows` | Exactly 24 data rows (one per business hour) |
| 3 | `hour_business_range` | `hour_business` column contains values 1..24 |
| 4 | `no_duplicate_hours` | No duplicate `hour_business` values |
| 5 | `no_nan_in_predictions` | No NaN in `y_pred` / price / forecast columns |
| 6 | `business_day_consistency` | `business_day` column contains the target date |
| 7 | `profile_delivery_allowed` | The profile used has `delivery_allowed: True` |
| 8 | `no_quarantined_models` | No quarantined models appear in `allowed_models` |
| 9 | `claim_guard_pass` | Calls `validate_delivery_claims.run_claim_guard()` |
| 10 | `no_git_tracked_artifacts` | Work directory is not tracked by git |
| 11 | `hour24_convention` | Hour 24 follows D+1 00:00:00 convention |
| 12 | `no_merge_suffixes` | No `_x`/`_y` suffix columns from bad merges |

Each check returns `{"passed": bool, "detail": str}`. The overall status is:

- **PASS**: All checks passed
- **WARN**: At most 2 checks failed (degraded but usable)
- **FAIL**: 3+ checks failed

## 3. Manifest Schema

Manifests are JSON files stored as `run_manifest.json` in the delivery output directory:

```json
{
    "run_id": "str — unique run identifier",
    "target_day": "str — YYYY-MM-DD",
    "profile": "str — profile name",
    "started_at": "str — ISO format start timestamp",
    "completed_at": "str — ISO format completion timestamp",
    "status": "str — PASS/FAIL/WARN",
    "delivery_status": "str — delivery-specific status label",
    "selected_training_days": "int — number of training days",
    "trusted_models": ["str — model names that passed trust gate"],
    "quarantined_models": ["str — model names that failed trust gate"],
    "fusion_method": "str — e.g. BGEW, equal_weight",
    "fallback": {
        "fallback_used": "bool",
        "fallback_method": "str — method name if fallback triggered"
    },
    "postflight": "dict — full postflight result dict",
    "metrics": "dict — key-value metrics",
    "warnings": ["str"],
    "errors": ["str"]
}
```

Key functions:

- `create_manifest(...)` — Build a manifest dict with all required fields
- `write_manifest(manifest, output_dir)` — Write to `output_dir/run_manifest.json`
- `read_manifest(manifest_path)` — Read and parse a manifest JSON file
- `validate_manifest_keys(manifest)` — Check all required keys are present

## 4. Report Format

The `generate_delivery_report(manifest, output_dir)` function produces two files:

### `delivery_report.md` (Markdown)

Sections:
- **Delivery Status**: Run ID, target day, status, profile, fusion method
- **Training Summary**: Number of training days
- **Model Pool**: Trusted and quarantined model lists
- **Postflight Results**: Per-check pass/fail table with details
- **Metrics**: Key-value metrics table
- **Warnings / Errors**: Lists of any issues
- **Output Files**: Paths to final output and submission-ready file

### `delivery_report.json` (JSON)

Contains the full manifest, report data, and generation timestamp for programmatic consumption.

### Terminal Report

`print_terminal_report(manifest)` prints a formatted, colored header report to stdout with all key information at a glance.

## 5. Integration

The P55 modules integrate with the existing delivery pipeline:

1. **P47 runner** produces `final_output.csv` and `metrics.json`
2. **P55 postflight** validates `final_output.csv` against 12 checks
3. **P55 manifest** creates a delivery manifest with all metadata
4. **P55 report** generates markdown + JSON reports for the run

Typical usage:

```python
from delivery.postflight import run_postflight
from delivery.manifest import create_manifest, write_manifest
from delivery.report import generate_delivery_report, print_terminal_report

# 1. Run postflight checks
postflight_result = run_postflight(
    output_path=".local_artifacts/delivery_run/final_output.csv",
    target_date="2026-07-05",
    profile_name="trusted_delivery",
    profile_def=profile_definition,
)

# 2. Create manifest
manifest = create_manifest(
    run_id="delivery-20260705-001",
    target_day="2026-07-05",
    profile="trusted_delivery",
    started_at="2026-07-05T08:00:00",
    completed_at="2026-07-05T08:30:00",
    status=postflight_result["status"],
    delivery_status="DELIVERY_READY",
    selected_training_days=30,
    trusted_models=["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
    quarantined_models=["stage3_business_fixed"],
    fusion_method="BGEW",
    fallback_used=False,
    fallback_method="",
    postflight=postflight_result,
    metrics={"sMAPE": 9.23, "MAE": 15.4},
    warnings=[],
    errors=postflight_result.get("errors", []),
)

# 3. Write manifest
write_manifest(manifest, ".local_artifacts/delivery_run")

# 4. Generate reports
generate_delivery_report(manifest, ".local_artifacts/delivery_run")
print_terminal_report(manifest)
```

## 6. Test Summary

| Category | Tests | Description |
|----------|-------|-------------|
| Postflight PASS | 1 | Valid 24-row output passes |
| Postflight FAIL | 5 | 23 rows, NaN, duplicates, missing file, non-delivery profile |
| Postflight quarantine | 1 | Detects quarantined model in allowed list |
| Postflight merge suffixes | 1 | Detects _x/_y bad merge columns |
| Manifest creation | 1 | Correct structure and defaults |
| Manifest write/read | 2 | Valid JSON, round-trip fidelity |
| Manifest missing file | 1 | Raises FileNotFoundError |
| Manifest required keys | 1 | All schema keys present |
| Report file creation | 1 | Both .md and .json created |
| Report sections | 1 | All required sections in markdown |
| Report JSON content | 1 | Manifest stored in JSON report |
| Terminal report | 1 | Runs without error |
| Report data fields | 1 | Expected fields present |
| Individual checks | 3 | File exists, 24 rows, hour range |
| **Total** | **21** | |

All tests can be run with:

```bash
python -m pytest tests/test_p55_postflight_manifest_report.py -v --tb=short
```
