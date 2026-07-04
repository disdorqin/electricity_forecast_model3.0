# P12 Report: Source Repo Clone + cfg05 REAL Export Run

> **Phase**: P12 — Clone epf-sota-experiment, run artifact export, build feature input, attempt REAL smoke
> **Generated**: 2026-07-04
> **Test count**: 630 total (617 prior + 13 P12 new), 0 failures

---

## Summary

P12 implements the full orchestration pipeline that:
1. Locates or clones the `epf-sota-experiment` source repo
2. Runs cfg05 artifact export from source
3. Builds/validates feature input for cfg05
4. Optionally runs the REAL smoke pipeline if both gates pass
5. Returns a structured summary with 17 required keys

## Key Components

### `scripts/run_p12_cfg05_source_clone_and_smoke.py`

The P12 orchestration script with a single entry point:

**`run_p12_cfg05_source_clone_and_smoke(...)`** — Returns `dict[str, Any]` with 17 keys:
- `source_repo_status` / `source_repo_path` — how the source was located
- `artifact_status` / `artifact_candidates` / `copied_artifact_path` — export results
- `input_status` / `input_candidates` / `prepared_input_path` — feature input results
- `real_smoke_attempted` / `real_smoke_status` / `prediction_rows` / `validator_passed` / `readiness_label` — smoke results
- `final_status` / `reason_codes` — aggregated status
- `forbidden_files_check` — pass-through placeholder

**Source repo discovery** (3 methods):
1. `--source-repo PATH` → `EXISTING_PATH` if directory exists
2. Default `clone_dir` already exists → `ALREADY_CLONED`
3. `--clone-url` provided → attempt `git clone`, returns `CLONED_OK` or `CLONE_FAILED`
4. None of the above → `NO_SOURCE_PROVIDED` → `CFG05_EXPORT_BLOCKED`

**Path safety**: Both `clone_dir` and `work_dir` are checked against forbidden prefixes (`data/`, `outputs/`, `ledgers/`, `reports/local/`). Relative paths must start with `.local_artifacts/`.

**CLI modes**:
- `--json` — output structured JSON report
- `--strict` — exit non-zero on any blocker (except `CFG05_REAL_READY_LOCAL`)
- `--verbose/-v` — debug logging

**Final status determination**:
| Condition | Status |
|-----------|--------|
| Readiness == `REAL_READY` | `CFG05_REAL_READY_LOCAL` |
| Artifact loadable + input schema ready | `CFG05_REAL_SMOKE_FAILED` |
| Artifact loadable only | `CFG05_ARTIFACT_FOUND_INPUT_BLOCKED` |
| Input schema ready only | `CFG05_ARTIFACT_BLOCKED_INPUT_FOUND` |
| Export blocked | `CFG05_EXPORT_BLOCKED` |
| Input blocked | `CFG05_INPUT_BLOCKED` |

### `tests/test_p12_cfg05_source_clone_and_smoke.py`

13 tests in `TestP12Orchestration`:

| # | Test | Assertion |
|---|------|-----------|
| 1 | `test_existing_source_repo_path_used` | Existing dir → `EXISTING_PATH` |
| 2 | `test_no_source_or_clone_returns_export_blocked` | No inputs → `CFG05_EXPORT_BLOCKED` |
| 3 | `test_unsafe_clone_dir_rejected` | `data/` clone_dir → blocked |
| 4 | `test_unsafe_work_dir_rejected` | `outputs/` work_dir → blocked |
| 5 | `test_fake_source_placeholder_not_real_ready` | Placeholder artifact → not `REAL_READY` |
| 6 | `test_fake_source_no_input_reports_input_blocked` | No features → no loadable artifact |
| 7 | `test_supplied_input_validated` | Valid CSV → `SCHEMA_READY` |
| 8 | `test_real_smoke_not_attempted_without_gates` | Gates fail → smoke not attempted |
| 9 | `test_non_strict_exit_0_with_blocker` | Non-strict → exit 0 |
| 10 | `test_strict_exit_nonzero_with_blocker` | Strict with blocker → exit != 0 |
| 11 | `test_summary_contains_all_required_keys` | 17 required keys present |
| 12 | `test_already_cloned_dir_detected` | Existing clone_dir → `ALREADY_CLONED` |
| 13 | `test_forbidden_files_check` | No `.csv`/`.pkl`/`.joblib` etc. in untracked |

## Source Repo Exploration

The source repo (`disdorqin/epf-sota-experiment`) was successfully cloned to `.local_artifacts/source_repos/epf-sota-experiment/`.

**Key findings**:
- **cfg05 confirmed champion**: 11.48% sMAPE_floor50 (from `docs/reports/dayahead_current_champion.md`)
- **No pre-saved model weights**: Models are trained by scripts and saved to `outputs/` (gitignored)
- **No CSV data files**: Feature data must be built from external Chinese CSV data
- **Training requires**: Raw CSV with columns (时刻, 日前电价, 直调负荷预测值, etc.)
- **Feature builder**: `src/common/feature_builder_dayahead.py` converts raw data to 42 CFG05 feature columns

**Result**: `CFG05_EXPORT_BLOCKED` — source code exists but external training data and model training are required to produce artifacts.

## Test Results

```
630 passed in 12.94s
```

## Files Created

| File | Description |
|------|-------------|
| `scripts/run_p12_cfg05_source_clone_and_smoke.py` | P12 orchestration script (405 lines) |
| `tests/test_p12_cfg05_source_clone_and_smoke.py` | 13 P12 tests (229 lines) |
| `docs/reports/p12_source_repo_clone_cfg05_real_export_run_report.md` | This report |

## Next Steps

1. **Obtain training data** — external Chinese CSV with load/price data
2. **Run model training** — `run_dayahead_tabular_model_search.py` to generate cfg05 model artifacts
3. **Re-run P12** — with generated artifacts, expect `CFG05_ARTIFACT_FOUND_INPUT_BLOCKED` or better
4. **Provide feature input** — CSV with 42 CFG05_FEATURE_COLUMNS to reach `CFG05_REAL_SMOKE_FAILED` or `CFG05_REAL_READY_LOCAL`
