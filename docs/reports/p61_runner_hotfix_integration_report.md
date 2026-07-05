# P61 Runner Hotfix Integration Report

> **Generated**: 2026-07-05
> **Status**: P61_COMPLETE

---

## 1. Overview

P61 fixes 7 delivery-blocking bugs in the P57 runner integration layer, ensuring
the safety supervisor modules (P52-P56) are correctly connected to the
one-command runner.

### Files

| File | Description |
|------|-------------|
| `scripts/run_delivery_local_chain.py` | Complete rewrite with all 7 bug fixes |
| `tests/test_p61_runner_hotfix_integration.py` | 40 tests covering all fixes |
| `docs/reports/p61_runner_hotfix_integration_report.md` | This report |

---

## 2. Bugs Fixed

| Bug | Description | Root Cause | Fix |
|-----|-------------|------------|-----|
| 1 | raw_data VALID → FAILED | `CFG05_RAW_DATA_VALID` was in the FAILED list | Now: VALID → PASSED, MISSING → FAILED |
| 2 | adaptive_training_days missing `trusted_models` + `actual_ledger_path` | Runner didn't pass required params to `select_complete_training_days()` | Both params now passed correctly |
| 3 | safety_preflight before ledger generation | Step order had safety before actual_ledger | Reordered: ledgers → safety → training days |
| 4 | Sentinel return structure parsed wrong | Runner expected `sentinel["model_statuses"]` but sentinel returns `sentinel["models"]` (list) | Now reads `sentinel["models"]` list of `{model_name, status}` |
| 5 | postflight call used `output_df=` not `output_path=` | Runner passed DataFrame, but API expects file path | Now: `run_postflight(output_path=, target_date=, profile_name=, profile_def=)` |
| 6 | Fallback ladder output not persisted | Runner didn't write `ladder["output"]` to CSV | Added `to_csv("final_output.csv")` |
| 7 | `--fusion-engine` did not dispatch P56 | No code path for regime_bgew | Added `step_regime_bgew_fusion()` + dispatch logic |

---

## 3. Step Order (Fixed)

Old (broken): raw → source → safety → training → trust → actual → fusion → ...

New (correct): raw → source → trust → actual_ledger → prediction_ledger → safety → training → fusion dispatch → fallback → postflight → manifest → ...

---

## 4. Test Results

**P61 tests: 40 passed, 0 failed**
**Full suite: 1401 passed, 0 failed**
