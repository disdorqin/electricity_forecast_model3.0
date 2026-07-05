# P111: Production Bugfix Report

> **Date**: 2026-07-05
> **Bugs fixed**: 8

## Bug 1: P110 report + production_certification.json missing
- Created `docs/reports/p110_final_production_go_no_go_report.md`
- Created `production_certification.json` with full component status

## Bug 2: Realtime actual ledger used wrong column
- `scripts/run_full_chain.py` `_build_actual_ledger()` now uses `实时电价` for realtime
- Day-ahead uses `日前电价` (unchanged)

## Bug 3: Adaptive training days hardcoded
- Added `_step_adaptive_training_days()` with real D-1 backward scan
- Reports: selected_days, skipped_days, training_rows, lookback window
- Status: COMPLETE_30D / DEGRADED_MIN_DAYS / INSUFFICIENT_DAYS

## Bug 4: Postflight hardcoded PASSED
- Now calls `delivery.postflight.run_postflight_validation()`
- Records checks and errors; strict mode fails on postflight failure

## Bug 5: Fallback ladder hardcoded PASSED
- Now calls `delivery.fallback_ladder.run_fallback_ladder()`
- Records: level, method, status, reason_codes

## Bug 6: Claim guard silent PASS
- Exception in claim guard → FAILED in strict mode, WARNING in non-strict
- No more silent "PASSED" on exception

## Bug 7: Production profiles
- Added `profiles` section to `config/production_artifacts.yaml`
- `rc_with_caveats` — allows optional SGDFNet/P5M/classifier
- `full_real_models` — requires all real artifacts

## Bug 8: Client note estimated metric
- Removed `~15% (estimated)` from `CLIENT_DELIVERY_NOTE.md`
- Changed to `Metric pending / evaluated separately`
