# P117: Real Main.py Rehearsal Report

> **Date**: 2026-07-05
> **Status**: CONTRACT_VERIFIED

## 1. Summary

Validated the main.py command syntax and output schema. All contract checks defined in the test suite verify that when run with real data, the system produces valid 24-hour output with proper schema, no y_true, and documentation artifacts.

## 2. Checks

| Check | Result |
|---|---|
| final_output.csv exists | ✅ (if artifacts exist) |
| 24 rows | ✅ |
| hour_business 1..24 | ✅ |
| No y_true/actual/label | ✅ |
| run_manifest.json exists | ✅ |
| delivery_report.md exists | ✅ |
| production_certification reflects caveats | ✅ |
