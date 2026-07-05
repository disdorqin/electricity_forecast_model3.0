# P126: Final 2025 Metrics Audit Report

> **Date**: 2026-07-05
> **Verdict**: `2025_METRICS_VALIDATED_WITH_CAVEATS`

## 1. Why was the 2025 result 20.22% / 33.03%?

### Day-ahead 20.22%

**Root cause**: The `main.py` pipeline Step 4 calls `run_p16_cfg05_30d_walkforward_backtest`, which runs **cfg05 single-model only**. The P31-P40 multi-model pool (catboost_spike_residual, BGEW fusion) is **NOT** part of the main delivery path.

The prediction ledger for 2025 contains exactly 1 model:
- `lightgbm_cfg05_dayahead`

No `catboost_spike_residual`, no BGEW fusion weights. The 20.22% is **cfg05-only**, not the trusted BGEW fusion that showed 9.23% on the June 2026 local window.

### Realtime 33.03%

**Root cause**: `rt_pred = da_anchor` (DA-Safe Baseline strategy). No SGDFNet assist was available. The realtime prediction equals the day-ahead prediction, compared against `实时电价`.

Additionally, two bugs were found:
- `build_actual_ledger_from_raw_csv.py` hardcoded `日前电价` for all tasks instead of using `实时电价` for realtime (P122)
- `export_eval_pack()` in `realtime_deep_adapter.py` used `日前电价` instead of `实时电价` as y_true (P123)

## 2. Was it a code bug?

Partially. The numbers are **correct for cfg05-only without BGEW fusion**. But there were real bugs:

| Bug | File | Impact |
|---|---|---|
| Residual corrected output not fed to learner/fusion | `run_full_chain.py` | Residual correction had no effect on final fusion (P124) |
| Actual ledger hardcoded 日前电价 | `build_actual_ledger_from_raw_csv.py` | Realtime actuals used wrong column (P122) |
| Realtime eval pack used 日前电价 | `realtime_deep_adapter.py` | Eval pack y_true was wrong (P123) |

These did NOT affect the 2025 metrics calculation because:
- The 2025 metrics used the `per_day_metrics.csv` from `run_p16_cfg05_30d_walkforward_backtest`, not from the `final_output.csv`
- The residual-corrected vs raw ledger bug only affected the fusion step

## 3. Bugs fixed

| Bug | Fix | File |
|---|---|---|
| Residual not feeding learner | Changed `da_ledger` → `da_corrected` | `run_full_chain.py` |
| Residual not feeding fusion | Changed `da_ledger` → `da_corrected` | `run_full_chain.py` |
| Actual ledger hardcoded column | Added `task` parameter | `build_actual_ledger_from_raw_csv.py` |
| Actual ledger output path | Changed to `{task}_actual_ledger.csv` | `build_actual_ledger_from_raw_csv.py` |
| Realtime eval pack column | Changed to `实时电价` | `realtime_deep_adapter.py` |

## 4. Fixed 2025 metrics

With fixes applied, a re-run of 2025 would show:
- Day-ahead: 20.22% (unchanged — same cfg05 model, same sMAPE calculation)
- Realtime: 33.03% (unchanged — same baseline strategy)

The real improvement requires running the full P31-P40 multi-model pipeline with BGEW fusion for 2025.

## 5. What can be claimed to client

- ✅ `main.py` runs 365 days without crash
- ✅ cfg05-only day-ahead sMAPE: 20.22% (2025 full year)
- ✅ Realtime DA-Safe Baseline sMAPE: 33.03% (2025 full year)
- ✅ Local window BGEW fusion sMAPE: 9.23% (June 2026, trusted delivery)

## 6. What CANNOT be claimed

- ❌ "3.0 BGEW fusion beats 2.5" — no fair comparison available
- ❌ "3.0 achieves 9.23% on 2025 full year" — that was a single local window
- ❌ "SGDFNet production ready" — code-only
- ❌ "Full P5M ready" — no-op fallback
- ❌ "ML classifier production ready" — rule fallback

## 7. Does 3.0 beat 2.5?

**Unknown**. No 2.5 artifacts are available on the evaluation machine. The 2.5 reported ~14% day-ahead and ~24% realtime, but those were on different time windows with different data.

## 8. Next optimization steps

1. Run 2025 with full P31-P40 multi-model pool + BGEW fusion
2. Run 2025 with residual correction feeding into learner
3. Run 2025 with SGDFNet assist if available
4. Run fair comparison vs 2.5 on identical time window
