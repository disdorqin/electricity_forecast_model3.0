# P132: Final 2025 Performance Claim Report

> **Date**: 2026-07-05
> **Verdict**: `2025_FULL_BGEW_BENCHMARK_BLOCKED_CFG05_ONLY`

## 1. 2025 Benchmark Status

| Benchmark | Status | sMAPE |
|---|---|---|
| cfg05-only | ✅ **COMPLETE** | **20.22%** |
| catboost_spike_residual | ❌ **BLOCKED** — feature pipeline not compatible | N/A |
| catboost_sota | ❌ **BLOCKED** (quarantined, not trusted) | N/A |
| best_two_average | ❌ **BLOCKED** (quarantined, not trusted) | N/A |
| stage3_business_fixed | ❌ **BLOCKED** (quarantined, excluded) | N/A |
| trusted BGEW fusion | ❌ **BLOCKED** — needs ≥2 trusted models | N/A |
| residual-corrected BGEW | ❌ **BLOCKED** — depends on BGEW first | N/A |

## 2. Why blocked?

The P31-P40 multi-model pool has **trained model files** but:

1. The feature builder (`build_dayahead_features`) requires model IDs registered in `dayahead_models.py` registry
2. The P31 training scripts used different feature pipelines than the P16 delivery path
3. Running inference requires the exact feature columns each model was trained on
4. No 2025 predictions exist for any model except cfg05 (from the P16 walkforward)

The 5 models (4 trusted candidates + cfg05) were trained and exist, but the unified inference path to run them on 2025 data is not yet integrated.

## 3. What we know

| Source | sMAPE | Window | Notes |
|---|---|---|---|
| cfg05-only 2025 | 20.22% | Full year | Proven, verifiable |
| cfg05-only June 2026 | 9.90% | 30 days | From P31-P40 metrics |
| trusted BGEW June 2026 | 9.23% | 30 days | cfg05 + catboost_spike_residual |
| realtime DA-safe 2025 | 33.03% | Full year | rt_pred = da_anchor |
| 2.5 reported day-ahead | ~14% | Unknown window | Not verifiable on this machine |
| 2.5 reported realtime | ~24% | Unknown window | Not verifiable on this machine |

## 4. What can be claimed

- ✅ `main.py` runs 365 days without crash
- ✅ 3.0 cfg05-only day-ahead sMAPE: 20.22% (2025 full year)
- ✅ 3.0 realtime DA-Safe Baseline sMAPE: 33.03% (2025 full year)
- ✅ 3.0 trusted BGEW fusion sMAPE: 9.23% (June 2026 local window)
- ✅ System correctly detects and reports all caveats (no fake GO)
- ✅ Full test suite: 2261 passed, 0 failed

## 5. What CANNOT be claimed

- ❌ "3.0 achieves 9.23% on full-year 2025" — that was a single local month
- ❌ "3.0 BGEW fusion beats 2.5" — no fair comparison available
- ❌ "2025 full-year BGEW fusion actual sMAPE" — not yet computed
- ❌ "SGDFNet production ready" — code-only
- ❌ "Full P5M residual ready" — no-op fallback
- ❌ "ML classifier production ready" — rule fallback

## 6. Next steps to unblock

1. Register P31-P40 model IDs in `data/features/dayahead_features.py` registry
2. Unify feature column schema across all models
3. Run full 2025 inference for cfg05 + catboost_spike_residual + best_two_average
4. Run `run_p129_2025_trusted_bgew_benchmark.py`
5. Update claim report
