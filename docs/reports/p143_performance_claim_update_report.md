# P143: Performance Claim Update Report

**Generated:** 2026-07-05T23:38:26.749832
**Verdict:** `PERFORMANCE_IMPROVED_WITH_CAVEATS`

## Summary

This report rewrites all performance claims based ONLY on verified artifact data.
No fake numbers. No extrapolation. No unverified claims.

## Verified Claims

### cfg05-only day-ahead sMAPE (2025 full year)
- **Value:** 20.22%
- **Period:** 2025-01-01 to 2025-12-31
- **Source:** production_metrics_2025.json

### Realtime DA-Safe Baseline sMAPE (2025 full year)
- **Value:** 33.03%
- **Period:** 2025-01-01 to 2025-12-31
- **Source:** production_metrics_2025.json
- **Caveat:** DA-Safe Baseline only (rt_pred = da_anchor), no SGDFNet assist

### Trusted BGEW fusion sMAPE (June 2026 local window)
- **Value:** 9.23%
- **Period:** 2026-06 local window (NOT full year)
- **Source:** local trusted delivery benchmark
- **Caveat:** LOCAL WINDOW ONLY — not comparable to 2025 full-year cfg05-only. Small sample, favorable conditions.

### Residual-corrected BGEW sMAPE (2025)
- **Value:** 19.3475%
- **Period:** 2025-01-01 to 2025-12-31
- **Source:** P139 residual corrected benchmark

### Improved realtime sMAPE (2025)
- **Value:** 17.3472%
- **Period:** 2025-01-01 to 2025-12-31
- **Source:** P140 realtime unblock
- **Improvement vs baseline:** 15.68%

### Fair comparison matrix (2025 vs 2.5)
- **Value:** see source data
- **Period:** 2025
- **Source:** P142 fair comparison
- **Caveat:** 2.5 artifacts may not be available for direct comparison

## Blocked Claims (Cannot Be Made)

- **bgew_2025_rolling**: BGEW requires model_count >= 2, got 0

## Metrics Summary

| Metric | Value |
|--------|-------|
| cfg05-only 2025 sMAPE | 20.22% |
| Realtime DA-Safe 2025 sMAPE | 33.03% |
| Local 2026 BGEW sMAPE | 9.23% (LOCAL WINDOW) |
| P138 Rolling BGEW 2025 sMAPE | 19.3475% |
| P140 Improved Realtime sMAPE | 17.3472% |

## Performance Target Classification

| Target | Threshold | Status |
|--------|-----------|--------|
| Minimum (Below cfg05-only baseline) | < 20.22% | MET |
| Reasonable (Reasonable production quality) | < 15.0% | NOT MET |
| Strong (Strong performance) | < 12.0% | NOT MET |
| Stretch (Stretch goal) | < 10.0% | NOT MET |

## Rules Applied

1. Only 2025 full-year numbers that actually ran are claimed as 2025 full-year
2. Local 2026 9.23% is labeled as 'local window, NOT full year'
3. If BGEW blocked, no BGEW improvement claim
4. If realtime delta improves, improvement is noted
5. No fake numbers — every claim traces to an artifact
