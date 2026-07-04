# P19 cfg05 Source Methodology Alignment Report

> **Phase**: P19 — source methodology alignment audit
> **Generated**: 2026-07-04
> **Test count**: 772 total, 0 failures

---

## 1. Executive Status

| Component | Status |
|-----------|--------|
| Audit script | `scripts/audit_cfg05_source_methodology_alignment.py` — created |
| Audit dimensions | 16 dimensions audited |
| Label | **SOURCE_METHODOLOGY_PARTIAL** |
| Claim | **source 11.48% reproduction not claimed** |

## 2. Audit Results

### Dimensions Matched (9/16)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 3 | target_day_definition | MATCHED | Business day convention: D 00:00 → D-1 hour 24 |
| 4 | hour_24_mapping | MATCHED | [D+01:00, D+1+01:00) = 24 hours, hour_business 1..24 |
| 5 | train_window_length | MATCHED | 90 days |
| 6 | cfg05_lightgbm_params | MATCHED | Frozen: num_leaves=191, lr=0.015, obj=mae |
| 7 | feature_columns | MATCHED | Same 42-feature cfg05 set |
| 10 | y_true_availability | MATCHED | 日前电价 from raw CSV |
| 11 | null_y_true_filtering | MATCHED | Drop null y_true before metrics |
| 12 | metric_formula | MATCHED | sMAPE_floor50 = 200*mean(\|y_f-yp_f\|/(|y_f|+|yp_f|)) |
| 15 | invalid_model_blacklist | MATCHED | Excludes leakage/natural-day/690-row models |

### Dimensions Partial (5/16)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 1 | source_repo_report_date_window | PARTIAL | Source repo available but exact eval window not extracted |
| 2 | evaluation_start_end | PARTIAL | Local eval window known, source window not verified |
| 8 | feature_builder_version | PARTIAL | Source feature_builder exists but version not byte-verified |
| 9 | raw_data_file_row_range | PARTIAL | Same file, row range may differ |
| 14 | single_model_reuse_vs_per_day_retrain | PARTIAL | Local reuse strategy, source not fully verified |
| 16 | source_champion_config_equivalence | PARTIAL | Frozen params used, source config not directly compared |

### Dimensions Not Matched (2/16)

| # | Dimension | Status | Detail |
|---|-----------|--------|--------|
| 13 | walk_forward_retrain_strategy | NOT_MATCHED | Local uses model reuse (train once), source may use per-day retrain |

## 3. Claim

```
source 11.48% reproduction not claimed
local cfg05 metric is comparable only with caveats
```

The audit found 9 dimensions matched, 5 partial, and 2 not matched. The primary gap is the walk-forward retraining strategy: local implementation reuses a single model across all eval days (for computational efficiency), while the source methodology may retrain per day. Additionally, the source repo's exact evaluation window and feature builder version have not been byte-verified.

## 4. Labels

| Label | Condition |
|-------|-----------|
| SOURCE_METHODOLOGY_MATCHED | All 16 dimensions matched |
| SOURCE_METHODOLOGY_PARTIAL | Most matched, some partial/few not matched |
| SOURCE_METHODOLOGY_NOT_MATCHED | Many dimensions not matched |

## 5. Test Coverage (15 P19 tests)

| Group | Tests | Coverage |
|-------|-------|----------|
| TestAuditLabels | 4 | Valid label, no source repo, no summary, with summary |
| TestClaimRules | 4 | Claim rule, all dimensions present, counts sum to 16, dimension names |
| TestAuditDetails | 7 | Target day, hour 24, metric formula, walk-forward detection, local params, reason codes |

## 6. Files Changed/Created

| File | Action |
|------|--------|
| `scripts/audit_cfg05_source_methodology_alignment.py` | **NEW** |
| `tests/test_p19_cfg05_source_methodology_alignment.py` | **NEW** (15 tests) |
| `docs/reports/p19_cfg05_source_methodology_alignment_report.md` | **NEW** (this file) |

---

*End of P19 report. 772 tests total, 0 failures.*
