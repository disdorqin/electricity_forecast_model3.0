# P25 Real cfg05 Chain Consolidated Report

> **Phase**: P25 — Real data execution + consolidated report
> **Generated**: 2026-07-04
> **Sprint**: P21–P25 Local Real Data Execution
> **Test count**: 839 total, 0 failures

---

## 1. Executive Status

**Overall**: cfg05 30-day real backtest COMPLETED on local machine. Full chain (prediction → residual → fusion → classifier) executed with real predictions. Source 11.48% reproduction NOT claimed (methodology gap: model reuse vs per-day retrain).

**Key achievement**: Went from `CFG05_BACKTEST_BLOCKED` (P16/P20) to `CFG05_BACKTEST_COMPLETE` with real data.

---

## 2. Local Data Discovery

| Resource | Status | Path |
|----------|--------|------|
| Raw CSV | FOUND | `../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv` (39408 rows, GBK) |
| Source repo | FOUND | `.local_artifacts/source_repos/epf-sota-experiment` (data_loader + feature_builder_dayahead) |
| P5M pack | NOT FOUND | Code exists in 2.0_exp but no trained .pkl/.joblib artifact |
| ExtremPriceClf | NOT FOUND | Code exists in 2.1/ExtremPriceClf but no trained model |
| Actual ledger | NOT FOUND | No actual_ledger.csv in work dir |

---

## 3. Real cfg05 Backtest Result

| Metric | Value |
|--------|-------|
| Final status | **CFG05_BACKTEST_COMPLETE** |
| Eval window | 2026-06-01 ~ 2026-06-30 |
| Attempted days | 30 |
| Complete days | **30/30** (all 24H) |
| Incomplete days | 0 |
| Eval rows | 696 (29 days × 24h; 1 day missing y_true) |
| Missing y_true rows | 24 (1 day) |
| Metric days | 29 |
| Training strategy | Model reuse (train once on pre-June data, predict all 30 days) |
| Feature builder | Source repo `feature_builder_dayahead.py` (extended features) |

---

## 4. Metrics

| Metric | Value |
|--------|-------|
| sMAPE_floor50 | **20.71%** |
| MAE | 68.04 |
| RMSE | 86.74 |
| n_observations | 696 |

**Comparison with source 11.48%**: Local metric is significantly higher. This is expected because:
1. **Training strategy gap**: Local uses model reuse (train once), source likely uses per-day walk-forward retrain
2. **Evaluation window**: Local uses June 2026, source window unknown
3. **Feature builder**: May differ in detail from source version
4. **No hyperparameter tuning**: Using frozen cfg05 params without window-specific adaptation

---

## 5. 24H Completeness

**All 30 days are COMPLETE_24H.** Every target day has exactly 24 rows with hour_business 1..24. The P15 hour-24 fix (canonical window [D+01:00, D+1+01:00)) is working correctly on real data.

---

## 6. Prediction Ledger Result

| Metric | Value |
|--------|-------|
| Final status | **CFG05_PREDICTION_LEDGER_READY_LOCAL** |
| Input rows | 720 (30 × 24) |
| Ledger rows | 720 |
| Target days | 30 |
| Complete days | 30 |
| Duplicate keys | 0 |
| Schema valid | True |
| Completeness | COMPLETE_24H |

---

## 7. Full Chain Result

| Metric | Value |
|--------|-------|
| Final status | **CFG05_FULL_CHAIN_READY_WITH_FALLBACKS** |
| Input prediction rows | 720 |
| Corrected rows | 720 |
| Fusion rows | 720 |
| Final rows | 720 |
| Validators passed | corrected_schema, fusion_schema, final_schema |
| Row counts | CONSISTENT (720 through all stages) |
| Readiness label | LOCAL_CHAIN_READY |

### Fallback Labels (Honest)

| Stage | Label |
|-------|-------|
| Residual | P5M_DATA_MISSING_NO_OP |
| Fusion | CFG05_SINGLE_REAL_MODEL_FUSION |
| Classifier | NEGATIVE_CLASSIFIER_RULE_FALLBACK |

---

## 8. Artifact Upgrade Scan

| Module | Status | Detail |
|--------|--------|--------|
| P5M residual | NO_UPGRADE | Source code in 2.0_exp/extreme/negative_price/ but no trained .pkl/.joblib |
| ExtremPriceClf | NO_UPGRADE | Code in 2.1/ExtremPriceClf/ but no trained model artifact |
| BGEW | BLOCKED | No actual_ledger.csv for weight training |
| Final scan status | **P24_NO_REAL_ARTIFACTS_FOUND** |

---

## 9. Source Methodology Alignment

P19 audit result: **SOURCE_METHODOLOGY_PARTIAL** (9/16 matched, 5 partial, 2 not matched)

Primary gaps:
- Walk-forward retrain strategy: local reuses model, source may retrain per day
- Source evaluation window not verified
- Feature builder version not byte-verified

---

## 10. What Can Be Claimed

- **Local cfg05 24H backtest completed**: 30/30 days, all 24H complete
- **Full chain runs end-to-end with real predictions**: 720 rows through all stages
- **Prediction ledger is canonical**: schema valid, no duplicates, 24H enforced
- **No data leakage into git**: all artifacts in .local_artifacts/ (gitignored)

---

## 11. What Cannot Be Claimed

- **Source 11.48% reproduction**: NOT claimed. Local sMAPE = 20.71%, methodology PARTIAL
- **P5M REAL**: No trained pack available
- **ExtremPriceClf deployed**: No trained artifact
- **BGEW production**: No actual ledger for training
- **Metrics comparable to source**: Different training strategy, different window

---

## 12. Remaining Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| Walk-forward strategy gap | Cannot claim 11.48% | Implement per-day retrain |
| No P5M trained pack | Residual stays NO_OP | Train/export P5M from 2.0_exp |
| No ExtremPriceClf artifact | Classifier stays RULE_FALLBACK | Train classifier model |
| No actual ledger | BGEW stays STRUCTURAL | Build actual_ledger from y_true |
| 1 day missing y_true | 24 rows excluded | Verify raw data completeness |

---

## 13. Next Sprint

1. **Per-day retrain**: Implement true walk-forward (retrain each day) to close methodology gap
2. **P5M pack export**: Train P5M residual model from 2.0_exp, export to .local_artifacts/
3. **Actual ledger build**: Extract y_true from raw CSV into actual_ledger.csv
4. **3-month backtest**: Run 90-day backtest with per-day retrain
5. **Methodology re-audit**: Re-run P19 after per-day retrain implementation
6. **Target**: Get sMAPE_floor50 closer to 11.48% with aligned methodology

---

## Readiness Matrix (Updated)

```
Day-ahead cfg05:        REAL_LOCAL / 24H_READY / BACKTESTED (30 days)
Realtime assist:        DRY_RUN / DA_ONLY (unchanged)
P5M residual:           DATA_MISSING_NO_OP (no trained pack found)
Fusion:                 CFG05_SINGLE_REAL_MODEL_LOCAL (real cfg05 input)
BGEW:                   STRUCTURAL (no actual ledger)
Negative classifier:    RULE_FALLBACK (no trained artifact)
Final output:           LOCAL_CHAIN_READY_WITH_FALLBACKS (real predictions)
```

---

*End of P25 consolidated report. 839 tests total, 0 failures.*
*P21–P25 Local Real Data Execution Sprint complete.*
