## P30 Source-Aligned cfg05 + Fusion Readiness Report

### 1. Executive Status

| Phase | Status |
|-------|--------|
| P26 per-day retrain | P26_PER_DAY_RETRAIN_COMPLETE (sMAPE 17.06%) |
| P27 feature alignment | FEATURE_ALIGNMENT_PARTIAL |
| P28 actual ledger | P28_ACTUAL_LEDGER_READY (720 rows) |
| P29 model pool | P29_MODEL_POOL_SINGLE_MODEL_ONLY |
| P30 learner | P30_LEARNER_BLOCKED_SINGLE_MODEL |

### 2. P25 Baseline Recap

- cfg05 train-once 30-day backtest: sMAPE_floor50 = 20.71%, MAE = 68.04, RMSE = 86.74
- 30/30 days COMPLETE_24H, 720 eval rows
- Full chain: prediction → residual (NO_OP) → fusion (SINGLE_MODEL) → final (RULE_FALLBACK)

### 3. P26 Per-Day Retrain Result

- Strategy: train fresh model each day on [D-90d, D), predict 24H for D
- No data leakage — training window strictly before target day
- 29/30 days with metrics (June 30 has null y_true)
- **sMAPE_floor50 = 17.06%** (vs P21 train-once 20.71%, improvement -3.65pp)
- MAE = 53.89 (vs P21 68.04)
- RMSE = 64.41 (vs P21 86.74)
- Per-day retrain significantly improved accuracy

### 4. P27 Feature Alignment Result

- cfg05: 42 features
- Source v2: 40 features
- Source v3: 54 features
- Label: FEATURE_ALIGNMENT_PARTIAL
- cfg05 is a near-superset of v2 and subset of v3
- 12 features from v3 not in cfg05 (volatility, ranks, change, interaction, exact spring festival)

### 5. P28 Actual Ledger Result

- 720 rows for June 2026 (30 days x 24 hours)
- 30/30 complete days
- 0 duplicate keys
- 24 null y_true rows (June 30 — last day not yet available)
- Schema valid

### 6. P29 Model Pool Readiness

| Model | Readiness | Notes |
|-------|-----------|-------|
| cfg05 | REAL_24H_READY | Backtested, artifact exists |
| best_two_average | TRAINABLE_LOCAL | Needs training |
| stage3_business_fixed | TRAINABLE_LOCAL | Needs training |
| catboost_spike_residual | TRAINABLE_LOCAL | Needs training |
| catboost_sota | TRAINABLE_LOCAL | Needs training |

Only 1 real model — cannot form fusion pool.

### 7. P30 Learner Result

- Status: P30_LEARNER_BLOCKED_SINGLE_MODEL
- Only cfg05 available — learner correctly refuses to fake multi-model fusion
- Design doc created for when >=2 models become available

### 8. Metrics Comparison

| Metric | P21 (train-once) | P26 (per-day retrain) | Source |
|--------|-------------------|------------------------|--------|
| sMAPE_floor50 | 20.71% | **17.06%** | 11.48% |
| MAE | 68.04 | **53.89** | — |
| RMSE | 86.74 | **64.41** | — |

### 9. Source 11.48% Claim Boundary

Cannot claim 11.48% reproduced because:
- Feature set mismatch (42 vs 54 cols)
- Training strategy may differ (per-day retrain vs expanding window)
- Feature builder version gap (v2 vs v3)

### 10. What Is Now Real

- Per-day retrain walk-forward engine (P26) — 17.06% sMAPE, improved from 20.71%
- Feature alignment audit (P27) — cfg05 42 cols, v2 40 cols, v3 54 cols
- Actual ledger with 720 rows (P28)
- Model pool readiness assessment (P29) — cfg05 REAL_24H_READY, 4 others TRAINABLE_LOCAL
- Fusion learner design with single-model guard (P30) — correctly blocked

### 11. What Is Still Blocked

- Multi-model fusion: need >=2 trained models
- P5M residual correction: no trained P5M pack
- ExtremPriceClf: no trained model artifact
- BGEW weight learning: blocked by single model
- Source 11.48% reproduction: feature gap + methodology gap

### 12. Next Sprint

P31-P35: Train additional models to unblock fusion
- Train catboost_sota locally using source repo adapter
- Validate 24H completeness for each new model
- Backtest each model on June 2026 window
- Re-run P30 with >=2 models to enable real fusion
- Compare fusion sMAPE vs cfg05-alone sMAPE
