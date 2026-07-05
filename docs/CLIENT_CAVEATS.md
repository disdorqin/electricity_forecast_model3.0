# Client Caveats

## Current Caveats (as of 2026-07-05)

1. **SGDFNet Assist — CODE_ONLY**
   - SGDFNet source exists in model2.0_exp repo but production runtime
     requires additional setup.
   - Realtime falls back to DA-Safe Baseline (rt_pred = da_anchor).
   - This is acceptable for production — SGDFNet is an optional enhancement.

2. **P5M Full Residual — NO_OP_FALLBACK**
   - Full 5-model residual stack is not yet assembled.
   - CatBoost spike residual is available as partial correction.
   - Residual correction is best-effort only.

3. **ML Classifier — RULE_FALLBACK**
   - ML classifier artifacts exist (negative_risk, spike_risk models)
     but are not yet in the automated production path.
   - Classification uses rule-based fallback.

## Forbidden Claims

These claims must NEVER appear in delivery context:

- "FINAL_REAL_INTEGRATED_GO" unless all artifacts pass
- "SGDFNet production ready" unless runtime verified
- "Full P5M ready" unless full stack assembled
- "ML classifier production ready" unless in ML path

## Upgrade Path to FINAL_GO

1. SGDFNet runtime verified with 24H output
2. P5M full stack trained and verified
3. ML classifier artifacts in automated path
4. 30-day rehearsal passes with all components
