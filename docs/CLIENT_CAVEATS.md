# Client Caveats

**Updated:** 2026-07-05T23:38:26.749832 (P143 rewrite)

## Current Caveats

1. **Local 2026 BGEW 9.23% — LOCAL WINDOW ONLY**
   - This number comes from a local window (June 2026), NOT a full-year evaluation.
   - It is NOT directly comparable to the 2025 full-year cfg05-only 20.22%.
   - Small sample, potentially favorable conditions.

2. **SGDFNet Assist — CODE_ONLY**
   - SGDFNet source exists in model2.0_exp repo but production runtime
     requires additional setup.
   - Realtime falls back to DA-Safe Baseline (rt_pred = da_anchor).

3. **P5M Full Residual — NO_OP_FALLBACK**
   - Full 5-model residual stack is not yet assembled.
   - CatBoost spike residual is available as partial correction.
   - Residual correction is best-effort only.

4. **ML Classifier — RULE_FALLBACK**
   - ML classifier artifacts exist but are not yet in the automated production path.
   - Classification uses rule-based fallback.

## Blocked Claims (Cannot Be Made)

- **bgew_2025_rolling**: BGEW requires model_count >= 2, got 0

## Forbidden Claims

These claims must NEVER appear in delivery context:

- "BGEW full-year 2025 sMAPE" unless P138 artifacts exist with model_count >= 2
- "3.0 beats 2.5" unless P142 fair comparison confirms
- "9.23% is full-year performance" — it is LOCAL WINDOW ONLY
- "SGDFNet production ready" unless runtime verified
- "Full P5M ready" unless full stack assembled
- "ML classifier production ready" unless in ML path

## Verified Claims (P143)

- cfg05-only day-ahead sMAPE (2025 full year): 20.22%
- Realtime DA-Safe Baseline sMAPE (2025 full year): 33.03%
- Trusted BGEW fusion sMAPE (June 2026 local window): 9.23%
- Residual-corrected BGEW sMAPE (2025): 19.3475%
- Improved realtime sMAPE (2025): 17.3472%
