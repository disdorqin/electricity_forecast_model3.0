# Chain Design After User Clarification

## User clarified architecture

The 3.0 backend is split into two prediction branches: day-ahead and realtime.

Execution chain:

1. model predictions
2. residual correction module
3. learner and fusion module
4. negative price classifier
5. final output

## Current command decision

3.0 should first assemble a safe production chain with these blocks:

- DAYAHEAD_MODEL_ADAPTERS
- REALTIME_MODEL_ADAPTERS
- PREDICTION_LEDGER
- P5M_RESIDUAL_CORRECTION
- CORRECTED_PREDICTION_LEDGER
- 2.5_LEDGER_LEARNER_AND_FUSION
- 2.5_NEGATIVE_PRICE_CLASSIFIER
- FINAL_OUTPUT

## Open issue: model count for learner

The learner needs multiple valid candidates to learn useful weights.

Day-ahead known branch-summary candidates include cfg05_dayahead_lgbm, catboost_sota, catboost residual candidate, best_two_average, and stage3_business_fixed_baseline.

Only candidates that pass no-leakage, business-day, and 720-row checks may enter fusion.

Realtime known candidates include DA-safe realtime assist and 2.5 SGDFNet.

Source review must confirm whether additional realtime SOTA models exist.

## Safe rule

If a task has fewer than two valid candidates, the learner must use single_model_passthrough. Do not manufacture ensemble weights when candidate diversity is insufficient.
