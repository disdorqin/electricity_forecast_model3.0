# DeepRT-Finalize-1 Prompt

## Source repository

```text
disdorqin/electricity_forecast_deep_sgdf_delta
```

## Role

This repository is the realtime electricity price branch / DeepRT experiment branch for the 2.0 system.

It is not:

```text
day-ahead branch
final fusion system
production chain
```

## Final positioning

```text
DA-Safe Realtime Assist Model
```

Default prediction:

```text
rt_pred = da_anchor
```

Deep / ML outputs are sidecar assist signals and should not overwrite DA by default.

## Required assist outputs

```text
da_error_prob
residual_direction_prob
residual_magnitude_bucket
uncertainty_score
correction_permission
reason_codes
```

## Confirmed repository modules

```text
models/deep_sgdf_delta/business_time.py
models/deep_sgdf_delta/metrics.py
models/deep_sgdf_delta/deep_rt_sota_features.py
models/deep_sgdf_delta/deep_rt_sota_dataset.py
models/deep_sgdf_delta/deep_rt_sota_model.py
scripts/train_working.py
docs/DEEP_RT_SOTA_2B_RESULTS.md
```

## Important facts

1. `business_time.py` is the single source of truth for business-day alignment.
2. `metrics.py` computes sMAPE on realtime price, not residual.
3. `DEEP_RT_SOTA_2B_RESULTS.md` reports NO_GO for direct deep RT replacement.
4. Residual is weakly autocorrelated and hard to predict.
5. DA anchor is a strong baseline and should remain the default.

## P0 hardening tasks

```text
1. Fix dataset test interface mismatch.
2. Disable hourly mode in production or fully implement hourly samples.
3. Fix MLP input dimension risk.
4. Export hourly predictions, not only daily mean predictions.
5. Formal mode must not fill target NaN with 0.
6. Fix or rename previous_7d_same_hour_mean because implementation is rolling mean, not strict same-hour mean.
7. Confirm whether latest DA-safe / RT-assist pack artifacts are pushed.
```

## P1 hardening tasks

```text
1. Mark TCN as experimental because it is not strict causal TCN.
2. Create feature NaN fill manifest.
3. Explicitly record forecast_price fallback to da_anchor.
4. Block synthetic risk features from formal metrics.
```

## Required final artifacts

```text
exported_models/rt_assist_pack/feature_manifest.json
exported_models/rt_assist_pack/predict_schema.json
exported_models/rt_assist_pack/model_card.md
exported_models/rt_assist_pack/config.yaml
exported_models/rt_assist_pack/optional classifier artifacts
scripts/export_rt_assist_pack.py
scripts/predict_rt_assist_pack.py
docs/DEEP_RT_FINAL_MODEL_CARD.md
docs/REALTIME_BRANCH_FINAL_SUMMARY.md
docs/CHAIN_HANDOFF_REALTIME_BRANCH.md
docs/DEEP_RT_FINALIZATION_RESULTS.md
```

## Required prediction command

```bash
python scripts/predict_rt_assist_pack.py \
  --model-dir exported_models/rt_assist_pack \
  --data-path <data.csv> \
  --start 2026-02-01 \
  --end 2026-02-28 \
  --out predictions/rt_assist_predictions.csv
```

## Required hourly output schema

```text
business_day
hour_business
ds
da_anchor
rt_pred
safe_correction
final_pred_source
da_error_prob_50
da_error_prob_100
da_error_prob_150
da_error_prob_200
prob_residual_up
prob_residual_down
prob_residual_neutral
expected_abs_residual
residual_magnitude_bucket
uncertainty_score
correction_permission
reason_codes
model_version
```

## Fallback rule

If no classifier is available or safe correction is not GO:

```text
rt_pred = da_anchor
final_pred_source = DA_ONLY
```

## Final verdict criteria

### READY_FOR_CHAIN_HANDOFF

```text
code interface stable
predict script callable
hourly schema emitted
DA-only fallback explicit
no leakage tests pass
```

### NOT_READY

```text
interface problems remain
tests broken
hourly output schema unavailable
fallback unclear
```
