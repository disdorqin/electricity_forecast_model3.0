# Day-Ahead Model Zoo Handoff

## Source repository

```text
disdorqin/epf-sota-experiment
```

## Role

This branch is the day-ahead model branch. It is not realtime, not the production chain, and not the final fusion system.

## Final champion

```text
model_id: cfg05
formal_name: lightgbm_cfg05_dayahead
sMAPE_floor50: 11.4838% / reported 11.48%
```

Configuration:

```text
model = LightGBM
window = 90d
objective = mae
num_leaves = 191
min_data_in_leaf = 30
learning_rate = 0.015
lambda_l1 = 0.1
lambda_l2 = 5.0
feature_fraction = 0.85
bagging_fraction = 0.95
bagging_freq = 5
n_estimators = 2000
```

## Expected default model zoo

```python
DEFAULT_FUSION_POOL = [
    "cfg05",
    "best_two_average",
    "stage3_business_fixed",
    "catboost_spike_residual",
    "catboost_sota",
]
```

## Candidate roles

| model_id | role | metric | use |
| --- | --- | ---: | --- |
| cfg05 | champion | 11.4838% | main day-ahead model |
| best_two_average | strong_candidate | about 11.85% | fusion input |
| stage3_business_fixed | strong_candidate | about 11.86% | fusion input |
| catboost_spike_residual | diversity_fallback | about 12.47% | diversity / fallback |
| catboost_sota | baseline_fallback | about 12.58% | baseline / fallback |

## Invalid models

```text
lgbm_spike_residual_1127
stage3_old_1164
lightgbm_90d_orig_1197
```

Reasons:

```text
lgbm_spike_residual_1127: target leakage, y_true used as prediction feature
stage3_old_1164: natural-day mapping error
lightgbm_90d_orig_1197: 690 rows only, missing hour 24
```

Any script request for invalid models should raise immediately.

## Expected packaged files

```text
src/registry/dayahead_models.py
scripts/run_dayahead_model_zoo.py
scripts/validate_dayahead_model_zoo.py
tests/test_dayahead_model_zoo_contract.py
docs/reports/dayahead_model_zoo.md
scripts/run_champion_cfg05.py
```

Current GitHub main check:

```text
src/registry/dayahead_models.py: NOT_FOUND_ON_MAIN
docs/reports/dayahead_model_zoo.md: NOT_FOUND_ON_MAIN
```

So execution AI must either locate the files on another branch/local copy or create the missing registry wrapper in 3.0.

## Unified output schema

```text
task
model_name
target_day
business_day
ds
hour_business
period
y_true
y_pred
```

Hard requirements:

```text
720 rows
task all dayahead
hour_business = 1..24
business_day D hour 24 = D+1 00:00
y_true identical across candidates
y_pred no NaN
no duplicate key
```

Join key:

```text
target_day
business_day
ds
hour_business
period
```

## Required checks before merge

```bash
python -m pytest tests/test_no_target_leakage.py
python scripts/check_stage3_business_day_mapping.py
python -m pytest tests/test_cfg05_champion_contract.py
python -m pytest tests/test_dayahead_model_zoo_contract.py
```

Reproduce champion:

```bash
python scripts/run_champion_cfg05.py
```

Expected:

```text
full_30d sMAPE_floor50 approximately 11.4838%
720 rows
hour_business 1-24
business-day mapping correct
no target leakage
```

## Fusion recommendation

1. Use cfg05 alone as baseline.
2. Then test cfg05 + best_two_average + stage3_business_fixed.
3. CatBoost models should be diversity/fallback, not naive simple average.
4. Fusion weights must be learned from rolling/search window, not full test 30 days.
5. Do not retune day-ahead models during 3.0 merge.
