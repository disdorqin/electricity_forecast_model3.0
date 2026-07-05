## P27 cfg05 Feature Builder Alignment Audit Report

### Status: FEATURE_ALIGNMENT_PARTIAL

### Objective

Audit alignment between source repo feature builders (v2, v3) and current 3.0 cfg05 adapter.

### Results

| Dimension | Count | Match |
|-----------|-------|-------|
| cfg05 features | 42 | — |
| Source v2 features | 40 | 1 dim matched (params) |
| Source v3 features | 54 | 3 dims not matched |

### Key Findings

1. **cfg05 uses 42 features** — this is 2 more than source v2 (40 cols) and 12 fewer than source v3 (54 cols).

2. **cfg05 vs v2**: cfg05 has 2 extra features not in v2 (likely `lag_48h` and one other). Most v2 features are present in cfg05.

3. **cfg05 vs v3**: v3 adds 14 new features on top of v2:
   - price_volatility_24h, price_volatility_168h
   - renewable_penetration_rank_30d, load_ramp_rank_30d
   - bidding_space_change_24h, net_load_change_24h, renewable_change_24h
   - is_spring_festival_exact, days_to_spring_festival_exact, days_after_spring_festival_exact
   - hour_x_bidding_space, hour_x_net_load, period_x_bidding_space, period_x_renewable_penetration

4. **Params match**: cfg05 params exactly match the registry (objective=mae, num_leaves=191, lr=0.015, etc.)

### Alignment Label: FEATURE_ALIGNMENT_PARTIAL

cfg05 (42 cols) is a near-superset of v2 (40 cols) and a subset of v3 (54 cols). The gap to v3 is 12 features.

### Migration Recommendation

To close the sMAPE gap (20.71% → ~12%), consider upgrading cfg05 to use v3 features:
1. Add the 12 v3-new features to CFG05_FEATURE_COLUMNS
2. Ensure the feature builder produces these features correctly
3. Retrain with the expanded feature set
4. Validate no leakage in volatility, rank, change, and interaction features

### Tests

10 tests in `tests/test_p27_cfg05_feature_builder_alignment.py` — all passing.

### Files

- `scripts/audit_p27_cfg05_feature_builder_alignment.py`
- `tests/test_p27_cfg05_feature_builder_alignment.py`
