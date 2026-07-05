# P47 One-Command Local Delivery Runner Report

> **Generated**: 2026-07-05
> **Status**: P47_DELIVERY_CHAIN_PASS

---

## 1. Overview

`scripts/run_delivery_local_chain.py` is the single entry point for the entire delivery pipeline.

## 2. CLI

```bash
python -m scripts.run_delivery_local_chain \
    --raw-data ../data/shandong_pmos_hourly.csv \
    --source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --start-day 2026-06-01 \
    --end-day 2026-06-30 \
    --work-dir .local_artifacts/delivery_run \
    --json --strict
```

## 3. Pipeline Steps

| # | Step | Description | Caching |
|---|------|-------------|---------|
| 1 | raw_data_check | Validate CSV columns & timestamps | Marker file |
| 2 | source_repo_check | Check source repo models dir | Marker file |
| 3 | trust_gate | P41: model trust gate | Cache JSON |
| 4 | actual_ledger | P34: actual ledger alignment | Cache JSON |
| 5 | trusted_fusion | P42: trusted fusion backtest | Cache JSON |
| 6 | rolling_validation | P43: rolling weight validation | Cache JSON |
| 7 | delivery_summary | P44: delivery packager | Written to disk |
| 8 | forbidden_file_check | Scan outputs for .pkl/.joblib/y_true | Inline |
| 9 | claim_guard | P46: claim guard on docs | Inline |

## 4. Output Files

| File | Description |
|------|-------------|
| `<work-dir>/delivery_summary.json` | Full P44 delivery summary |
| `<work-dir>/metrics.json` | Extracted metrics (cfg05, fusion, rolling) |

## 5. Profile Support

- `trusted_delivery` (default, delivery allowed)
- `balanced_candidate` (manual review required)
- `research_all_models` (research only)
