# Client Acceptance Checklist — Electricity Price Forecasting System v3.0.0-rc1

## Pre-Run Checks
- [ ] Python 3.11+ installed
- [ ] Dependencies: `pip install pandas numpy pyyaml lightgbm catboost joblib`
- [ ] Raw data CSV available (Chinese format with 时刻, 日前电价, 实时电价 columns)
- [ ] Source repo cloned: `.local_artifacts/source_repos/epf-sota-experiment`
- [ ] (Optional) SGDFNet source: `../electricity_forecast_model2.0_exp/SGDFNet`

## System Verification
- [ ] `python --version` ≥ 3.11
- [ ] `python -m pytest tests/ -q` — all passing
- [ ] `python main.py --help` — displays usage
- [ ] `python -m scripts.run_p97_production_artifact_registry --json` — registry works

## Production Run
- [ ] `python main.py ... --strict --strict-no-leakage` — exits 0 or with caveats
- [ ] `final_output.csv` generated with 24 rows
- [ ] `dayahead_price` column has no NaN
- [ ] `realtime_price` column has no NaN
- [ ] No `y_true`/`actual`/`label` in output
- [ ] `run_manifest.json` generated
- [ ] `delivery_report.md` generated

## Acceptance
- [ ] System produces 24-hour price predictions
- [ ] System handles missing artifacts gracefully (no crash)
- [ ] System reports honest caveats (not fake GO)

## Sign-off
- **Date:** _______________
- **Tester:** _______________
- **Verdict:** CLIENT_DELIVERY_READY_RC ☐ / CLIENT_DELIVERY_BLOCKED ☐
