# Client Demo Commands

## Quick Smoke Test

```bash
python main.py --target-start 2026-06-01 --target-end 2026-06-01 --json --strict
```

## Full Production Chain

```bash
python main.py \
    --raw-data data/shandong_pmos_hourly.csv \
    --dayahead-source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --fusion-engine period_bgew \
    --work-dir .local_artifacts/production_run \
    --target-start 2026-06-01 \
    --target-end 2026-06-30 \
    --strict --strict-no-leakage \
    --json
```

## Check Artifact Status

```bash
python -m scripts.run_p97_production_artifact_registry --json
```

## Run Tests

```bash
python -m pytest tests/ -q
```
