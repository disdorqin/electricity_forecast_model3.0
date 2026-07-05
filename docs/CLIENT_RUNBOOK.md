# Client Runbook

## One-Command Production Run

```bash
python main.py \
    --raw-data data/shandong_pmos_hourly.csv \
    --dayahead-source-repo .local_artifacts/source_repos/epf-sota-experiment \
    --profile trusted_delivery \
    --fusion-engine period_bgew \
    --work-dir .local_artifacts/production_run \
    --strict --strict-no-leakage \
    --json
```

## Output Files

| File | Description |
|------|-------------|
| `final_output.csv` | 24-hour fused price predictions |
| `delivery_report.md` | Human-readable delivery report |
| `run_manifest.json` | Run metadata and status |
| `production_artifact_status.md` | Artifact availability status |

## Data Requirements

- Raw CSV with Chinese columns: 时刻, 日前电价, 实时电价, etc.
- Source repo with trained model artifacts

## Troubleshooting

- Check `run_manifest.json` for step-by-step status
- Check `production_artifact_status.md` for missing artifacts
- Run with `--json` for machine-readable output
