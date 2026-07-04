# Next Step Source Review Checklist

## Purpose

Before code migration, review all source repositories and produce a grounded source review report.

## Repositories to review

- disdorqin/epf-sota-experiment
- realtime SOTA repository or local path: MISSING
- disdorqin/electricity_forecast_model2.0_exp, branch tune-timemixer
- electricity_forecast_model2.5 repository or local path: MISSING

## Questions to answer

### Day-ahead

1. Which valid model outputs exist?
2. How many valid candidates can enter the learner?
3. Does cfg05 remain the trusted champion?
4. Which CatBoost outputs are safe candidates?
5. Which outputs must be excluded?

### Realtime

1. Where is the realtime SOTA repository?
2. Does it contain the RT assist pack?
3. What are the exact input and output schemas?
4. Can it output standard ledger rows?
5. Where is the 2.5 SGDFNet model and how does it run?

### Residual

1. Are P5M plugin directories present?
2. Do P5M tests pass?
3. Is canonical pack available locally?
4. If not, which smoke results must be marked DATA-MISSING?

### 2.5 chain

1. Where are ledger_full and ledger_full_range?
2. Where is the learner?
3. Where is the negative price classifier?
4. How does 2.5 store prediction ledger and actual ledger?
5. What must be changed to support the new adapters?

## Output

Create:

```text
docs/reports/source_review_report.md
```

Use MISSING or DATA-MISSING whenever evidence is absent.
