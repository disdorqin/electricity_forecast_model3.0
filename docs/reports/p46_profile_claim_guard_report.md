# P46 Profile Registry + Claim Guard Report

> **Generated**: 2026-07-05
> **Status**: P46_CLAIM_GUARD_PASS

---

## 1. Profile Registry

File: `config/fusion_profiles.yaml`

Three profiles defined:

| Profile | Models | Delivery | Default |
|---------|--------|----------|---------|
| `trusted_delivery` | cfg05 + catboost_spike_residual | ✅ Yes | ✅ Yes |
| `balanced_candidate` | cfg05 + best_two_average + catboost_sota + catboost_spike_residual | ❌ No (manual review) | ❌ |
| `research_all_models` | All 5 including stage3 | ❌ No | ❌ |

## 2. Claim Guard

File: `scripts/validate_delivery_claims.py`

Scans all markdown files in `docs/reports/` and `README.md` for forbidden delivery claims:

| Forbidden Pattern | Label | Severity |
|-------------------|-------|----------|
| `2.97% production` | production_sMAPE_2_97 | Violation |
| `69.96% production` | production_improvement_69_96 | Violation |
| `stage3 production readiness` | stage3_production_readiness | Violation |
| `11.48% reproduced/reproduc` | source_11_48_reproduced | Violation |

If any of "research only", "not delivery", or "stage3 leakage" caveats are present nearby, the severity is downgraded to **warning**.

## 3. Scan Results

- Files scanned: all 10 reports in `docs/reports/` + `README.md`
- Violations: 0
- Warnings: 0
- Default profile: `trusted_delivery`

## 4. Usage

```bash
# Quick check
python -m scripts.validate_delivery_claims

# JSON output
python -m scripts.validate_delivery_claims --json

# Strict mode (exit non-zero on any violation)
python -m scripts.validate_delivery_claims --strict
```
