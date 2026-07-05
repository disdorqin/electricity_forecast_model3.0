# P51: 2.5 Mechanism Alignment Audit

> **Generated**: 2026-07-05
> **Status**: AUDIT_COMPLETE

---

## 1. 2.5 Five-Stage Pipeline

| Stage | 2.5 Implementation | 3.0 Equivalent | 3.0 Absorption Plan |
|-------|-------------------|-----------------|---------------------|
| `ledger_predict` | 7 models (3 DA + 4 RT) predict target-day 24h | P41 trust gate selects model pool | Already in P47 runner — no change needed |
| `ledger_weight` | Adaptive 30d complete training day selection + dynamic weights | BGEW in `fusion/weights.py` | **P52: adaptive_training_days.py** — port adaptive selection |
| `ledger_fuse` | Task/period/model weighted fusion | `fusion/engine.py` with BGEW | **P56: trust_gated_regime_bgew.py** — add regime awareness |
| `ledger_classifier` | Realtime -80 classification + correction | DA_ONLY/DRY_RUN — not real | **Not absorbed** — 3.0 realtime is DRY_RUN, must document honestly |
| `final_outputs` | submission_ready.csv 24 rows, 6 columns | P44 delivery packager | Already in P47 — enhance with fallback |

**Decision:**
- `ledger_predict` — Already matches, no change
- `ledger_weight` — **Absorb with adaptation** (P52)
- `ledger_fuse` — **Absorb with innovation** (P56)
- `ledger_classifier` — **Cannot absorb** — 3.0 RT is DA_ONLY/DRY_RUN, must label honestly
- `final_outputs` — Already matches, enhance with postflight (P55)

---

## 2. 2.5 Adaptive Complete Training Days

| Property | 2.5 | 3.0 Adaptation |
|----------|-----|----------------|
| Target day | D-1 start scanning | Same |
| Required days | 30 | 30 (configurable) |
| Max lookback | 180 | 180 (configurable) |
| Min degraded | N/A (hard requirement in 2.5) | 7 days (new 3.0 addition) |
| Prediction check | 3 models x 24h (DA), 4 models x 24h (RT) | N trusted models x 24h (DA only) |
| Actual check | 24h per task | 24h (DA only for day-ahead task) |
| Status | Implicit (fail if <30) | **Explicit**: COMPLETE_30D / DEGRADED_MIN_DAYS / INSUFFICIENT_DAYS / NO_VALID_DAYS |
| RT check | Required | **Skipped** — 3.0 realtime is DRY_RUN |

**Decision:** Absorb with modifications:
- Add explicit status codes (new in 3.0)
- Add degraded mode (required_days=30, min_days_for_degraded=7)
- DA-only (no RT check since 3.0 RT is not real)
- Configurable parameters via CLI

---

## 3. 2.5 Delivery Status Three-Tier

| Status | 2.5 Definition | 3.0 Adaptation |
|--------|---------------|----------------|
| `NORMAL` | 5 stages complete, postflight PASS | Trusted fusion success + postflight PASS |
| `DEGRADED_DELIVERED` | Normal failed, emergency fallback produced valid output | Fallback ladder produced valid 24H output |
| `FAILED_NO_DELIVERY` | All fallback failed | All 6 fallback levels failed |

**Decision:** **Fully absorb** — this is the core delivery status model for 3.0. The exit code convention (0=NORMAL, 2=DEGRADED, 1=FAILED) is also absorbed.

---

## 4. 2.5 Postflight / Fallback / Manifest / Report

| Mechanism | 2.5 | 3.0 Absorption |
|-----------|-----|----------------|
| **Postflight** | `delivery_quality.validate_daily_submission()` — checks 24 rows, columns, NaN, manifest consistency | **P55: delivery/postflight.py** — same checks + profile check + claim guard |
| **Emergency Fallback** | `emergency_fallback.try_emergency_fallback()` — historical median | **P54: delivery/fallback_ladder.py** — 6-level ladder (not just historical median) |
| **Manifest** | `run_manifest.json` with stages, delivery_status, fallback info | **P55: delivery/manifest.py** — same concept, adapted for 3.0 profile system |
| **Delivery Report** | `delivery_report.md` + `delivery_report.json` with formatted output | **P55: delivery/report.py** — same dual output, 3.0-aware |

**Decision:**
- **Postflight** — Absorb with enhancements (add profile check, claim guard, git tracking check)
- **Fallback** — **Enhance beyond 2.5**: 6-level ladder vs 2.5's single-level emergency_fallback
- **Manifest** — Absorb, add profile metadata
- **Report** — Absorb dual format (JSON + Markdown)

---

## 5. What Can Be Copied Directly

| Item | Source | Target |
|------|--------|--------|
| Historical median computation | `emergency_fallback.py` | `delivery/fallback_ladder.py` |
| Submission validation logic | `delivery_quality.validate_daily_submission()` | `delivery/postflight.py` |
| Manifest file I/O (UTF-8) | `delivery_report.py` | `delivery/manifest.py` |
| Exit code mapping (0/1/2) | `main.py` | `delivery/report.py` |
| Terminal report formatting | `delivery_report.py` | `delivery/report.py` |
| Adaptive scanning logic | `ledger_weight.py` | `fusion/adaptive_training_days.py` |
| `NORMAL`/`DEGRADED_DELIVERED`/`FAILED_NO_DELIVERY` | Global convention | Global convention |

---

## 6. What Cannot Be Copied (Must Be Rebuilt or Skipped)

| Item | Reason | 3.0 Action |
|------|--------|------------|
| RT (realtime) model pool | 3.0 RT is DA_ONLY/DRY_RUN, not real RT predictions | Skip RT checks, label as DRY_RUN |
| LEDGER directory structure | 2.5 uses `outputs/ledger/{task}/` with parquet; 3.0 uses `.local_artifacts/` with CSV ledgers | Use 3.0's existing CSV ledger paths |
| Model list | 2.5 uses lightgbm/timesfm/timemixer for DA; 3.0 uses cfg05/catboost variants | Use 3.0 profile-based model selection |
| 5-stage chain naming | 2.5 uses `ledger_predict → weight → fuse → classifier → final` | 3.0 uses `trust_gate → fusion → rolling → summary → postflight → manifest` |
| Realtime classifier | 3.0 does NOT have realtime -80 classifier | Skip — document as future work |
| Data sync pipeline | 2.5 has `sync_dataset` pipeline | Not needed — 3.0 uses raw CSV directly |
| Backfill pipeline | 2.5 has `ledger_backfill` | Not needed for 3.0's scope |

---

## 7. 3.0 Innovation / Transformation Points

| Innovation | 2.5 Baseline | 3.0 Enhancement |
|------------|-------------|-----------------|
| **Profile system** | No profile concept | 3 profiles: trusted_delivery / balanced_candidate / research_all_models |
| **Trust gate** | No runtime guard | P41 + P53 leakage sentinel with quarantine |
| **Claim guard** | No automated claim checking | P46 automated forbidden-claims scanning |
| **Fallback ladder** | Single-level emergency_fallback | 6-level ladder with different methods |
| **Regime-aware fusion** | Simple inverse-MAE BGEW | Period (3) + Regime (4) dimensional weighting |
| **Fusion safety floor** | No floor | cfg05_floor = 0.30, min/max weight bounds |
| **Degraded mode** | Hard fail if <30 days | Degraded allowed with min_days_for_degraded=7 |
| **Delivery status** | Implicit | Explicit + manifest tracking |
| **Git safety checks** | Manual (`git status`) | Automated in postflight |

---

## 8. Integration Plan Summary

| Phase | What | Depends On |
|-------|------|-----------|
| P52 | `fusion/adaptive_training_days.py` | Nothing |
| P53 | `safety/leakage_sentinel.py` | Nothing |
| P54 | `delivery/fallback_ladder.py` | P53 (model status checks) |
| P55 | `delivery/postflight.py`, `manifest.py`, `report.py` | Nothing |
| P56 | `fusion/trust_gated_regime_bgew.py` | P52 (training days) |
| P57 | Runner integration (`run_delivery_local_chain.py`) | P52-P56 |
| P58 | Failure injection tests | P52-P56 |
| P59 | Docs update | P57 |
| P60 | Final safety freeze audit | P52-P59 |

---

## 9. Key Findings

1. **2.5 → 3.0 absorption is feasible** — Core delivery concepts (adaptive days, postflight, fallback, manifest, report) transfer cleanly
2. **3.0 must NOT copy realtime classifier** — 3.0 RT is DA_ONLY/DRY_RUN, honest labeling is required
3. **3.0's profile system is more mature** — Profile-based model selection is an improvement over 2.5's hardcoded model lists
4. **3.0's safety mechanisms are stronger** — Leakage sentinel, claim guard, and fallback ladder exceed 2.5's capabilities
5. **The main gap is explicit delivery status tracking** — 3.0's P47 runner lacks NORMAL/DEGRADED/FAILED status output

---

## 10. Action Items for P52-P60

| # | Action | Phase |
|---|--------|-------|
| 1 | Implement adaptive training day selector for 3.0 | P52 |
| 2 | Implement runtime leakage sentinel | P53 |
| 3 | Implement 6-level fallback ladder | P54 |
| 4 | Implement postflight + manifest + report | P55 |
| 5 | Implement regime-aware BGEW fusion | P56 |
| 6 | Integrate all modules into runner | P57 |
| 7 | Write failure injection tests | P58 |
| 8 | Update documentation | P59 |
| 9 | Final safety freeze audit | P60 |
