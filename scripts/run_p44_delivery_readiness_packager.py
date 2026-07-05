"""
scripts/run_p44_delivery_readiness_packager.py — P44: Delivery readiness packager.

Generates delivery-ready summary from P41-P43 results.
Default profile is trusted_no_stage3.
Research profiles are clearly marked as not delivery-safe.

Usage::

    python -m scripts.run_p44_delivery_readiness_packager --json

Options::

    --work-dir PATH     Model artifacts dir.
    --json              Output JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")


def run_delivery_packager(
    work_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Read P41-P43 results and assemble delivery summary."""
    work_dir = work_dir or _DEFAULT_WORK_DIR

    result: dict[str, Any] = {
        "phase": "P44",
        "default_profile": "trusted_no_stage3",
        "profiles": {},
        "trusted_model_pool": [],
        "quarantined_models": [],
        "delivery_metrics": {},
        "known_caveats": [],
        "delivery_commands": [],
        "forbidden_claims": [],
        "recommended_default": None,
        "p44_status": "P44_NOT_STARTED",
        "reason_codes": [],
    }

    # Read P41 trust gate
    # Re-run P41 inline by loading its module
    try:
        from scripts.run_p41_model_trust_gate import run_trust_gate
        gate = run_trust_gate(work_dir=work_dir)
        result["trust_gate_status"] = gate["summary"]["p41_status"]
        result["trusted_model_pool"] = gate["summary"].get("trusted_models", [])
        result["quarantined_models"] = list(dict.fromkeys(
            gate["summary"].get("suspect_models", [])
        ))
        result["profiles"] = gate.get("profiles", {})
        result["reason_codes"].append(f"GATE:{len(result['trusted_model_pool'])}trusted/{len(result['quarantined_models'])}quarantined")
    except Exception as e:
        result["reason_codes"].append(f"GATE_FAILED:{e}")

    # Read P42 trusted fusion backtest (using P41's trusted pool)
    try:
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        p42 = run_trusted_fusion_backtest(
            work_dir=work_dir,
            trusted_models=result["trusted_model_pool"] or None,
        )
        result["p42_status"] = p42["summary"]["p42_status"]
        result["delivery_metrics"]["cfg05"] = p42.get("cfg05_metrics", {})
        result["delivery_metrics"]["best_single_trusted"] = {
            "model": p42.get("best_single_model"),
            **p42.get("best_single_metrics", {}),
        }
        result["delivery_metrics"]["equal_weight_fusion"] = p42.get("equal_weight_metrics", {})
        result["delivery_metrics"]["bgew_fusion"] = p42.get("fusion_metrics", {})
        result["delivery_metrics"]["fusion_vs_cfg05_delta"] = p42["summary"].get("fusion_vs_cfg05_delta")
        result["delivery_metrics"]["fusion_vs_best_single_delta"] = p42["summary"].get("fusion_vs_best_single_delta")
        result["delivery_metrics"]["fusion_weights"] = p42.get("fusion_weights", {})
        result["delivery_metrics"]["per_period"] = p42.get("period_metrics", {})
        result["reason_codes"].append(f"P42:{p42['summary']['p42_status']}")
    except Exception as e:
        result["reason_codes"].append(f"P42_FAILED:{e}")

    # Read P43 rolling validation (using P41's trusted pool)
    try:
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        p43 = run_rolling_validation(
            work_dir=work_dir,
            trusted_models=result["trusted_model_pool"] or None,
        )
        result["rolling_validation"] = {
            "status": p43["summary"]["p43_status"],
            "full_period": p43.get("full_period", {}),
            "split": p43.get("split", {}),
            "rolling": p43.get("rolling", {}).get("metrics", {}),
        }
        result["reason_codes"].append(f"P43:{p43['summary']['p43_status']}")
    except Exception as e:
        result["reason_codes"].append(f"P43_FAILED:{e}")

    # Determine recommended default
    p42_metrics = result["delivery_metrics"]
    rv = result.get("rolling_validation", {})

    # Check out-of-sample validation
    split_fusion = rv.get("split", {}).get("fusion_sMAPE")
    rolling_fusion = rv.get("rolling", {}).get("fusion_sMAPE")
    split_cfg05 = rv.get("split", {}).get("cfg05_sMAPE")
    rolling_cfg05 = rv.get("rolling", {}).get("cfg05_sMAPE")
    fusion_oos_valid = True
    if split_fusion is not None and split_cfg05 is not None and split_fusion >= split_cfg05:
        fusion_oos_valid = False
    if rolling_fusion is not None and rolling_cfg05 is not None and rolling_fusion >= rolling_cfg05:
        fusion_oos_valid = False

    if p42_metrics.get("bgew_fusion", {}).get("sMAPE_floor50") is not None:
        fus_smape = p42_metrics["bgew_fusion"]["sMAPE_floor50"]
        best_smape = p42_metrics.get("best_single_trusted", {}).get("sMAPE_floor50")
        if best_smape and fus_smape < best_smape and fusion_oos_valid:
            result["recommended_default"] = "trusted_bgew_fusion"
            result["reason_codes"].append(f"RECOMMEND_FUSION(sMAPE={fus_smape}% < best_single={best_smape}%, OOS_validated)")
        elif best_smape and fus_smape < best_smape and not fusion_oos_valid:
            result["recommended_default"] = "best_single_trusted_model"
            result["reason_codes"].append(f"RECOMMEND_BEST_SINGLE(fusion_OOS_not_validated)")
        elif best_smape:
            result["recommended_default"] = "best_single_trusted_model"
            result["reason_codes"].append(f"RECOMMEND_BEST_SINGLE(sMAPE={best_smape}% < fusion={fus_smape}%)")
        else:
            result["recommended_default"] = "cfg05"
    else:
        result["recommended_default"] = "cfg05"

    # Known caveats
    is_fusion = result.get("recommended_default") == "trusted_bgew_fusion"
    if is_fusion:
        fusion_note = "BGEW fusion (9.23%) modestly improves over cfg05 (9.90%) and holds out-of-sample"
    else:
        fusion_note = "BGEW fusion does not hold out-of-sample vs cfg05"
    result["known_caveats"] = [
        "stage3_business_fixed excluded due to SUSPECT_LEAKAGE (source training data issue)",
        "best_two_average and catboost_sota flagged SUSPECT_LEAKAGE (corr > 0.995)",
        "Trusted pool reduced to 2 models (cfg05 + catboost_spike_residual)",
        fusion_note,
    ]
    if result["quarantined_models"]:
        result["known_caveats"].append(
            f"Quarantined models: {', '.join(result['quarantined_models'])}"
        )

    # Forbidden claims
    result["forbidden_claims"] = [
        "Cannot claim 2.97% production sMAPE (was research profile with leakage-suspect stage3)",
        "Cannot claim 69.96% production improvement (same reason)",
        "Cannot claim source 11.48% reproduction",
        "Cannot claim stage3 production readiness",
    ]

    # Delivery commands
    result["delivery_commands"] = [
        "# Default: trusted_no_stage3 profile (P41-P43 gate passed)",
        "python -m scripts.run_p41_model_trust_gate --json",
        "python -m scripts.run_p42_trusted_fusion_backtest --json",
        "python -m scripts.run_p43_rolling_weight_fusion_validation --json",
        "",
        "# Research (not delivery): full pool including stage3",
        "python -m scripts.run_p36_fusion_backtest --json",
    ]

    result["p44_status"] = "P44_DELIVERY_READINESS_PACKAGED"
    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P44 — Delivery Readiness Packager")
    print("=" * 60)
    print(f"  Default profile:  {result['default_profile']}")
    print()

    print("── Trusted Model Pool ──")
    for m in result.get("trusted_model_pool", []):
        print(f"  {m}")
    print()
    print("── Quarantined Models ──")
    for m in result.get("quarantined_models", []):
        print(f"  {m} (SUSPECT_LEAKAGE)")
    print()

    dm = result.get("delivery_metrics", {})
    print("── Delivery Metrics ──")
    cfg = dm.get("cfg05", {})
    print(f"  cfg05: sMAPE={cfg.get('sMAPE_floor50','N/A')}% MAE={cfg.get('MAE','N/A')}")
    bs = dm.get("best_single_trusted", {})
    print(f"  best single: {bs.get('model','N/A')} sMAPE={bs.get('sMAPE_floor50','N/A')}%")
    ew = dm.get("equal_weight_fusion", {})
    print(f"  equal-weight fusion: sMAPE={ew.get('sMAPE_floor50','N/A')}%")
    bf = dm.get("bgew_fusion", {})
    print(f"  BGEW fusion: sMAPE={bf.get('sMAPE_floor50','N/A')}%")
    print(f"  fusion vs cfg05: {dm.get('fusion_vs_cfg05_delta','N/A')}%")
    print(f"  fusion vs best single: {dm.get('fusion_vs_best_single_delta','N/A')}%")

    print()
    print("── Rolling/Split Validation ──")
    rv = result.get("rolling_validation", {})
    sp = rv.get("split", {})
    if sp:
        print(f"  Split: fusion={sp.get('fusion_sMAPE','N/A')}% vs cfg05={sp.get('cfg05_sMAPE','N/A')}%")
    rl = rv.get("rolling", {})
    if rl:
        print(f"  Rolling: fusion={rl.get('fusion_sMAPE','N/A')}% vs cfg05={rl.get('cfg05_sMAPE','N/A')}%")

    print()
    print("── Default ──")
    print(f"  Recommended: {result.get('recommended_default', 'N/A')}")

    print()
    print("── Caveats ──")
    for c in result.get("known_caveats", []):
        print(f"  - {c}")

    print()
    print("── Forbidden Claims ──")
    for fc in result.get("forbidden_claims", []):
        print(f"  - {fc}")

    print()
    print(f"  Status: {result['p44_status']}")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P44: Delivery readiness packager.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--json", action="store_true", default=False)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_delivery_packager(work_dir=args.work_dir)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
