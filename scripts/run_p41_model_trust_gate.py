"""
scripts/run_p41_model_trust_gate.py — P41: Model trust gate.

Evaluates each model against suspicious-leakage criteria and assigns
trust labels (TRUSTED vs SUSPECT_LEAKAGE).

Rules (any triggers SUSPECT_LEAKAGE):
  - within_1pct_ratio > 0.50
  - corr_y_pred_y_true > 0.995
  - sMAPE_floor50 < 1.0
  - MAE < 3.0

Usage::

    python -m scripts.run_p41_model_trust_gate --json

Options::

    --work-dir PATH   Model artifacts dir.
    --json            Output JSON report.
    --strict          Exit non-zero if any model is SUSPECT_LEAKAGE.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")

# Trust labels
TRUSTED = "TRUSTED"
SUSPECT_LEAKAGE = "SUSPECT_LEAKAGE"

# Models (full set)
_ALL_MODELS = [
    "lightgbm_cfg05_dayahead",
    "best_two_average",
    "stage3_business_fixed",
    "catboost_sota",
    "catboost_spike_residual",
]

# Leakage suspicion thresholds (any triggers SUSPECT_LEAKAGE)
_WITHIN_1PCT_THRESHOLD = 0.50     # > 50% of predictions within 1% of actual
_CORR_THRESHOLD = 0.995          # correlation > 0.995
_SMAPE_THRESHOLD = 1.0           # sMAPE_floor50 < 1.0%
_MAE_THRESHOLD = 3.0             # MAE < 3.0 CNY


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sMAPE with floor of 50."""
    y_true_f = np.maximum(y_true, 50.0)
    y_pred_f = np.maximum(y_pred, 50.0)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(200.0 * np.mean(np.abs(y_true_f - y_pred_f) / denom))


def run_trust_gate(
    work_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate all models and assign trust labels."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledger")

    result: dict[str, Any] = {
        "phase": "P41",
        "models": {},
        "summary": {
            "trusted_count": 0,
            "suspect_count": 0,
            "p41_status": "P41_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Load ledgers
    pred_path = os.path.join(ledger_dir, "prediction_ledger_30d.csv")
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")

    if not os.path.isfile(pred_path) or not os.path.isfile(actual_path):
        result["reason_codes"].append("LEDGER_DATA_MISSING")
        result["summary"]["p41_status"] = "P41_DATA_MISSING"
        return result

    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)

    # Merge
    merged = pred.merge(
        actual[["task", "target_day", "business_day", "hour_business", "y_true"]],
        on=["task", "target_day", "business_day", "hour_business"],
        how="inner",
    )
    merged = merged.dropna(subset=["y_true"])
    result["total_eval_rows"] = len(merged[merged["model_name"] == _ALL_MODELS[0]])
    result["reason_codes"].append(f"EVAL_ROWS:{result['total_eval_rows']}")

    if result["total_eval_rows"] == 0:
        result["summary"]["p41_status"] = "P41_NO_DATA"
        return result

    # Per-model evaluation
    suspect_models = []
    trusted_models = []

    for model_name in _ALL_MODELS:
        md = merged[merged["model_name"] == model_name]
        if len(md) < 24:
            result["models"][model_name] = {
                "trust_label": "INSUFFICIENT_DATA",
                "n": len(md),
            }
            continue

        yt = md["y_true"].values
        yp = md["y_pred"].values
        diff = np.abs(yt - yp)
        within_1pct = (diff / (np.abs(yt) + 1.0) < 0.01).sum()
        within_1pct_ratio = within_1pct / len(md)
        corr = float(np.corrcoef(yt, yp)[0, 1]) if len(md) > 2 else 0.0
        s = smape_floor50(yt, yp)
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        exact_matches = int((diff < 0.001).sum())

        # Determine trust label
        reasons = []
        if within_1pct_ratio > _WITHIN_1PCT_THRESHOLD:
            reasons.append(f"within_1pct_ratio={within_1pct_ratio:.4f}>{_WITHIN_1PCT_THRESHOLD}")
        if corr > _CORR_THRESHOLD:
            reasons.append(f"corr={corr:.4f}>{_CORR_THRESHOLD}")
        if s < _SMAPE_THRESHOLD:
            reasons.append(f"sMAPE={s:.4f}<{_SMAPE_THRESHOLD}")
        if mae < _MAE_THRESHOLD:
            reasons.append(f"MAE={mae:.2f}<{_MAE_THRESHOLD}")

        trust_label = SUSPECT_LEAKAGE if reasons else TRUSTED
        if trust_label == SUSPECT_LEAKAGE:
            suspect_models.append(model_name)
        else:
            trusted_models.append(model_name)

        # Coverage check
        coverage_days = int(md["business_day"].nunique())
        per_day = md.groupby("business_day").size()
        all_24h = bool((per_day == 24).all()) if len(per_day) > 0 else False

        result["models"][model_name] = {
            "trust_label": trust_label,
            "n": len(md),
            "sMAPE_floor50": round(s, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "within_1pct": int(within_1pct),
            "within_1pct_ratio": round(within_1pct_ratio, 4),
            "exact_matches": exact_matches,
            "corr_y_pred_y_true": round(corr, 4),
            "coverage_days": coverage_days,
            "all_24h": all_24h,
            "suspicion_reasons": reasons,
        }
        result["reason_codes"].append(
            f"{model_name}:{trust_label}(sMAPE={s:.2f},MAE={mae:.1f},"
            f"within1pct={within_1pct_ratio:.2f},corr={corr:.3f})"
        )

    result["summary"]["trusted_count"] = len(trusted_models)
    result["summary"]["suspect_count"] = len(suspect_models)
    result["summary"]["trusted_models"] = trusted_models
    result["summary"]["suspect_models"] = suspect_models

    # Build profiles
    result["profiles"] = {
        "research_all_models": {
            "description": "All models including leakage-suspect ones",
            "stage3_included": True,
            "delivery_allowed": False,
            "model_pool": list(_ALL_MODELS),
        },
        "trusted_no_stage3": {
            "description": "Only TRUSTED models, excludes SUSPECT_LEAKAGE",
            "stage3_included": False,
            "delivery_allowed": True,
            "model_pool": trusted_models,
        },
    }

    result["summary"]["p41_status"] = "P41_GATE_COMPLETE"
    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P41 — Model Trust Gate")
    print("=" * 60)
    print(f"  Total eval rows:  {result.get('total_eval_rows', 0)}")
    print()

    for model_name, m in result.get("models", {}).items():
        label = m.get("trust_label", "N/A")
        if label == "INSUFFICIENT_DATA":
            print(f"  {model_name:<35} {label}")
            continue
        s = m.get("sMAPE_floor50", "N/A")
        mae = m.get("MAE", "N/A")
        w1p = m.get("within_1pct_ratio", 0)
        corr = m.get("corr_y_pred_y_true", 0)
        reasons = m.get("suspicion_reasons", [])
        flag = " *** SUSPECT ***" if reasons else ""
        print(f"  {model_name:<35} {label}{flag}")
        print(f"    sMAPE={s}% MAE={mae} within_1pct={w1p:.2%} corr={corr:.4f}")
        if reasons:
            for r in reasons:
                print(f"      -> {r}")

    print()
    print("── Profiles ──")
    for profile, pdef in result.get("profiles", {}).items():
        print(f"  {profile}: delivery_allowed={pdef['delivery_allowed']}")
        print(f"    models: {pdef['model_pool']}")

    print()
    print(f"  Trusted:  {result['summary']['trusted_count']}")
    print(f"  Suspect:  {result['summary']['suspect_count']}")
    print(f"  Status:   {result['summary']['p41_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P41: Model trust gate.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_trust_gate(work_dir=args.work_dir)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["summary"]["suspect_count"] == 0:
            return 0
        logger.error("P41: %d suspect model(s) found", result["summary"]["suspect_count"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
