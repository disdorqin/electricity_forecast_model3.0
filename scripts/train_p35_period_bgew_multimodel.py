"""
scripts/train_p35_period_bgew_multimodel.py — P35: Period BGEW weight learner.

Trains per-period Bayesian-Gamma-Exponential-Weighted (BGEW) weights using
sMAPE_floor50 metrics from the 30-day backtest.

Usage::

    python -m scripts.train_p35_period_bgew_multimodel

Options::

    --work-dir PATH         Model artifacts dir.
    --alpha FLOAT           BGEW alpha parameter (default: 0.5).
    --min-weight FLOAT      Minimum weight per model (default: 0.05).
    --max-weight FLOAT      Maximum weight per model (default: 0.75).
    --cfg05-min-prior FLOAT Minimum weight for cfg05 (default: 0.30).
    --force                 Overwrite existing.
    --json                  Output JSON report.
    --strict                Exit non-zero on failures.
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
_PERIODS = ["1_8", "9_16", "17_24"]
_ALL_MODELS = [
    "lightgbm_cfg05_dayahead", "best_two_average", "stage3_business_fixed",
    "catboost_sota", "catboost_spike_residual",
]


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sMAPE with floor of 50."""
    y_true_f = np.maximum(y_true, 50.0)
    y_pred_f = np.maximum(y_pred, 50.0)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(200.0 * np.mean(np.abs(y_true_f - y_pred_f) / denom))


def learn_period_bgew(
    work_dir: Optional[str] = None,
    alpha: float = 0.5,
    min_weight: float = 0.05,
    max_weight: float = 0.75,
    cfg05_min_prior: float = 0.30,
    force: bool = False,
) -> dict[str, Any]:
    """Train per-period BGEW weights from prediction + actual ledgers."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledger")

    result: dict[str, Any] = {
        "phase": "P35",
        "alpha": alpha,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "cfg05_min_prior": cfg05_min_prior,
        "periods": {},
        "summary": {
            "models_with_data": 0,
            "p35_status": "P35_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Load ledgers
    pred_path = os.path.join(ledger_dir, "prediction_ledger_30d.csv")
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")

    if not os.path.isfile(pred_path):
        result["reason_codes"].append(f"PREDICTION_LEDGER_MISSING:{pred_path}")
        result["summary"]["p35_status"] = "P35_DATA_MISSING"
        return result
    if not os.path.isfile(actual_path):
        result["reason_codes"].append(f"ACTUAL_LEDGER_MISSING:{actual_path}")
        result["summary"]["p35_status"] = "P35_DATA_MISSING"
        return result

    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)

    # Merge
    merged = pred.merge(
        actual[["task", "target_day", "business_day", "hour_business", "y_true"]],
        on=["task", "target_day", "business_day", "hour_business"],
        how="inner",
    )
    # Drop rows with NaN actuals (e.g., last day hour 24 not yet known)
    before = len(merged)
    merged = merged.dropna(subset=["y_true"])
    result["reason_codes"].append(f"DROPPED_NAN_ACTUALS:{before - len(merged)}rows")
    merged["period"] = merged["hour_business"].apply(
        lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
    )

    result["total_merged_rows"] = len(merged)
    result["reason_codes"].append(f"MERGED:{len(merged)}rows")

    if len(merged) == 0:
        result["summary"]["p35_status"] = "P35_NO_MERGE_DATA"
        return result

    # Compute per-model, per-period metrics
    model_count = 0
    for model_name in _ALL_MODELS:
        model_df = merged[merged["model_name"] == model_name]
        if len(model_df) < 24:
            continue
        model_count += 1

        for period in _PERIODS:
            pdf = model_df[model_df["period"] == period]
            if len(pdf) < 10:
                continue
            y_true = pdf["y_true"].values
            y_pred = pdf["y_pred"].values
            s = smape_floor50(y_true, y_pred)
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            if period not in result["periods"]:
                result["periods"][period] = {}
            result["periods"][period][model_name] = {
                "sMAPE_floor50": round(s, 4),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "n": len(pdf),
            }

    result["summary"]["models_with_data"] = model_count
    result["reason_codes"].append(f"METRICS_COMPUTED:{model_count}models")

    # Compute BGEW weights per period
    weights = {}
    for period in _PERIODS:
        pdata = result["periods"].get(period, {})
        if len(pdata) < 2:
            result["reason_codes"].append(f"INSUFFICIENT_MODELS_FOR_PERIOD:{period}({len(pdata)})")
            continue

        # Compute scores: score_m = exp(-alpha * smape_m)
        scores = {}
        for model_name, metrics in pdata.items():
            smape = metrics["sMAPE_floor50"]
            scores[model_name] = np.exp(-alpha * smape)

        total_score = sum(scores.values())
        if total_score <= 0:
            result["reason_codes"].append(f"ZERO_TOTAL_SCORE:{period}")
            continue

        raw_weights = {m: s / total_score for m, s in scores.items()}
        # Apply constraints
        constrained = {}
        for m, w in raw_weights.items():
            w = max(w, min_weight)
            if m == "lightgbm_cfg05_dayahead":
                w = max(w, cfg05_min_prior)
            constrained[m] = w

        # Re-normalize after constraints
        total_c = sum(constrained.values())
        if total_c > 0:
            constrained = {m: w / total_c for m, w in constrained.items()}

        # Apply max constraint
        for m in constrained:
            if constrained[m] > max_weight:
                constrained[m] = max_weight

        # Final re-normalize
        total_f = sum(constrained.values())
        if total_f > 0:
            constrained = {m: w / total_f for m, w in constrained.items()}

        weights[period] = constrained
        result["periods"][period]["weights"] = constrained
        result["reason_codes"].append(
            f"WEIGHTS_{period}:{','.join(f'{m}={w:.3f}' for m, w in sorted(constrained.items()))}"
        )

    # Save weights
    weights_path = os.path.join(work_dir, "period_bgew_weights.json")
    try:
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        result["weights_path"] = weights_path
        result["reason_codes"].append(f"WEIGHTS_SAVED:{weights_path}")
    except Exception as e:
        result["reason_codes"].append(f"WEIGHTS_SAVE_FAILED:{e}")

    # Determine status
    if len(weights) >= 2:
        result["summary"]["p35_status"] = "P35_LEARNER_READY"
    elif len(weights) > 0:
        result["summary"]["p35_status"] = "P35_LEARNER_PARTIAL"
    else:
        result["summary"]["p35_status"] = "P35_LEARNER_FAILED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P35 — Period BGEW Weight Learner")
    print("=" * 60)
    print(f"  Alpha:            {result['alpha']}")
    print(f"  Min weight:       {result['min_weight']}")
    print(f"  Max weight:       {result['max_weight']}")
    print(f"  cfg05 min prior:  {result['cfg05_min_prior']}")
    print(f"  Merged rows:      {result.get('total_merged_rows', 0)}")
    print(f"  Models w/ data:   {result['summary']['models_with_data']}")
    print()

    for period in _PERIODS:
        pdata = result["periods"].get(period, {})
        if not pdata:
            continue
        print(f"── Period {period} ──")
        weights = pdata.get("weights", {})
        for model_name, metrics in sorted(pdata.items()):
            if model_name == "weights":
                continue
            w = weights.get(model_name, 0)
            smape = metrics.get("sMAPE_floor50", "N/A")
            n = metrics.get("n", 0)
            print(f"  {model_name:<25} sMAPE={smape:<8} n={n:<4} weight={w:.4f}")

    print()
    print(f"  Status:           {result['summary']['p35_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P35: Period BGEW learner.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.75)
    parser.add_argument("--cfg05-min-prior", type=float, default=0.30)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    result = learn_period_bgew(
        work_dir=args.work_dir, alpha=args.alpha,
        min_weight=args.min_weight, max_weight=args.max_weight,
        cfg05_min_prior=args.cfg05_min_prior, force=args.force,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict and result["summary"]["p35_status"] != "P35_LEARNER_READY":
        logger.error("P35: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
