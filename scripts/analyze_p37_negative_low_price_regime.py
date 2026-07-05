"""
scripts/analyze_p37_negative_low_price_regime.py — P37: Negative/low-price regime analysis.

Analyzes negative and low-price hours in the backtest period, comparing
how cfg05 and fusion perform during extreme price regimes.

Usage::

    python -m scripts.analyze_p37_negative_low_price_regime

Options::

    --work-dir PATH     Model artifacts dir.
    --low-threshold FLOAT  Low-price threshold (default: 100.0 CNY/MWh).
    --json              Output JSON report.
    --strict            Exit non-zero on failures.
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


def analyze_regime(
    work_dir: Optional[str] = None,
    low_threshold: float = 100.0,
) -> dict[str, Any]:
    """Analyze negative and low-price regimes."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledger")

    result: dict[str, Any] = {
        "phase": "P37",
        "low_threshold": low_threshold,
        "summary": {
            "negative_hours": 0,
            "low_price_hours": 0,
            "p37_status": "P37_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Load actuals
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")
    if not os.path.isfile(actual_path):
        result["reason_codes"].append(f"ACTUAL_LEDGER_MISSING:{actual_path}")
        result["summary"]["p37_status"] = "P37_DATA_MISSING"
        return result

    actual = pd.read_csv(actual_path)
    actual = actual.dropna(subset=["y_true"])

    # Identify regimes
    neg_mask = actual["y_true"] < 0
    low_mask = (actual["y_true"] >= 0) & (actual["y_true"] < low_threshold)
    normal_mask = actual["y_true"] >= low_threshold

    result["summary"]["total_hours"] = len(actual)
    result["summary"]["negative_hours"] = int(neg_mask.sum())
    result["summary"]["low_price_hours"] = int(low_mask.sum())
    result["summary"]["normal_hours"] = int(normal_mask.sum())

    result["reason_codes"].append(
        f"REGIMES:neg={result['summary']['negative_hours']}_"
        f"low={result['summary']['low_price_hours']}_"
        f"normal={result['summary']['normal_hours']}"
    )

    # Per-period breakdown
    period_breakdown = {}
    for period in ["1_8", "9_16", "17_24"]:
        pdf = actual[actual["period"] == period]
        pneg = int((pdf["y_true"] < 0).sum())
        plow = int(((pdf["y_true"] >= 0) & (pdf["y_true"] < low_threshold)).sum())
        period_breakdown[period] = {
            "total": len(pdf),
            "negative": pneg,
            "low_price": plow,
            "neg_pct": round(pneg / len(pdf) * 100, 2) if len(pdf) > 0 else 0,
        }
    result["period_breakdown"] = period_breakdown

    # Load fusion results for regime-level metrics
    fusion_path = os.path.join(work_dir, "fusion_backtest_30d.csv")
    if os.path.isfile(fusion_path):
        fusion = pd.read_csv(fusion_path)
        fusion = fusion.dropna(subset=["y_true"])

        for regime_name, mask in [
            ("negative", fusion["y_true"] < 0),
            ("low_price", (fusion["y_true"] >= 0) & (fusion["y_true"] < low_threshold)),
            ("normal", fusion["y_true"] >= low_threshold),
        ]:
            rdf = fusion[mask]
            if len(rdf) < 3:
                continue
            yt = rdf["y_true"].values
            yp = rdf["fused_price"].values
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            result.setdefault("regime_metrics", {})[regime_name] = {
                "n": len(rdf),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
            }

    # Also get cfg05 regime metrics
    pred_path = os.path.join(ledger_dir, "prediction_ledger_30d.csv")
    if os.path.isfile(pred_path):
        pred = pd.read_csv(pred_path)
        cfg05_pred = pred[pred["model_name"] == "lightgbm_cfg05_dayahead"]
        if len(cfg05_pred) > 0:
            merged = cfg05_pred.merge(
                actual[["target_day", "business_day", "hour_business", "y_true"]],
                on=["target_day", "business_day", "hour_business"],
                how="inner",
            )
            merged = merged.dropna(subset=["y_true"])
            for regime_name, mask in [
                ("negative", merged["y_true"] < 0),
                ("low_price", (merged["y_true"] >= 0) & (merged["y_true"] < low_threshold)),
                ("normal", merged["y_true"] >= low_threshold),
            ]:
                rdf = merged[mask]
                if len(rdf) < 3:
                    continue
                yt = rdf["y_true"].values
                yp = rdf["y_pred"].values
                result.setdefault("cfg05_regime_metrics", {})[regime_name] = {
                    "n": len(rdf),
                    "MAE": round(float(np.mean(np.abs(yt - yp))), 2),
                    "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2),
                }

    result["summary"]["p37_status"] = "P37_REGIME_ANALYZED"
    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P37 — Negative/Low-Price Regime Analysis")
    print("=" * 60)
    print(f"  Low threshold:     {result['low_threshold']} CNY/MWh")
    print()
    s = result["summary"]
    print(f"  Total hours:       {s.get('total_hours', 0)}")
    print(f"  Negative:          {s['negative_hours']}")
    print(f"  Low price (0-{result['low_threshold']}): {s['low_price_hours']}")
    print(f"  Normal:            {s['normal_hours']}")

    print()
    print("── Per Period ──")
    for period, pb in result.get("period_breakdown", {}).items():
        print(f"  {period}: total={pb['total']} neg={pb['negative']}({pb['neg_pct']}%) low={pb['low_price']}")

    if result.get("cfg05_regime_metrics"):
        print()
        print("── cfg05 Regime Metrics ──")
        for regime, m in result["cfg05_regime_metrics"].items():
            print(f"  {regime:<12} n={m['n']:<4} MAE={m['MAE']:<8} RMSE={m['RMSE']:<8}")

    if result.get("regime_metrics"):
        print()
        print("── Fusion Regime Metrics ──")
        for regime, m in result["regime_metrics"].items():
            print(f"  {regime:<12} n={m['n']:<4} MAE={m['MAE']:<8} RMSE={m['RMSE']:<8}")

    print()
    print(f"  Status: {result['summary']['p37_status']}")
    print()
    for rc in result["reason_codes"]:
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P37: Regime analysis.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--low-threshold", type=float, default=100.0)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = analyze_regime(work_dir=args.work_dir, low_threshold=args.low_threshold)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict:
        if result["summary"]["p37_status"] == "P37_REGIME_ANALYZED":
            return 0
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
