"""
scripts/run_p36_fusion_backtest.py — P36: Fusion backtest.

Compares period-BGEW fused predictions vs cfg05-alone across the 30-day
backtest period. Reports sMAPE_floor50, MAE, RMSE for both.

Usage::

    python -m scripts.run_p36_fusion_backtest

Options::

    --work-dir PATH   Model artifacts dir.
    --force           Overwrite existing.
    --json            Output JSON report.
    --strict          Exit non-zero if fusion doesn't improve.
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


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_f = np.maximum(y_true, 50.0)
    y_pred_f = np.maximum(y_pred, 50.0)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(200.0 * np.mean(np.abs(y_true_f - y_pred_f) / denom))


def run_fusion_backtest(
    work_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run fusion backtest comparing fused vs cfg05-alone."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledger")

    result: dict[str, Any] = {
        "phase": "P36",
        "cfg05_metrics": {},
        "fusion_metrics": {},
        "period_metrics": {},
        "improvement": {},
        "summary": {
            "cfg05_sMAPE": None,
            "fusion_sMAPE": None,
            "sMAPE_improvement": None,
            "p36_status": "P36_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Load data
    pred_path = os.path.join(ledger_dir, "prediction_ledger_30d.csv")
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")
    weights_path = os.path.join(work_dir, "period_bgew_weights.json")

    for p in [pred_path, actual_path, weights_path]:
        if not os.path.isfile(p):
            result["reason_codes"].append(f"MISSING:{p}")
            result["summary"]["p36_status"] = "P36_DATA_MISSING"
            return result

    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)
    with open(weights_path) as f:
        weights = json.load(f)

    # Merge
    merged = pred.merge(
        actual[["task", "target_day", "business_day", "hour_business", "y_true"]],
        on=["task", "target_day", "business_day", "hour_business"],
        how="inner",
    )
    merged = merged.dropna(subset=["y_true"])

    # Add period
    merged["period"] = merged["hour_business"].apply(
        lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
    )

    result["total_rows"] = len(merged)
    result["reason_codes"].append(f"MERGED:{len(merged)}rows")

    if len(merged) == 0:
        result["summary"]["p36_status"] = "P36_NO_DATA"
        return result

    # ── cfg05 alone metrics ──
    cfg05 = merged[merged["model_name"] == "lightgbm_cfg05_dayahead"]
    if len(cfg05) > 0:
        yt = cfg05["y_true"].values
        yp = cfg05["y_pred"].values
        result["cfg05_metrics"] = {
            "sMAPE_floor50": round(smape_floor50(yt, yp), 4),
            "MAE": round(float(np.mean(np.abs(yt - yp))), 2),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2),
            "n": len(cfg05),
        }
        result["summary"]["cfg05_sMAPE"] = result["cfg05_metrics"]["sMAPE_floor50"]

    # ── Fused predictions ──
    fused_rows = []
    for (td, bd, hb), group in merged.groupby(["target_day", "business_day", "hour_business"]):
        period = group["period"].iloc[0]
        w = weights.get(period, {})
        if not w:
            continue

        y_true_val = group["y_true"].iloc[0]
        fused_pred = 0.0
        total_w = 0.0
        included = []
        excluded = []
        model_weights = {}

        for _, row in group.iterrows():
            mname = row["model_name"]
            mw = w.get(mname, 0)
            if mw > 0 and not np.isnan(row["y_pred"]):
                fused_pred += mw * row["y_pred"]
                total_w += mw
                included.append(mname)
                model_weights[mname] = mw
            else:
                excluded.append(mname)

        if total_w > 0:
            fused_pred = fused_pred / total_w
        else:
            continue

        fused_rows.append({
            "target_day": td,
            "business_day": bd,
            "hour_business": hb,
            "period": period,
            "y_true": y_true_val,
            "fused_price": round(fused_pred, 4),
            "weights_json": json.dumps(model_weights),
            "included_models": ";".join(sorted(set(included))),
            "excluded_models": ";".join(sorted(set(excluded))),
            "fusion_method": "bgew_skeleton",
            "learner_version": "p35_v1",
            "readiness_mode": "REAL",
            "reason_codes": "BGEW_PERIOD",
        })

    if len(fused_rows) == 0:
        result["summary"]["p36_status"] = "P36_NO_FUSION"
        return result

    fused_df = pd.DataFrame(fused_rows)
    result["fused_rows"] = len(fused_df)
    result["reason_codes"].append(f"FUSED:{len(fused_df)}rows")

    # ── Fusion metrics ──
    yt_fused = fused_df["y_true"].values
    yp_fused = fused_df["fused_price"].values
    result["fusion_metrics"] = {
        "sMAPE_floor50": round(smape_floor50(yt_fused, yp_fused), 4),
        "MAE": round(float(np.mean(np.abs(yt_fused - yp_fused))), 2),
        "RMSE": round(float(np.sqrt(np.mean((yt_fused - yp_fused) ** 2))), 2),
        "n": len(fused_df),
    }
    result["summary"]["fusion_sMAPE"] = result["fusion_metrics"]["sMAPE_floor50"]

    # ── Per-period metrics ──
    for period in ["1_8", "9_16", "17_24"]:
        pdf = fused_df[fused_df["period"] == period]
        if len(pdf) < 10:
            continue
        cfg05_p = cfg05[cfg05["period"] == period]
        if len(cfg05_p) < 10:
            continue
        result["period_metrics"][period] = {
            "cfg05_sMAPE": round(smape_floor50(cfg05_p["y_true"].values, cfg05_p["y_pred"].values), 4),
            "fusion_sMAPE": round(smape_floor50(pdf["y_true"].values, pdf["fused_price"].values), 4),
            "n": len(pdf),
        }

    # ── Improvement ──
    cfg_s = result["cfg05_metrics"].get("sMAPE_floor50")
    fus_s = result["fusion_metrics"].get("sMAPE_floor50")
    if cfg_s and fus_s:
        imp = ((cfg_s - fus_s) / cfg_s) * 100
        result["improvement"] = {
            "sMAPE_improvement_pct": round(imp, 2),
            "direction": "IMPROVED" if imp > 0 else "DEGRADED",
        }
        result["summary"]["sMAPE_improvement"] = round(imp, 2)

    # ── Save fusion output ──
    fusion_path = os.path.join(work_dir, "fusion_backtest_30d.csv")
    try:
        fused_df.to_csv(fusion_path, index=False)
        result["fusion_path"] = fusion_path
        result["reason_codes"].append(f"FUSION_SAVED:{fusion_path}")
    except Exception as e:
        result["reason_codes"].append(f"FUSION_SAVE_FAILED:{e}")

    # ── Status ──
    if result.get("improvement", {}).get("direction") == "IMPROVED":
        result["summary"]["p36_status"] = "P36_FUSION_IMPROVED"
    else:
        result["summary"]["p36_status"] = "P36_FUSION_NOT_IMPROVED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P36 — Fusion Backtest")
    print("=" * 60)
    print(f"  Total rows:       {result.get('total_rows', 0)}")
    print(f"  Fused rows:       {result.get('fused_rows', 0)}")

    print()
    print("── cfg05 Alone ──")
    for k, v in result.get("cfg05_metrics", {}).items():
        print(f"  {k}: {v}")

    print()
    print("── Fused (BGEW) ──")
    for k, v in result.get("fusion_metrics", {}).items():
        print(f"  {k}: {v}")

    if result.get("improvement"):
        print()
        i = result["improvement"]
        print(f"── Improvement ──")
        print(f"  sMAPE change:     {i['sMAPE_improvement_pct']}% ({i['direction']})")

    print()
    print("── Per Period ──")
    for period, pm in result.get("period_metrics", {}).items():
        print(f"  {period}: cfg05={pm['cfg05_sMAPE']}% fusion={pm['fusion_sMAPE']}%")

    print()
    print(f"  Status:           {result['summary']['p36_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P36: Fusion backtest.")
    parser.add_argument("--work-dir", type=str, default=None)
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
    result = run_fusion_backtest(work_dir=args.work_dir, force=args.force)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict and result["summary"]["p36_status"] == "P36_FUSION_NOT_IMPROVED":
        logger.error("P36: Fusion did not improve over cfg05 alone")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
