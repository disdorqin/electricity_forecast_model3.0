"""
scripts/run_p42_trusted_fusion_backtest.py — P42: Trusted no-stage3 fusion backtest.

Runs BGEW fusion on trusted model pool (excluding stage3_business_fixed).
Compares: cfg05-alone, best single trusted, equal-weight, BGEW fusion.

Usage::

    python -m scripts.run_p42_trusted_fusion_backtest --json

Options::

    --work-dir PATH     Model artifacts dir.
    --alpha FLOAT       BGEW alpha (default: 0.5).
    --json              Output JSON report.
    --strict            Exit non-zero if fusion doesn't improve.
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

# Trusted pool (excludes stage3_business_fixed)
_TRUSTED_MODELS = [
    "lightgbm_cfg05_dayahead",
    "best_two_average",
    "catboost_sota",
    "catboost_spike_residual",
]

_PERIODS = ["1_8", "9_16", "17_24"]


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_f = np.maximum(y_true, 50.0)
    y_pred_f = np.maximum(y_pred, 50.0)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(200.0 * np.mean(np.abs(y_true_f - y_pred_f) / denom))


def run_trusted_fusion_backtest(
    work_dir: Optional[str] = None,
    alpha: float = 0.5,
    trusted_models: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run trusted fusion backtest (no stage3)."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledger")
    model_list = trusted_models if trusted_models is not None else list(_TRUSTED_MODELS)

    result: dict[str, Any] = {
        "phase": "P42",
        "profile": "trusted_no_stage3",
        "excluded_models": ["stage3_business_fixed"],
        "trusted_models": list(model_list),
        "alpha": alpha,
        "per_model_metrics": {},
        "cfg05_metrics": {},
        "best_single_metrics": {},
        "equal_weight_metrics": {},
        "fusion_metrics": {},
        "period_metrics": {},
        "summary": {
            "cfg05_sMAPE": None,
            "best_single_sMAPE": None,
            "equal_weight_sMAPE": None,
            "fusion_sMAPE": None,
            "fusion_vs_cfg05_delta": None,
            "fusion_vs_best_single_delta": None,
            "p42_status": "P42_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Load data
    pred_path = os.path.join(ledger_dir, "prediction_ledger_30d.csv")
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")

    if not os.path.isfile(pred_path) or not os.path.isfile(actual_path):
        result["reason_codes"].append("DATA_MISSING")
        result["summary"]["p42_status"] = "P42_DATA_MISSING"
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
    merged["period"] = merged["hour_business"].apply(
        lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
    )

    result["total_rows"] = len(merged[merged["model_name"] == _TRUSTED_MODELS[0]])
    result["reason_codes"].append(f"MERGED:{result['total_rows']}rows")

    if result["total_rows"] == 0:
        result["summary"]["p42_status"] = "P42_NO_DATA"
        return result

    # ── cfg05 alone ──
    cfg05 = merged[merged["model_name"] == "lightgbm_cfg05_dayahead"]
    yt_c = cfg05["y_true"].values
    yp_c = cfg05["y_pred"].values
    result["cfg05_metrics"] = {
        "sMAPE_floor50": round(smape_floor50(yt_c, yp_c), 4),
        "MAE": round(float(np.mean(np.abs(yt_c - yp_c))), 2),
        "RMSE": round(float(np.sqrt(np.mean((yt_c - yp_c) ** 2))), 2),
        "n": len(cfg05),
    }
    result["summary"]["cfg05_sMAPE"] = result["cfg05_metrics"]["sMAPE_floor50"]

    # ── Per-model metrics on trusted pool ──
    for model_name in model_list:
        md = merged[merged["model_name"] == model_name]
        yt = md["y_true"].values
        yp = md["y_pred"].values
        result["per_model_metrics"][model_name] = {
            "sMAPE_floor50": round(smape_floor50(yt, yp), 4),
            "MAE": round(float(np.mean(np.abs(yt - yp))), 2),
            "RMSE": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2),
            "n": len(md),
        }

    # ── Best single trusted model ──
    best_model = min(
        model_list,
        key=lambda m: result["per_model_metrics"][m]["sMAPE_floor50"],
    )
    result["best_single_model"] = best_model
    result["best_single_metrics"] = dict(result["per_model_metrics"][best_model])
    result["summary"]["best_single_sMAPE"] = result["best_single_metrics"]["sMAPE_floor50"]
    result["reason_codes"].append(f"BEST_SINGLE:{best_model}")

    # ── Equal-weight fusion ──
    ew_rows = []
    for (td, bd, hb), group in merged.groupby(["target_day", "business_day", "hour_business"]):
        trusted = group[group["model_name"].isin(model_list)]
        if len(trusted) == 0:
            continue
        ew_pred = trusted["y_pred"].mean()
        ew_rows.append({
            "y_true": group["y_true"].iloc[0],
            "fused_price": ew_pred,
        })
    ew_df = pd.DataFrame(ew_rows)
    if len(ew_df) > 0:
        result["equal_weight_metrics"] = {
            "sMAPE_floor50": round(smape_floor50(ew_df["y_true"].values, ew_df["fused_price"].values), 4),
            "MAE": round(float(np.mean(np.abs(ew_df["y_true"].values - ew_df["fused_price"].values))), 2),
            "RMSE": round(float(np.sqrt(np.mean((ew_df["y_true"].values - ew_df["fused_price"].values) ** 2))), 2),
            "n": len(ew_df),
        }
        result["summary"]["equal_weight_sMAPE"] = result["equal_weight_metrics"]["sMAPE_floor50"]

    # ── Learn BGEW weights on trusted pool ──
    weights = {}
    for period in _PERIODS:
        pdata = merged[merged["period"] == period]
        scores = {}
        for m in model_list:
            md = pdata[pdata["model_name"] == m]
            if len(md) < 10:
                continue
            s = smape_floor50(md["y_true"].values, md["y_pred"].values)
            scores[m] = np.exp(-alpha * s)
        total = sum(scores.values())
        if total <= 0:
            continue
        raw_w = {m: s / total for m, s in scores.items()}
        # Apply min/max constraints
        constrained = {}
        for m, w in raw_w.items():
            w = max(w, 0.05)
            if m == "lightgbm_cfg05_dayahead":
                w = max(w, 0.30)
            constrained[m] = w
        total_c = sum(constrained.values())
        constrained = {m: w / total_c for m, w in constrained.items()}
        for m in constrained:
            if constrained[m] > 0.75:
                constrained[m] = 0.75
        total_f = sum(constrained.values())
        weights[period] = {m: round(w / total_f, 4) for m, w in constrained.items()}

    result["fusion_weights"] = weights

    # ── Apply BGEW fusion ──
    fused_rows = []
    for (td, bd, hb), group in merged.groupby(["target_day", "business_day", "hour_business"]):
        period = group["period"].iloc[0]
        w = weights.get(period, {})
        if not w:
            continue
        fused_pred = 0.0
        total_w = 0.0
        for _, row in group.iterrows():
            mname = row["model_name"]
            if mname not in model_list:
                continue
            mw = w.get(mname, 0)
            if mw > 0 and not np.isnan(row["y_pred"]):
                fused_pred += mw * row["y_pred"]
                total_w += mw
        if total_w > 0:
            fused_rows.append({
                "y_true": group["y_true"].iloc[0],
                "fused_price": round(fused_pred / total_w, 4),
                "period": period,
            })

    if len(fused_rows) == 0:
        result["summary"]["p42_status"] = "P42_NO_FUSION"
        return result

    fused_df = pd.DataFrame(fused_rows)
    result["fusion_rows"] = len(fused_df)

    yt_f = fused_df["y_true"].values
    yp_f = fused_df["fused_price"].values
    result["fusion_metrics"] = {
        "sMAPE_floor50": round(smape_floor50(yt_f, yp_f), 4),
        "MAE": round(float(np.mean(np.abs(yt_f - yp_f))), 2),
        "RMSE": round(float(np.sqrt(np.mean((yt_f - yp_f) ** 2))), 2),
        "n": len(fused_df),
    }
    result["summary"]["fusion_sMAPE"] = result["fusion_metrics"]["sMAPE_floor50"]

    # ── Per-period metrics ──
    for period in _PERIODS:
        pdf = fused_df[fused_df["period"] == period]
        cfg05_p = cfg05[cfg05["period"] == period]
        if len(pdf) < 10 or len(cfg05_p) < 10:
            continue
        cfg_s = smape_floor50(cfg05_p["y_true"].values, cfg05_p["y_pred"].values)
        fus_s = smape_floor50(pdf["y_true"].values, pdf["fused_price"].values)
        result["period_metrics"][period] = {
            "cfg05_sMAPE": round(cfg_s, 4),
            "fusion_sMAPE": round(fus_s, 4),
            "n": len(pdf),
        }

    # ── Improvement ──
    cfg_s = result["cfg05_metrics"]["sMAPE_floor50"]
    fus_s = result["fusion_metrics"]["sMAPE_floor50"]
    best_s = result["best_single_metrics"]["sMAPE_floor50"]

    result["summary"]["fusion_vs_cfg05_delta"] = round(
        ((cfg_s - fus_s) / cfg_s) * 100, 2
    )
    result["summary"]["fusion_vs_best_single_delta"] = round(
        ((best_s - fus_s) / best_s) * 100, 2
    )

    # Fusion vs best single
    if fus_s < best_s:
        result["summary"]["p42_status"] = "TRUSTED_FUSION_IMPROVED"
        result["reason_codes"].append(
            f"FUSION_BETTER_THAN_BEST_SINGLE(fusion={fus_s}% vs {best_s}%)"
        )
    elif fus_s <= cfg_s:
        result["summary"]["p42_status"] = "TRUSTED_FUSION_IMPROVED_VS_CFG05"
        result["reason_codes"].append(
            f"FUSION_BETTER_THAN_CFG05_BUT_NOT_BEST_SINGLE"
        )
    else:
        result["summary"]["p42_status"] = "TRUSTED_FUSION_NOT_IMPROVED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P42 — Trusted Fusion Backtest (no stage3)")
    print("=" * 60)
    print(f"  Profile:          {result['profile']}")
    print(f"  Excluded:         {result['excluded_models']}")
    print(f"  Trusted models:   {result['trusted_models']}")
    print()

    print("── cfg05 Alone ──")
    for k, v in result.get("cfg05_metrics", {}).items():
        print(f"  {k}: {v}")

    print()
    print("── Best Single Trusted ──")
    print(f"  Model: {result.get('best_single_model', 'N/A')}")
    for k, v in result.get("best_single_metrics", {}).items():
        print(f"  {k}: {v}")

    print()
    print("── Equal-Weight Fusion ──")
    for k, v in result.get("equal_weight_metrics", {}).items():
        print(f"  {k}: {v}")

    print()
    print("── BGEW Fusion ──")
    for k, v in result.get("fusion_metrics", {}).items():
        print(f"  {k}: {v}")
    print(f"  vs cfg05:         {result['summary'].get('fusion_vs_cfg05_delta', 'N/A')}%")
    print(f"  vs best single:   {result['summary'].get('fusion_vs_best_single_delta', 'N/A')}%")

    if result.get("fusion_weights"):
        print()
        print("── BGEW Weights ──")
        for period, w in result["fusion_weights"].items():
            w_str = ", ".join(f"{k}={v}" for k, v in sorted(w.items()))
            print(f"  {period}: {w_str}")

    print()
    print(f"  Status: {result['summary']['p42_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P42: Trusted fusion backtest.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_trusted_fusion_backtest(
        work_dir=args.work_dir, alpha=args.alpha,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict:
        if "IMPROVED" in result["summary"]["p42_status"]:
            return 0
        logger.error("P42: Fusion did not improve")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
