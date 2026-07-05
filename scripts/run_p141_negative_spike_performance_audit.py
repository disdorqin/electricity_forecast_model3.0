#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P141 - Negative / Spike Specialized Performance Audit
======================================================
Reads cfg05 2025 full-year dayahead predictions (with y_true merged),
classifies each hour into price-regime categories, and computes
per-category / per-period / per-month error metrics.

Outputs land in .local_artifacts/p141_negative_spike/.

Main entry: run_p141_negative_spike_audit(predictions_path, raw_data_path, output_dir)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical sMAPE with floor 50
# ---------------------------------------------------------------------------

def compute_smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
) -> float:
    """sMAPE where both y_true and y_pred are floored at *floor*."""
    y_true_f = np.maximum(np.asarray(y_true, dtype=float), floor)
    y_pred_f = np.maximum(np.asarray(y_pred, dtype=float), floor)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    if mask.sum() == 0:
        return 0.0
    return float(
        200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask])
    )


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

CATEGORY_PRIORITY = ["negative", "spike", "low_price", "high_price", "normal"]


def classify_hour(y_true: float) -> str:
    """Return the *most specific* category for a single y_true value.

    Priority order:
        negative  -> y_true < 0
        spike     -> y_true > 500
        low_price -> 0 <= y_true < 50
        high_price-> y_true >= 300
        normal    -> 0 <= y_true <= 500  (everything else)
    """
    if y_true < 0:
        return "negative"
    if y_true > 500:
        return "spike"
    if y_true < 50:
        return "low_price"
    if y_true >= 300:
        return "high_price"
    return "normal"


def get_period(hour: int) -> str:
    """Map hour_business (1-24) to period label."""
    if 1 <= hour <= 8:
        return "1_8"
    if 9 <= hour <= 16:
        return "9_16"
    if 17 <= hour <= 24:
        return "17_24"
    return "unknown"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> Dict[str, Any]:
    """Return dict with sMAPE_floor50, MAE, RMSE, count for a slice."""
    n = len(y_true_arr)
    if n == 0:
        return {"smape_floor50": None, "mae": None, "rmse": None, "count": 0}
    smape = compute_smape_floor50(y_true_arr, y_pred_arr)
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    return {"smape_floor50": round(smape, 4), "mae": round(mae, 4),
            "rmse": round(rmse, 4), "count": n}


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

def run_p141_negative_spike_audit(
    predictions_path: str,
    raw_data_path: str,
    output_dir: str,
) -> dict:
    """Run the full negative/spike performance audit.

    Parameters
    ----------
    predictions_path : str
        Path to cfg05 all_predictions.csv (must contain y_true column).
    raw_data_path : str
        Path to shandong_pmos_hourly.csv (GBK).
    output_dir : str
        Directory where output artefacts are written.

    Returns
    -------
    dict
        Audit summary dict (also written to audit_summary.json).
    """
    # -- 1. Load predictions ------------------------------------------------
    pred_df = pd.read_csv(predictions_path)
    required_cols = {"ds", "hour_business", "period", "y_pred", "y_true"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"predictions CSV missing columns: {missing}")

    pred_df["ds"] = pd.to_datetime(pred_df["ds"])
    pred_df["hour_business"] = pred_df["hour_business"].astype(int)

    # -- 2. Classify each hour ----------------------------------------------
    pred_df["category"] = pred_df["y_true"].apply(classify_hour)
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
    pred_df["month"] = pred_df["ds"].dt.month

    # -- 3. Ensure output dir -----------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- 4. Per-category metrics --------------------------------------------
    category_metrics: Dict[str, Dict] = {}
    for cat in CATEGORY_PRIORITY:
        mask = pred_df["category"] == cat
        sub = pred_df.loc[mask]
        metrics = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
        category_metrics[cat] = metrics
        # Write individual JSON
        fname = f"{cat}_hours_metrics.json" if cat != "normal" else "normal_hours_metrics.json"
        _write_json(out / fname, {"category": cat, **metrics})

    # -- 5. Per-period metrics ----------------------------------------------
    period_metrics: Dict[str, Dict] = {}
    for period_label in ["1_8", "9_16", "17_24"]:
        mask = pred_df["period"] == period_label
        sub = pred_df.loc[mask]
        period_metrics[period_label] = compute_metrics(
            sub["y_true"].values, sub["y_pred"].values
        )

    # -- 6. Per-month metrics -----------------------------------------------
    month_metrics: Dict[str, Dict] = {}
    for m in sorted(pred_df["month"].unique()):
        mask = pred_df["month"] == m
        sub = pred_df.loc[mask]
        month_metrics[str(m)] = compute_metrics(
            sub["y_true"].values, sub["y_pred"].values
        )

    # -- 7. Top-50 worst error hours ----------------------------------------
    top50 = (
        pred_df.nlargest(50, "abs_error")[
            ["ds", "y_true", "y_pred", "abs_error", "category"]
        ]
        .copy()
    )
    top50["ds"] = top50["ds"].astype(str)
    top50.to_csv(out / "top_50_error_hours.csv", index=False)

    # -- 8. Hourly heatmap (24 hours x avg error) ---------------------------
    heatmap_rows: List[Dict[str, Any]] = []
    for h in range(1, 25):
        mask = pred_df["hour_business"] == h
        sub = pred_df.loc[mask]
        avg_err = float(sub["abs_error"].mean()) if len(sub) > 0 else 0.0
        cnt = int(mask.sum())
        heatmap_rows.append({
            "hour": h,
            "avg_error": round(avg_err, 4),
            "count": cnt,
        })
    _write_json(out / "hourly_heatmap.json", heatmap_rows)

    # -- 9. Audit summary ---------------------------------------------------
    overall = compute_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)

    # Where does error come from?
    total_abs_err = pred_df["abs_error"].sum()
    cat_error_share: Dict[str, float] = {}
    for cat in CATEGORY_PRIORITY:
        mask = pred_df["category"] == cat
        cat_abs = pred_df.loc[mask, "abs_error"].sum()
        cat_error_share[cat] = round(float(cat_abs / total_abs_err * 100), 2) if total_abs_err > 0 else 0.0

    worst_cat = max(cat_error_share, key=cat_error_share.get)  # type: ignore[arg-type]

    summary: Dict[str, Any] = {
        "total_hours": len(pred_df),
        "overall": overall,
        "category_breakdown": category_metrics,
        "category_error_share_pct": cat_error_share,
        "worst_error_category": worst_cat,
        "period_breakdown": period_metrics,
        "month_breakdown": month_metrics,
        "negative_hours_count": category_metrics.get("negative", {}).get("count", 0),
        "spike_hours_count": category_metrics.get("spike", {}).get("count", 0),
        "key_findings": [
            f"{cat_error_share.get('negative', 0):.1f}% of total absolute error comes from negative-price hours",
            f"{cat_error_share.get('spike', 0):.1f}% of total absolute error comes from spike-price hours",
            f"Worst category by error share: {worst_cat}",
            f"Top-50 worst hours written to top_50_error_hours.csv",
        ],
    }
    _write_json(out / "audit_summary.json", summary)

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_paths() -> Tuple[str, str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    predictions = str(repo_root / ".local_artifacts" / "p2025_full" / "dayahead" / "all_predictions.csv")
    raw_data = str(repo_root / "data" / "shandong_pmos_hourly.csv")
    output_dir = str(repo_root / ".local_artifacts" / "p141_negative_spike")
    return predictions, raw_data, output_dir


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        pred_path, raw_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        pred_path, raw_path, out_dir = _default_paths()

    print(f"[P141] predictions : {pred_path}")
    print(f"[P141] raw data    : {raw_path}")
    print(f"[P141] output dir  : {out_dir}")

    result = run_p141_negative_spike_audit(pred_path, raw_path, out_dir)

    print(f"\n[P141] Overall sMAPE_floor50 : {result['overall']['smape_floor50']}")
    print(f"[P141] Negative hours        : {result['negative_hours_count']}")
    print(f"[P141] Spike hours           : {result['spike_hours_count']}")
    print(f"[P141] Worst error category  : {result['worst_error_category']}")
    for finding in result["key_findings"]:
        print(f"  - {finding}")
    print(f"\n[P141] Outputs written to {out_dir}")
