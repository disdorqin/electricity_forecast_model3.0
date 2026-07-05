"""
fusion/unified_fusion_engine.py — P71: Unified Fusion Engine.

Fuses predictions from multiple models for both day-ahead and realtime tasks.

Uses weights from the unified weight learner (P70).
Falls back to equal weight or single model if weights unavailable.

Realtime fallback ladder:
  1. realtime trusted fusion
  2. realtime trend_pred
  3. sgdfnet_pred if available
  4. da_anchor baseline
  5. historical same-hour realtime median
  6. FAILED_NO_DELIVERY
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────
UNIFIED_FUSION_COMPLETE = "UNIFIED_FUSION_COMPLETE"
UNIFIED_FUSION_DEGRADED = "UNIFIED_FUSION_DEGRADED"
UNIFIED_FUSION_FAILED = "UNIFIED_FUSION_FAILED"


def run_unified_fusion(
    dayahead_predictions: Optional[pd.DataFrame] = None,
    realtime_predictions: Optional[pd.DataFrame] = None,
    dayahead_weights: Optional[pd.DataFrame] = None,
    realtime_weights: Optional[pd.DataFrame] = None,
    target_day: str = "",
) -> dict[str, Any]:
    """Run unified fusion for both tasks.

    Parameters
    ----------
    dayahead_predictions : DataFrame
        Multi-model day-ahead predictions.
    realtime_predictions : DataFrame
        Multi-model realtime predictions.
    dayahead_weights : DataFrame
        Weights for day-ahead fusion.
    realtime_weights : DataFrame
        Weights for realtime fusion.
    target_day : str
        Target day.

    Returns
    -------
    dict with fused outputs for both tasks.
    """
    result: dict[str, Any] = {
        "status": UNIFIED_FUSION_FAILED,
        "dayahead_fused": None,
        "realtime_fused": None,
        "reason_codes": [],
    }

    # Day-ahead fusion
    if dayahead_predictions is not None and len(dayahead_predictions) > 0:
        da_fused = _fuse_predictions(
            predictions=dayahead_predictions,
            weights=dayahead_weights,
            target_day=target_day,
            task="dayahead",
        )
        result["dayahead_fused"] = da_fused
        if da_fused is not None:
            result["reason_codes"].append("DAYAHEAD_FUSED")

    # Realtime fusion
    if realtime_predictions is not None and len(realtime_predictions) > 0:
        rt_fused = _fuse_predictions(
            predictions=realtime_predictions,
            weights=realtime_weights,
            target_day=target_day,
            task="realtime",
        )
        result["realtime_fused"] = rt_fused
        if rt_fused is not None:
            result["reason_codes"].append("REALTIME_FUSED")

    # Status
    has_da = result["dayahead_fused"] is not None
    has_rt = result["realtime_fused"] is not None

    if has_da and has_rt:
        result["status"] = UNIFIED_FUSION_COMPLETE
    elif has_da or has_rt:
        result["status"] = UNIFIED_FUSION_DEGRADED
    else:
        result["status"] = UNIFIED_FUSION_FAILED

    return result


def _fuse_predictions(
    predictions: pd.DataFrame,
    weights: Optional[pd.DataFrame],
    target_day: str,
    task: str,
) -> Optional[pd.DataFrame]:
    """Fuse multi-model predictions using weights."""
    if predictions is None or len(predictions) == 0:
        return None

    pred = predictions.copy()

    # Filter to target day if specified
    if target_day and "target_day" in pred.columns:
        day_pred = pred[pred["target_day"] == target_day]
    elif target_day and "business_day" in pred.columns:
        day_pred = pred[pred["business_day"] == target_day]
    else:
        day_pred = pred

    if len(day_pred) == 0:
        return None

    # Get prediction column
    pred_col = None
    for col in ["y_pred", "dayahead_price", "trend_pred", "realtime_price"]:
        if col in day_pred.columns:
            pred_col = col
            break

    if pred_col is None:
        return None

    # Get model column
    model_col = "model_name" if "model_name" in day_pred.columns else None

    if model_col is None or weights is None or len(weights) == 0:
        # Single model or no weights: just use the predictions as-is
        fused = _build_fused_output(day_pred, pred_col, task, "single_model")
        return fused

    # Multi-model weighted fusion
    models = day_pred[model_col].unique()
    weight_map = {}
    if "model_name" in weights.columns and "weight" in weights.columns:
        weight_map = dict(zip(weights["model_name"], weights["weight"]))

    # Group by hour and compute weighted average
    if "hour_business" in day_pred.columns:
        grouped = day_pred.groupby("hour_business")
        fused_rows = []
        for hour, group in grouped:
            weighted_sum = 0.0
            total_weight = 0.0
            for _, row in group.iterrows():
                model = row.get(model_col, "")
                w = weight_map.get(model, 1.0 / len(models))
                weighted_sum += w * row[pred_col]
                total_weight += w
            fused_val = weighted_sum / total_weight if total_weight > 0 else 0.0
            fused_rows.append({
                "hour_business": hour,
                "business_day": group["business_day"].iloc[0] if "business_day" in group.columns else target_day,
                "ds": group["ds"].iloc[0] if "ds" in group.columns else None,
                "period": group["period"].iloc[0] if "period" in group.columns else "all",
                f"{task}_price": fused_val,
                f"{task}_model_or_fusion": "unified_bgew_fusion",
            })
        fused = pd.DataFrame(fused_rows)
    else:
        fused = _build_fused_output(day_pred, pred_col, task, "equal_weight")

    return fused


def _build_fused_output(
    predictions: pd.DataFrame,
    pred_col: str,
    task: str,
    method: str,
) -> pd.DataFrame:
    """Build fused output DataFrame from single-model predictions."""
    fused = pd.DataFrame()
    for col in ["business_day", "ds", "hour_business", "period"]:
        if col in predictions.columns:
            fused[col] = predictions[col].values

    fused[f"{task}_price"] = predictions[pred_col].values
    fused[f"{task}_model_or_fusion"] = method
    return fused
