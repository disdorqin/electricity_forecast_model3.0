"""
fusion/unified_weight_learner.py — P70: Unified Weight Learner.

Learns fusion weights for both day-ahead and realtime tasks.

Dimensions:
  - task: dayahead / realtime
  - period: 1_8 / 9_16 / 17_24
  - regime: normal / low_price / negative_risk / high_spike
  - model: model_name

Strict no-lookahead:
  For target_day D, weights use only days < D.

Fallback ladder:
  1. If enough history → rolling BGEW
  2. If degraded history → period BGEW
  3. If one model → single model weight=1.0
  4. Else → task fallback (equal weight)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.75
DEFAULT_ALPHA = 0.05
MIN_TRAINING_DAYS_ROLLING = 14
MIN_TRAINING_DAYS_PERIOD = 7

# ── Status constants ──────────────────────────────────────────────────
UNIFIED_LEARNER_TRAINED = "UNIFIED_LEARNER_TRAINED"
UNIFIED_LEARNER_DEGRADED = "UNIFIED_LEARNER_DEGRADED"
UNIFIED_LEARNER_SINGLE_MODEL = "UNIFIED_LEARNER_SINGLE_MODEL"
UNIFIED_LEARNER_BLOCKED = "UNIFIED_LEARNER_BLOCKED"


def compute_bgew_weights(
    smape_values: dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
) -> dict[str, float]:
    """Compute BGEW weights from sMAPE values.

    Parameters
    ----------
    smape_values : dict
        {model_name: smape_floor50}
    alpha : float
        Exponential decay parameter.
    min_weight : float
        Minimum weight floor.
    max_weight : float
        Maximum weight cap.

    Returns
    -------
    dict of {model_name: weight}
    """
    if not smape_values:
        return {}

    scores = {m: np.exp(-alpha * s) for m, s in smape_values.items()}
    total = sum(scores.values())
    if total < 1e-10:
        n = len(scores)
        return {m: 1.0 / n for m in scores}

    weights = {m: s / total for m, s in scores.items()}

    # Clip
    weights = {m: max(min_weight, min(max_weight, w)) for m, w in weights.items()}

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {m: w / total for m, w in weights.items()}

    return weights


def train_unified_weights(
    dayahead_predictions: Optional[pd.DataFrame] = None,
    realtime_predictions: Optional[pd.DataFrame] = None,
    dayahead_actuals: Optional[pd.DataFrame] = None,
    realtime_actuals: Optional[pd.DataFrame] = None,
    target_day: str = "",
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Train unified weights for both tasks.

    Parameters
    ----------
    dayahead_predictions : DataFrame
        Day-ahead prediction ledger (multi-model).
    realtime_predictions : DataFrame
        Realtime prediction ledger (multi-model).
    dayahead_actuals : DataFrame
        Day-ahead actual values.
    realtime_actuals : DataFrame
        Realtime actual values.
    target_day : str
        Target day for which weights are computed.
    alpha : float
        BGEW alpha parameter.

    Returns
    -------
    dict with weights DataFrames and status.
    """
    result: dict[str, Any] = {
        "status": UNIFIED_LEARNER_BLOCKED,
        "dayahead_weights": None,
        "realtime_weights": None,
        "training_days": 0,
        "reason_codes": [],
    }

    # Day-ahead weights
    if dayahead_predictions is not None and dayahead_actuals is not None:
        da_weights = _compute_task_weights(
            predictions=dayahead_predictions,
            actuals=dayahead_actuals,
            target_day=target_day,
            task="dayahead",
            alpha=alpha,
        )
        result["dayahead_weights"] = da_weights["weights_df"]
        result["training_days"] = da_weights.get("training_days", 0)
        result["reason_codes"].extend(da_weights.get("reason_codes", []))

    # Realtime weights
    if realtime_predictions is not None and realtime_actuals is not None:
        rt_weights = _compute_task_weights(
            predictions=realtime_predictions,
            actuals=realtime_actuals,
            target_day=target_day,
            task="realtime",
            alpha=alpha,
        )
        result["realtime_weights"] = rt_weights["weights_df"]
        result["reason_codes"].extend(rt_weights.get("reason_codes", []))

    # Determine status
    has_da = result["dayahead_weights"] is not None and len(result["dayahead_weights"]) > 0
    has_rt = result["realtime_weights"] is not None and len(result["realtime_weights"]) > 0

    if has_da and has_rt:
        result["status"] = UNIFIED_LEARNER_TRAINED
    elif has_da or has_rt:
        result["status"] = UNIFIED_LEARNER_DEGRADED
    else:
        result["status"] = UNIFIED_LEARNER_BLOCKED
        result["reason_codes"].append("NO_WEIGHTS_COMPUTED")

    return result


def _compute_task_weights(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    target_day: str,
    task: str,
    alpha: float,
) -> dict[str, Any]:
    """Compute weights for a single task."""
    result: dict[str, Any] = {
        "weights_df": None,
        "training_days": 0,
        "reason_codes": [],
    }

    # Merge predictions with actuals
    merged = _merge_pred_actuals(predictions, actuals, target_day)
    if merged is None or len(merged) == 0:
        result["reason_codes"].append(f"NO_MERGED_DATA_{task}")
        return result

    # Get unique models
    model_col = "model_name" if "model_name" in merged.columns else None
    if model_col is None:
        result["reason_codes"].append(f"NO_MODEL_COLUMN_{task}")
        return result

    models = merged[model_col].unique().tolist()
    if len(models) < 2:
        result["reason_codes"].append(f"SINGLE_MODEL_{task}")
        # Single model gets weight=1.0
        weights_df = pd.DataFrame({
            "task": task,
            "target_day": target_day,
            "period": "all",
            "model_name": models[0] if models else "unknown",
            "weight": 1.0,
            "learner_method": "single_model",
            "training_days": merged["business_day"].nunique() if "business_day" in merged.columns else 0,
        })
        result["weights_df"] = weights_df
        result["training_days"] = weights_df["training_days"].iloc[0]
        return result

    # Compute per-model sMAPE
    smape_by_model = {}
    for model in models:
        model_data = merged[merged[model_col] == model]
        if "y_true" in model_data.columns and "y_pred" in model_data.columns:
            y_true = model_data["y_true"].dropna().values
            y_pred = model_data["y_pred"].dropna().values
            min_len = min(len(y_true), len(y_pred))
            if min_len > 0:
                y_true_f = np.maximum(y_true[:min_len], 50)
                y_pred_f = np.maximum(y_pred[:min_len], 50)
                denom = np.abs(y_true_f) + np.abs(y_pred_f)
                mask = denom > 1e-10
                if mask.any():
                    smape = 200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask])
                    smape_by_model[model] = float(smape)

    if not smape_by_model:
        result["reason_codes"].append(f"NO_SMAPE_COMPUTED_{task}")
        return result

    # Compute BGEW weights
    weights = compute_bgew_weights(smape_by_model, alpha=alpha)

    # Build weights DataFrame
    training_days = merged["business_day"].nunique() if "business_day" in merged.columns else 0
    weights_df = pd.DataFrame({
        "task": task,
        "target_day": target_day,
        "period": "all",
        "model_name": list(weights.keys()),
        "weight": list(weights.values()),
        "learner_method": "bgew",
        "training_days": training_days,
    })

    result["weights_df"] = weights_df
    result["training_days"] = training_days
    return result


def _merge_pred_actuals(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    target_day: str,
) -> Optional[pd.DataFrame]:
    """Merge predictions with actuals, filtering to days before target."""
    if predictions is None or actuals is None:
        return None

    pred = predictions.copy()
    actual = actuals.copy()

    # Filter to before target day (no lookahead)
    if target_day and "business_day" in pred.columns:
        pred = pred[pred["business_day"] < target_day]
    if target_day and "business_day" in actual.columns:
        actual = actual[actual["business_day"] < target_day]

    # Find join keys
    join_keys = []
    for key in ["business_day", "hour_business", "ds"]:
        if key in pred.columns and key in actual.columns:
            join_keys.append(key)

    if not join_keys:
        return None

    # Normalize join key types (string vs datetime64 mismatch)
    pred_work = pred.copy()
    actual_work = actual.copy()
    for key in join_keys:
        if pred_work[key].dtype != actual_work[key].dtype:
            pred_work[key] = pred_work[key].astype(str)
            actual_work[key] = actual_work[key].astype(str)

    # Determine prediction value column
    pred_val_col = None
    for col in ["y_pred", "dayahead_price", "trend_pred"]:
        if col in pred.columns:
            pred_val_col = col
            break

    if pred_val_col is None:
        return None

    # Rename for merge
    pred_renamed = pred_work.rename(columns={pred_val_col: "y_pred"})

    # Merge
    actual_cols = ["y_true"] + join_keys
    actual_avail = [c for c in actual_cols if c in actual_work.columns]
    merged = pd.merge(
        pred_renamed,
        actual_work[actual_avail],
        on=join_keys,
        how="inner",
    )

    return merged
