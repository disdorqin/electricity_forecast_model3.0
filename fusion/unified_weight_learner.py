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

# ── 2.5-style dimensional constants ─────────────────────────────────
PERIODS = ["1_8", "9_16", "17_24"]
REGIMES = ["normal", "low_price", "negative_risk", "high_spike"]
LEARNER_FULL_DIMENSIONAL = "LEARNER_FULL_DIMENSIONAL"
LEARNER_PERIOD_ONLY = "LEARNER_PERIOD_ONLY"
LEARNER_TASK_ONLY = "LEARNER_TASK_ONLY"


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


# ── Regime classification ─────────────────────────────────────────────

def classify_regime(prices: pd.Series) -> pd.Series:
    """Classify electricity prices into market regimes.

    Parameters
    ----------
    prices : pd.Series
        Price values (can be actual or predicted).

    Returns
    -------
    pd.Series of str
        Regime labels: 'negative_risk', 'low_price', 'high_spike', or 'normal'.
    """
    conditions = [
        prices < 0,
        prices < 50,
        prices > 500,
    ]
    choices = ["negative_risk", "low_price", "high_spike"]
    return np.select(conditions, choices, default="normal")


# ── sMAPE helper ──────────────────────────────────────────────────────

def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sMAPE with floor-50 clipping."""
    y_true_f = np.maximum(y_true.astype(float), 50)
    y_pred_f = np.maximum(y_pred.astype(float), 50)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    if not mask.any():
        return float("inf")
    return float(200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask]))


# ── Dimensional weight training (2.5-style) ──────────────────────────

def train_dimensional_weights(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    target_day: str,
    task: str,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Train weights across task x period x regime dimensions.

    For each period, computes period-level BGEW weights.  Then for each
    regime within that period, computes regime-specific BGEW weights if
    enough data exists; otherwise falls back to period-level weights.

    No-lookahead invariant: for target_day D, only days < D are used.

    Parameters
    ----------
    predictions : DataFrame
        Multi-model prediction ledger.
    actuals : DataFrame
        Actual values ledger.
    target_day : str
        Target day for which weights are computed.
    task : str
        Task name ('dayahead' or 'realtime').
    alpha : float
        BGEW exponential decay parameter.

    Returns
    -------
    dict with keys:
        weights_df : DataFrame or None
        training_days : int
        reason_codes : list of str
        lookback_start : str
        lookback_end : str
        dimensional_level : str or None
    """
    result: dict[str, Any] = {
        "weights_df": None,
        "training_days": 0,
        "reason_codes": [],
        "lookback_start": "",
        "lookback_end": "",
        "dimensional_level": None,
    }

    # Merge predictions with actuals
    merged = _merge_pred_actuals(predictions, actuals, target_day)
    if merged is None or len(merged) == 0:
        result["reason_codes"].append(f"NO_MERGED_DATA_{task}")
        return result

    # No-lookahead: ensure only days < target_day
    if target_day and "business_day" in merged.columns:
        merged = merged[merged["business_day"] < target_day]

    if len(merged) == 0:
        result["reason_codes"].append(f"NO_DATA_BEFORE_TARGET_{task}")
        return result

    # Identify models
    models = merged["model_name"].unique().tolist() if "model_name" in merged.columns else []
    if not models:
        result["reason_codes"].append(f"NO_MODELS_{task}")
        return result

    # Track lookback window
    if "business_day" in merged.columns:
        sorted_days = sorted(merged["business_day"].unique())
        result["lookback_start"] = str(sorted_days[0]) if len(sorted_days) > 0 else ""
        result["lookback_end"] = str(sorted_days[-1]) if len(sorted_days) > 0 else ""

    # Period hour ranges
    _PERIOD_HOURS = {
        "1_8": (1, 8),
        "9_16": (9, 16),
        "17_24": (17, 24),
    }

    all_rows: list[dict[str, Any]] = []
    has_full_regime = True
    has_period_level = False

    for period in PERIODS:
        hour_start, hour_end = _PERIOD_HOURS[period]

        # Filter to period hours
        if "hour_business" in merged.columns:
            period_mask = (
                (merged["hour_business"] >= hour_start)
                & (merged["hour_business"] <= hour_end)
            )
            period_data = merged[period_mask]
        else:
            period_data = merged

        n_period_days = (
            period_data["business_day"].nunique()
            if "business_day" in period_data.columns
            else 0
        )

        if n_period_days < MIN_TRAINING_DAYS_PERIOD:
            has_full_regime = False
            result["reason_codes"].append(f"INSUFFICIENT_PERIOD_DATA_{task}_{period}")
            continue

        has_period_level = True

        # ── Period-level sMAPE & BGEW weights (fallback) ──
        period_smape: dict[str, float] = {}
        for model in models:
            model_data = period_data[period_data["model_name"] == model]
            if len(model_data) > 0 and "y_true" in model_data.columns and "y_pred" in model_data.columns:
                y_true = model_data["y_true"].dropna().values
                y_pred = model_data["y_pred"].dropna().values
                min_len = min(len(y_true), len(y_pred))
                if min_len > 0:
                    period_smape[model] = _compute_smape(y_true[:min_len], y_pred[:min_len])

        period_weights = compute_bgew_weights(period_smape, alpha=alpha) if len(period_smape) > 1 else {}

        # ── Regime-level weights within this period ──
        regime_trained = False
        if len(period_data) > 0 and "y_true" in period_data.columns:
            regimes = classify_regime(period_data["y_true"])

            for regime in REGIMES:
                regime_mask = regimes == regime
                regime_data = period_data[regime_mask]

                n_regime_days = (
                    regime_data["business_day"].nunique()
                    if "business_day" in regime_data.columns
                    else 0
                )

                if n_regime_days < MIN_TRAINING_DAYS_PERIOD:
                    has_full_regime = False
                    # Fall back to period-level weights
                    if period_weights:
                        for model_name, w in period_weights.items():
                            all_rows.append({
                                "task": task,
                                "target_day": target_day,
                                "business_day": target_day,
                                "period": period,
                                "regime": regime,
                                "model_name": model_name,
                                "weight": w,
                                "learner_method": "bgew_period_fallback",
                                "training_days": n_period_days,
                                "lookback_start": result["lookback_start"],
                                "lookback_end": result["lookback_end"],
                                "reason_codes": f"REGIME_FALLBACK_{regime}",
                            })
                    continue

                # Compute regime-level sMAPE & BGEW weights
                regime_smape: dict[str, float] = {}
                for model in models:
                    model_data = regime_data[regime_data["model_name"] == model]
                    if len(model_data) > 0 and "y_true" in model_data.columns and "y_pred" in model_data.columns:
                        y_true = model_data["y_true"].dropna().values
                        y_pred = model_data["y_pred"].dropna().values
                        min_len = min(len(y_true), len(y_pred))
                        if min_len > 0:
                            regime_smape[model] = _compute_smape(y_true[:min_len], y_pred[:min_len])

                regime_weights = compute_bgew_weights(regime_smape, alpha=alpha) if len(regime_smape) > 1 else {}

                if regime_weights:
                    regime_trained = True
                    for model_name, w in regime_weights.items():
                        all_rows.append({
                            "task": task,
                            "target_day": target_day,
                            "business_day": target_day,
                            "period": period,
                            "regime": regime,
                            "model_name": model_name,
                            "weight": w,
                            "learner_method": "bgew_regime",
                            "training_days": n_regime_days,
                            "lookback_start": result["lookback_start"],
                            "lookback_end": result["lookback_end"],
                            "reason_codes": "",
                        })
                elif period_weights:
                    has_full_regime = False
                    for model_name, w in period_weights.items():
                        all_rows.append({
                            "task": task,
                            "target_day": target_day,
                            "business_day": target_day,
                            "period": period,
                            "regime": regime,
                            "model_name": model_name,
                            "weight": w,
                            "learner_method": "bgew_period_fallback",
                            "training_days": n_period_days,
                            "lookback_start": result["lookback_start"],
                            "lookback_end": result["lookback_end"],
                            "reason_codes": f"REGIME_FALLBACK_{regime}",
                        })

        # If no regime was trained for this period, emit period-level weights for all regimes
        if not regime_trained and period_weights:
            for regime in REGIMES:
                for model_name, w in period_weights.items():
                    all_rows.append({
                        "task": task,
                        "target_day": target_day,
                        "business_day": target_day,
                        "period": period,
                        "regime": regime,
                        "model_name": model_name,
                        "weight": w,
                        "learner_method": "bgew_period",
                        "training_days": n_period_days,
                        "lookback_start": result["lookback_start"],
                        "lookback_end": result["lookback_end"],
                        "reason_codes": "PERIOD_LEVEL_ONLY",
                    })

    # Determine dimensional level
    if has_full_regime and has_period_level:
        result["dimensional_level"] = LEARNER_FULL_DIMENSIONAL
    elif has_period_level:
        result["dimensional_level"] = LEARNER_PERIOD_ONLY
    else:
        result["dimensional_level"] = LEARNER_TASK_ONLY

    # Build output DataFrame
    if all_rows:
        weights_df = pd.DataFrame(all_rows)
        result["weights_df"] = weights_df
        result["training_days"] = int(weights_df["training_days"].max()) if len(weights_df) > 0 else 0
    else:
        result["reason_codes"].append(f"NO_WEIGHTS_COMPUTED_{task}")

    return result


def train_unified_weights(
    dayahead_predictions: Optional[pd.DataFrame] = None,
    realtime_predictions: Optional[pd.DataFrame] = None,
    dayahead_actuals: Optional[pd.DataFrame] = None,
    realtime_actuals: Optional[pd.DataFrame] = None,
    target_day: str = "",
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Train unified weights for both tasks using 2.5-style dimensional learning.

    Calls ``train_dimensional_weights`` for each task (dayahead, realtime),
    producing weights across period x regime dimensions.  If only one model
    is available per task, weight=1 is assigned with SINGLE_MODEL_FALLBACK.

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
        "lookback_start": "",
        "lookback_end": "",
        "dimensional_level": None,
    }

    trained_levels: list[str] = []

    # ── Day-ahead dimensional weights ──
    if dayahead_predictions is not None and dayahead_actuals is not None:
        da_result = train_dimensional_weights(
            predictions=dayahead_predictions,
            actuals=dayahead_actuals,
            target_day=target_day,
            task="dayahead",
            alpha=alpha,
        )
        if da_result["weights_df"] is not None and len(da_result["weights_df"]) > 0:
            result["dayahead_weights"] = da_result["weights_df"]
            result["training_days"] = max(result["training_days"], da_result["training_days"])
            result["reason_codes"].extend(da_result["reason_codes"])
            result["lookback_start"] = da_result.get("lookback_start", "")
            result["lookback_end"] = da_result.get("lookback_end", "")
            if da_result.get("dimensional_level"):
                trained_levels.append(da_result["dimensional_level"])
        else:
            # Check for single-model fallback
            merged = _merge_pred_actuals(dayahead_predictions, dayahead_actuals, target_day)
            if merged is not None and "model_name" in merged.columns:
                models = merged["model_name"].unique().tolist()
                if len(models) == 1:
                    result["dayahead_weights"] = _single_model_fallback_df(
                        "dayahead", target_day, models[0], merged,
                    )
                    result["reason_codes"].append("SINGLE_MODEL_FALLBACK_dayahead")
                    trained_levels.append("SINGLE_MODEL_FALLBACK")

    # ── Realtime dimensional weights ──
    if realtime_predictions is not None and realtime_actuals is not None:
        rt_result = train_dimensional_weights(
            predictions=realtime_predictions,
            actuals=realtime_actuals,
            target_day=target_day,
            task="realtime",
            alpha=alpha,
        )
        if rt_result["weights_df"] is not None and len(rt_result["weights_df"]) > 0:
            result["realtime_weights"] = rt_result["weights_df"]
            result["training_days"] = max(result["training_days"], rt_result["training_days"])
            result["reason_codes"].extend(rt_result["reason_codes"])
            if not result["lookback_start"]:
                result["lookback_start"] = rt_result.get("lookback_start", "")
                result["lookback_end"] = rt_result.get("lookback_end", "")
            if rt_result.get("dimensional_level"):
                trained_levels.append(rt_result["dimensional_level"])
        else:
            merged = _merge_pred_actuals(realtime_predictions, realtime_actuals, target_day)
            if merged is not None and "model_name" in merged.columns:
                models = merged["model_name"].unique().tolist()
                if len(models) == 1:
                    result["realtime_weights"] = _single_model_fallback_df(
                        "realtime", target_day, models[0], merged,
                    )
                    result["reason_codes"].append("SINGLE_MODEL_FALLBACK_realtime")
                    trained_levels.append("SINGLE_MODEL_FALLBACK")

    # ── Determine overall status ──
    has_da = result["dayahead_weights"] is not None and len(result["dayahead_weights"]) > 0
    has_rt = result["realtime_weights"] is not None and len(result["realtime_weights"]) > 0

    if has_da and has_rt:
        if LEARNER_FULL_DIMENSIONAL in trained_levels:
            result["status"] = UNIFIED_LEARNER_TRAINED
            result["dimensional_level"] = LEARNER_FULL_DIMENSIONAL
        elif LEARNER_PERIOD_ONLY in trained_levels:
            result["status"] = UNIFIED_LEARNER_TRAINED
            result["dimensional_level"] = LEARNER_PERIOD_ONLY
        elif LEARNER_TASK_ONLY in trained_levels:
            result["status"] = UNIFIED_LEARNER_DEGRADED
            result["dimensional_level"] = LEARNER_TASK_ONLY
        else:
            result["status"] = UNIFIED_LEARNER_DEGRADED
    elif has_da or has_rt:
        result["status"] = UNIFIED_LEARNER_DEGRADED
        if trained_levels:
            result["dimensional_level"] = trained_levels[0]
    else:
        result["status"] = UNIFIED_LEARNER_BLOCKED
        result["reason_codes"].append("NO_WEIGHTS_COMPUTED")

    return result


def _single_model_fallback_df(
    task: str, target_day: str, model_name: str, merged: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single-model fallback weights DataFrame (weight=1 for all period/regime combos)."""
    rows = []
    training_days = merged["business_day"].nunique() if "business_day" in merged.columns else 0
    lookback_start = ""
    lookback_end = ""
    if "business_day" in merged.columns:
        sorted_days = sorted(merged["business_day"].unique())
        lookback_start = str(sorted_days[0]) if len(sorted_days) > 0 else ""
        lookback_end = str(sorted_days[-1]) if len(sorted_days) > 0 else ""

    for period in PERIODS:
        for regime in REGIMES:
            rows.append({
                "task": task,
                "target_day": target_day,
                "business_day": target_day,
                "period": period,
                "regime": regime,
                "model_name": model_name,
                "weight": 1.0,
                "learner_method": "SINGLE_MODEL_FALLBACK",
                "training_days": training_days,
                "lookback_start": lookback_start,
                "lookback_end": lookback_end,
                "reason_codes": "SINGLE_MODEL_FALLBACK",
            })
    return pd.DataFrame(rows)


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
