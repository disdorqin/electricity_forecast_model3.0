"""
fusion/unified_weight_learner.py — P70/P94: Unified Weight Learner.

Learns fusion weights for both day-ahead and realtime tasks.

P94 update: learner_policy distinguishes day-ahead vs realtime.

    learner_policy = {
        "dayahead": "period_regime_bgew",    # task x period x regime
        "realtime": "pooled_30d_bgew",        # task-level pooled 24H
    }

Day-ahead (unchanged):
    Dimensions: task x period x regime
    Uses train_dimensional_weights() for period/regime-specific BGEW.

Realtime (changed):
    Uses pooled_30d_bgew:
        For target_day D, use complete days D-30 .. D-1
        Use all hours 1..24
        rows ~ 720
        Compute sMAPE_floor50 for each candidate model
        BGEW over model dimension only (no period/regime split)

    If only rt_da_anchor available:
        model_name = rt_da_anchor
        weight = 1.0
        learner_method = realtime_single_model_safe_baseline
        reason_codes = SGDFNET_ASSIST_DISABLED

Fallback ladder (both tasks):
  1. If enough history → BGEW
  2. If degraded history → period BGEW
  3. If one model → single model weight=1.0
  4. Else → task fallback (equal weight)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.realtime_state import (
    SGDFNET_ASSIST_DISABLED,
    LEARNER_POLICY_DAYAHEAD,
    LEARNER_POLICY_REALTIME,
    LEARNER_POLICY_REALTIME_SINGLE,
    REALTIME_LEARNER_POOLED_TRAINED,
    REALTIME_LEARNER_SINGLE_MODEL,
    REALTIME_LEARNER_BLOCKED,
)

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

# ── Learner policy (P94) ──────────────────────────────────────────────
LEARNER_POLICY: dict[str, str] = {
    "dayahead": LEARNER_POLICY_DAYAHEAD,    # period_regime_bgew
    "realtime": LEARNER_POLICY_REALTIME,     # pooled_30d_bgew
}

# ── Pooled learner constants ──────────────────────────────────────────
POOLED_LOOKBACK_DAYS = 30
POOLED_MIN_TRAINING_DAYS = 7


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
    learner_policy: Optional[dict[str, str]] = None,
    hard_reject_bad_assist: bool = False,
) -> dict[str, Any]:
    """Train unified weights using the task-specific learner policy.

    P94: learner_policy determines the method per task.

    Day-ahead (LEARNER_POLICY_DAYAHEAD = period_regime_bgew):
        Dimensional learning across period x regime (unchanged).

    Realtime (LEARNER_POLICY_REALTIME = pooled_30d_bgew):
        Pooled 30-day learning without period/regime split.

    Parameters
    ----------
    dayahead_predictions : DataFrame
        Day-ahead prediction ledger (multi-model).
    realtime_predictions : DataFrame
        Realtime prediction ledger (can have 1 or 2 models).
    dayahead_actuals : DataFrame
        Day-ahead actual values.
    realtime_actuals : DataFrame
        Realtime actual values.
    target_day : str
        Target day for which weights are computed.
    alpha : float
        BGEW alpha parameter.
    learner_policy : dict, optional
        Override policy. Defaults to LEARNER_POLICY.
    hard_reject_bad_assist : bool
        If True, completely exclude bad assist models.

    Returns
    -------
    dict with weights DataFrames and status.
    """
    policy = learner_policy or LEARNER_POLICY
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

    # ── Day-ahead: period_regime_bgew (dimensional) ──
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
            # Single model fallback
            merged = _merge_pred_actuals(dayahead_predictions, dayahead_actuals, target_day)
            if merged is not None and "model_name" in merged.columns:
                models = merged["model_name"].unique().tolist()
                if len(models) == 1:
                    result["dayahead_weights"] = _single_model_fallback_df(
                        "dayahead", target_day, models[0], merged,
                    )
                    result["reason_codes"].append("SINGLE_MODEL_FALLBACK_dayahead")
                    trained_levels.append("SINGLE_MODEL_FALLBACK")

    # ── Realtime: pooled_30d_bgew ──
    if realtime_predictions is not None and realtime_actuals is not None:
        rt_policy = policy.get("realtime", LEARNER_POLICY_REALTIME)

        if rt_policy == LEARNER_POLICY_REALTIME:
            # Use pooled 30-day BGEW
            rt_result = train_pooled_30d_bgew(
                predictions=realtime_predictions,
                actuals=realtime_actuals,
                target_day=target_day,
                task="realtime",
                alpha=alpha,
                hard_reject_bad_assist=hard_reject_bad_assist,
            )
            if rt_result["weights_df"] is not None and len(rt_result["weights_df"]) > 0:
                result["realtime_weights"] = rt_result["weights_df"]
                result["training_days"] = max(result["training_days"], rt_result["training_days"])
                result["reason_codes"].extend(rt_result.get("reason_codes", []))
                if not result["lookback_start"]:
                    result["lookback_start"] = rt_result.get("lookback_start", "")
                    result["lookback_end"] = rt_result.get("lookback_end", "")
                trained_levels.append(f"REALTIME_POOLED:{rt_result['status']}")
            else:
                # Single model fallback
                merged = _merge_pred_actuals(realtime_predictions, realtime_actuals, target_day)
                if merged is not None and "model_name" in merged.columns:
                    models = merged["model_name"].unique().tolist()
                    if len(models) == 1:
                        result["realtime_weights"] = _single_model_fallback_df(
                            "realtime", target_day, models[0], merged,
                        )
                        result["reason_codes"].append("SINGLE_MODEL_FALLBACK_realtime")
                        trained_levels.append("SINGLE_MODEL_FALLBACK")
        else:
            # Fallback to dimensional (legacy)
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
                result["reason_codes"].extend(rt_result.get("reason_codes", []))
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


def train_pooled_30d_bgew(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    target_day: str,
    task: str,
    alpha: float = DEFAULT_ALPHA,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    hard_reject_bad_assist: bool = False,
) -> dict[str, Any]:
    """Train pooled 30-day BGEW weights for realtime task.

    P94: Realtime uses pooled_30d_bgew instead of period_regime_bgew.

    For target_day D:
      - Use complete days D-30 .. D-1 (no-lookahead)
      - Use all hours 1..24 (pooled, no period/regime split)
      - rows ~ 720 (30 days x 24 hours)
      - Compute sMAPE_floor50 for each model
      - BGEW over model dimension only

    Parameters
    ----------
    predictions : DataFrame
        Prediction ledger with model_name, business_day, y_pred, etc.
    actuals : DataFrame
        Actual values ledger.
    target_day : str
        Target day (no-lookahead: only days < target_day used).
    task : str
        Task name (should be 'realtime').
    alpha : float
        BGEW exponential decay parameter.
    min_weight : float
        Minimum weight floor.
    max_weight : float
        Maximum weight cap.
    hard_reject_bad_assist : bool
        If True, completely exclude an assist model whose sMAPE is
        significantly worse than the baseline (instead of just low weight).

    Returns
    -------
    dict with keys:
        weights_df : DataFrame or None
        training_days : int
        training_rows : int
        reason_codes : list of str
        lookback_start : str
        lookback_end : str
        status : str
    """
    result: dict[str, Any] = {
        "weights_df": None,
        "training_days": 0,
        "training_rows": 0,
        "reason_codes": [],
        "lookback_start": "",
        "lookback_end": "",
        "status": REALTIME_LEARNER_BLOCKED,
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

    # Count training days
    training_days = merged["business_day"].nunique() if "business_day" in merged.columns else 0
    result["training_days"] = training_days
    result["training_rows"] = len(merged)

    if training_days < POOLED_MIN_TRAINING_DAYS:
        result["reason_codes"].append(f"INSUFFICIENT_TRAINING_DAYS:{training_days}_lt_{POOLED_MIN_TRAINING_DAYS}")
        return result

    # Single model case
    if len(models) < 2:
        model_name = models[0] if models else "rt_da_anchor"
        result["status"] = REALTIME_LEARNER_SINGLE_MODEL
        result["reason_codes"].append("SGDFNET_ASSIST_DISABLED" if "sgdfnet" not in model_name else "SINGLE_MODEL")
        weights_df = pd.DataFrame({
            "task": task,
            "target_day": target_day,
            "period": "all",
            "regime": "all",
            "model_name": [model_name],
            "weight": [1.0],
            "learner_method": LEARNER_POLICY_REALTIME_SINGLE,
            "training_days": training_days,
            "training_rows": len(merged),
            "lookback_start": result["lookback_start"],
            "lookback_end": result["lookback_end"],
            "reason_codes": ";".join(result["reason_codes"]) if result["reason_codes"] else "",
        })
        result["weights_df"] = weights_df
        return result

    # Compute pooled sMAPE for each model
    smape_by_model: dict[str, float] = {}
    for model in models:
        model_data = merged[merged["model_name"] == model]
        if "y_true" not in model_data.columns or "y_pred" not in model_data.columns:
            continue
        y_true = model_data["y_true"].dropna().values
        y_pred = model_data["y_pred"].dropna().values
        min_len = min(len(y_true), len(y_pred))
        if min_len > 0:
            smape_by_model[model] = _compute_smape(y_true[:min_len], y_pred[:min_len])

    if not smape_by_model:
        result["reason_codes"].append("NO_SMAPE_COMPUTED_POOLED")
        return result

    # Compute BGEW weights
    weights = compute_bgew_weights(smape_by_model, alpha=alpha, min_weight=min_weight, max_weight=max_weight)

    # Hard reject bad assist
    if hard_reject_bad_assist and len(weights) > 1:
        baseline_model = "rt_da_anchor"
        for model_name, w in list(weights.items()):
            if model_name != baseline_model and w < min_weight:
                # This assist model is too poor — reject entirely
                weights[model_name] = 0.0
                result["reason_codes"].append(f"HARD_REJECT_BAD_ASSIST:{model_name}")

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {m: w / total for m, w in weights.items()}

    # Build weights DataFrame
    all_rows = []
    for model_name, w in weights.items():
        if w <= 0:
            continue
        all_rows.append({
            "task": task,
            "target_day": target_day,
            "period": "all",
            "regime": "all",
            "model_name": model_name,
            "weight": w,
            "learner_method": LEARNER_POLICY_REALTIME,
            "training_days": training_days,
            "training_rows": len(merged),
            "lookback_start": result["lookback_start"],
            "lookback_end": result["lookback_end"],
            "reason_codes": "",
        })

    if not all_rows:
        result["reason_codes"].append("NO_WEIGHTS_AFTER_FILTER")
        return result

    weights_df = pd.DataFrame(all_rows)
    result["weights_df"] = weights_df
    result["status"] = REALTIME_LEARNER_POOLED_TRAINED
    result["reason_codes"].append("POOLED_30D_BGEW_TRAINED")
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
