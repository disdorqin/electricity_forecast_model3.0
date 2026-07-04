"""
fusion/weights.py — Weight computation strategies for the P4 fusion engine.

Strategies:
    equal_weight       — Each included model gets 1/N.
    prior_weight       — User-supplied prior dict, normalised to sum 1.
                        Missing models get a small fallback weight.
    bgew_skeleton      — Rolling window inverse-error weighting using past
                        actuals.  Falls back to equal_weight when actuals
                        are not available.

All compute_* functions return (weights_dict, list_of_reason_codes).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FALLBACK_WEIGHT: float = 1e-6
"""Small weight assigned to models absent from a prior-weights dict."""


def equal_weight(model_names: list[str]) -> tuple[dict[str, float], list[str]]:
    """Equal weight for every model.

    Returns
    -------
    tuple[dict[str, float], list[str]]
        ({model: weight, ...}, reason_codes).
    """
    if not model_names:
        return {}, ["EQUAL_WEIGHT_NO_MODELS"]

    w = 1.0 / len(model_names)
    weights = {m: w for m in model_names}
    return weights, ["EQUAL_WEIGHT"]


def prior_weight(
    model_names: list[str],
    prior: Optional[dict[str, float]] = None,
) -> tuple[dict[str, float], list[str]]:
    """Weight by a user-supplied prior dict, normalised to sum 1.

    Models present in *model_names* but missing from *prior* receive a
    small *FALLBACK_WEIGHT*.  Models in *prior* not in *model_names* are
    silently ignored.

    Returns
    -------
    tuple[dict[str, float], list[str]]
        ({model: weight, ...}, reason_codes).
    """
    reasons: list[str] = []

    if not model_names:
        return {}, ["PRIOR_WEIGHT_NO_MODELS"]

    if not prior:
        reasons.append("PRIOR_NOT_PROVIDED_FALLBACK_EQUAL")
        ew, ew_reasons = equal_weight(model_names)
        return ew, reasons + ew_reasons

    # Build raw weight dict — prior values for known models, fallback for others
    raw: dict[str, float] = {}
    for m in model_names:
        if m in prior:
            raw[m] = prior[m]
        else:
            raw[m] = FALLBACK_WEIGHT
            reasons.append(f"PRIOR_MISSING_MODEL_{m}_FALLBACK")

    # Normalise to sum 1
    total = sum(raw.values())
    if total <= 0:
        reasons.append("PRIOR_ZERO_TOTAL_FALLBACK_EQUAL")
        ew, ew_reasons = equal_weight(model_names)
        return ew, reasons + ew_reasons

    weights = {m: v / total for m, v in raw.items()}

    # Log / report
    prior_keys = set(prior.keys())
    group_keys = set(model_names)
    ignored = prior_keys - group_keys
    if ignored:
        reasons.append(f"PRIOR_IGNORED_MODELS_{'_'.join(sorted(ignored))}")

    reasons.append("PRIOR_WEIGHT")
    return weights, reasons


def bgew_skeleton(
    model_names: list[str],
    corrected_df: pd.DataFrame,
    actuals_df: Optional[pd.DataFrame] = None,
    window: int = 30,
    min_history: int = 7,
) -> tuple[dict[str, float], list[str]]:
    """BGEW skeleton: rolling inverse-error weighting from past actuals.

    For each model, compute the inverse mean-absolute-error over the most
    recent *window* business days where both predictions and actuals exist.
    Weights are then normalised to sum 1.

    **Future-awareness**: only ``business_day < target_day`` rows from
    ``corrected_df`` are used — no future leakage.

    If ``actuals_df`` is None or has insufficient history, falls back to
    equal_weight with an appropriate reason code.

    Parameters
    ----------
    model_names : list[str]
        Models to compute weights for.
    corrected_df : pd.DataFrame
        Corrected prediction output with at minimum
        ``model_name, business_day, hour_business, y_pred_corrected``.
    actuals_df : pd.DataFrame, optional
        Actuals DataFrame with at minimum
        ``business_day, hour_business, y_true``.
    window : int
        Rolling window size in business days (default 30).
    min_history : int
        Minimum business days of history required (default 7).

    Returns
    -------
    tuple[dict[str, float], list[str]]
        ({model: weight, ...}, reason_codes).
    """
    reasons: list[str] = []

    if not model_names:
        return {}, ["BGEW_NO_MODELS"]

    if actuals_df is None or len(actuals_df) == 0:
        reasons.append("ACTUAL_LEDGER_MISSING_EQUAL_WEIGHT")
        ew, ew_reasons = equal_weight(model_names)
        return ew, reasons + ew_reasons

    # Ensure datetime
    actuals = actuals_df.copy()
    if "business_day" in actuals.columns:
        actuals["business_day"] = pd.to_datetime(actuals["business_day"])

    corrected = corrected_df.copy()
    if "business_day" in corrected.columns:
        corrected["business_day"] = pd.to_datetime(corrected["business_day"])

    # Determine the set of business days in corrected output
    corrected_days = sorted(corrected["business_day"].unique())
    if len(corrected_days) == 0:
        reasons.append("BGEW_NO_CORRECTED_DAYS_FALLBACK_EQUAL")
        ew, ew_reasons = equal_weight(model_names)
        return ew, reasons + ew_reasons

    # For future-awareness: use the earliest target_day in corrected data
    # as the cut — only train on business_day < cut
    earliest_target = corrected_days[0]

    # Filter actuals to pre-cut rows
    train_actuals = actuals[actuals["business_day"] < earliest_target].copy()
    if len(train_actuals) < min_history:
        reasons.append(
            f"BGEW_INSUFFICIENT_HISTORY_{len(train_actuals)}_FALLBACK_EQUAL"
        )
        ew, ew_reasons = equal_weight(model_names)
        return ew, reasons + ew_reasons

    # Limit to rolling window
    latest_train_day = train_actuals["business_day"].max()
    window_start = latest_train_day - pd.Timedelta(days=window)
    train_actuals = train_actuals[train_actuals["business_day"] >= window_start]

    # Merge each model's predictions with actuals
    model_errors: dict[str, list[float]] = {m: [] for m in model_names}

    for model in model_names:
        model_preds = corrected[corrected["model_name"] == model][
            ["business_day", "hour_business", "y_pred_corrected"]
        ].copy()
        if len(model_preds) == 0:
            model_errors[model] = [1e6]  # huge error → zero weight
            continue

        merged = model_preds.merge(
            train_actuals[["business_day", "hour_business", "y_true"]],
            on=["business_day", "hour_business"],
            how="inner",
        )
        if len(merged) == 0:
            model_errors[model] = [1e6]
            continue

        abs_errors = np.abs(
            merged["y_pred_corrected"].values - merged["y_true"].values
        )
        model_errors[model] = abs_errors.tolist()

    # Compute inverse-MAE weights
    mae_scores: dict[str, float] = {}
    for model in model_names:
        errs = model_errors[model]
        mae = float(np.mean(errs)) if len(errs) > 0 else 1e6
        # Inverse: low MAE → high weight
        inv = 1.0 / max(mae, 1e-10)
        mae_scores[model] = inv

    # Normalise
    total_inv = sum(mae_scores.values())
    if total_inv <= 0:
        reasons.append("BGEW_ZERO_INVERSE_FALLBACK_EQUAL")
        ew, ew_reasons = equal_weight(model_names)
        return ew, reasons + ew_reasons

    weights = {m: mae_scores[m] / total_inv for m in model_names}
    reasons.append("BGEW_SKELETON")

    # Report effective history
    n_train_days = len(train_actuals["business_day"].unique())
    reasons.append(f"BGEW_TRAIN_DAYS_{n_train_days}")

    return weights, reasons


def compute_weights(
    method: str,
    model_names: list[str],
    *,
    prior: Optional[dict[str, float]] = None,
    actuals_df: Optional[pd.DataFrame] = None,
    corrected_df: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, float], list[str]]:
    """Dispatch to the appropriate weight strategy.

    Parameters
    ----------
    method : str
        One of ``"equal_weight"``, ``"prior_weight"``, ``"bgew_skeleton"``.
    model_names : list[str]
        Models to compute weights for.
    prior : dict, optional
        Prior weights dict (required for ``prior_weight``).
    actuals_df : pd.DataFrame, optional
        Actuals DataFrame (required for ``bgew_skeleton``).
    corrected_df : pd.DataFrame, optional
        Corrected prediction DataFrame (required for ``bgew_skeleton``).

    Returns
    -------
    tuple[dict[str, float], list[str]]
    """
    if method == "equal_weight":
        return equal_weight(model_names)
    elif method == "prior_weight":
        return prior_weight(model_names, prior=prior)
    elif method == "bgew_skeleton":
        return bgew_skeleton(
            model_names,
            corrected_df=corrected_df if corrected_df is not None else pd.DataFrame(),
            actuals_df=actuals_df,
        )
    else:
        raise ValueError(f"Unknown fusion method: {method}")
