"""
fusion/trust_gated_regime_bgew.py — Trust-Gated Adaptive Regime BGEW (P56).

An innovative but safe fusion engine that combines 2.5's adaptive ledger learner
with 3.0's model trust states.  Uses regime-aware period-based BGEW weighting
with trust gating, safety constraints, and a robust fallback chain.

**Model eligibility (gate check):**
    - ALLOWED: TRUSTED, DELIVERY_ALLOWED, COMPLETE_24H
    - BALANCED PROFILE ALSO ALLOWS: CONSERVATIVE_QUARANTINE
    - BLOCKED: SUSPECT_LEAKAGE, DRY_RUN, STUB, DATA_MISSING, INVALID_24H

**Weight dimensions:**
    - Period-based (3): 1_8, 9_16, 17_24
    - Regime-based (4): normal, low_price, negative_risk, high_spike

**Fallback chain:**
    1. Full regime+period BGEW (>= min_training_days_for_regime)
    2. Period-only BGEW (>= min_training_days_for_period)
    3. Equal weight
    4. Return None (caller falls back to cfg05)

Usage:
    from fusion.trust_gated_regime_bgew import run_trust_gated_regime_bgew

    result = run_trust_gated_regime_bgew(
        target_date="2026-07-05",
        trusted_models=["cfg05", "catboost_spike_residual"],
        prediction_ledger_path="ledgers/prediction_ledger.csv",
        actual_ledger_path="ledgers/actual_ledger.csv",
    )
    if result["success"]:
        output_df = result["output"]
        print(result["regime"], result["weights"])
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from data.business_day import add_business_time_columns, infer_period

logger = logging.getLogger(__name__)

# ── Trust state constants ────────────────────────────────────────────

TRUST_STATE_TRUSTED: str = "TRUSTED"
TRUST_STATE_DELIVERY_ALLOWED: str = "DELIVERY_ALLOWED"
TRUST_STATE_COMPLETE_24H: str = "COMPLETE_24H"
TRUST_STATE_SUSPECT_LEAKAGE: str = "SUSPECT_LEAKAGE"
TRUST_STATE_CONSERVATIVE_QUARANTINE: str = "CONSERVATIVE_QUARANTINE"
TRUST_STATE_DRY_RUN: str = "DRY_RUN"
TRUST_STATE_STUB: str = "STUB"
TRUST_STATE_DATA_MISSING: str = "DATA_MISSING"
TRUST_STATE_INVALID_24H: str = "INVALID_24H"

ALLOWED_TRUST_STATES: set[str] = {
    TRUST_STATE_TRUSTED,
    TRUST_STATE_DELIVERY_ALLOWED,
    TRUST_STATE_COMPLETE_24H,
}

ALLOWED_TRUST_STATES_BALANCED: set[str] = ALLOWED_TRUST_STATES | {
    TRUST_STATE_CONSERVATIVE_QUARANTINE,
}

# ── Regime constants ─────────────────────────────────────────────────

REGIME_NORMAL: str = "normal"
REGIME_LOW_PRICE: str = "low_price"
REGIME_NEGATIVE_RISK: str = "negative_risk"
REGIME_HIGH_SPIKE: str = "high_spike"

ALL_REGIMES: list[str] = [
    REGIME_NORMAL,
    REGIME_LOW_PRICE,
    REGIME_NEGATIVE_RISK,
    REGIME_HIGH_SPIKE,
]

# ── Default thresholds ───────────────────────────────────────────────

DEFAULT_MIN_WEIGHT: float = 0.05
DEFAULT_MAX_WEIGHT: float = 0.75
DEFAULT_CFG05_FLOOR: float = 0.30
CFG05_MODEL_NAME: str = "cfg05"

VALID_PERIODS: list[str] = ["1_8", "9_16", "17_24"]


# ── Trust Gate ───────────────────────────────────────────────────────


def _apply_trust_gate(
    model_names: list[str],
    trusted_models: list[str],
    profile_name: str = "trusted_delivery",
    model_trust_states: Optional[dict[str, str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Apply the trust gate to filter eligible models.

    Parameters
    ----------
    model_names : list[str]
        All candidate model names.
    trusted_models : list[str]
        Models that have TRUSTED status (always pass the gate).
    profile_name : str
        Delivery profile.  ``"balanced_candidate"`` also allows
        CONSERVATIVE_QUARANTINE.
    model_trust_states : dict, optional
        Mapping of model_name -> trust_state for models not in
        *trusted_models*.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        (allowed_models, blocked_models, gate_warnings).
    """
    allowed: list[str] = []
    blocked: list[str] = []
    warnings: list[str] = []

    allowed_states = (
        ALLOWED_TRUST_STATES_BALANCED
        if profile_name == "balanced_candidate"
        else ALLOWED_TRUST_STATES
    )

    model_trust_states = model_trust_states or {}

    for model in model_names:
        if model in trusted_models:
            allowed.append(model)
            continue

        state = model_trust_states.get(model, TRUST_STATE_DATA_MISSING)

        if state in allowed_states:
            allowed.append(model)
            if state == TRUST_STATE_CONSERVATIVE_QUARANTINE:
                warnings.append(
                    f"Model '{model}' passed gate with "
                    f"CONSERVATIVE_QUARANTINE (profile={profile_name})"
                )
        else:
            blocked.append(model)
            warnings.append(
                f"Model '{model}' blocked by trust gate: "
                f"state={state}, profile={profile_name}"
            )

    return allowed, blocked, warnings


# ── Regime Classification ────────────────────────────────────────────


def classify_regime(
    ensemble_median: float,
    recent_p90: float,
    historical_same_hour_median: float,
) -> str:
    """Classify a day into a market regime.

    Parameters
    ----------
    ensemble_median : float
        Median of all trusted model predictions for the target day.
    recent_p90 : float
        90th percentile of daily ensemble medians over the training window.
    historical_same_hour_median : float
        Median of historical actuals across the training window.

    Returns
    -------
    str
        Regime label: ``"normal"``, ``"low_price"``, ``"negative_risk"``,
        or ``"high_spike"``.

    Notes
    -----
    Classification order (first match wins):
        1. negative_risk: historical median < 0 OR ensemble median < 0
        2. low_price: ensemble median < 100 CNY
        3. high_spike: ensemble median > recent_p90
        4. normal: everything else
    """
    if historical_same_hour_median < 0 or ensemble_median < 0:
        return REGIME_NEGATIVE_RISK
    if ensemble_median < 100:
        return REGIME_LOW_PRICE
    if ensemble_median > recent_p90:
        return REGIME_HIGH_SPIKE
    return REGIME_NORMAL


# ── Metric Computation Helpers ───────────────────────────────────────


def _compute_smape(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute sMAPE (Symmetric Mean Absolute Percentage Error).

    sMAPE = 100 * mean(2 * |y_pred - y_true| / (|y_pred| + |y_true| + epsilon))

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    y_true : np.ndarray
        Actual values.

    Returns
    -------
    float
        sMAPE value in percent.
    """
    epsilon = 1e-10
    numerator = 2.0 * np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    ape = numerator / denominator
    return float(np.mean(ape) * 100.0)


def _compute_period_smape(
    model_name: str,
    training_data: pd.DataFrame,
    period: str,
) -> float:
    """Compute sMAPE within a specific period for a model.

    Parameters
    ----------
    model_name : str
        Model name.
    training_data : pd.DataFrame
        Merged training data with columns ``model_name``, ``period``,
        ``y_pred``, ``y_true``.
    period : str
        Period label (e.g. ``"1_8"``).

    Returns
    -------
    float
        Period sMAPE value.  Returns 100.0 if insufficient data.
    """
    mask = (
        (training_data["model_name"] == model_name)
        & (training_data["period"] == period)
    )
    subset = training_data[mask]
    if len(subset) < 2:
        return 100.0

    y_pred = subset["y_pred"].values.astype(float)
    y_true = subset["y_true"].values.astype(float)
    return _compute_smape(y_pred, y_true)


def _compute_regime_smape(
    model_name: str,
    training_data: pd.DataFrame,
    period: str,
    regime: str,
) -> float:
    """Compute sMAPE within a specific regime and period for a model.

    Parameters
    ----------
    model_name : str
        Model name.
    training_data : pd.DataFrame
        Training data with ``regime`` column.
    period : str
        Period label.
    regime : str
        Regime label.

    Returns
    -------
    float
        Regime sMAPE value.  Returns 100.0 if insufficient data.
    """
    mask = (
        (training_data["model_name"] == model_name)
        & (training_data["period"] == period)
        & (training_data["regime"] == regime)
    )
    subset = training_data[mask]
    if len(subset) < 2:
        return 100.0

    y_pred = subset["y_pred"].values.astype(float)
    y_true = subset["y_true"].values.astype(float)
    return _compute_smape(y_pred, y_true)


def _compute_period_rmse_volatility(
    model_name: str,
    training_data: pd.DataFrame,
    period: str,
) -> float:
    """Compute the RMSE coefficient of variation within a period.

    Groups by ``business_day``, computes daily RMSE, then returns the
    coefficient of variation (std / mean) of those daily RMSE values.

    Parameters
    ----------
    model_name : str
        Model name.
    training_data : pd.DataFrame
        Training data.
    period : str
        Period label.

    Returns
    -------
    float
        RMSE volatility (coefficient of variation).  Returns 10.0 if
        insufficient data.
    """
    mask = (
        (training_data["model_name"] == model_name)
        & (training_data["period"] == period)
    )
    subset = training_data[mask]
    if len(subset) < 4:
        return 10.0

    daily_rmse: list[float] = []
    for _day, day_group in subset.groupby("business_day"):
        y_pred = day_group["y_pred"].values.astype(float)
        y_true = day_group["y_true"].values.astype(float)
        if len(y_pred) < 2:
            continue
        mse = float(np.mean((y_pred - y_true) ** 2))
        daily_rmse.append(float(np.sqrt(mse)))

    if len(daily_rmse) < 2:
        return 10.0

    arr = np.array(daily_rmse)
    mean_rmse = float(np.mean(arr))
    std_rmse = float(np.std(arr, ddof=1))

    if mean_rmse < 1e-10:
        return 0.0

    return std_rmse / mean_rmse


# ── Score Computation ────────────────────────────────────────────────


def _compute_regime_scores(
    models: list[str],
    training_data: pd.DataFrame,
    regime: str,
    period: str,
    alpha: float = 5.0,
    min_regime_hours: int = 20,
) -> dict[str, float]:
    """Compute per-model regime scores for a specific regime and period.

    If there are insufficient regime training hours, returns a neutral
    score (1.0) for every model — the product is then governed by base
    and stability scores alone.

    Parameters
    ----------
    models : list[str]
        Model names.
    training_data : pd.DataFrame
        Training data with ``regime`` column.
    regime : str
        Regime label.
    period : str
        Period label.
    alpha : float
        Exponential scaling factor (default 5.0).
    min_regime_hours : int
        Minimum training hours for regime-specific scoring.

    Returns
    -------
    dict[str, float]
        {model_name: regime_score}.
    """
    regime_mask = (
        (training_data["period"] == period)
        & (training_data["regime"] == regime)
    )
    n_regime_hours = int(regime_mask.sum())

    scores: dict[str, float] = {}

    if n_regime_hours >= min_regime_hours:
        for model in models:
            smape = _compute_regime_smape(model, training_data, period, regime)
            scores[model] = float(np.exp(-alpha * smape / 100.0))
    else:
        # Neutral score — does not affect the product
        for model in models:
            scores[model] = 1.0

    return scores


def _compute_period_scores(
    models: list[str],
    training_data: pd.DataFrame,
    period: str,
    alpha: float = 5.0,
) -> dict[str, float]:
    """Compute per-model period base scores.

    base_score = exp(-alpha * period_smape / 100)

    Parameters
    ----------
    models : list[str]
        Model names.
    training_data : pd.DataFrame
        Training data.
    period : str
        Period label.
    alpha : float
        Exponential scaling factor.

    Returns
    -------
    dict[str, float]
        {model_name: base_score}.
    """
    scores: dict[str, float] = {}
    for model in models:
        smape = _compute_period_smape(model, training_data, period)
        scores[model] = float(np.exp(-alpha * smape / 100.0))
    return scores


# ── Weight Normalization ─────────────────────────────────────────────


def _normalize_weights(
    weights: dict[str, float],
    min_w: float = 0.05,
    max_w: float = 0.75,
    cfg05_floor: float = 0.30,
    profile_name: str = "trusted_delivery",
) -> dict[str, float]:
    """Normalize weights with min/max constraints and cfg05 floor.

    Workflow:
        1. Normalize raw scores to sum 1.
        2. Clip to [min_w, max_w].
        3. Apply cfg05_floor for ``trusted_delivery`` profile.
        4. Renormalize to sum 1.
        5. Re-clip (renormalization may push some above max_w).
        6. Final renormalize.

    Parameters
    ----------
    weights : dict[str, float]
        Raw model scores (must be non-negative).
    min_w : float
        Minimum permitted weight (default 0.05).
    max_w : float
        Maximum permitted weight (default 0.75).
    cfg05_floor : float
        Minimum weight for cfg05 (default 0.30).
    profile_name : str
        Profile name.  cfg05_floor only applies to
        ``"trusted_delivery"``.

    Returns
    -------
    dict[str, float]
        Constrained, normalized weights.
    """
    if not weights:
        return {}

    model_names = list(weights.keys())
    raw = np.array([weights[m] for m in model_names], dtype=float)

    total = float(raw.sum())
    if total <= 0:
        eq = 1.0 / len(model_names)
        return {m: eq for m in model_names}

    # Redistribution algorithm: clamp weights to [min_w, max_w], apply
    # cfg05_floor, and redistribute surplus/deficit among unclamped
    # models proportionally.  This avoids the oscillation that a simple
    # clip-then-renormalise loop exhibits when the surplus is large.
    w = raw / total
    n = len(model_names)
    cfg05_idx = (
        model_names.index(CFG05_MODEL_NAME)
        if profile_name == "trusted_delivery" and CFG05_MODEL_NAME in model_names
        else None
    )

    for _ in range(50):
        prev = w.copy()

        # ── cfg05 floor ──────────────────────────────────────────────
        if cfg05_idx is not None and w[cfg05_idx] < cfg05_floor:
            deficit = cfg05_floor - w[cfg05_idx]
            w[cfg05_idx] = cfg05_floor
            # Take deficit proportionally from every other model
            others = np.ones(n, dtype=bool)
            others[cfg05_idx] = False
            other_sum = float(w[others].sum())
            if other_sum > 1e-15:
                scale = (other_sum - deficit) / other_sum
                w[others] *= max(scale, 0.0)

        # ── max_w clamp (redistribute excess to unclamped) ────────────
        max_mask = w > max_w
        if np.any(max_mask):
            excess = float((w[max_mask] - max_w).sum())
            w[max_mask] = max_w
            free = ~max_mask
            free_sum = float(w[free].sum())
            if free_sum > 1e-15:
                w[free] += excess * w[free] / free_sum

        # ── min_w clamp (draw deficit from unclamped) ─────────────────
        min_mask = w < min_w
        if np.any(min_mask):
            deficit = float((min_w - w[min_mask]).sum())
            w[min_mask] = min_w
            free = ~min_mask
            free_sum = float(w[free].sum())
            if free_sum > 1e-15:
                scale = max((free_sum - deficit) / free_sum, 0.0)
                w[free] *= scale

        # ── Normalise (final safety pass) ─────────────────────────────
        s = float(w.sum())
        if s > 0:
            w = w / s

        # ── Convergence check ────────────────────────────────────────
        if float(np.max(np.abs(w - prev))) < 1e-15:
            break

    return {m: float(w[i]) for i, m in enumerate(model_names)}


# ── Output Builder ───────────────────────────────────────────────────


def _build_24h_output(
    target_date: str,
    fused_prices: list[float],
    method: str,
) -> pd.DataFrame:
    """Build a standard 24-row output DataFrame.

    Columns: ``business_day``, ``ds``, ``hour_business``, ``period``,
    ``dayahead_price``, ``realtime_price``.

    Parameters
    ----------
    target_date : str
        Target date string (``YYYY-MM-DD``).
    fused_prices : list[float]
        24 fused price values (hour 1 through 24).
    method : str
        Fusion method description (e.g. ``"regime_bgew"``).

    Returns
    -------
    pd.DataFrame
        24-row DataFrame.
    """
    n_hours = min(len(fused_prices), 24)

    if n_hours == 0:
        return pd.DataFrame(columns=[
            "business_day", "ds", "hour_business", "period",
            "dayahead_price", "realtime_price",
        ])

    df = pd.DataFrame({
        "ds": pd.date_range(
            f"{target_date} 01:00",
            periods=n_hours,
            freq="h",
        ),
    })
    df = add_business_time_columns(df, timestamp_col="ds")

    df["dayahead_price"] = fused_prices[:n_hours]
    df["realtime_price"] = [None] * n_hours

    return df[[
        "business_day", "ds", "hour_business", "period",
        "dayahead_price", "realtime_price",
    ]].reset_index(drop=True)


# ── Ledger Loading ───────────────────────────────────────────────────


def _load_prediction_ledger(path: str) -> pd.DataFrame:
    """Load prediction ledger from a CSV file path.

    Parameters
    ----------
    path : str
        Path to the CSV.

    Returns
    -------
    pd.DataFrame
        Prediction ledger.  Empty DataFrame on failure.
    """
    try:
        df = pd.read_csv(path)
        for col in ["business_day", "ds", "target_day"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    except Exception as exc:
        logger.warning("Could not load prediction ledger from %s: %s", path, exc)
        return pd.DataFrame()


def _load_actual_ledger(path: str) -> pd.DataFrame:
    """Load actual ledger from a CSV file path.

    Parameters
    ----------
    path : str
        Path to the CSV.

    Returns
    -------
    pd.DataFrame
        Actual ledger.  Empty DataFrame on failure.
    """
    try:
        df = pd.read_csv(path)
        for col in ["business_day", "ds", "target_day"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    except Exception as exc:
        logger.warning("Could not load actual ledger from %s: %s", path, exc)
        return pd.DataFrame()


# ── Training Data Preparation ────────────────────────────────────────


def _prepare_training_data(
    prediction_ledger: pd.DataFrame,
    actual_ledger: pd.DataFrame,
    target_date: str,
    models: list[str],
    window: int = 30,
) -> pd.DataFrame:
    """Prepare merged training data with no future leakage.

    Merges predictions with actuals on ``(target_day, hour_business)``
    for days where ``business_day < target_date`` (strictly historical).

    Parameters
    ----------
    prediction_ledger : pd.DataFrame
        Prediction ledger with at minimum columns:
        ``model_name``, ``business_day``, ``target_day``,
        ``hour_business``, ``y_pred``.
    actual_ledger : pd.DataFrame
        Actual ledger with at minimum columns:
        ``target_day``, ``hour_business``, ``y_true``.
    target_date : str
        Target date (``YYYY-MM-DD``).  Only data with
        ``business_day < target_date`` is used.
    models : list[str]
        Model names to include.
    window : int
        Rolling window in calendar days (default 30).

    Returns
    -------
    pd.DataFrame
        Merged training data with columns:
        ``business_day``, ``target_day``, ``hour_business``, ``period``,
        ``model_name``, ``y_pred``, ``y_true``.
    """
    target = pd.Timestamp(target_date)

    preds = prediction_ledger[
        (prediction_ledger["model_name"].isin(models))
        & (prediction_ledger["business_day"] < target)
    ].copy()

    if len(preds) == 0:
        return pd.DataFrame()

    latest_pred_day = preds["business_day"].max()
    window_start = latest_pred_day - pd.Timedelta(days=window)
    preds = preds[preds["business_day"] >= window_start]

    if len(preds) == 0:
        return pd.DataFrame()

    actuals = actual_ledger[
        (actual_ledger["business_day"] < target)
        & (actual_ledger["business_day"] >= window_start)
    ].copy()

    if len(actuals) == 0:
        return pd.DataFrame()

    merged = preds.merge(
        actuals[["target_day", "hour_business", "y_true"]],
        on=["target_day", "hour_business"],
        how="inner",
    )

    if len(merged) == 0:
        return pd.DataFrame()

    if "period" not in merged.columns:
        merged["period"] = merged["hour_business"].apply(infer_period)

    return merged.reset_index(drop=True)


def _add_regime_labels(
    training_data: pd.DataFrame,
    recent_p90: float,
) -> pd.DataFrame:
    """Add regime labels to each row of training data.

    For each ``business_day``, classifies the regime using the ensemble
    prediction median and the actual median, then labels every row from
    that day.

    Parameters
    ----------
    training_data : pd.DataFrame
        Merged training data.
    recent_p90 : float
        90th percentile of daily ensemble medians.

    Returns
    -------
    pd.DataFrame
        Training data with an added ``regime`` column.
    """
    df = training_data.copy()
    df["regime"] = REGIME_NORMAL

    for day, day_group in df.groupby("business_day"):
        day_ensemble_median = float(day_group["y_pred"].median())
        day_actual_median = float(day_group["y_true"].median())
        regime = classify_regime(
            day_ensemble_median,
            recent_p90,
            day_actual_median,
        )
        df.loc[day_group.index, "regime"] = regime

    return df


def _compute_recent_p90(training_data: pd.DataFrame) -> float:
    """Compute the 90th percentile of daily ensemble prediction medians.

    Parameters
    ----------
    training_data : pd.DataFrame
        Training data with columns ``business_day`` and ``y_pred``.

    Returns
    -------
    float
        90th percentile of daily ensemble medians.  Returns 0.0 if
        insufficient data.
    """
    daily_medians = (
        training_data.groupby("business_day")["y_pred"]
        .median()
        .values
    )
    if len(daily_medians) == 0:
        return 0.0
    return float(np.percentile(daily_medians, 90))


# ── Fused Price Builder ──────────────────────────────────────────────


def _build_fused_prices(
    target_preds: pd.DataFrame,
    allowed_models: list[str],
    period_weights: dict[str, dict[str, float]],
) -> list[float]:
    """Build fused prices for a 24-hour day given per-period weights.

    Parameters
    ----------
    target_preds : pd.DataFrame
        Predictions for the target day (columns ``hour_business``,
        ``model_name``, ``y_pred``).
    allowed_models : list[str]
        Models that passed the trust gate.
    period_weights : dict[str, dict[str, float]]
        Nested dict: period -> model_name -> weight.

    Returns
    -------
    list[float]
        24 fused price values.
    """
    n_hours = 24
    fused: list[float] = []

    for hour in range(1, n_hours + 1):
        period = infer_period(hour)
        p_weights = period_weights.get(period, {})

        hour_preds = target_preds[target_preds["hour_business"] == hour]
        if len(hour_preds) == 0:
            fused.append(0.0)
            continue

        hour_map = hour_preds.set_index("model_name")["y_pred"].to_dict()
        total = 0.0
        weight_sum = 0.0
        for model in allowed_models:
            w = p_weights.get(model, 0.0)
            p = hour_map.get(model)
            if p is not None:
                total += w * float(p)
                weight_sum += w

        fused.append(total / weight_sum if weight_sum > 0 else 0.0)

    return fused


# ── Main Fusion Engine ───────────────────────────────────────────────


def run_trust_gated_regime_bgew(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger_path: str,
    actual_ledger_path: str,
    profile_name: str = "trusted_delivery",
    alpha: float = 5.0,
    beta: float = 3.0,
    min_training_days_for_regime: int = 10,
    min_training_days_for_period: int = 5,
    model_trust_states: Optional[dict[str, str]] = None,
    prediction_ledger: Optional[pd.DataFrame] = None,
    actual_ledger: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Run trust-gated regime-aware BGEW fusion.

    This is the main entry-point for P56.  It orchestrates trust gating,
    regime classification, BGEW weight computation, and fallback handling.

    Parameters
    ----------
    target_date : str
        Target date (``YYYY-MM-DD``).
    trusted_models : list[str]
        Model names that have TRUSTED status (from P41 trust gate).
    prediction_ledger_path : str
        Path to prediction ledger CSV.
    actual_ledger_path : str
        Path to actual ledger CSV.
    profile_name : str
        Delivery profile (default ``"trusted_delivery"``).
        ``"balanced_candidate"`` allows CONSERVATIVE_QUARANTINE models.
    alpha : float
        Exponential scaling for sMAPE (default 5.0).
    beta : float
        Exponential scaling for RMSE volatility (default 3.0).
    min_training_days_for_regime : int
        Minimum training days required for regime-level BGEW (default 10).
    min_training_days_for_period : int
        Minimum training days required for period-level BGEW (default 5).
    model_trust_states : dict, optional
        Mapping of model_name -> trust_state for models not in
        *trusted_models*.
    prediction_ledger : pd.DataFrame, optional
        Pre-loaded prediction ledger (bypasses file load).
    actual_ledger : pd.DataFrame, optional
        Pre-loaded actual ledger (bypasses file load).

    Returns
    -------
    dict
        Result dictionary with keys:

        - **success** (*bool*) -- Whether fusion produced output.
        - **delivery_status** (*str*) -- Status code.
        - **method** (*str*) -- ``"regime_bgew"`` / ``"period_bgew"`` /
          ``"equal_weight"`` / ``"failed"``.
        - **output** (*pd.DataFrame or None*) -- 24-row output or None.
        - **weights** (*dict*) -- Per-model weights (period 1_8).
        - **regime** (*str*) -- Detected regime for the target day.
        - **regime_details** (*dict*) -- Regime classification details.
        - **training_days_used** (*int*) -- Number of training days.
        - **warnings** (*list*) -- Warning messages.
        - **errors** (*list*) -- Error messages.
        - **fallback_chain** (*list*) -- Sequence of fallback attempts.
    """
    warnings: list[str] = []
    errors: list[str] = []
    fallback_chain: list[dict[str, Any]] = []

    result: dict[str, Any] = {
        "success": False,
        "delivery_status": "FAILED",
        "method": "failed",
        "output": None,
        "weights": {},
        "regime": REGIME_NORMAL,
        "regime_details": {},
        "training_days_used": 0,
        "warnings": warnings,
        "errors": errors,
        "fallback_chain": fallback_chain,
        "fused_prices": [],
    }

    # ── Step 1: Load Ledgers ─────────────────────────────────────────────

    if prediction_ledger is not None:
        pl = prediction_ledger
    else:
        pl = _load_prediction_ledger(prediction_ledger_path)

    if actual_ledger is not None:
        al = actual_ledger
    else:
        al = _load_actual_ledger(actual_ledger_path)

    if len(pl) == 0:
        errors.append("Prediction ledger is empty or could not be loaded")
        result["delivery_status"] = "LEDGER_LOAD_FAILED"
        result["errors"] = errors
        result["fallback_chain"] = fallback_chain
        return result

    if len(al) == 0:
        errors.append("Actual ledger is empty or could not be loaded")
        result["delivery_status"] = "LEDGER_LOAD_FAILED"
        result["errors"] = errors
        result["fallback_chain"] = fallback_chain
        return result

    # ── Step 2: Discover models & Trust Gate ─────────────────────────────

    # Discover all models present in the ledger for the target date,
    # then gate them together with the explicitly trusted models.
    target_dt = pd.Timestamp(target_date)
    ledger_models = (
        pl[pl["target_day"] == target_dt]["model_name"].unique().tolist()
        if "target_day" in pl.columns
        else []
    )
    # Ensure trusted_models are always included even if not in ledger
    all_candidates = list(dict.fromkeys(trusted_models + ledger_models))

    allowed_models, blocked_models, gate_warnings = _apply_trust_gate(
        all_candidates,
        trusted_models,
        profile_name=profile_name,
        model_trust_states=model_trust_states,
    )
    warnings.extend(gate_warnings)

    if not allowed_models:
        errors.append("No models passed the trust gate")
        result["delivery_status"] = "TRUST_GATE_BLOCKED"
        result["errors"] = errors
        result["fallback_chain"] = fallback_chain
        return result

    # ── Step 3: Prepare Training Data ────────────────────────────────────

    training_data = _prepare_training_data(
        pl, al, target_date, allowed_models, window=30,
    )

    if len(training_data) == 0:
        errors.append("No training data available after merge")
        result["delivery_status"] = "NO_TRAINING_DATA"
        result["errors"] = errors
        result["fallback_chain"] = fallback_chain
        return result

    n_training_days = len(training_data["business_day"].unique())
    result["training_days_used"] = n_training_days

    # ── Step 4: Detect Regime for Target Day ────────────────────────────

    target_preds = pl[
        (pl["target_day"] == pd.Timestamp(target_date))
        & (pl["model_name"].isin(allowed_models))
    ].copy()

    if len(target_preds) == 0:
        errors.append(
            f"No predictions found for target date {target_date} "
            f"among allowed models: {allowed_models}"
        )
        result["delivery_status"] = "NO_TARGET_PREDICTIONS"
        result["errors"] = errors
        result["fallback_chain"] = fallback_chain
        return result

    recent_p90 = _compute_recent_p90(training_data)
    historical_actual_median = float(training_data["y_true"].median())

    ensemble_median = float(target_preds["y_pred"].median())

    regime = classify_regime(
        ensemble_median,
        recent_p90,
        historical_actual_median,
    )
    result["regime"] = regime
    result["regime_details"] = {
        "ensemble_median": ensemble_median,
        "recent_p90": recent_p90,
        "historical_actual_median": historical_actual_median,
        "classifier_input": {
            "ensemble_median": ensemble_median,
            "recent_p90": recent_p90,
            "historical_same_hour_median": historical_actual_median,
        },
    }

    training_data = _add_regime_labels(training_data, recent_p90)

    regime_training_days = len(
        training_data[
            training_data["regime"] == regime
        ]["business_day"].unique()
    )

    # ── Step 5: Compute Weights (with fallback chain) ───────────────────

    method: str = "failed"
    period_weights: dict[str, dict[str, float]] = {}

    # --- Level 1: Full regime + period BGEW ---
    if n_training_days >= min_training_days_for_regime:
        fb: dict[str, Any] = {
            "level": 1,
            "method": "regime_bgew",
            "success": True,
            "reason": (
                f"Training days={n_training_days} >= "
                f"{min_training_days_for_regime}"
            ),
        }
        try:
            for period in VALID_PERIODS:
                regime_scores = _compute_regime_scores(
                    allowed_models,
                    training_data,
                    regime,
                    period,
                    alpha=alpha,
                )
                base_scores = _compute_period_scores(
                    allowed_models,
                    training_data,
                    period,
                    alpha=alpha,
                )
                stability_scores: dict[str, float] = {}
                for model in allowed_models:
                    vol = _compute_period_rmse_volatility(
                        model, training_data, period,
                    )
                    stability_scores[model] = float(np.exp(-beta * vol))

                combined: dict[str, float] = {}
                for model in allowed_models:
                    base = base_scores.get(model, 0.01)
                    regime_s = regime_scores.get(model, 0.01)
                    stability = stability_scores.get(model, 0.01)
                    combined[model] = base * stability * regime_s

                period_weights[period] = _normalize_weights(
                    combined, profile_name=profile_name,
                )

            fb["regime_training_days"] = int(regime_training_days)
            fb["periods_weights"] = {
                p: str(period_weights.get(p, {}))
                for p in VALID_PERIODS
            }
            method = "regime_bgew"
            warnings.append(
                f"Regime BGEW succeeded: regime={regime}, "
                f"training_days={n_training_days}, "
                f"regime_training_days={regime_training_days}"
            )
        except Exception as exc:
            fb["success"] = False
            fb["reason"] = f"Exception: {exc}"
            period_weights = {}

        fallback_chain.append(fb)

    # --- Level 2: Period-only BGEW ---
    if method == "failed" and n_training_days >= min_training_days_for_period:
        fb = {
            "level": 2,
            "method": "period_bgew",
            "success": True,
            "reason": (
                f"Training days={n_training_days} >= "
                f"{min_training_days_for_period}, "
                f"regime BGEW skipped/failed"
            ),
        }
        try:
            for period in VALID_PERIODS:
                base_scores = _compute_period_scores(
                    allowed_models,
                    training_data,
                    period,
                    alpha=alpha,
                )
                stability_scores = {}
                for model in allowed_models:
                    vol = _compute_period_rmse_volatility(
                        model, training_data, period,
                    )
                    stability_scores[model] = float(np.exp(-beta * vol))

                combined = {}
                for model in allowed_models:
                    base = base_scores.get(model, 0.01)
                    stability = stability_scores.get(model, 0.01)
                    combined[model] = base * stability

                period_weights[period] = _normalize_weights(
                    combined, profile_name=profile_name,
                )

            fb["periods_weights"] = {
                p: str(period_weights.get(p, {}))
                for p in VALID_PERIODS
            }
            method = "period_bgew"

        except Exception as exc:
            fb["success"] = False
            fb["reason"] = f"Exception: {exc}"
            period_weights = {}

        fallback_chain.append(fb)

    # --- Level 3: Equal weight ---
    if method == "failed":
        fb = {
            "level": 3,
            "method": "equal_weight",
            "success": True,
            "reason": "All BGEW methods failed; falling back to equal weight",
        }
        try:
            n = len(allowed_models)
            eq_w = 1.0 / n if n > 0 else 0.0
            for period in VALID_PERIODS:
                period_weights[period] = {m: eq_w for m in allowed_models}
            method = "equal_weight"

        except Exception as exc:
            fb["success"] = False
            fb["reason"] = f"Exception: {exc}"
            period_weights = {}

        fallback_chain.append(fb)

    # --- Level 4: Complete failure ---
    if method == "failed":
        fallback_chain.append({
            "level": 4,
            "method": "failed",
            "success": False,
            "reason": "All fusion methods exhausted",
        })
        errors.append("All fusion methods failed")
        result["delivery_status"] = "ALL_FUSION_METHODS_FAILED"
        result["method"] = "failed"
        result["fallback_chain"] = fallback_chain
        return result

    # ── Step 6: Build Fused Prices ──────────────────────────────────────

    fused_prices = _build_fused_prices(
        target_preds, allowed_models, period_weights,
    )

    # ── Step 7: Build Output DataFrame ──────────────────────────────────

    output = _build_24h_output(target_date, fused_prices, method)

    canonical_weights = period_weights.get("1_8", {})

    result["success"] = True
    result["delivery_status"] = (
        "DELIVERY_READY"
        if method in ("regime_bgew", "period_bgew")
        else "FALLBACK"
    )
    result["method"] = method
    result["output"] = output
    result["weights"] = canonical_weights
    result["fused_prices"] = fused_prices
    result["fallback_chain"] = fallback_chain

    return result
