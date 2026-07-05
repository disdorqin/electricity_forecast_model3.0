"""
src/registry/dayahead_models.py — Day-ahead model registry.

Central registry of approved day-ahead models, the champion, fusion pool,
and invalid models. This is the single source of truth for day-ahead
model identity in 3.0.

Usage:
    from src.registry.dayahead_models import (
        CHAMPION_MODEL_ID,
        DEFAULT_FUSION_POOL,
        is_valid_model,
        is_invalid_model,
        get_model_config,
    )
"""

from __future__ import annotations

from typing import Any

# ──────────────────────────────────────────────
# Champion
# ──────────────────────────────────────────────

CHAMPION_MODEL_ID: str = "cfg05"
"""The current day-ahead champion model ID."""

CHAMPION_SMAPE_FLOOR50: float = 11.4838
"""Champion sMAPE_floor50 (from Source Review Stage 1)."""

# ──────────────────────────────────────────────
# Default fusion pool (ordered by rank)
# ──────────────────────────────────────────────

DEFAULT_FUSION_POOL: list[dict[str, Any]] = [
    {
        "model_id": "cfg05",
        "formal_name": "lightgbm_cfg05_dayahead",
        "sMAPE_floor50": 11.48,
        "description": "LightGBM cfg05 champion (90d, mae, nl=191, lr=0.015)",
    },
    {
        "model_id": "best_two_average",
        "formal_name": "best_two_average",
        "sMAPE_floor50": 11.85,
        "description": "Trusted champion (leak-free average of trial_02 + trial_24)",
    },
    {
        "model_id": "stage3_business_fixed",
        "formal_name": "stage3_business_fixed",
        "sMAPE_floor50": 11.86,
        "description": "Stage3 with corrected business_day mapping",
    },
    {
        "model_id": "catboost_spike_residual",
        "formal_name": "catboost_spike_residual",
        "sMAPE_floor50": 12.47,
        "description": "CatBoost with spike residual correction",
    },
    {
        "model_id": "catboost_sota",
        "formal_name": "catboost_sota",
        "sMAPE_floor50": 12.58,
        "description": "CatBoost SOTA baseline",
    },
]

# ──────────────────────────────────────────────
# Invalid models (permanently banned)
# ──────────────────────────────────────────────

INVALID_MODELS: dict[str, str] = {
    "lgbm_spike_residual_1127": "target leakage: y_true used as prediction feature",
    "stage3_old_1164": "natural-day business_day mapping error (used ds.date())",
    "lightgbm_90d_orig_1197": "690 rows only — missing hour 24, invalid output shape",
}
"""Mapping of invalid model IDs to their invalidation reason."""

# ──────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "cfg05": {
        "model_id": "cfg05",
        "formal_name": "lightgbm_cfg05_dayahead",
        "model_type": "LightGBM",
        "task": "dayahead",
        "window": "90d",
        "params": {
            "objective": "mae",
            "num_leaves": 191,
            "min_data_in_leaf": 30,
            "learning_rate": 0.015,
            "lambda_l1": 0.1,
            "lambda_l2": 5.0,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.95,
            "bagging_freq": 5,
            "n_estimators": 2000,
        },
        "champion": True,
        "sMAPE_floor50": 11.4838,
        "feature_columns": [
            # Base (24)
            "hour", "month", "day_of_week", "is_weekend",
            "lag_price_target", "lag_price_week",
            "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
            "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
            "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
            "morning_trend", "is_info_fresh",
            # Extended lags (5)
            "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
            # Same-hour stats (5)
            "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
            "same_hour_max_7d", "same_hour_min_7d",
            # Momentum + ranks (3)
            "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
            # Calendar (5)
            "is_spring_festival_window", "days_to_spring_festival",
            "days_after_spring_festival", "is_month_start", "is_month_end",
            # v3: Volatility (2)
            "price_volatility_24h", "price_volatility_168h",
            # v3: Additional ranks (2)
            "renewable_penetration_rank_30d", "load_ramp_rank_30d",
            # v3: Change features (3)
            "bidding_space_change_24h", "net_load_change_24h", "renewable_change_24h",
            # v3: Exact spring festival (3)
            "is_spring_festival_exact", "days_to_spring_festival_exact",
            "days_after_spring_festival_exact",
            # v3: Interaction features (4)
            "hour_x_bidding_space", "hour_x_net_load",
            "period_x_bidding_space", "period_x_renewable_penetration",
        ],
    },
    "best_two_average": {
        "model_id": "best_two_average",
        "formal_name": "best_two_average",
        "model_type": "ensemble_average",
        "task": "dayahead",
        "construction": "simple average of LightGBM trial_02 + trial_24",
        "champion": False,
        "sMAPE_floor50": 11.85,
    },
    "stage3_business_fixed": {
        "model_id": "stage3_business_fixed",
        "formal_name": "stage3_business_fixed",
        "model_type": "LightGBM",
        "task": "dayahead",
        "champion": False,
        "sMAPE_floor50": 11.86,
    },
    "catboost_spike_residual": {
        "model_id": "catboost_spike_residual",
        "formal_name": "catboost_spike_residual",
        "model_type": "CatBoost",
        "task": "dayahead",
        "champion": False,
        "sMAPE_floor50": 12.47,
    },
    "catboost_sota": {
        "model_id": "catboost_sota",
        "formal_name": "catboost_sota",
        "model_type": "CatBoost",
        "task": "dayahead",
        "champion": False,
        "sMAPE_floor50": 12.58,
    },
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def is_valid_model(model_id: str) -> bool:
    """Check if a model_id is in the valid (non-invalidated) set."""
    return model_id in MODEL_CONFIGS and model_id not in INVALID_MODELS


def is_invalid_model(model_id: str) -> bool:
    """Check if a model_id has been invalidated."""
    return model_id in INVALID_MODELS


def get_model_config(model_id: str) -> dict[str, Any]:
    """Get the configuration dict for a model, or raise KeyError.

    Parameters
    ----------
    model_id : str
        The model identifier.

    Returns
    -------
    dict
        Model configuration.

    Raises
    ------
    KeyError
        If model_id is not in the registry.
    ValueError
        If model_id has been invalidated.
    """
    if model_id in INVALID_MODELS:
        raise ValueError(
            f"Model '{model_id}' is INVALID and must not be used. "
            f"Reason: {INVALID_MODELS[model_id]}"
        )
    if model_id not in MODEL_CONFIGS:
        raise KeyError(f"Model '{model_id}' not found in day-ahead registry")
    return dict(MODEL_CONFIGS[model_id])


def list_valid_models() -> list[str]:
    """Return list of all valid (non-invalidated) model IDs."""
    return [m for m in MODEL_CONFIGS if m not in INVALID_MODELS]
