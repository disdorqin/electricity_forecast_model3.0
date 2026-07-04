"""
data/features/dayahead_features.py — Day-ahead feature pipeline.

Provides the canonical feature builder for the cfg05 champion and
other day-ahead models in the zoo.  Feature definitions are sourced
from the adapter layer (``models/adapters/cfg05_dayahead_lgbm.py``).

Business columns (business_day, target_day, ds, hour_business, period)
are always preserved through the pipeline so downstream adapters and
runners can build standard-schema output.

Usage:
    from data.features.dayahead_features import (
        build_dayahead_features,
        get_dayahead_feature_columns,
        validate_dayahead_feature_frame,
    )

    df_with_features = build_dayahead_features(raw_df, model_id="cfg05")
    cols = get_dayahead_feature_columns("cfg05")
    validate_dayahead_feature_frame(df_with_features)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.registry.dayahead_models import (
    get_model_config,
    is_invalid_model,
    list_valid_models,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Deny-list for feature leakage
# ──────────────────────────────────────────────

DENY_LIST: set[str] = {"y_true", "residual", "error", "abs_error"}

# ──────────────────────────────────────────────
# Business columns to always preserve
# ──────────────────────────────────────────────

BUSINESS_COLUMNS: list[str] = [
    "business_day",
    "ds",
    "hour_business",
    "period",
]

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def get_dayahead_feature_columns(model_id: str = "cfg05") -> list[str]:
    """Return the canonical feature column list for a day-ahead model.

    Parameters
    ----------
    model_id : str
        Model identifier. Must be registered and valid.

    Returns
    -------
    list[str]
        Feature column names.

    Raises
    ------
    ValueError
        If model_id is invalid (banned).
    KeyError
        If model_id is not in the registry.
    """
    if is_invalid_model(model_id):
        raise ValueError(
            f"Model '{model_id}' is INVALID and must not enter the feature pipeline. "
        )
    cfg = get_model_config(model_id)
    fcols: list[str] | None = cfg.get("feature_columns")
    if fcols is not None:
        return list(fcols)

    # Fallback: import from adapter module
    if model_id == "cfg05":
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        return list(CFG05_FEATURE_COLUMNS)

    logger.warning(f"No feature_columns defined for '{model_id}' in registry. Returning empty.")
    return []


def build_dayahead_features(
    df: pd.DataFrame,
    model_id: str = "cfg05",
    fill_strategy: str = "zero",
) -> pd.DataFrame:
    """Build a feature matrix suitable for a day-ahead model.

    The returned DataFrame preserves business columns
    (business_day, target_day, ds, hour_business, period) followed
    by the model-specific feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input with at least the columns required by the model,
        plus ``ds`` for business-time inference.
    model_id : str
        Target model identifier.
    fill_strategy : str
        How to fill missing features: ``"zero"`` (default), ``"nan"``.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with business columns + feature columns.
        Never contains y_true, residual, error, or abs_error columns.

    Raises
    ------
    ValueError
        If model_id is invalid.
    KeyError
        If model_id is not registered.
    """
    if is_invalid_model(model_id):
        raise ValueError(
            f"Model '{model_id}' is INVALID and must not enter the feature pipeline. "
            f"Reason: {get_model_config(model_id)}"
        )

    df = df.copy()

    # Ensure ds is datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # Ensure business-time columns
    if "hour_business" not in df.columns or "business_day" not in df.columns:
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")

    # Preserve business columns
    present_business = [c for c in BUSINESS_COLUMNS if c in df.columns]

    # Get target feature columns
    target_features = get_dayahead_feature_columns(model_id)

    # Select available features
    available_features = [c for c in target_features if c in df.columns]
    missing_features = [c for c in target_features if c not in df.columns]

    if missing_features:
        logger.warning(
            f"Feature pipeline: {len(missing_features)}/{len(target_features)} "
            f"features missing for '{model_id}'. "
            f"First 5 missing: {missing_features[:5]}"
        )

    # Build output: business columns + available features
    out_columns = present_business + available_features
    result = df[out_columns].copy()

    # Fill missing features
    for col in missing_features:
        if fill_strategy == "zero":
            result[col] = 0.0
        else:
            result[col] = np.nan

    # ── Deny-list enforcement ──────────────────
    leaked = _check_deny_list(result)
    if leaked:
        raise ValueError(
            f"Feature pipeline for '{model_id}' produced denied columns: {leaked}. "
            f"The following columns are forbidden as prediction features: {sorted(DENY_LIST)}"
        )

    # Reorder: business columns first, then features
    ordered_cols = present_business + target_features
    ordered_cols = [c for c in ordered_cols if c in result.columns]
    result = result[ordered_cols]

    logger.info(
        f"build_dayahead_features('{model_id}'): "
        f"{len(available_features)} features available, "
        f"{len(missing_features)} missing (filled with {fill_strategy})"
    )

    return result


def validate_dayahead_feature_frame(
    df: pd.DataFrame,
    model_id: str = "cfg05",
    strict: bool = False,
) -> list[str]:
    """Validate a day-ahead feature DataFrame.

    Checks:
      1. Business columns are present.
      2. No denied columns (y_true, residual, error, abs_error).
      3. Feature columns match expected list (in non-strict mode, only warns).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to validate.
    model_id : str
        Model identifier for expected feature list.
    strict : bool
        If True, raise on missing features. If False, return warnings.

    Returns
    -------
    list[str]
        List of warning/error messages. Empty = clean.
    """
    issues: list[str] = []

    # Check business columns
    for col in BUSINESS_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing business column: '{col}'")

    # Check deny list
    leaked = _check_deny_list(df)
    if leaked:
        msg = f"Denied columns found: {leaked}"
        issues.append(msg)
        if strict:
            raise ValueError(msg)

    # Check features
    expected = get_dayahead_feature_columns(model_id)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        msg = f"Missing {len(missing)}/{len(expected)} feature columns for '{model_id}'"
        issues.append(msg)
        if strict:
            raise ValueError(msg)

    return issues


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _check_deny_list(df: pd.DataFrame) -> list[str]:
    """Check which denied terms appear in column names.

    Returns list of offending column names (empty = clean).
    """
    leaked: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        for deny_term in DENY_LIST:
            if deny_term in col_lower:
                leaked.append(col)
                break
    return leaked


def report_missing_features(
    df: pd.DataFrame,
    model_id: str = "cfg05",
) -> dict[str, Any]:
    """Generate a structured report of which features are present vs missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    model_id : str
        Model identifier.

    Returns
    -------
    dict
        Report with keys: model_id, total_features, present, missing, ratio.
    """
    expected = get_dayahead_feature_columns(model_id)
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    return {
        "model_id": model_id,
        "total_features": len(expected),
        "present": present,
        "missing": missing,
        "ratio": len(present) / max(len(expected), 1),
    }
