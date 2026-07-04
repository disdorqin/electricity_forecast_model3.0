"""
data/features/realtime_features.py — Realtime assist input pipeline.

Provides column normalisation, validation, and input preparation
for the DA-Safe Realtime Assist Model.

Key design:
    - Input is normalised to canonical column names (ds, da_anchor, rt_actual).
    - ``da_anchor`` is always required for production predictions.
    - ``rt_actual`` is optional for production, required for eval.
    - Reason codes track which transformations were applied.
    - Business-time columns are added if missing.

Usage:
    from data.features.realtime_features import (
        build_realtime_assist_input,
        validate_realtime_assist_input,
        normalize_realtime_columns,
    )

    df, reason_codes = build_realtime_assist_input(raw_df)
    validate_realtime_assist_input(df, production=True)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data.business_day import add_business_time_columns

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Column alias map
# ──────────────────────────────────────────────

TIMESTAMP_ALIASES = {"times", "timestamp", "date_time", "datetime", "time"}
DA_ANCHOR_ALIASES = {"da_price", "forecast_price", "da_forecast", "dayahead_price"}
RT_ACTUAL_ALIASES = {"rt_price", "realtime_price", "actual_rt"}

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def normalize_realtime_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalise DataFrame column names to canonical realtime schema.

    Attempts to locate:
      - ``ds`` from timestamp aliases (times, timestamp, ...)
      - ``da_anchor`` from DA price aliases (da_price, forecast_price, ...)
      - ``rt_actual`` from RT price aliases (rt_price, ...)

    Parameters
    ----------
    df : pd.DataFrame
        Raw input with possibly non-canonical column names.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - DataFrame with canonical column names.
        - List of reason codes describing transformations applied.
    """
    df = df.copy()
    reason_codes: list[str] = []

    # ── Normalise timestamp column ─────────────
    if "ds" not in df.columns:
        for alias in TIMESTAMP_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: "ds"})
                reason_codes.append(f"TIMESTAMP_FROM_{alias.upper()}")
                break
        if "ds" not in df.columns:
            # Try common variations
            for col in df.columns:
                if col.lower() in TIMESTAMP_ALIASES:
                    df = df.rename(columns={col: "ds"})
                    reason_codes.append(f"TIMESTAMP_FROM_{col.upper()}")
                    break

    # ── Normalise da_anchor column ─────────────
    if "da_anchor" not in df.columns:
        for alias in DA_ANCHOR_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: "da_anchor"})
                reason_codes.append(f"DA_ANCHOR_FROM_{alias.upper()}")
                break
        # Fallback: forecast_price with explicit reason
        if "da_anchor" not in df.columns and "forecast_price" in df.columns:
            df = df.rename(columns={"forecast_price": "da_anchor"})
            reason_codes.append("DA_ANCHOR_FROM_FORECAST_PRICE")

    # ── Normalise rt_actual column ─────────────
    if "rt_actual" not in df.columns:
        for alias in RT_ACTUAL_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: "rt_actual"})
                reason_codes.append(f"RT_ACTUAL_FROM_{alias.upper()}")
                break

    # ── Ensure ds is datetime ──────────────────
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    return df, reason_codes


def build_realtime_assist_input(
    df: pd.DataFrame,
    production: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a standardised realtime assist input DataFrame.

    Steps:
      1. Normalise column names (alias → canonical).
      2. Add business-time columns if missing.
      3. Validate required columns.
      4. Return with metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    production : bool
        If True (default), rt_actual is optional.
        If False (eval mode), rt_actual is required.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - Normalised DataFrame with at minimum ds, da_anchor, business_day,
          hour_business, period.
        - Metadata dict with keys:
            - ``reason_codes``: list of transformation codes
            - ``has_rt_actual``: whether rt_actual was found
            - ``da_anchor_source``: how da_anchor was derived
            - ``production``: production flag
            - ``errors``: list of non-fatal issues
    """
    meta: dict[str, Any] = {
        "reason_codes": [],
        "has_rt_actual": False,
        "da_anchor_source": "unknown",
        "production": production,
        "errors": [],
    }

    # Normalise columns
    df, reason_codes = normalize_realtime_columns(df)
    meta["reason_codes"] = reason_codes

    # Track da_anchor source
    for code in reason_codes:
        if code.startswith("DA_ANCHOR_FROM_"):
            meta["da_anchor_source"] = code.replace("DA_ANCHOR_FROM_", "").lower()

    # Track rt_actual
    meta["has_rt_actual"] = "rt_actual" in df.columns

    if not meta["has_rt_actual"] and not production:
        meta["errors"].append(
            "RT_ACTUAL_MISSING_FOR_EVAL: rt_actual not found but required in eval mode"
        )
    elif not meta["has_rt_actual"] and production:
        meta["reason_codes"].append("RT_ACTUAL_MISSING_FOR_PRODUCTION")

    # Add business-time columns if missing
    has_business_time = (
        "business_day" in df.columns and "hour_business" in df.columns
    )
    if not has_business_time:
        if "ds" in df.columns:
            df = add_business_time_columns(df, timestamp_col="ds")
            meta["reason_codes"].append("BUSINESS_TIME_ADDED")
        else:
            meta["errors"].append("Cannot add business time: ds column missing")

    return df, meta


def validate_realtime_assist_input(
    df: pd.DataFrame,
    production: bool = True,
) -> list[str]:
    """Validate a realtime assist input DataFrame.

    Checks:
      1. ``ds`` column exists and is datetime.
      2. ``da_anchor`` column exists.
      3. ``business_day`` and ``hour_business`` exist.
      4. In eval mode (production=False), ``rt_actual`` must exist.

    Parameters
    ----------
    df : pd.DataFrame
        Input to validate.
    production : bool
        If True, rt_actual is optional (default).
        If False, rt_actual is required.

    Returns
    -------
    list[str]
        List of validation errors. Empty list = valid.
    """
    errors: list[str] = []

    if "ds" not in df.columns:
        errors.append("Missing required column: 'ds'")
    elif not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        errors.append("'ds' column is not datetime type")

    if "da_anchor" not in df.columns:
        errors.append(
            "Missing required column: 'da_anchor'. "
            "At least one of 'da_anchor', 'da_price', or 'forecast_price' is needed."
        )

    if "business_day" not in df.columns:
        errors.append("Missing business_day — call build_realtime_assist_input() first")

    if "hour_business" not in df.columns:
        errors.append("Missing hour_business — call build_realtime_assist_input() first")

    if not production and "rt_actual" not in df.columns:
        errors.append(
            "Eval mode requires 'rt_actual' but column is missing. "
            "Available columns for rt_actual: rt_price, realtime_price, actual_rt"
        )

    return errors
