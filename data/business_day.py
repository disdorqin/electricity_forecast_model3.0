"""
data/business_day.py — Business day / business hour utilities.

Canonical rules (single source of truth):

    1. timestamp D 00:00:00  →  business_day = D-1, hour_business = 24
    2. timestamp D HH:00:00  →  business_day = D,   hour_business = HH
       (where HH = 1..23)
    3. hour_business is ALWAYS 1..24 (never 0..23).
    4. hour_business = 24 means the midnight hour of business_day D,
       which is physically D+1 00:00.
    5. Period mapping:
       - hour_business 1..8   → "1_8"
       - hour_business 9..16  → "9_16"
       - hour_business 17..24 → "17_24"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Core mapping functions
# ──────────────────────────────────────────────


def business_day_from_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """Map a timestamp to its business day.

    Parameters
    ----------
    ts : pd.Timestamp
        The input timestamp.

    Returns
    -------
    pd.Timestamp
        The business day (date only, time=00:00:00).
    """
    if ts.hour == 0:
        # 00:00 → business_day = D-1
        return (ts - pd.Timedelta(days=1)).normalize()
    else:
        # 01:00~23:00 → business_day = D
        return ts.normalize()


def hour_business_from_timestamp(ts: pd.Timestamp) -> int:
    """Map a timestamp to its business hour (1..24).

    Parameters
    ----------
    ts : pd.Timestamp
        The input timestamp.

    Returns
    -------
    int
        Business hour in 1..24.
    """
    if ts.hour == 0:
        return 24
    return ts.hour


def hour_business_to_timestamp(business_day: pd.Timestamp, hour_business: int) -> pd.Timestamp:
    """Convert (business_day, hour_business) back to a wall-clock timestamp.

    Parameters
    ----------
    business_day : pd.Timestamp
        The business day (date).
    hour_business : int
        Business hour (1..24).

    Returns
    -------
    pd.Timestamp
        The corresponding wall-clock timestamp.
    """
    if hour_business == 24:
        return business_day + pd.Timedelta(days=1)  # D+1 00:00
    return business_day + pd.Timedelta(hours=hour_business)  # D HH:00


def infer_period(hour_business: int) -> str:
    """Map hour_business (1..24) to period label.

    Parameters
    ----------
    hour_business : int
        Business hour in 1..24.

    Returns
    -------
    str
        "1_8", "9_16", or "17_24".
    """
    if 1 <= hour_business <= 8:
        return "1_8"
    elif 9 <= hour_business <= 16:
        return "9_16"
    elif 17 <= hour_business <= 24:
        return "17_24"
    raise ValueError(f"hour_business must be in 1..24, got {hour_business}")


# ──────────────────────────────────────────────
# Batch DataFrame operations
# ──────────────────────────────────────────────


def add_business_time_columns(
    df: pd.DataFrame,
    timestamp_col: str = "ds",
) -> pd.DataFrame:
    """Add business_day, hour_business, and period columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain ``timestamp_col``.
    timestamp_col : str
        Name of the timestamp column (default: "ds").

    Returns
    -------
    pd.DataFrame
        DataFrame with ``business_day``, ``hour_business``, ``period`` added.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    df["business_day"] = ts.apply(business_day_from_timestamp)
    df["hour_business"] = ts.apply(hour_business_from_timestamp).astype(int)
    df["period"] = df["hour_business"].apply(infer_period)

    return df


def standardize_business_columns(
    df: pd.DataFrame,
    timestamp_col: str = "ds",
    inplace: bool = False,
) -> pd.DataFrame:
    """Standardize business_day, hour_business, period columns.

    If the columns already exist, validates their correctness.
    If missing, infers them from ``timestamp_col``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str
        Name of the timestamp column (default: "ds").
    inplace : bool
        If True, modify the input DataFrame in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized business time columns.
    """
    if not inplace:
        df = df.copy()

    has_bd = "business_day" in df.columns
    has_hb = "hour_business" in df.columns

    if has_bd and has_hb:
        # Validate existing columns
        df["business_day"] = pd.to_datetime(df["business_day"]).dt.normalize()
        df["hour_business"] = df["hour_business"].astype(int)
        # Verify range
        invalid_hours = df[~df["hour_business"].between(1, 24)]
        if len(invalid_hours) > 0:
            raise ValueError(
                f"hour_business out of range [1,24]: {invalid_hours['hour_business'].unique()}"
            )
        # Infer period if missing
        if "period" not in df.columns:
            df["period"] = df["hour_business"].apply(infer_period)
    else:
        # Infer from timestamp
        df = add_business_time_columns(df, timestamp_col=timestamp_col)

    return df


def validate_daily_predictions(df: pd.DataFrame) -> list[str]:
    """Validate a 24-hour prediction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction DataFrame with at least ``business_day`` and ``hour_business``.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list = valid.
    """
    errors: list[str] = []

    if "business_day" not in df.columns:
        errors.append("Missing 'business_day' column")
    if "hour_business" not in df.columns:
        errors.append("Missing 'hour_business' column")

    if errors:
        return errors

    hb = df["hour_business"]

    # Check unique hours
    unique_hours = sorted(hb.unique())
    expected = list(range(1, 25))
    missing_hours = [h for h in expected if h not in unique_hours]
    extra_hours = [h for h in unique_hours if h not in expected]
    if missing_hours:
        errors.append(f"Missing hour_business values: {missing_hours}")
    if extra_hours:
        errors.append(f"Extra hour_business values (outside 1..24): {extra_hours}")

    # Check hour_business range
    if hb.min() < 1 or hb.max() > 24:
        errors.append(f"hour_business out of range [1,24]: min={hb.min()}, max={hb.max()}")

    # Check duplicates
    if hb.duplicated().any():
        errors.append(f"Duplicate hour_business values: {hb[hb.duplicated()].tolist()}")

    # Check NaN in y_pred
    if "y_pred" in df.columns:
        nan_count = df["y_pred"].isna().sum()
        if nan_count > 0:
            errors.append(f"y_pred has {nan_count} NaN values")

    return errors


def validate_no_target_leakage(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Check that feature columns do not contain leakage-causing terms.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    feature_columns : list[str]
        List of feature column names to check.

    Returns
    -------
    list[str]
        List of leaked column names. Empty list = safe.
    """
    DENYLIST = {"y_true", "residual", "error", "abs_error", "future_y", "target_actual", "oracle"}
    leaked = []
    for col in feature_columns:
        for term in DENYLIST:
            if term in col.lower():
                leaked.append(col)
                break
    return leaked
