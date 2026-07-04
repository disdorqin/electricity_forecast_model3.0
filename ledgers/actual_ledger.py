"""
ledgers/actual_ledger.py — Actuals ledger with no-leakage training filter.

The actual ledger accumulates real price actuals day by day.
The ``filter_actuals_for_training`` function ensures that only historical
actuals (``business_day < target_day``) are returned, preventing future leakage.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.schema import ACTUAL_LEDGER_COLUMNS, ACTUAL_LEDGER_KEY
from ledgers.store import append_ledger, add_run_metadata, validate_ledger_keys

logger = logging.getLogger(__name__)


def append_actuals_to_ledger(
    actuals_df: pd.DataFrame,
    ledger_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Append actuals to the actual ledger.

    Parameters
    ----------
    actuals_df : pd.DataFrame
        Actuals data.  Must contain at minimum the ``ACTUAL_LEDGER_KEY``
        columns plus ``y_true``.
    ledger_df : pd.DataFrame, optional
        Existing actual ledger.  If None, a new ledger is created.
    run_id : str, optional
        Run identifier stamped into every row.

    Returns
    -------
    pd.DataFrame
        Deduplicated actual ledger.
    """
    if len(actuals_df) == 0:
        return ledger_df.copy() if ledger_df is not None else pd.DataFrame(
            columns=ACTUAL_LEDGER_COLUMNS
        )

    available = [c for c in ACTUAL_LEDGER_COLUMNS if c in actuals_df.columns]
    new_rows = actuals_df[available].copy()

    new_rows = add_run_metadata(new_rows, run_id=run_id)

    for c in ACTUAL_LEDGER_COLUMNS:
        if c not in new_rows.columns:
            new_rows[c] = None

    new_rows = new_rows[ACTUAL_LEDGER_COLUMNS]

    if ledger_df is None or len(ledger_df) == 0:
        return new_rows

    return append_ledger(ledger_df, new_rows, key_cols=ACTUAL_LEDGER_KEY, keep="latest")


def validate_actual_ledger(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate an actual ledger DataFrame.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues).
    """
    issues: list[str] = []

    missing = [c for c in ACTUAL_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if len(df) == 0:
        return True, ["Empty ledger"]

    null_y = df["y_true"].isna().sum()
    if null_y > 0:
        issues.append(f"{null_y} rows with null y_true")

    key_valid, key_issues = validate_ledger_keys(df, ACTUAL_LEDGER_KEY)
    if not key_valid:
        issues.extend(key_issues)

    return len(issues) == 0, issues


def filter_actuals_for_training(
    actual_ledger_df: pd.DataFrame,
    target_day: str,
    window: int = 30,
) -> pd.DataFrame:
    """Filter actuals to historical rows suitable for weight training.

    Future-awareness guarantee: only rows where
    ``business_day < target_day`` are returned.

    Parameters
    ----------
    actual_ledger_df : pd.DataFrame
        Full actual ledger.
    target_day : str
        Target day (YYYY-MM-DD).  Actuals with ``business_day >= target_day``
        are excluded.
    window : int
        Maximum number of past calendar days to include (default 30).

    Returns
    -------
    pd.DataFrame
        Filtered actuals with no future leakage.
    """
    if len(actual_ledger_df) == 0:
        return actual_ledger_df.copy()

    target = pd.Timestamp(target_day)

    # Ensure datetime
    df = actual_ledger_df.copy()
    if "business_day" in df.columns:
        df["business_day"] = pd.to_datetime(df["business_day"])

    # FUTURE-AWARENESS: strict < target_day
    train = df[df["business_day"] < target].copy()

    # Apply rolling window
    if len(train) > 0 and window > 0:
        latest = train["business_day"].max()
        window_start = latest - pd.Timedelta(days=window)
        train = train[train["business_day"] >= window_start]

    return train.reset_index(drop=True)
