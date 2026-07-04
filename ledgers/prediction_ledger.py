"""
ledgers/prediction_ledger.py — Prediction and corrected-prediction ledgers.

Both ledgers follow the same append + validate pattern using the generic
store module.  The prediction ledger accumulates P2 standard prediction
output; the corrected ledger accumulates P3 corrected prediction output.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    PREDICTION_LEDGER_KEY,
    CORRECTED_LEDGER_COLUMNS,
    CORRECTED_LEDGER_KEY,
)
from ledgers.store import (
    append_ledger,
    add_run_metadata,
    validate_ledger_keys,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prediction ledger (P2 output → ledger)
# ──────────────────────────────────────────────


def append_predictions_to_ledger(
    predictions_df: pd.DataFrame,
    ledger_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Append P2 standard predictions to the prediction ledger.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        P2 standard prediction output.  Must contain at minimum the
        ``PREDICTION_LEDGER_KEY`` columns plus ``y_pred``.
    ledger_df : pd.DataFrame, optional
        Existing prediction ledger.  If None, a new ledger is created.
    run_id : str, optional
        Run identifier stamped into every row.

    Returns
    -------
    pd.DataFrame
        Deduplicated prediction ledger.
    """
    if len(predictions_df) == 0:
        return ledger_df.copy() if ledger_df is not None else pd.DataFrame(
            columns=PREDICTION_LEDGER_COLUMNS
        )

    # Select only the columns we need for the ledger
    available = [c for c in PREDICTION_LEDGER_COLUMNS if c in predictions_df.columns]
    new_rows = predictions_df[available].copy()

    # Add run metadata
    new_rows = add_run_metadata(new_rows, run_id=run_id)

    # Ensure all ledger columns exist
    for c in PREDICTION_LEDGER_COLUMNS:
        if c not in new_rows.columns:
            new_rows[c] = None

    # Reorder to canonical column order
    new_rows = new_rows[PREDICTION_LEDGER_COLUMNS]

    if ledger_df is None or len(ledger_df) == 0:
        return new_rows

    return append_ledger(ledger_df, new_rows, key_cols=PREDICTION_LEDGER_KEY, keep="latest")


def validate_prediction_ledger(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a prediction ledger DataFrame.

    Checks:
    - All ``PREDICTION_LEDGER_COLUMNS`` are present.
    - No duplicate ``PREDICTION_LEDGER_KEY`` rows.
    - ``y_pred`` is not null.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues).
    """
    issues: list[str] = []

    missing = [c for c in PREDICTION_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if len(df) == 0:
        return True, ["Empty ledger"]

    null_y = df["y_pred"].isna().sum()
    if null_y > 0:
        issues.append(f"{null_y} rows with null y_pred")

    key_valid, key_issues = validate_ledger_keys(df, PREDICTION_LEDGER_KEY)
    if not key_valid:
        issues.extend(key_issues)

    return len(issues) == 0, issues


# ──────────────────────────────────────────────
# Corrected prediction ledger (P3 output → ledger)
# ──────────────────────────────────────────────


def append_corrected_predictions_to_ledger(
    corrected_df: pd.DataFrame,
    ledger_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Append P3 corrected predictions to the corrected ledger.

    Parameters
    ----------
    corrected_df : pd.DataFrame
        P3 corrected prediction output.  Must contain at minimum the
        ``CORRECTED_LEDGER_KEY`` columns plus ``y_pred_corrected``.
    ledger_df : pd.DataFrame, optional
        Existing corrected ledger.  If None, a new ledger is created.
    run_id : str, optional
        Run identifier stamped into every row.

    Returns
    -------
    pd.DataFrame
        Deduplicated corrected ledger.
    """
    if len(corrected_df) == 0:
        return ledger_df.copy() if ledger_df is not None else pd.DataFrame(
            columns=CORRECTED_LEDGER_COLUMNS
        )

    available = [c for c in CORRECTED_LEDGER_COLUMNS if c in corrected_df.columns]
    new_rows = corrected_df[available].copy()

    new_rows = add_run_metadata(new_rows, run_id=run_id)

    for c in CORRECTED_LEDGER_COLUMNS:
        if c not in new_rows.columns:
            new_rows[c] = None

    new_rows = new_rows[CORRECTED_LEDGER_COLUMNS]

    if ledger_df is None or len(ledger_df) == 0:
        return new_rows

    return append_ledger(ledger_df, new_rows, key_cols=CORRECTED_LEDGER_KEY, keep="latest")


def validate_corrected_ledger(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a corrected prediction ledger DataFrame.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues).
    """
    issues: list[str] = []

    missing = [c for c in CORRECTED_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if len(df) == 0:
        return True, ["Empty ledger"]

    null_corrected = df["y_pred_corrected"].isna().sum()
    if null_corrected > 0:
        issues.append(f"{null_corrected} rows with null y_pred_corrected")

    key_valid, key_issues = validate_ledger_keys(df, CORRECTED_LEDGER_KEY)
    if not key_valid:
        issues.extend(key_issues)

    return len(issues) == 0, issues
