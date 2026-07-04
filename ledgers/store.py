"""
ledgers/store.py — Generic ledger store with idempotent append / dedup.

All ledger operations use a common append pattern:
    load → append (dedup by key) → save

No ledger files are tracked in git.  Tests use pytest's tmp_path exclusively.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_ledger(
    path: Optional[str] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load a ledger from *path* (CSV or Parquet).

    If *path* is None or does not exist, an empty DataFrame with *columns*
    (or an empty DataFrame with no columns) is returned.
    """
    if path is None or not os.path.isfile(path):
        if columns:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
        return pd.DataFrame()

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if columns is not None:
        for c in columns:
            if c not in df.columns:
                df[c] = None
    return df


def save_ledger(df: pd.DataFrame, path: str) -> None:
    """Save *df* to *path* (inferred from extension: .parquet or .csv).

    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def append_ledger(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_cols: list[str],
    keep: str = "latest",
) -> pd.DataFrame:
    """Append *new_df* to *existing_df*, deduplicating on *key_cols*.

    Parameters
    ----------
    existing_df : pd.DataFrame
        Current ledger content (may be empty).
    new_df : pd.DataFrame
        New rows to append.
    key_cols : list[str]
        Columns that define uniqueness.
    keep : str
        ``"latest"`` (default): keep the row with the most recent ``updated_at``
        (or last occurrence if tie / no timestamp column).
        ``"first"``: keep the first occurrence from *existing_df*.

    Returns
    -------
    pd.DataFrame
        Merged and deduplicated ledger.
    """
    if len(new_df) == 0:
        return existing_df.copy()

    if len(existing_df) == 0:
        # Ensure key columns are present
        if key_cols:
            missing = [c for c in key_cols if c not in new_df.columns]
            if missing:
                raise ValueError(f"new_df missing key columns: {missing}")
        return new_df.copy()

    # Normalise business_day to datetime if present (CSV round-trip → string)
    if "business_day" in existing_df.columns:
        existing_df["business_day"] = pd.to_datetime(existing_df["business_day"])
    if "business_day" in new_df.columns:
        new_df["business_day"] = pd.to_datetime(new_df["business_day"])

    combined = pd.concat([existing_df, new_df], ignore_index=True)

    if not key_cols:
        return combined

    if keep == "latest":
        if "updated_at" in combined.columns:
            # Fill NaN updated_at with current time so they sort last
            now = datetime.datetime.utcnow().isoformat()
            combined["updated_at"] = combined["updated_at"].fillna(now)
            # Sort so most recent updated_at wins
            combined = combined.sort_values("updated_at", ascending=True)
        # Keep last occurrence per key group
        deduped = combined.groupby(key_cols, as_index=False).last()
    elif keep == "first":
        deduped = combined.groupby(key_cols, as_index=False).first()
    else:
        raise ValueError(f"Unknown keep strategy: {keep}")

    return deduped.reset_index(drop=True)


def validate_ledger_keys(
    df: pd.DataFrame,
    key_cols: list[str],
) -> tuple[bool, list[str]]:
    """Check that *df* has no duplicate rows on *key_cols*.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages).
    """
    errors: list[str] = []

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        return False, [f"Missing key columns: {missing}"]

    if len(df) == 0:
        return True, []

    dups = df.duplicated(subset=key_cols, keep=False)
    if dups.any():
        n_dups = dups.sum()
        errors.append(f"{n_dups} duplicate rows on key {key_cols}")
        return False, errors

    return True, []


def add_run_metadata(
    df: pd.DataFrame,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Add (or overwrite) run metadata columns.

    Adds ``run_id``, ``created_at``, ``updated_at`` as new columns if they do
    not exist; does not overwrite existing values unless the column is missing.
    ``updated_at`` is always set to the current UTC timestamp.
    """
    result = df.copy()
    now = datetime.datetime.utcnow().isoformat()

    if run_id is not None:
        result["run_id"] = run_id
    elif "run_id" not in result.columns:
        result["run_id"] = None

    if "created_at" not in result.columns:
        result["created_at"] = now

    # Always set updated_at so dedup can distinguish successive appends
    result["updated_at"] = now

    return result
