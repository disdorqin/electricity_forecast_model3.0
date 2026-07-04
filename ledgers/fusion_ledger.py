"""
ledgers/fusion_ledger.py — Fusion ledger.

Accumulates P4 fusion output day by day, deduplicated on the fusion ledger key.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.schema import FUSION_LEDGER_COLUMNS, FUSION_LEDGER_KEY
from ledgers.store import append_ledger, add_run_metadata, validate_ledger_keys

logger = logging.getLogger(__name__)


def append_fusion_to_ledger(
    fusion_df: pd.DataFrame,
    ledger_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Append P4 fusion output to the fusion ledger.

    Parameters
    ----------
    fusion_df : pd.DataFrame
        P4 fusion output.  Must contain at minimum the ``FUSION_LEDGER_KEY``
        columns plus ``fused_price``.
    ledger_df : pd.DataFrame, optional
        Existing fusion ledger.  If None, a new ledger is created.
    run_id : str, optional
        Run identifier stamped into every row.

    Returns
    -------
    pd.DataFrame
        Deduplicated fusion ledger.
    """
    if len(fusion_df) == 0:
        return ledger_df.copy() if ledger_df is not None else pd.DataFrame(
            columns=FUSION_LEDGER_COLUMNS
        )

    available = [c for c in FUSION_LEDGER_COLUMNS if c in fusion_df.columns]
    new_rows = fusion_df[available].copy()

    new_rows = add_run_metadata(new_rows, run_id=run_id)

    for c in FUSION_LEDGER_COLUMNS:
        if c not in new_rows.columns:
            new_rows[c] = None

    new_rows = new_rows[FUSION_LEDGER_COLUMNS]

    if ledger_df is None or len(ledger_df) == 0:
        return new_rows

    return append_ledger(ledger_df, new_rows, key_cols=FUSION_LEDGER_KEY, keep="latest")


def validate_fusion_ledger(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a fusion ledger DataFrame.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues).
    """
    issues: list[str] = []

    missing = [c for c in FUSION_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if len(df) == 0:
        return True, ["Empty ledger"]

    null_fused = df["fused_price"].isna().sum()
    if null_fused > 0:
        issues.append(f"{null_fused} rows with null fused_price")

    key_valid, key_issues = validate_ledger_keys(df, FUSION_LEDGER_KEY)
    if not key_valid:
        issues.extend(key_issues)

    return len(issues) == 0, issues
