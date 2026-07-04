"""
ledgers/weight_ledger.py — Weight ledger.

Extracts per-model per-hour weights from P4 fusion output (``weights_json``)
and accumulates them in a weight ledger.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from data.schema import (
    FUSION_LEDGER_COLUMNS,
    WEIGHT_LEDGER_COLUMNS,
    WEIGHT_LEDGER_KEY,
)
from ledgers.store import append_ledger, add_run_metadata, validate_ledger_keys

logger = logging.getLogger(__name__)

# Columns that should be copied from the fusion row into each weight row
_FUSION_COLUMNS_FOR_WEIGHT: list[str] = [
    "task",
    "target_day",
    "business_day",
    "ds",
    "hour_business",
    "period",
    "fusion_method",
    "learner_version",
    "reason_codes",
    "run_id",
    "created_at",
    "updated_at",
]


def extract_weight_rows(fusion_df: pd.DataFrame) -> pd.DataFrame:
    """Extract individual weight rows from a fusion output DataFrame.

    Each row in the fusion output has a ``weights_json`` column containing
    ``{"model_name": weight, ...}``.  This function expands that dict so that
    each (fusion_row, model_name) pair becomes its own row.

    Parameters
    ----------
    fusion_df : pd.DataFrame
        P4 fusion output with ``weights_json`` column.

    Returns
    -------
    pd.DataFrame
        Weight rows conforming to ``WEIGHT_LEDGER_COLUMNS``.
    """
    if len(fusion_df) == 0:
        return pd.DataFrame(columns=WEIGHT_LEDGER_COLUMNS)

    if "weights_json" not in fusion_df.columns:
        raise ValueError("fusion_df must contain 'weights_json' column")

    rows: list[dict] = []

    # Copy fusion-level metadata into each weight row
    fusion_cols = [c for c in _FUSION_COLUMNS_FOR_WEIGHT if c in fusion_df.columns]

    for _, frow in fusion_df.iterrows():
        try:
            weights = json.loads(frow["weights_json"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping row with invalid weights_json: %s", frow.get("weights_json"))
            continue

        if not isinstance(weights, dict):
            continue

        for model_name, weight_value in weights.items():
            row = {c: frow[c] for c in fusion_cols}
            row["model_name"] = model_name
            row["weight"] = float(weight_value)
            row["weight_source"] = frow.get("fusion_method", "unknown")
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=WEIGHT_LEDGER_COLUMNS)

    result = pd.DataFrame(rows)

    # Ensure all columns exist
    for c in WEIGHT_LEDGER_COLUMNS:
        if c not in result.columns:
            result[c] = None

    return result[WEIGHT_LEDGER_COLUMNS]


def append_weights_to_ledger(
    weight_df: pd.DataFrame,
    ledger_df: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Append weight rows to the weight ledger.

    Parameters
    ----------
    weight_df : pd.DataFrame
        Weight rows (as produced by ``extract_weight_rows``).
    ledger_df : pd.DataFrame, optional
        Existing weight ledger.  If None, a new ledger is created.
    run_id : str, optional
        Run identifier stamped into every row.

    Returns
    -------
    pd.DataFrame
        Deduplicated weight ledger.
    """
    if len(weight_df) == 0:
        return ledger_df.copy() if ledger_df is not None else pd.DataFrame(
            columns=WEIGHT_LEDGER_COLUMNS
        )

    new_rows = weight_df.copy()
    new_rows = add_run_metadata(new_rows, run_id=run_id)

    for c in WEIGHT_LEDGER_COLUMNS:
        if c not in new_rows.columns:
            new_rows[c] = None

    new_rows = new_rows[WEIGHT_LEDGER_COLUMNS]

    if ledger_df is None or len(ledger_df) == 0:
        return new_rows

    return append_ledger(ledger_df, new_rows, key_cols=WEIGHT_LEDGER_KEY, keep="latest")


def validate_weight_ledger(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a weight ledger DataFrame.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues).
    """
    issues: list[str] = []

    missing = [c for c in WEIGHT_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if len(df) == 0:
        return True, ["Empty ledger"]

    null_weight = df["weight"].isna().sum()
    if null_weight > 0:
        issues.append(f"{null_weight} rows with null weight")

    key_valid, key_issues = validate_ledger_keys(df, WEIGHT_LEDGER_KEY)
    if not key_valid:
        issues.extend(key_issues)

    return len(issues) == 0, issues
