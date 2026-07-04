"""
pipelines/ledger_fusion.py — Ledger-based fusion runner.

Reads from the corrected ledger and actual ledger, runs P4 fusion,
and writes to the fusion and weight ledgers.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.schema import FUSION_LEDGER_COLUMNS, WEIGHT_LEDGER_COLUMNS
from ledgers.fusion_ledger import append_fusion_to_ledger
from ledgers.weight_ledger import extract_weight_rows, append_weights_to_ledger
from ledgers.actual_ledger import filter_actuals_for_training
from fusion.engine import run_fusion

logger = logging.getLogger(__name__)


def run_ledger_fusion(
    corrected_ledger_df: pd.DataFrame,
    actual_ledger_df: Optional[pd.DataFrame] = None,
    method: str = "equal_weight",
    allow_dry_run: bool = False,
    run_id: Optional[str] = None,
    existing_fusion_ledger: Optional[pd.DataFrame] = None,
    existing_weight_ledger: Optional[pd.DataFrame] = None,
    readiness_status: Optional[dict[str, str]] = None,
    **fusion_kwargs,
) -> dict:
    """Run fusion using ledger-stored data.

    Parameters
    ----------
    corrected_ledger_df : pd.DataFrame
        Corrected prediction ledger.
    actual_ledger_df : pd.DataFrame, optional
        Actual ledger.  If provided, only rows with
        ``business_day < target_day`` are passed to the fusion engine.
    method : str
        Fusion method (default ``"equal_weight"``).
    allow_dry_run : bool
        Passed through to ``run_fusion``.
    run_id : str, optional
        Run identifier.
    existing_fusion_ledger : pd.DataFrame, optional
        Existing fusion ledger to append to.
    existing_weight_ledger : pd.DataFrame, optional
        Existing weight ledger to append to.
    **fusion_kwargs
        Additional kwargs forwarded to ``run_fusion``.

    Returns
    -------
    dict
        Summary with fusion/weight row counts, run_id.
    """
    summary: dict = {
        "fusion_rows": 0,
        "weight_rows": 0,
        "fusion_ledger_size": 0,
        "weight_ledger_size": 0,
        "method": method,
        "run_id": run_id,
    }

    if len(corrected_ledger_df) == 0:
        summary["error"] = "Empty corrected ledger"
        return summary

    # Prepare actuals for training: no-leakage filter
    actuals_for_fusion = None
    if actual_ledger_df is not None and len(actual_ledger_df) > 0:
        # Determine target days from corrected ledger
        target_days = sorted(corrected_ledger_df["target_day"].unique())
        if len(target_days) > 0:
            # Use the earliest target day as the cut — only past actuals
            earliest_target = str(target_days[0])
            actuals_for_fusion = filter_actuals_for_training(
                actual_ledger_df, target_day=earliest_target, window=90,
            )
            if len(actuals_for_fusion) == 0:
                logger.info(
                    "No historical actuals found for target_day %s; "
                    "running fusion without actuals", earliest_target,
                )

    # Run fusion
    fusion_df = run_fusion(
        corrected_ledger_df,
        method=method,
        actuals_df=actuals_for_fusion,
        allow_dry_run=allow_dry_run,
        readiness_status=readiness_status,
        **fusion_kwargs,
    )

    summary["fusion_rows"] = len(fusion_df)

    # Append to fusion ledger
    result_fusion = append_fusion_to_ledger(
        fusion_df, ledger_df=existing_fusion_ledger, run_id=run_id,
    )
    summary["fusion_ledger_size"] = len(result_fusion)

    # Extract and append weight rows
    weight_df = extract_weight_rows(fusion_df)
    summary["weight_rows"] = len(weight_df)

    result_weights = append_weights_to_ledger(
        weight_df, ledger_df=existing_weight_ledger, run_id=run_id,
    )
    summary["weight_ledger_size"] = len(result_weights)

    return summary
