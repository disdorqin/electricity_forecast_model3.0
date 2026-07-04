"""
pipelines/ledger_backfill.py — Synthetic-friendly backfill pipeline.

Accumulates predictions, corrected predictions, and actuals into their
respective ledgers.  Supports both in-memory (DataFrame) and file-based modes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    CORRECTED_LEDGER_COLUMNS,
    ACTUAL_LEDGER_COLUMNS,
)
from ledgers.prediction_ledger import (
    append_predictions_to_ledger,
    append_corrected_predictions_to_ledger,
)
from ledgers.actual_ledger import append_actuals_to_ledger
from ledgers.store import load_ledger, save_ledger

logger = logging.getLogger(__name__)


def run_ledger_backfill(
    prediction_df: Optional[pd.DataFrame] = None,
    corrected_df: Optional[pd.DataFrame] = None,
    actuals_df: Optional[pd.DataFrame] = None,
    ledger_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Run a ledger backfill with the provided DataFrames.

    Parameters
    ----------
    prediction_df : pd.DataFrame, optional
        P2 standard prediction output.
    corrected_df : pd.DataFrame, optional
        P3 corrected prediction output.
    actuals_df : pd.DataFrame, optional
        Actuals data.
    ledger_dir : str, optional
        If provided, ledgers are loaded from and saved to this directory.
    run_id : str, optional
        Run identifier.

    Returns
    -------
    dict
        Summary with counts of appended rows, deduped counts, and run_id.
    """
    summary: dict = {
        "prediction_rows": 0,
        "corrected_rows": 0,
        "actual_rows": 0,
        "prediction_ledger_size": 0,
        "corrected_ledger_size": 0,
        "actual_ledger_size": 0,
        "run_id": run_id,
    }

    # ── Prediction ledger ──────────────────────────────────────────────
    if prediction_df is not None and len(prediction_df) > 0:
        if ledger_dir is not None:
            path = os.path.join(ledger_dir, "prediction_ledger.csv")
            existing = load_ledger(path, columns=PREDICTION_LEDGER_COLUMNS)
        else:
            existing = None

        result = append_predictions_to_ledger(prediction_df, ledger_df=existing, run_id=run_id)
        summary["prediction_rows"] = len(result) - (len(existing) if existing is not None else 0)
        summary["prediction_ledger_size"] = len(result)

        if ledger_dir is not None:
            save_ledger(result, path)

    # ── Corrected ledger ───────────────────────────────────────────────
    if corrected_df is not None and len(corrected_df) > 0:
        if ledger_dir is not None:
            path = os.path.join(ledger_dir, "corrected_ledger.csv")
            existing = load_ledger(path, columns=CORRECTED_LEDGER_COLUMNS)
        else:
            existing = None

        result = append_corrected_predictions_to_ledger(
            corrected_df, ledger_df=existing, run_id=run_id,
        )
        summary["corrected_rows"] = len(result) - (len(existing) if existing is not None else 0)
        summary["corrected_ledger_size"] = len(result)

        if ledger_dir is not None:
            save_ledger(result, path)

    # ── Actual ledger ──────────────────────────────────────────────────
    if actuals_df is not None and len(actuals_df) > 0:
        if ledger_dir is not None:
            path = os.path.join(ledger_dir, "actual_ledger.csv")
            existing = load_ledger(path, columns=ACTUAL_LEDGER_COLUMNS)
        else:
            existing = None

        result = append_actuals_to_ledger(actuals_df, ledger_df=existing, run_id=run_id)
        summary["actual_rows"] = len(result) - (len(existing) if existing is not None else 0)
        summary["actual_ledger_size"] = len(result)

        if ledger_dir is not None:
            save_ledger(result, path)

    return summary
