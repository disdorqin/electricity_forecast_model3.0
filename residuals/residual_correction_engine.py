"""
residuals/residual_correction_engine.py — P69: Unified Residual Correction Engine.

Applies residual correction to both day-ahead and realtime predictions.
Uses P5M artifact if available, otherwise falls back to no-op.

Strict rules:
  - residual_delta must come from model or past window only
  - Never use current target_day y_true for correction
  - No-op fallback must be explicitly labeled
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from residuals import (
    RESIDUAL_CORRECTION_APPLIED,
    RESIDUAL_NO_OP_FALLBACK,
    RESIDUAL_BLOCKED_NO_DATA,
    run_residual_correction,
)

logger = logging.getLogger(__name__)


def run_full_chain_residual_correction(
    dayahead_predictions: Optional[pd.DataFrame] = None,
    realtime_predictions: Optional[pd.DataFrame] = None,
    actual_ledger_path: str = "",
    work_dir: str = "",
) -> dict[str, Any]:
    """Run residual correction for both dayahead and realtime tasks.

    Parameters
    ----------
    dayahead_predictions : DataFrame, optional
        Day-ahead prediction ledger.
    realtime_predictions : DataFrame, optional
        Realtime prediction ledger.
    actual_ledger_path : str
        Path to actual ledger.
    work_dir : str
        Working directory.

    Returns
    -------
    dict with corrected predictions for both tasks.
    """
    result: dict[str, Any] = {
        "dayahead": {"status": "NOT_RUN", "output": None},
        "realtime": {"status": "NOT_RUN", "output": None},
        "overall_status": "NOT_RUN",
        "reason_codes": [],
    }

    # Day-ahead correction
    if dayahead_predictions is not None and len(dayahead_predictions) > 0:
        da_result = run_residual_correction(
            predictions=dayahead_predictions,
            actual_ledger_path=actual_ledger_path,
            task="dayahead",
            work_dir=work_dir,
        )
        result["dayahead"] = da_result
    else:
        result["dayahead"]["status"] = RESIDUAL_BLOCKED_NO_DATA
        result["reason_codes"].append("NO_DAYAHEAD_PREDICTIONS")

    # Realtime correction
    if realtime_predictions is not None and len(realtime_predictions) > 0:
        rt_result = run_residual_correction(
            predictions=realtime_predictions,
            actual_ledger_path=actual_ledger_path,
            task="realtime",
            work_dir=work_dir,
        )
        result["realtime"] = rt_result
    else:
        result["realtime"]["status"] = RESIDUAL_BLOCKED_NO_DATA
        result["reason_codes"].append("NO_REALTIME_PREDICTIONS")

    # Overall status
    da_ok = result["dayahead"].get("status") in (
        RESIDUAL_CORRECTION_APPLIED, RESIDUAL_NO_OP_FALLBACK
    )
    rt_ok = result["realtime"].get("status") in (
        RESIDUAL_CORRECTION_APPLIED, RESIDUAL_NO_OP_FALLBACK
    )

    if da_ok and rt_ok:
        result["overall_status"] = "RESIDUAL_CORRECTION_COMPLETE"
    elif da_ok:
        result["overall_status"] = "RESIDUAL_PARTIAL_DAYAHEAD_ONLY"
    else:
        result["overall_status"] = "RESIDUAL_CORRECTION_FAILED"

    return result
