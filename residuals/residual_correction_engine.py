"""
residuals/residual_correction_engine.py — P69: Unified Residual Correction Engine.

Applies residual correction to both day-ahead and realtime predictions.
Uses P5M adapter (when --residual-source-repo is provided) or built-in
P5M artifact if available, otherwise falls back to no-op.

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
    residual_source_repo: str = "",
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
    residual_source_repo : str
        Path to the 2.0 source repo for P5M residual artifacts.
        When provided, a :class:`~adapters.residual_p5m_adapter.ResidualP5MAdapter`
        is created to discover and load correction models.

    Returns
    -------
    dict with corrected predictions for both tasks.
    """
    # ── Build P5M adapter if source repo is provided ─────────────────
    p5m_adapter = _build_p5m_adapter(
        residual_source_repo=residual_source_repo,
        work_dir=work_dir,
    )

    result: dict[str, Any] = {
        "dayahead": {"status": "NOT_RUN", "output": None},
        "realtime": {"status": "NOT_RUN", "output": None},
        "overall_status": "NOT_RUN",
        "reason_codes": [],
    }
    if p5m_adapter is not None:
        result["p5m_adapter_status"] = p5m_adapter.status

    # Day-ahead correction
    if dayahead_predictions is not None and len(dayahead_predictions) > 0:
        da_result = run_residual_correction(
            predictions=dayahead_predictions,
            actual_ledger_path=actual_ledger_path,
            task="dayahead",
            work_dir=work_dir,
            p5m_adapter=p5m_adapter,
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
            p5m_adapter=p5m_adapter,
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


def _build_p5m_adapter(
    residual_source_repo: str,
    work_dir: str,
) -> Any:
    """Create and initialise a ResidualP5MAdapter if source repo is given.

    Returns
    -------
    ResidualP5MAdapter or None
        *None* when *residual_source_repo* is empty.
    """
    if not residual_source_repo:
        return None

    try:
        from adapters.residual_p5m_adapter import (
            ResidualP5MAdapter,
            RESIDUAL_P5M_CODE_ONLY,
            RESIDUAL_P5M_NO_OP,
        )
    except ImportError:
        logger.warning(
            "adapters.residual_p5m_adapter not importable. "
            "Falling back to built-in residual correction."
        )
        return None

    adapter = ResidualP5MAdapter(
        residual_source_repo=residual_source_repo,
        work_dir=work_dir,
    )

    # Discover artifacts
    artifacts = adapter.find_artifacts()
    logger.info("P5M adapter artifact scan: %s", artifacts)

    # Load correction model (sets adapter._status)
    load_result = adapter.load_correction_model()
    adapter_status = load_result.get("status", RESIDUAL_P5M_NO_OP)

    if adapter_status == RESIDUAL_P5M_CODE_ONLY:
        logger.warning(
            "P5M residual stack code found but no serialised model weights "
            "in source repo '%s'. Using no-op fallback.",
            residual_source_repo,
        )
    elif adapter_status == RESIDUAL_P5M_NO_OP:
        logger.info(
            "No P5M artifacts found in source repo '%s'. "
            "Using no-op fallback.",
            residual_source_repo,
        )
    else:
        logger.info(
            "P5M adapter loaded model: %s (source=%s)",
            load_result.get("model_info", {}).get("model_name", "?"),
            load_result.get("model_info", {}).get("source", "?"),
        )

    return adapter
