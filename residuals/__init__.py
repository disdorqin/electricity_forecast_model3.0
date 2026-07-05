"""
residuals/residual_correction_engine.py — P69: Unified Residual Correction.

Applies residual correction to both day-ahead and realtime predictions.

Rules:
  - If P5M artifact exists → use real P5M correction
  - If not → use safe no-op fallback
  - residual_delta can only come from model or past window
  - Never use current target_day y_true for correction
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────
RESIDUAL_CORRECTION_APPLIED = "RESIDUAL_CORRECTION_APPLIED"
RESIDUAL_NO_OP_FALLBACK = "RESIDUAL_NO_OP_FALLBACK"
RESIDUAL_BLOCKED_NO_DATA = "RESIDUAL_BLOCKED_NO_DATA"
RESIDUAL_BLOCKED_LEAKAGE = "RESIDUAL_BLOCKED_LEAKAGE"


def run_residual_correction(
    predictions: pd.DataFrame,
    actual_ledger_path: str = "",
    task: str = "dayahead",
    work_dir: str = "",
    p5m_adapter: Any = None,
) -> dict[str, Any]:
    """Apply residual correction to predictions.

    Parameters
    ----------
    predictions : DataFrame
        Prediction ledger with y_pred column.
    actual_ledger_path : str
        Path to actual ledger for historical residual learning.
    task : str
        "dayahead" or "realtime".
    work_dir : str
        Working directory for artifacts.
    p5m_adapter : ResidualP5MAdapter, optional
        If provided, delegate correction to this adapter (from
        ``adapters.residual_p5m_adapter``).  The adapter handles
        artifact discovery, model loading, and safety guardrails.

    Returns
    -------
    dict with corrected predictions and status.
    """
    result: dict[str, Any] = {
        "task": task,
        "status": RESIDUAL_NO_OP_FALLBACK,
        "correction_applied": False,
        "reason_codes": [],
        "output": None,
    }

    if predictions is None or len(predictions) == 0:
        result["status"] = RESIDUAL_BLOCKED_NO_DATA
        result["reason_codes"].append("NO_PREDICTIONS")
        return result

    # ── Delegate to P5M adapter if provided ──────────────────────────
    if p5m_adapter is not None:
        try:
            adapter_result = p5m_adapter.apply_correction(
                predictions=predictions, task=task
            )
            # Map adapter status to engine status constants
            adapter_status = adapter_result.get("status", "")
            if adapter_result.get("correction_applied"):
                result["status"] = RESIDUAL_CORRECTION_APPLIED
                result["correction_applied"] = True
                result["reason_codes"].append("P5M_ADAPTER_USED")
            else:
                result["status"] = RESIDUAL_NO_OP_FALLBACK
            result["reason_codes"].extend(adapter_result.get("reason_codes", []))
            result["output"] = adapter_result.get("output")
            result["rows"] = adapter_result.get("rows", len(predictions))
            result["adapter_status"] = adapter_status
            result["model_info"] = adapter_result.get("model_info", {})
            return result
        except Exception as e:
            logger.warning("P5M adapter failed: %s. Falling back to built-in.", e)
            result["reason_codes"].append(f"P5M_ADAPTER_FAILED:{e}")
            # Fall through to built-in logic below

    corrected = predictions.copy()

    # Check for P5M artifact
    p5m_artifact = os.path.join(work_dir, "p5m_residual_model.pkl") if work_dir else ""
    if p5m_artifact and os.path.isfile(p5m_artifact):
        try:
            corrected = _apply_p5m_correction(corrected, p5m_artifact, task)
            result["status"] = RESIDUAL_CORRECTION_APPLIED
            result["correction_applied"] = True
            result["reason_codes"].append("P5M_ARTIFACT_USED")
        except Exception as e:
            result["reason_codes"].append(f"P5M_FAILED:{e}")
            corrected = _apply_noop_fallback(corrected, task)
    else:
        corrected = _apply_noop_fallback(corrected, task)
        result["reason_codes"].append("NO_P5M_ARTIFACT")

    result["output"] = corrected
    result["rows"] = len(corrected)
    return result


def _apply_p5m_correction(
    predictions: pd.DataFrame,
    artifact_path: str,
    task: str,
) -> pd.DataFrame:
    """Apply P5M residual correction from saved artifact."""
    import pickle
    with open(artifact_path, "rb") as f:
        model = pickle.load(f)

    corrected = predictions.copy()
    price_col = "y_pred" if "y_pred" in corrected.columns else "dayahead_price"

    if price_col in corrected.columns:
        X = corrected[[price_col]].fillna(0).values
        residual_delta = model.predict(X) if hasattr(model, "predict") else np.zeros(len(X))
        corrected["y_pred_raw"] = corrected[price_col]
        corrected["residual_delta"] = residual_delta
        corrected["y_pred_corrected"] = corrected[price_col] + residual_delta
        corrected["residual_model_name"] = "p5m_residual"
        corrected["residual_status"] = RESIDUAL_CORRECTION_APPLIED
    else:
        corrected["y_pred_raw"] = np.nan
        corrected["residual_delta"] = 0.0
        corrected["y_pred_corrected"] = np.nan
        corrected["residual_model_name"] = "p5m_residual"
        corrected["residual_status"] = RESIDUAL_BLOCKED_NO_DATA

    return corrected


def _apply_noop_fallback(
    predictions: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Apply no-op fallback: corrected = original prediction."""
    corrected = predictions.copy()

    price_col = None
    for col in ["y_pred", "dayahead_price", "trend_pred", "realtime_price"]:
        if col in corrected.columns:
            price_col = col
            break

    if price_col:
        corrected["y_pred_raw"] = corrected[price_col]
        corrected["residual_delta"] = 0.0
        corrected["y_pred_corrected"] = corrected[price_col]
        corrected["residual_model_name"] = "noop_fallback"
        corrected["residual_status"] = RESIDUAL_NO_OP_FALLBACK
    else:
        corrected["y_pred_raw"] = np.nan
        corrected["residual_delta"] = 0.0
        corrected["y_pred_corrected"] = np.nan
        corrected["residual_model_name"] = "noop_fallback"
        corrected["residual_status"] = RESIDUAL_NO_OP_FALLBACK

    return corrected
