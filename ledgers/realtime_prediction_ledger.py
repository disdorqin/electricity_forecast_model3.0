"""
ledgers/realtime_prediction_ledger.py — P93: Realtime Two-Candidate Prediction Ledger.

Manages the realtime prediction ledger with two candidate models:
  1. rt_da_anchor  (DA-Safe Baseline) — ALWAYS available
  2. sgdfnet_rt_assist (SGDFNet Assist) — OPTIONAL

Schema:
    task, model_name, target_day, business_day, ds, hour_business, period,
    y_pred, source_confidence, model_version, run_id, created_at, updated_at,
    da_error_prob, residual_direction_prob, uncertainty_score,
    correction_permission, reason_codes

Rules:
    - rt_da_anchor must always be present.
    - sgdfnet_rt_assist is optional.
    - y_true is forbidden in the prediction ledger.
    - If SGDFNet is unavailable, only rt_da_anchor entries exist.
    - If both models are available, both entries exist for learner fusion.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    PREDICTION_LEDGER_KEY,
)
from ledgers.store import (
    add_run_metadata,
    append_ledger,
    validate_ledger_keys,
)
from models.realtime_state import (
    SGDFNET_ASSIST_DISABLED,
    DA_SAFE_BASELINE_ACTIVE,
)

logger = logging.getLogger(__name__)

# ── Realtime prediction ledger columns ─────────────────────────────────
REALTIME_LEDGER_COLUMNS = PREDICTION_LEDGER_COLUMNS + [
    "da_error_prob",
    "residual_direction_prob",
    "uncertainty_score",
    "correction_permission",
    "reason_codes",
]

REALTIME_LEDGER_KEY = PREDICTION_LEDGER_KEY  # same unique key

# ── Model names ────────────────────────────────────────────────────────
RT_DA_ANCHOR_MODEL = "rt_da_anchor"
SGDFNET_ASSIST_MODEL = "sgdfnet_rt_assist"


def build_realtime_ledger(
    da_anchor_predictions: pd.DataFrame,
    sgdfnet_predictions: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Build a realtime prediction ledger from the two candidate models.

    Parameters
    ----------
    da_anchor_predictions : pd.DataFrame
        DA-Safe Baseline predictions. Must contain at minimum:
        ``business_day, hour_business, ds, y_pred``.
    sgdfnet_predictions : pd.DataFrame, optional
        SGDFNet assist predictions. If None or empty, only the
        DA-Safe Baseline is included.
    run_id : str, optional
        Run identifier.

    Returns
    -------
    pd.DataFrame
        Realtime prediction ledger with both candidates (if available).
    """
    # ── Build DA-Safe Baseline entries ──
    da_df = da_anchor_predictions.copy()
    n = len(da_df)

    da_rows = pd.DataFrame({
        "task": "realtime",
        "model_name": RT_DA_ANCHOR_MODEL,
        "target_day": da_df.get("target_day", da_df.get("business_day", "")),
        "business_day": da_df["business_day"] if "business_day" in da_df.columns else "",
        "ds": da_df["ds"] if "ds" in da_df.columns else pd.NaT,
        "hour_business": da_df["hour_business"] if "hour_business" in da_df.columns else 0,
        "period": da_df["period"] if "period" in da_df.columns else "",
        "y_pred": da_df["y_pred"] if "y_pred" in da_df.columns else
                  da_df.get("trend_pred", da_df.get("da_anchor", np.nan)),
        "source_confidence": da_df.get("source_confidence",
                                        da_df.get("trend_confidence", 0.5)),
        "model_version": da_df.get("model_version", "1.0.0"),
        "da_error_prob": da_df.get("da_error_prob", np.nan),
        "residual_direction_prob": da_df.get("residual_direction_prob", np.nan),
        "uncertainty_score": da_df.get("uncertainty_score", np.nan),
        "correction_permission": da_df.get("correction_permission", False),
        "reason_codes": DA_SAFE_BASELINE_ACTIVE,
    })

    if "target_day" in da_df.columns:
        da_rows["target_day"] = da_df["target_day"]
    elif "business_day" in da_df.columns:
        da_rows["target_day"] = da_df["business_day"]

    all_rows = [da_rows]

    # ── Build SGDFNet Assist entries (optional) ──
    if sgdfnet_predictions is not None and len(sgdfnet_predictions) > 0:
        sg_df = sgdfnet_predictions.copy()
        m = len(sg_df)

        # Align to same row count
        min_len = min(n, m)

        sg_rows = pd.DataFrame({
            "task": "realtime",
            "model_name": SGDFNET_ASSIST_MODEL,
            "target_day": sg_df.get("target_day", sg_df.get("business_day", ""))[:min_len]
                if len(sg_df) >= min_len else "",
            "business_day": sg_df["business_day"][:min_len]
                if "business_day" in sg_df.columns and len(sg_df) >= min_len else "",
            "ds": sg_df["ds"][:min_len].values
                if "ds" in sg_df.columns and len(sg_df) >= min_len else pd.NaT,
            "hour_business": sg_df["hour_business"][:min_len].values
                if "hour_business" in sg_df.columns and len(sg_df) >= min_len else 0,
            "period": sg_df["period"][:min_len].values
                if "period" in sg_df.columns and len(sg_df) >= min_len else "",
            "y_pred": sg_df["rt_pred"][:min_len].values
                if "rt_pred" in sg_df.columns and len(sg_df) >= min_len else
                sg_df.get("sgdfnet_pred", da_rows["y_pred"].values[:min_len]),
            "source_confidence": sg_df.get("source_confidence", 0.4)[:min_len]
                if len(sg_df) >= min_len else 0.0,
            "model_version": sg_df.get("model_version", "2.0.0"),
            "da_error_prob": sg_df.get("da_error_prob", np.nan)[:min_len]
                if len(sg_df) >= min_len else np.nan,
            "residual_direction_prob": sg_df.get("residual_direction_prob", np.nan)[:min_len]
                if len(sg_df) >= min_len else np.nan,
            "uncertainty_score": sg_df.get("uncertainty_score", np.nan)[:min_len]
                if len(sg_df) >= min_len else np.nan,
            "correction_permission": sg_df.get("correction_permission", False)[:min_len]
                if len(sg_df) >= min_len else False,
            "reason_codes": sg_df.get("reason_codes", SGDFNET_ASSIST_DISABLED),
        })
        all_rows.append(sg_rows)
    else:
        logger.info("SGDFNet assist unavailable — using DA-Safe Baseline only")

    # Combine
    ledger = pd.concat(all_rows, ignore_index=True)

    # Add run metadata
    if run_id:
        ledger = add_run_metadata(ledger, run_id=run_id)

    # Ensure all ledger columns exist
    for col in REALTIME_LEDGER_COLUMNS:
        if col not in ledger.columns:
            ledger[col] = None

    return ledger[REALTIME_LEDGER_COLUMNS]


def append_to_realtime_ledger(
    new_predictions: pd.DataFrame,
    existing_ledger: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """Append realtime predictions to the ledger with dedup.

    Parameters
    ----------
    new_predictions : pd.DataFrame
        New predictions to append.
    existing_ledger : pd.DataFrame, optional
        Existing ledger.
    run_id : str, optional
        Run identifier.

    Returns
    -------
    pd.DataFrame
        Deduplicated ledger.
    """
    if len(new_predictions) == 0:
        return existing_ledger.copy() if existing_ledger is not None else pd.DataFrame(
            columns=REALTIME_LEDGER_COLUMNS
        )

    new_rows = new_predictions.copy()
    new_rows = add_run_metadata(new_rows, run_id=run_id)

    for col in REALTIME_LEDGER_COLUMNS:
        if col not in new_rows.columns:
            new_rows[col] = None

    new_rows = new_rows[REALTIME_LEDGER_COLUMNS]

    if existing_ledger is None or len(existing_ledger) == 0:
        return new_rows

    return append_ledger(existing_ledger, new_rows, key_cols=REALTIME_LEDGER_KEY, keep="latest")


def validate_realtime_ledger(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a realtime prediction ledger.

    Checks:
    - All REALTIME_LEDGER_COLUMNS present
    - No duplicate REALTIME_LEDGER_KEY rows
    - y_pred has no NaN
    - rt_da_anchor is always present (if data exists)
    - y_true is absent

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues).
    """
    issues: list[str] = []

    missing = [c for c in REALTIME_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, issues

    if len(df) == 0:
        return True, ["Empty ledger"]

    # Check no y_true
    if "y_true" in df.columns:
        issues.append("y_true found in prediction ledger — forbidden")

    # Check no NaN in y_pred
    null_y = df["y_pred"].isna().sum()
    if null_y > 0:
        issues.append(f"{null_y} rows with null y_pred")

    # Check rt_da_anchor present
    models = df["model_name"].unique().tolist() if "model_name" in df.columns else []
    if RT_DA_ANCHOR_MODEL not in models and len(df) > 0:
        issues.append(f"{RT_DA_ANCHOR_MODEL} must always be present")

    # Check key uniqueness
    key_valid, key_issues = validate_ledger_keys(df, REALTIME_LEDGER_KEY)
    if not key_valid:
        issues.extend(key_issues)

    return len(issues) == 0, issues


def extract_rt_candidates(
    ledger: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Extract individual candidate DataFrames from the ledger.

    Returns dict: {model_name: model_dataframe}
    """
    result: dict[str, pd.DataFrame] = {}
    if "model_name" not in ledger.columns:
        return result

    for model in ledger["model_name"].unique():
        model_data = ledger[ledger["model_name"] == model].copy()
        result[model] = model_data

    return result
