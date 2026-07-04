"""
pipelines/residual_correction.py — Residual correction pipeline.

Applies residual correction (P5M / negative / low-valley) to standard
prediction output.  Produces corrected prediction output with delta
tracking and correction metadata.

Default behaviour: no-op (y_pred_corrected == y_pred_raw).
DATA-MISSING is the explicit default — real correction requires
risk data or a canonical pack.

Flow:
    prediction output (standard schema)
        → apply_residual_correction()
        → corrected prediction output (corrected schema)

Usage:
    from pipelines.residual_correction import apply_residual_correction

    corrected = apply_residual_correction(predictions_df)
    corrected = apply_residual_correction(
        predictions_df, correction_profile="aggressive", risk_df=risk_data
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    CORRECTED_UNIQUE_KEY,
    CORRECTED_MERGE_KEY,
    CORRECTED_REQUIRED_KEYS,
    PREDICTION_OUTPUT_COLUMNS,
    EVAL_ONLY_COLUMNS,
    VALID_PERIODS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Default no-op module metadata
# ──────────────────────────────────────────────

NOOP_MODULE = "p5m_residual_noop"
NOOP_VERSION = "0.0.0"
NOOP_RISK_SOURCE = "DATA_MISSING"
NOOP_REASON = "DATA_MISSING_NO_OP"

VALID_PROFILES = ("conservative", "moderate", "aggressive")


def _validate_input(df: pd.DataFrame) -> list[str]:
    """Check that input has the minimum required columns.

    Returns list of missing column names (empty = valid).
    Business-time columns (business_day, hour_business, period) are
    derived from ``ds`` and not required in the input.
    """
    missing = []
    # ds is required to derive business time columns
    if "ds" not in df.columns:
        missing.append("ds")
    # Require at least y_pred or y_pred_raw
    if "y_pred" not in df.columns and "y_pred_raw" not in df.columns:
        missing.append("y_pred or y_pred_raw")
    return missing


def _resolve_risk_merge_key(
    pred_columns: list[str],
    risk_columns: list[str],
) -> tuple[list[str], str] | tuple[None, None]:
    """Resolve the best available merge key between predictions and risk data.

    Full merge key (6 columns):
        task, model_name, target_day, business_day, ds, hour_business

    Tries progressively shorter key combinations so that risk data
    without the full key set is still usable.

    Returns
    -------
    tuple[list[str], str] or tuple[None, None]
        (merge_key_columns, key_quality) where key_quality is one of
        ``"full"``, ``"partial"``, ``"degraded"``, or
        ``(None, None)`` when no merge is possible.
    """
    # Full key available in both?
    available = [c for c in CORRECTED_MERGE_KEY if c in pred_columns and c in risk_columns]

    if all(c in available for c in CORRECTED_MERGE_KEY):
        return list(CORRECTED_MERGE_KEY), "full"

    # Partial: task + model_name + target_day + business_day + hour_business
    partial_key = ["task", "model_name", "target_day", "business_day", "hour_business"]
    if all(c in available for c in partial_key):
        return partial_key, "partial"

    # Partial without model_name: task + target_day + business_day + hour_business
    partial_no_mn = ["task", "target_day", "business_day", "hour_business"]
    if all(c in available for c in partial_no_mn):
        return partial_no_mn, "partial"

    # Degraded: business_day + hour_business only
    degraded_key = ["business_day", "hour_business"]
    if all(c in available for c in degraded_key):
        return degraded_key, "degraded"

    # No viable merge key
    return None, None


def _merge_risk_data(
    predictions_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Key-based merge of risk data onto predictions.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Prediction DataFrame with business-time columns resolved.
    risk_df : pd.DataFrame
        Risk data DataFrame to merge.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (predictions with risk columns merged in, merge_stats).
        merge_stats includes merge_key, key_quality, n_risk_rows,
        n_matched, n_unmatched_risk_rows, n_pred_rows_without_risk.
    """
    stats: dict[str, object] = {
        "merge_key": None,
        "key_quality": None,
        "n_risk_rows": len(risk_df),
        "n_matched": 0,
        "n_unmatched_risk_rows": 0,
        "n_pred_rows_without_risk": 0,
    }

    merge_key, key_quality = _resolve_risk_merge_key(
        list(predictions_df.columns), list(risk_df.columns),
    )

    if merge_key is None or key_quality is None:
        logger.warning(
            "Risk DataFrame has no columns matching the merge key. "
            "Skipping risk merge. Risk columns required for merge: "
            "business_day + hour_business (minimal). "
            f"Risk columns available: {list(risk_df.columns)}"
        )
        stats["key_quality"] = "none"
        stats["n_unmatched_risk_rows"] = len(risk_df)
        stats["n_pred_rows_without_risk"] = len(predictions_df)
        return predictions_df, stats

    stats["merge_key"] = merge_key
    stats["key_quality"] = key_quality

    # Identify risk columns to bring across (not already in predictions)
    risk_merge_cols = [c for c in risk_df.columns if c not in predictions_df.columns]

    if not risk_merge_cols:
        logger.warning("Risk DataFrame has no additional columns beyond merge key.")
        stats["n_unmatched_risk_rows"] = len(risk_df)
        stats["n_pred_rows_without_risk"] = len(predictions_df)
        return predictions_df, stats

    n_pred_before = len(predictions_df)
    n_risk_before = len(risk_df)

    # Left-merge risk data onto predictions
    merged = predictions_df.merge(
        risk_df[merge_key + risk_merge_cols],
        on=merge_key,
        how="left",
        suffixes=("", "_risk"),
    )

    # After left merge, rows that didn't match will have NaN in risk columns
    risk_indicator_col = risk_merge_cols[0]
    n_matched = merged[risk_indicator_col].notna().sum()
    n_pred_without_risk = n_pred_before - n_matched

    # Count unmatched risk rows (risk rows that didn't match any prediction)
    # Do a right-anti join conceptually
    risk_merged_inner = risk_df[merge_key + risk_merge_cols].merge(
        predictions_df[merge_key],
        on=merge_key,
        how="inner",
    )
    n_unmatched_risk = n_risk_before - len(risk_merged_inner)

    stats["n_matched"] = int(n_matched)
    stats["n_unmatched_risk_rows"] = int(n_unmatched_risk)
    stats["n_pred_rows_without_risk"] = int(n_pred_without_risk)

    logger.info(
        f"Risk merge ({key_quality} key, {merge_key}): "
        f"{n_matched}/{n_pred_before} prediction rows matched, "
        f"{n_unmatched_risk}/{n_risk_before} risk rows unmatched"
    )

    return merged, stats


def apply_residual_correction(
    predictions_df: pd.DataFrame,
    correction_profile: str = "conservative",
    risk_df: Optional[pd.DataFrame] = None,
    canonical_pack_path: Optional[str] = None,
    production: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply residual correction to prediction output.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Prediction output in standard schema (at minimum the key columns
        plus ``y_pred``).
    correction_profile : str
        Correction aggressiveness: ``"conservative"`` (default),
        ``"moderate"``, or ``"aggressive"``.
    risk_df : pd.DataFrame, optional
        Risk scores DataFrame (e.g. with ``negative_prob``, ``spike_prob``).
        If None, no correction is applied (DATA-MISSING no-op).
    canonical_pack_path : str, optional
        Path to a canonical prediction pack CSV. If provided, the P5M
        adapter may load additional context from this file.
    production : bool
        If True (default), y_true must not be present.

    Returns
    -------
    pd.DataFrame
        Corrected prediction output with the corrected schema.

    Raises
    ------
    ValueError
        If input is missing required columns or profile is invalid.
    """
    if correction_profile not in VALID_PROFILES:
        raise ValueError(
            f"Unknown correction profile: '{correction_profile}'. "
            f"Must be one of: {VALID_PROFILES}"
        )

    df = predictions_df.copy()
    n = len(df)

    if n == 0:
        logger.warning("Empty input DataFrame, returning empty corrected output")
        return pd.DataFrame(columns=CORRECTED_PREDICTION_COLUMNS)

    # ── Production check ───────────────────────
    if production:
        leaked = [c for c in EVAL_ONLY_COLUMNS if c in df.columns]
        if leaked:
            raise ValueError(
                f"Production corrected output must not contain eval-only columns: {leaked}"
            )

    # ── Validate input schema ──────────────────
    missing = _validate_input(df)
    if missing:
        raise ValueError(
            f"Input missing required columns for residual correction: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # ── Ensure standard columns ────────────────
    # Normalise y_pred → y_pred_raw
    if "y_pred_raw" not in df.columns and "y_pred" in df.columns:
        df["y_pred_raw"] = df["y_pred"].values.copy()

    # Ensure business-time columns
    if "business_day" not in df.columns or "hour_business" not in df.columns:
        if "ds" in df.columns:
            from data.business_day import add_business_time_columns
            df["ds"] = pd.to_datetime(df["ds"])
            df = add_business_time_columns(df, timestamp_col="ds")

    # Ensure key columns exist with defaults
    for col in ["task", "model_name", "target_day", "period"]:
        if col not in df.columns:
            if col == "task":
                df[col] = "dayahead"
            elif col == "model_name":
                df[col] = "unknown"
            elif col == "target_day" and "ds" in df.columns:
                df[col] = df["ds"].dt.date.astype(str)
            elif col == "period" and "hour_business" in df.columns:
                df[col] = df["hour_business"].apply(
                    lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
                )

    # Ensure ds is datetime
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])

    # Ensure business_day is datetime
    if "business_day" in df.columns:
        df["business_day"] = pd.to_datetime(df["business_day"])

    y_pred_raw = df["y_pred_raw"].values.astype(float)

    # ── Decide: real correction or no-op? ──────
    # The P5M adapter is our primary correction module.
    # Without risk data or canonical pack, we do a DATA-MISSING no-op.

    correction_module = NOOP_MODULE
    risk_source = NOOP_RISK_SOURCE
    reason_codes = [NOOP_REASON]
    correction_applied = False
    correction_version = NOOP_VERSION

    p5m_adapter = None

    # Try to determine if we have P5M-capable data
    has_risk_data = risk_df is not None and (
        "negative_prob" in risk_df.columns or "risk_source" in risk_df.columns
    )

    has_canonical_pack = (
        canonical_pack_path is not None and Path(canonical_pack_path).exists()
    )

    if has_risk_data or has_canonical_pack:
        # Attempt real correction via P5M adapter
        try:
            from models.adapters.p5m_residual_plugin import P5MResidualPluginAdapter

            p5m_adapter = P5MResidualPluginAdapter(profile=correction_profile)
            p5m_adapter.load()

            # ── Key-based risk merge (not positional) ─────────
            input_for_adapter = df.copy()
            merge_stats: dict[str, object] = {}
            if risk_df is not None:
                input_for_adapter, merge_stats = _merge_risk_data(
                    input_for_adapter, risk_df,
                )
                # Record merge outcome in reason codes
                key_quality = merge_stats.get("key_quality", "none")
                n_matched = merge_stats.get("n_matched", 0)
                n_unmatched = merge_stats.get("n_unmatched_risk_rows", 0)
                n_missing = merge_stats.get("n_pred_rows_without_risk", 0)

                reason_codes.append(f"RISK_MERGE_{key_quality.upper()}_KEY")
                if isinstance(n_matched, int) and n_matched > 0:
                    reason_codes.append("RISK_ROW_MATCHED")
                if isinstance(n_missing, int) and n_missing > 0:
                    reason_codes.append("RISK_ROW_MISSING_NO_OP")
                if isinstance(n_unmatched, int) and n_unmatched > 0:
                    reason_codes.append(f"RISK_UNMATCHED_{n_unmatched}")

            # Call adapter
            adapter_result = p5m_adapter.predict(df=input_for_adapter)
            y_pred_corrected = adapter_result["y_pred"].values.astype(float)

            # Detect whether adapter actually modified anything
            if not np.allclose(y_pred_corrected, y_pred_raw):
                correction_applied = True
                correction_module = "p5m_residual_plugin"
                correction_version = p5m_adapter.model_version
                risk_source = "NEGATIVE_RISK" if has_risk_data else "CANONICAL_PACK"
                reason_codes.insert(0, "P5M_ADAPTER_CORRECTION")
                if has_risk_data:
                    reason_codes.append("RISK_DATA_AVAILABLE")
                if has_canonical_pack:
                    reason_codes.append("CANONICAL_PACK_AVAILABLE")
            else:
                # Adapter ran but didn't change values
                risk_source = "ADAPTER_NO_EFFECT"
                reason_codes = ["ADAPTER_NO_EFFECT"]
                y_pred_corrected = y_pred_raw.copy()

        except Exception as e:
            logger.warning(
                f"P5M adapter correction failed: {e}. "
                "Falling back to DATA-MISSING no-op."
            )
            y_pred_corrected = y_pred_raw.copy()
            reason_codes = [f"DATA_MISSING_NO_OP_ADAPTER_ERROR"]
            risk_source = "ADAPTER_ERROR"
    else:
        # DATA-MISSING: no risk data, no canonical pack → pure no-op
        logger.info(
            "Residual correction: no risk data or canonical pack available. "
            "Applying DATA-MISSING no-op."
        )
        y_pred_corrected = y_pred_raw.copy()

    # Compute residual delta
    residual_delta = y_pred_corrected - y_pred_raw

    # ── Build output ───────────────────────────
    # Preserve all key columns from input
    out = pd.DataFrame({
        "task": df.get("task", "dayahead"),
        "model_name": df.get("model_name", "unknown"),
        "target_day": df.get("target_day", df["ds"].dt.date.astype(str) if "ds" in df.columns else "unknown"),
        "business_day": df.get("business_day", pd.NaT),
        "ds": df.get("ds", pd.NaT),
        "hour_business": df["hour_business"].astype(int),
        "period": df.get("period", df["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
        )),
    })

    out["y_pred_raw"] = y_pred_raw
    out["y_pred_corrected"] = y_pred_corrected
    out["residual_delta"] = residual_delta
    out["correction_applied"] = correction_applied
    out["correction_module"] = correction_module
    out["risk_source"] = risk_source
    out["reason_codes"] = ";".join(reason_codes)
    out["correction_version"] = correction_version
    out["source_confidence"] = df.get("source_confidence", np.nan)
    out["model_version"] = df.get("model_version", "unknown")

    # Sort and validate
    out = out.sort_values(
        ["business_day", "hour_business", "model_name"]
    ).reset_index(drop=True)

    out = out[CORRECTED_PREDICTION_COLUMNS]

    logger.info(
        f"Residual correction applied: module={correction_module}, "
        f"applied={correction_applied}, rows={len(out)}, "
        f"risk_source={risk_source}, reason={';'.join(reason_codes)}"
    )

    return out


def get_corrected_schema_columns() -> list[str]:
    """Return the corrected prediction schema column list."""
    return list(CORRECTED_PREDICTION_COLUMNS)


def is_data_missing_noop(corrected_df: pd.DataFrame) -> bool:
    """Check if a corrected DataFrame is a DATA-MISSING no-op.

    Returns True if all rows have correction_applied == False and
    risk_source == "DATA_MISSING".
    """
    if len(corrected_df) == 0:
        return True
    return (
        (~corrected_df["correction_applied"]).all()
        and (corrected_df["risk_source"] == "DATA_MISSING").all()
    )
