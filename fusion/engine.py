"""
fusion/engine.py — Fusion engine for the 3.0 system.

Consumes corrected prediction output (P3/P3.5) and produces fused
prediction output with configurable weight strategies and
readiness-aware model eligibility.

Usage:
    from fusion.engine import run_fusion

    fused = run_fusion(corrected_df, method="equal_weight")
    fused = run_fusion(corrected_df, method="prior_weight",
                       prior_weights={"cfg05": 0.6, "best_two_average": 0.4})
    fused = run_fusion(corrected_df, method="bgew_skeleton",
                       actuals_df=actuals)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    FUSION_OUTPUT_COLUMNS,
    FUSION_UNIQUE_KEY,
    FUSION_GROUPING_KEY,
    FUSION_REQUIRED_INPUT_COLUMNS,
    VALID_FUSION_METHODS,
    EVAL_ONLY_COLUMNS,
)
from fusion.weights import compute_weights

logger = logging.getLogger(__name__)

# ── Readiness state constants (mirrored from scripts.component_readiness_check)

READY_REAL = "READY_REAL"
READY_DRY_RUN = "READY_DRY_RUN"
READY_STUB = "READY_STUB"
DATA_MISSING = "DATA_MISSING"
NOT_READY = "NOT_READY"


def _auto_readiness() -> dict[str, str]:
    """Auto-detect readiness state of available model adapters.

    Returns {model_name: readiness_state, ...}.
    Mirrors logic from scripts.component_readiness_check but in-process.
    """
    import importlib
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent

    states: dict[str, str] = {}

    # cfg05
    try:
        mod = importlib.import_module("models.adapters.cfg05_dayahead_lgbm")
        if hasattr(mod, "CFG05DayaheadAdapter"):
            states["cfg05"] = READY_DRY_RUN
            # Check for real artifact
            for p in [
                base / "models" / "cfg05" / "model.txt",
                base / "models" / "cfg05" / "model.pkl",
            ]:
                if p.exists():
                    states["cfg05"] = READY_REAL
                    break
        else:
            states["cfg05"] = NOT_READY
    except Exception:
        states["cfg05"] = NOT_READY

    # best_two_average, stage3_business_fixed, catboost models
    for model_id in [
        "best_two_average", "stage3_business_fixed",
        "catboost_spike_residual", "catboost_sota",
    ]:
        try:
            from src.registry.dayahead_models import is_valid_model
            if is_valid_model(model_id):
                states[model_id] = READY_STUB
            else:
                states[model_id] = NOT_READY
        except Exception:
            states[model_id] = NOT_READY

    # p5m_residual_plugin — this is a correction module, not a fusion model
    # It should not appear as a fusion input model.

    return states


def _apply_readiness_gate(
    model_names: list[str],
    allow_dry_run: bool = False,
    readiness_status: Optional[dict[str, str]] = None,
) -> tuple[list[str], list[str], str]:
    """Determine which models pass the readiness gate.

    Parameters
    ----------
    model_names : list[str]
        Candidate model names from the corrected DataFrame.
    allow_dry_run : bool
        If True, READY_DRY_RUN models are included.
    readiness_status : dict, optional
        Pre-computed readiness map.  Auto-detected if None.

    Returns
    -------
    tuple[list[str], list[str], str]
        (included_models, excluded_models, readiness_mode).
        readiness_mode is ``"REAL"`` if all included are READY_REAL,
        ``"DRY_RUN"`` if any included is READY_DRY_RUN.
    """
    if readiness_status is None:
        readiness_status = _auto_readiness()

    included: list[str] = []
    excluded: list[str] = []
    has_dry_run = False
    has_real = False

    for m in model_names:
        state = readiness_status.get(m, NOT_READY)

        if state == READY_REAL:
            included.append(m)
            has_real = True
        elif state == READY_DRY_RUN:
            if allow_dry_run:
                included.append(m)
                has_dry_run = True
            else:
                excluded.append(m)
        else:
            # READY_STUB, DATA_MISSING, NOT_READY → excluded
            excluded.append(m)

    readiness_mode = "DRY_RUN" if (has_dry_run or not has_real) else "REAL"
    return included, excluded, readiness_mode


def _validate_corrected_input(df: pd.DataFrame) -> list[str]:
    """Check that input contains all required corrected columns.

    Returns list of missing column names (empty = valid).
    """
    missing = []
    for col in FUSION_REQUIRED_INPUT_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    return missing


def _check_duplicate_group_rows(df: pd.DataFrame) -> list[str]:
    """Check for duplicate model rows within a fusion group.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []
    dup_mask = df.duplicated(
        subset=FUSION_GROUPING_KEY + ["model_name"],
        keep=False,
    )
    n_dup = dup_mask.sum()
    if n_dup > 0:
        examples = df[dup_mask].head(5)
        errors.append(
            f"Found {n_dup} rows with duplicate "
            f"(fusion_key, model_name). "
            f"Examples: {examples[FUSION_GROUPING_KEY + ['model_name']].values.tolist()}"
        )
    return errors


def run_fusion(
    corrected_df: pd.DataFrame,
    method: str = "equal_weight",
    actuals_df: Optional[pd.DataFrame] = None,
    prior_weights: Optional[dict[str, float]] = None,
    allow_dry_run: bool = False,
    readiness_status: Optional[dict[str, str]] = None,
    production: bool = True,
    learner_version: str = "0.1.0-skeleton",
    **kwargs: Any,
) -> pd.DataFrame:
    """Run fusion on corrected prediction output.

    Parameters
    ----------
    corrected_df : pd.DataFrame
        Corrected prediction output (corrected schema).
    method : str
        Fusion method: ``"equal_weight"``, ``"prior_weight"``, or
        ``"bgew_skeleton"`` (default ``"equal_weight"``).
    actuals_df : pd.DataFrame, optional
        Actuals DataFrame for ``bgew_skeleton``.
    prior_weights : dict, optional
        Prior weights dict for ``prior_weight``.
    allow_dry_run : bool
        If True, include READY_DRY_RUN models (default False).
    readiness_status : dict, optional
        Pre-computed readiness map.  Auto-detected if None.
    production : bool
        If True (default), y_true must not be present.
    learner_version : str
        Version string for the weight learner.

    Returns
    -------
    pd.DataFrame
        Fusion output in FUSION_OUTPUT_COLUMNS schema.

    Raises
    ------
    ValueError
        If input is missing required columns, has duplicate group rows,
        or method is invalid.
    """
    if method not in VALID_FUSION_METHODS:
        raise ValueError(
            f"Unknown fusion method: '{method}'. "
            f"Must be one of: {VALID_FUSION_METHODS}"
        )

    df = corrected_df.copy()
    n = len(df)

    if n == 0:
        logger.warning("Empty corrected DataFrame, returning empty fusion output")
        return pd.DataFrame(columns=FUSION_OUTPUT_COLUMNS)

    # Production check
    if production:
        leaked = [c for c in EVAL_ONLY_COLUMNS if c in df.columns]
        if leaked:
            raise ValueError(
                f"Production fusion must not contain eval-only columns: {leaked}"
            )

    # Validate input
    missing = _validate_corrected_input(df)
    if missing:
        raise ValueError(
            f"Corrected input missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Check for duplicate group rows
    dup_errors = _check_duplicate_group_rows(df)
    if dup_errors:
        raise ValueError("; ".join(dup_errors))

    reasons: list[str] = [f"FUSION_{method.upper()}"]

    # Ensure datetime columns
    for col in ["business_day", "ds"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Detect unique model names present
    all_model_names = sorted(df["model_name"].unique())

    # Apply readiness gate
    included_models, excluded_models, readiness_mode = _apply_readiness_gate(
        all_model_names,
        allow_dry_run=allow_dry_run,
        readiness_status=readiness_status,
    )

    if not included_models:
        logger.warning("No models passed the readiness gate — returning empty fusion")
        return pd.DataFrame(columns=FUSION_OUTPUT_COLUMNS)

    if excluded_models:
        reasons.append(f"ELIGIBLE_EXCLUDED_{'_'.join(sorted(excluded_models))}")
        reasons.append(f"READINESS_{readiness_mode}")

    if readiness_mode == "DRY_RUN":
        reasons.append("FUSION_DRY_RUN_INPUT")

    # Filter to included models only
    df = df[df["model_name"].isin(included_models)].copy()

    # Build fusion groups
    groups = df.groupby(FUSION_GROUPING_KEY, sort=True)

    output_rows: list[dict[str, Any]] = []

    for group_key, group_df in groups:
        group_model_names = sorted(group_df["model_name"].unique())

        # Compute weights for this group
        weights, weight_reasons = compute_weights(
            method,
            group_model_names,
            prior=prior_weights,
            actuals_df=actuals_df,
            corrected_df=df,  # pass full corrected df for BGEW
        )

        # Compute fused price
        y_vals = group_df.set_index("model_name")["y_pred_corrected"]
        fused = sum(weights.get(m, 0.0) * y_vals[m] for m in group_model_names if m in y_vals.index)

        # Build row
        task_val = group_df["task"].iloc[0]
        td_val = group_df["target_day"].iloc[0]
        bd_val = group_df["business_day"].iloc[0]
        ds_val = group_df["ds"].iloc[0]
        hb_val = int(group_df["hour_business"].iloc[0])
        period_val = group_df["period"].iloc[0]

        row: dict[str, Any] = {
            "task": task_val,
            "target_day": str(td_val.date()) if hasattr(td_val, "date") else str(td_val),
            "business_day": bd_val,
            "ds": ds_val,
            "hour_business": hb_val,
            "period": period_val,
            "fused_price": float(fused),
            "weights_json": json.dumps(weights),
            "included_models": ";".join(included_models),
            "excluded_models": ";".join(excluded_models),
            "fusion_method": method,
            "learner_version": learner_version,
            "readiness_mode": readiness_mode,
            "reason_codes": ";".join(reasons + weight_reasons),
        }
        output_rows.append(row)

    if not output_rows:
        return pd.DataFrame(columns=FUSION_OUTPUT_COLUMNS)

    out = pd.DataFrame(output_rows)

    # Sort and reorder
    out = out.sort_values(
        ["business_day", "hour_business"]
    ).reset_index(drop=True)

    out = out[FUSION_OUTPUT_COLUMNS]

    logger.info(
        f"Fusion complete: method={method}, readiness={readiness_mode}, "
        f"rows={len(out)}, included={included_models}, "
        f"excluded={excluded_models}"
    )

    return out
