"""
delivery/fallback_ladder.py — P54: Fallback Ladder for 3.0 delivery.

Implements the delivery fallback mechanism with 6 levels:

    1. trusted_bgew_fusion       — BGEW fusion using trusted models
    2. trusted_equal_weight      — equal weight among trusted models
    3. best_trusted_single_model — best single trusted model output
    4. cfg05_baseline            — cfg05 baseline (always available as champion)
    5. historical_same_hour_median — median of historical same-hour prices
    6. FAILED_NO_DELIVERY        — no valid output

Each level attempts to produce 24 rows of hourly day-ahead delivery prices.
The first level that passes postflight validation is selected.

Delivery status rules:
    - NORMAL:            level 1 succeeds AND postflight PASS
    - DEGRADED_DELIVERED: levels 2-5 produce valid 24H output with no NaN
    - FAILED_NO_DELIVERY: all fallback levels failed
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.business_day import (
    hour_business_from_timestamp,
    infer_period,
)
from data.loaders import load_table
from fusion.weights import bgew_skeleton

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DELIVERY_OUTPUT_COLUMNS: list[str] = [
    "business_day",
    "ds",
    "hour_business",
    "period",
    "dayahead_price",
    "realtime_price",
]

FALLBACK_LEVEL_NAMES: dict[int, str] = {
    1: "trusted_bgew_fusion",
    2: "trusted_equal_weight",
    3: "best_trusted_single_model",
    4: "cfg05_baseline",
    5: "historical_same_hour_median",
    6: "FAILED_NO_DELIVERY",
}


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _extract_price_col(df: pd.DataFrame) -> str:
    """Determine which price column to use — corrected over raw.

    Returns ``"y_pred_corrected"`` if available, else ``"y_pred"``.
    """
    if "y_pred_corrected" in df.columns:
        return "y_pred_corrected"
    return "y_pred"


def _build_output_from_predictions(
    predictions: pd.DataFrame,
    target_date: str,
    price_col: str,
) -> pd.DataFrame | None:
    """Build a 24-row delivery output DataFrame from hourly model predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions for a single model or fused result. Must contain
        ``business_day``, ``ds``, ``hour_business``, and the *price_col*.
    target_date : str
        Target date (YYYY-MM-DD) — used for consistency checks.
    price_col : str
        Name of the column to extract as ``dayahead_price``.

    Returns
    -------
    pd.DataFrame | None
        24-row delivery output in ``DELIVERY_OUTPUT_COLUMNS`` order, or None
        if data is insufficient.
    """
    required = ["business_day", "ds", "hour_business", price_col]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        logger.warning("Missing required columns for output: %s", missing)
        return None

    df = predictions.copy()
    df["business_day"] = pd.to_datetime(df["business_day"])
    df["ds"] = pd.to_datetime(df["ds"])

    rows: list[dict[str, Any]] = []
    for hb in range(1, 25):
        hour_data = df[df["hour_business"] == hb]
        if len(hour_data) == 0:
            logger.warning("Missing hour_business %d — cannot build 24-row output", hb)
            return None

        row = hour_data.iloc[0]
        price = row[price_col]

        rows.append({
            "business_day": row["business_day"],
            "ds": row["ds"],
            "hour_business": int(hb),
            "period": infer_period(int(hb)),
            "dayahead_price": float(price) if pd.notna(price) else None,
            "realtime_price": None,
        })

    if len(rows) != 24:
        return None

    output = pd.DataFrame(rows)
    output = output.sort_values("hour_business").reset_index(drop=True)
    return output[DELIVERY_OUTPUT_COLUMNS]


def _validate_fallback_output(df: pd.DataFrame, target_date: str) -> tuple[bool, list[str]]:
    """Validate a fallback output DataFrame.

    Checks:
    - 24 rows present
    - ``hour_business`` values 1..24, all present, no duplicates
    - No NaN in ``dayahead_price``
    - Schema includes all ``DELIVERY_OUTPUT_COLUMNS``

    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame to validate.
    target_date : str
        Target date (YYYY-MM-DD) — used in error messages.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, list_of_issues)``.
    """
    issues: list[str] = []

    if df is None:
        return False, ["Output is None"]

    # Schema check
    required_cols = set(DELIVERY_OUTPUT_COLUMNS)
    missing = required_cols - set(df.columns)
    if missing:
        issues.append(f"Missing schema columns: {sorted(missing)}")
        return False, issues

    # Row count
    if len(df) != 24:
        issues.append(f"Expected 24 rows, got {len(df)}")

    # Hour completeness
    if "hour_business" in df.columns:
        hours = sorted(df["hour_business"].unique())
        expected = list(range(1, 25))
        if hours != expected:
            missing_hours = [h for h in expected if h not in hours]
            extra_hours = [h for h in hours if h not in expected]
            if missing_hours:
                issues.append(f"Missing hour_business values: {missing_hours}")
            if extra_hours:
                issues.append(f"Extra hour_business values: {extra_hours}")

        if df["hour_business"].duplicated().any():
            dupes = df[df["hour_business"].duplicated()]["hour_business"].tolist()
            issues.append(f"Duplicate hour_business values: {dupes}")

    # NaN
    if "dayahead_price" in df.columns:
        nan_count = int(df["dayahead_price"].isna().sum())
        if nan_count > 0:
            issues.append(f"dayahead_price has {nan_count} NaN value(s)")

    return len(issues) == 0, issues


def _run_postflight(
    output: pd.DataFrame,
) -> tuple[str, list[str]]:
    """Run postflight validation on the output.

    Validates the same structural properties as ``_validate_fallback_output``
    plus any additional delivery-stage checks.

    Returns
    -------
    tuple[str, list[str]]
        ``("PASS" | "FAIL", warnings)``.
    """
    if output is None or len(output) == 0:
        return "FAIL", ["Empty output — nothing to validate"]

    valid, issues = _validate_fallback_output(output, "")
    if valid:
        return "PASS", []

    return "FAIL", issues


def _count_input_rows(
    ledger: pd.DataFrame,
    target_date: str,
    trusted_models: list[str],
) -> int:
    """Count prediction rows for *target_date* and *trusted_models*."""
    if ledger is None or len(ledger) == 0:
        return 0
    mask = (
        (ledger.get("task", "") == "dayahead")
        & (ledger.get("target_day", "").astype(str) == target_date)
        & (ledger.get("model_name", "").isin(trusted_models))
    )
    return int(mask.sum())


# ──────────────────────────────────────────────
# Fallback level implementations
# ──────────────────────────────────────────────


def _try_trusted_bgew_fusion(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger: pd.DataFrame,
    actual_ledger: pd.DataFrame,
) -> dict | None:
    """Level 1: BGEW fusion using trusted models with inverse-error weighting.

    Returns a dict with keys ``level``, ``method``, ``success``, ``reason``,
    and optionally ``output`` (DataFrame). Returns None when no data is
    available to attempt this level.
    """
    if prediction_ledger is None or len(prediction_ledger) == 0:
        return None
    if actual_ledger is None or len(actual_ledger) == 0:
        return None
    if not trusted_models:
        return None

    # Filter to target date and trusted models
    mask = (
        (prediction_ledger["task"] == "dayahead")
        & (prediction_ledger["target_day"].astype(str) == target_date)
        & (prediction_ledger["model_name"].isin(trusted_models))
    )
    df = prediction_ledger[mask].copy()

    if len(df) == 0:
        return {
            "level": 1,
            "method": "trusted_bgew_fusion",
            "success": False,
            "reason": "No prediction data for trusted models on target date",
        }

    price_col = _extract_price_col(df)
    df["business_day"] = pd.to_datetime(df["business_day"])
    df["ds"] = pd.to_datetime(df["ds"])

    # Compute BGEW weights
    try:
        weights, weight_reasons = bgew_skeleton(
            trusted_models,
            corrected_df=df,
            actuals_df=actual_ledger,
        )
    except Exception as exc:
        logger.warning("BGEW weight computation failed: %s", exc)
        return {
            "level": 1,
            "method": "trusted_bgew_fusion",
            "success": False,
            "reason": f"BGEW weight computation failed: {exc}",
        }

    if not weights or all(w == 0 for w in weights.values()):
        return {
            "level": 1,
            "method": "trusted_bgew_fusion",
            "success": False,
            "reason": "BGEW weights are all zero",
        }

    # Apply weights per hour to produce fused price
    output_rows: list[dict[str, Any]] = []
    hours = sorted(df["hour_business"].unique())

    for hb in hours:
        hour_data = df[df["hour_business"] == hb]
        fused = 0.0
        n_contrib = 0

        for model in trusted_models:
            model_row = hour_data[hour_data["model_name"] == model]
            if len(model_row) == 0:
                continue
            price = model_row[price_col].iloc[0]
            if pd.notna(price):
                fused += weights.get(model, 0.0) * float(price)
                n_contrib += 1

        if n_contrib == 0:
            continue

        ref = hour_data.iloc[0]
        output_rows.append({
            "business_day": ref["business_day"],
            "ds": ref["ds"],
            "hour_business": int(hb),
            "period": infer_period(int(hb)),
            "dayahead_price": float(fused),
            "realtime_price": None,
        })

    if len(output_rows) == 0:
        return {
            "level": 1,
            "method": "trusted_bgew_fusion",
            "success": False,
            "reason": "No output rows produced by BGEW fusion",
        }

    output = pd.DataFrame(output_rows)
    output = output.sort_values("hour_business").reset_index(drop=True)
    output = output[DELIVERY_OUTPUT_COLUMNS]

    valid, issues = _validate_fallback_output(output, target_date)
    if not valid:
        return {
            "level": 1,
            "method": "trusted_bgew_fusion",
            "success": False,
            "reason": f"Output validation failed: {'; '.join(issues)}",
        }

    return {
        "level": 1,
        "method": "trusted_bgew_fusion",
        "success": True,
        "reason": f"BGEW fusion produced valid 24-hour output (weights: {weight_reasons})",
        "output": output,
    }


def _try_trusted_equal_weight(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger: pd.DataFrame,
) -> dict | None:
    """Level 2: Equal-weight average of all trusted model predictions.

    Returns a dict with keys ``level``, ``method``, ``success``, ``reason``,
    and optionally ``output`` (DataFrame). Returns None when no data is
    available to attempt this level.
    """
    if prediction_ledger is None or len(prediction_ledger) == 0:
        return None
    if not trusted_models:
        return None

    mask = (
        (prediction_ledger["task"] == "dayahead")
        & (prediction_ledger["target_day"].astype(str) == target_date)
        & (prediction_ledger["model_name"].isin(trusted_models))
    )
    df = prediction_ledger[mask].copy()

    if len(df) == 0:
        return {
            "level": 2,
            "method": "trusted_equal_weight",
            "success": False,
            "reason": "No prediction data for trusted models on target date",
        }

    price_col = _extract_price_col(df)
    df["business_day"] = pd.to_datetime(df["business_day"])
    df["ds"] = pd.to_datetime(df["ds"])

    output_rows: list[dict[str, Any]] = []
    hours = sorted(df["hour_business"].unique())

    for hb in hours:
        hour_data = df[df["hour_business"] == hb]
        prices = pd.to_numeric(hour_data[price_col], errors="coerce")
        valid_prices = prices.dropna()

        if len(valid_prices) == 0:
            continue

        avg_price = float(valid_prices.mean())
        ref = hour_data.iloc[0]
        output_rows.append({
            "business_day": ref["business_day"],
            "ds": ref["ds"],
            "hour_business": int(hb),
            "period": infer_period(int(hb)),
            "dayahead_price": avg_price,
            "realtime_price": None,
        })

    if len(output_rows) != 24:
        return {
            "level": 2,
            "method": "trusted_equal_weight",
            "success": False,
            "reason": f"Only {len(output_rows)} of 24 hours produced",
        }

    output = pd.DataFrame(output_rows)
    output = output.sort_values("hour_business").reset_index(drop=True)
    output = output[DELIVERY_OUTPUT_COLUMNS]

    valid, issues = _validate_fallback_output(output, target_date)
    if not valid:
        return {
            "level": 2,
            "method": "trusted_equal_weight",
            "success": False,
            "reason": f"Output validation failed: {'; '.join(issues)}",
        }

    return {
        "level": 2,
        "method": "trusted_equal_weight",
        "success": True,
        "reason": f"Equal-weight average of {len(trusted_models)} trusted models produced valid 24-hour output",
        "output": output,
    }


def _try_best_trusted_single(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger: pd.DataFrame,
    actual_ledger: pd.DataFrame,
) -> dict | None:
    """Level 3: Best single model output from the trusted pool.

    Uses recent historical MAE against actuals to select the best-performing
    model. Falls back to the first trusted model if actuals are unavailable.

    Returns a dict with keys ``level``, ``method``, ``success``, ``reason``,
    and optionally ``output`` (DataFrame). Returns None when no data is
    available to attempt this level.
    """
    if prediction_ledger is None or len(prediction_ledger) == 0:
        return None
    if not trusted_models:
        return None

    # Default to first trusted model
    selected_model: str = trusted_models[0]
    selection_reason: str = f"Defaulted to first trusted model: {trusted_models[0]}"

    # If actuals are available, select best model by MAE
    if actual_ledger is not None and len(actual_ledger) > 0:
        model_errors: dict[str, float] = {}
        price_col_candidates = _extract_price_col(prediction_ledger)

        for model in trusted_models:
            model_preds = prediction_ledger[
                prediction_ledger["model_name"] == model
            ].copy()

            if len(model_preds) == 0:
                continue

            model_preds["business_day"] = pd.to_datetime(model_preds["business_day"])
            act = actual_ledger.copy()
            act["business_day"] = pd.to_datetime(act["business_day"])

            merged = model_preds.merge(
                act[["business_day", "hour_business", "y_true"]],
                on=["business_day", "hour_business"],
                how="inner",
            )

            if len(merged) == 0:
                continue

            pred_vals = pd.to_numeric(merged[price_col_candidates], errors="coerce").values
            true_vals = pd.to_numeric(merged["y_true"], errors="coerce").values
            errors = np.abs(pred_vals - true_vals)
            errors = errors[~np.isnan(errors)]

            if len(errors) > 0:
                model_errors[model] = float(np.mean(errors))

        if model_errors:
            selected_model = min(model_errors, key=model_errors.get)
            selection_reason = (
                f"Selected '{selected_model}' with MAE={model_errors[selected_model]:.4f} "
                f"over {len(model_errors)} models"
            )

    # Extract predictions for selected model
    mask = (
        (prediction_ledger["task"] == "dayahead")
        & (prediction_ledger["target_day"].astype(str) == target_date)
        & (prediction_ledger["model_name"] == selected_model)
    )
    model_data = prediction_ledger[mask].copy()

    if len(model_data) == 0:
        return {
            "level": 3,
            "method": "best_trusted_single_model",
            "success": False,
            "reason": f"No predictions found for selected model '{selected_model}'",
        }

    price_col = _extract_price_col(model_data)
    output = _build_output_from_predictions(model_data, target_date, price_col)

    if output is None:
        return {
            "level": 3,
            "method": "best_trusted_single_model",
            "success": False,
            "reason": "Failed to build 24-hour output from best single model",
        }

    return {
        "level": 3,
        "method": "best_trusted_single_model",
        "success": True,
        "reason": selection_reason,
        "output": output,
    }


def _try_cfg05_baseline(
    target_date: str,
    prediction_ledger: pd.DataFrame,
) -> dict | None:
    """Level 4: cfg05 baseline (always available as champion model).

    Returns a dict with keys ``level``, ``method``, ``success``, ``reason``,
    and optionally ``output`` (DataFrame). Returns None when no data is
    available to attempt this level.
    """
    if prediction_ledger is None or len(prediction_ledger) == 0:
        return None

    # Find cfg05 model in the ledger (case-insensitive match on "cfg05")
    cfg05_models = [
        m for m in prediction_ledger["model_name"].unique()
        if "cfg05" in str(m).lower()
    ]

    if not cfg05_models:
        return {
            "level": 4,
            "method": "cfg05_baseline",
            "success": False,
            "reason": "No cfg05 model found in prediction ledger",
        }

    cfg05_model = cfg05_models[0]

    mask = (
        (prediction_ledger["task"] == "dayahead")
        & (prediction_ledger["target_day"].astype(str) == target_date)
        & (prediction_ledger["model_name"] == cfg05_model)
    )
    cfg05_data = prediction_ledger[mask].copy()

    if len(cfg05_data) == 0:
        return {
            "level": 4,
            "method": "cfg05_baseline",
            "success": False,
            "reason": f"No cfg05 predictions for target date {target_date}",
        }

    price_col = _extract_price_col(cfg05_data)
    output = _build_output_from_predictions(cfg05_data, target_date, price_col)

    if output is None:
        return {
            "level": 4,
            "method": "cfg05_baseline",
            "success": False,
            "reason": "Failed to build 24-hour output from cfg05 baseline",
        }

    return {
        "level": 4,
        "method": "cfg05_baseline",
        "success": True,
        "reason": f"cfg05 baseline ('{cfg05_model}') produced valid 24-hour output",
        "output": output,
    }


def _try_historical_median(
    target_date: str,
    raw_data_path: str | None,
) -> dict | None:
    """Level 5: Median of historical same-hour prices from raw data.

    Reads raw data via ``data.loaders.load_table``, computes per-hour
    median dayahead price from historical rows (before *target_date*),
    and builds a 24-row output.

    Returns a dict with keys ``level``, ``method``, ``success``, ``reason``,
    and optionally ``output`` (DataFrame). Returns None when no data is
    available to attempt this level.
    """
    if raw_data_path is None:
        return None

    # Load raw data
    try:
        raw_df, meta = load_table(
            raw_data_path,
            parse_dates=True,
            add_business_time=True,
        )
    except Exception as exc:
        return {
            "level": 5,
            "method": "historical_same_hour_median",
            "success": False,
            "reason": f"Failed to load raw data from '{raw_data_path}': {exc}",
        }

    if len(raw_df) == 0:
        return {
            "level": 5,
            "method": "historical_same_hour_median",
            "success": False,
            "reason": "Raw data file is empty",
        }

    # Identify the price column
    price_col: str | None = None
    for candidate in ["da_anchor", "y_pred", "dayahead_price", "price"]:
        if candidate in raw_df.columns:
            price_col = candidate
            break

    if price_col is None:
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = str(numeric_cols[0])
        else:
            return {
                "level": 5,
                "method": "historical_same_hour_median",
                "success": False,
                "reason": "No suitable price column found in raw data",
            }

    # Use only historical data (before target_date) to avoid leakage
    target_ts = pd.Timestamp(target_date)
    if "business_day" in raw_df.columns:
        raw_df["business_day"] = pd.to_datetime(raw_df["business_day"])
        hist_data = raw_df[raw_df["business_day"] < target_ts].copy()
    else:
        hist_data = raw_df.copy()

    if len(hist_data) == 0:
        return {
            "level": 5,
            "method": "historical_same_hour_median",
            "success": False,
            "reason": "No historical data available before target date",
        }

    # Ensure hour_business column exists
    if "hour_business" not in hist_data.columns and "ds" in hist_data.columns:
        hist_data["hour_business"] = hist_data["ds"].apply(hour_business_from_timestamp)

    if "hour_business" not in hist_data.columns:
        return {
            "level": 5,
            "method": "historical_same_hour_median",
            "success": False,
            "reason": "Cannot determine hour_business — no ds or hour_business column",
        }

    # Compute per-hour medians
    medians = hist_data.groupby("hour_business")[price_col].median()

    # Build output
    output_rows: list[dict[str, Any]] = []
    for hb in range(1, 25):
        median_price = medians.get(hb)
        if pd.isna(median_price) or median_price is None:
            continue

        # Compute wall-clock timestamp
        if hb == 24:
            ds = target_ts + pd.Timedelta(days=1)
        else:
            ds = target_ts + pd.Timedelta(hours=int(hb))

        output_rows.append({
            "business_day": target_ts,
            "ds": ds,
            "hour_business": hb,
            "period": infer_period(hb),
            "dayahead_price": float(median_price),
            "realtime_price": None,
        })

    if len(output_rows) != 24:
        return {
            "level": 5,
            "method": "historical_same_hour_median",
            "success": False,
            "reason": f"Only {len(output_rows)} of 24 hours available from historical data",
        }

    output = pd.DataFrame(output_rows)
    output = output.sort_values("hour_business").reset_index(drop=True)
    output = output[DELIVERY_OUTPUT_COLUMNS]

    valid, issues = _validate_fallback_output(output, target_date)
    if not valid:
        return {
            "level": 5,
            "method": "historical_same_hour_median",
            "success": False,
            "reason": f"Output validation failed: {'; '.join(issues)}",
        }

    return {
        "level": 5,
        "method": "historical_same_hour_median",
        "success": True,
        "reason": f"Historical median computed from {len(hist_data)} rows across "
        f"{hist_data['hour_business'].nunique()} hours",
        "output": output,
    }


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────


def run_fallback_ladder(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger_path: str,
    actual_ledger_path: str,
    raw_data_path: str | None = None,
) -> dict:
    """Try each fallback level in order and return the final delivery result.

    Parameters
    ----------
    target_date : str
        Target date in ``YYYY-MM-DD`` format.
    trusted_models : list[str]
        List of trusted model names (e.g. ``["lightgbm_cfg05_dayahead",
        "catboost_spike_residual"]``).
    prediction_ledger_path : str
        Path to prediction ledger CSV (or corrected ledger containing
        model predictions with ``y_pred`` / ``y_pred_corrected``).
    actual_ledger_path : str
        Path to actual ledger CSV.
    raw_data_path : str, optional
        Path to raw CSV/Excel file for historical median fallback.

    Returns
    -------
    dict
        Delivery result with keys:

        - **success** (*bool*) — whether delivery was achieved
        - **delivery_status** (*str*) — ``"NORMAL"`` | ``"DEGRADED_DELIVERED"`` |
          ``"FAILED_NO_DELIVERY"``
        - **fallback_level** (*int*) — winning level (1-6)
        - **fallback_method** (*str*) — winning method name
        - **reason** (*str*) — explanation of the outcome
        - **output** (*pd.DataFrame | None*) — 24-row final output
        - **output_file** (*str | None*) — not yet persisted (placeholder)
        - **warnings** (*list[str]*) — accumulated warnings
        - **errors** (*list[str]*) — accumulated errors
        - **attempts** (*list[dict]*) — per-level attempt log
    """
    # ── Load ledgers ───────────────────────────
    try:
        prediction_ledger = pd.read_csv(prediction_ledger_path)
    except Exception as exc:
        prediction_ledger = pd.DataFrame()
        logger.warning("Failed to read prediction ledger: %s", exc)

    try:
        actual_ledger = pd.read_csv(actual_ledger_path)
    except Exception as exc:
        actual_ledger = pd.DataFrame()
        logger.warning("Failed to read actual ledger: %s", exc)

    # Normalise datetime columns
    for ledger in (prediction_ledger, actual_ledger):
        if len(ledger) > 0:
            for col in ("business_day", "ds", "target_day"):
                if col in ledger.columns:
                    ledger[col] = pd.to_datetime(ledger[col], errors="coerce")

    warnings: list[str] = []
    errors: list[str] = []
    attempts: list[dict] = []
    final_output: pd.DataFrame | None = None
    final_level: int = 6
    final_method: str = "FAILED_NO_DELIVERY"
    final_reason: str = "All fallback levels failed"
    delivery_status: str = "FAILED_NO_DELIVERY"

    # ── Level 1: Trusted BGEW Fusion ───────────
    result_1 = _try_trusted_bgew_fusion(
        target_date, trusted_models, prediction_ledger, actual_ledger,
    )
    if result_1 is not None:
        if result_1.get("success") and result_1.get("output") is not None:
            postflight_status, pw = _run_postflight(result_1["output"])
            if postflight_status == "PASS":
                final_output = result_1["output"]
                final_level = 1
                final_method = "trusted_bgew_fusion"
                final_reason = result_1["reason"]
                delivery_status = "NORMAL"
            else:
                warnings.append(f"Level 1 postflight FAIL: {'; '.join(pw)}")
        if not result_1.get("success"):
            warnings.append(f"Level 1 failed: {result_1.get('reason', 'unknown')}")
        attempts.append({
            "level": 1,
            "method": "trusted_bgew_fusion",
            "success": result_1.get("success", False),
            "reason": result_1.get("reason", "unknown"),
        })

    # ── Level 2: Trusted Equal Weight ──────────
    if final_output is None:
        result_2 = _try_trusted_equal_weight(
            target_date, trusted_models, prediction_ledger,
        )
        if result_2 is not None:
            if result_2.get("success") and result_2.get("output") is not None:
                postflight_status, pw = _run_postflight(result_2["output"])
                if postflight_status == "PASS":
                    final_output = result_2["output"]
                    final_level = 2
                    final_method = "trusted_equal_weight"
                    final_reason = result_2["reason"]
                    delivery_status = "DEGRADED_DELIVERED"
                else:
                    warnings.append(f"Level 2 postflight FAIL: {'; '.join(pw)}")
            if not result_2.get("success"):
                warnings.append(f"Level 2 failed: {result_2.get('reason', 'unknown')}")
            attempts.append({
                "level": 2,
                "method": "trusted_equal_weight",
                "success": result_2.get("success", False),
                "reason": result_2.get("reason", "unknown"),
            })

    # ── Level 3: Best Trusted Single Model ─────
    if final_output is None:
        result_3 = _try_best_trusted_single(
            target_date, trusted_models, prediction_ledger, actual_ledger,
        )
        if result_3 is not None:
            if result_3.get("success") and result_3.get("output") is not None:
                postflight_status, pw = _run_postflight(result_3["output"])
                if postflight_status == "PASS":
                    final_output = result_3["output"]
                    final_level = 3
                    final_method = "best_trusted_single_model"
                    final_reason = result_3["reason"]
                    delivery_status = "DEGRADED_DELIVERED"
                else:
                    warnings.append(f"Level 3 postflight FAIL: {'; '.join(pw)}")
            if not result_3.get("success"):
                warnings.append(f"Level 3 failed: {result_3.get('reason', 'unknown')}")
            attempts.append({
                "level": 3,
                "method": "best_trusted_single_model",
                "success": result_3.get("success", False),
                "reason": result_3.get("reason", "unknown"),
            })

    # ── Level 4: cfg05 Baseline ────────────────
    if final_output is None:
        result_4 = _try_cfg05_baseline(target_date, prediction_ledger)
        if result_4 is not None:
            if result_4.get("success") and result_4.get("output") is not None:
                postflight_status, pw = _run_postflight(result_4["output"])
                if postflight_status == "PASS":
                    final_output = result_4["output"]
                    final_level = 4
                    final_method = "cfg05_baseline"
                    final_reason = result_4["reason"]
                    delivery_status = "DEGRADED_DELIVERED"
                else:
                    warnings.append(f"Level 4 postflight FAIL: {'; '.join(pw)}")
            if not result_4.get("success"):
                warnings.append(f"Level 4 failed: {result_4.get('reason', 'unknown')}")
            attempts.append({
                "level": 4,
                "method": "cfg05_baseline",
                "success": result_4.get("success", False),
                "reason": result_4.get("reason", "unknown"),
            })

    # ── Level 5: Historical Median ─────────────
    if final_output is None:
        result_5 = _try_historical_median(target_date, raw_data_path)
        if result_5 is not None:
            if result_5.get("success") and result_5.get("output") is not None:
                postflight_status, pw = _run_postflight(result_5["output"])
                if postflight_status == "PASS":
                    final_output = result_5["output"]
                    final_level = 5
                    final_method = "historical_same_hour_median"
                    final_reason = result_5["reason"]
                    delivery_status = "DEGRADED_DELIVERED"
                else:
                    warnings.append(f"Level 5 postflight FAIL: {'; '.join(pw)}")
            if not result_5.get("success"):
                warnings.append(f"Level 5 failed: {result_5.get('reason', 'unknown')}")
            attempts.append({
                "level": 5,
                "method": "historical_same_hour_median",
                "success": result_5.get("success", False),
                "reason": result_5.get("reason", "unknown"),
            })

    # ── Compile result ─────────────────────────
    if final_output is not None:
        reason_suffix = f" (level {final_level}: {final_method})"
        final_reason = final_reason + reason_suffix

    result: dict[str, Any] = {
        "success": final_output is not None,
        "delivery_status": delivery_status,
        "fallback_level": final_level,
        "fallback_method": final_method,
        "reason": final_reason,
        "output": final_output,
        "output_file": None,
        "warnings": warnings,
        "errors": errors,
        "attempts": attempts,
    }

    logger.info(
        "Fallback ladder result: level=%d method=%s status=%s",
        final_level,
        final_method,
        delivery_status,
    )

    return result
