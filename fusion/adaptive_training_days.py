"""
fusion/adaptive_training_days.py — P52 Adaptive Complete Training Day Selector.

Adapted from 2.5's ``ledger_weight.py`` adaptive scanning logic (2.5 used
``expected_models`` and a two-level status PASS/FAIL).  For 3.0 the function
accepts *trusted_models* (models with TRUSTED or DELIVERY_ALLOWED status)
and produces a richer output dict with four status levels:

    * COMPLETE_30D        — >= required_days (default 30) complete days found.
    * DEGRADED_MIN_DAYS   — >= min_days_for_degraded (default 7) but < required_days.
    * INSUFFICIENT_DAYS   — > 0 but < min_days_for_degraded.
    * NO_VALID_DAYS       — zero days, or ledgers not found.

A **complete training day** for 3.0 (day-ahead only) satisfies:

    1. Prediction ledger: every trusted model has hour_business 1..24,
       no NaN in y_pred, no duplicate (task, model_name, target_day,
       business_day, hour_business) keys.
    2. Actual ledger: hour_business 1..24, no NaN in y_true, no duplicate
       (task, target_day, business_day, hour_business) keys.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_EXPECTED_HOURS: set[int] = set(range(1, 25))
"""Business hours 1..24 that must be present for a complete day."""


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────


def select_complete_training_days(
    target_date: str,
    trusted_models: list[str],
    prediction_ledger_path: str,
    actual_ledger_path: str,
    required_days: int = 30,
    max_lookback_days: int = 180,
    min_days_for_degraded: int = 7,
) -> dict[str, Any]:
    """Select the most recent *required_days* complete training days.

    Scans backwards from ``target_date - 1`` day, checking each calendar
    day for completeness in both the prediction and actual ledgers.

    Parameters
    ----------
    target_date : str
        The forecast target date D (YYYY-MM-DD).  Scanning begins at D-1.
    trusted_models : list[str]
        Model names that must have complete 24-hour predictions for a day
        to be considered complete.
    prediction_ledger_path : str
        Path to the prediction ledger parquet file.
    actual_ledger_path : str
        Path to the actual ledger parquet file.
    required_days : int
        Number of complete training days desired (default 30).
    max_lookback_days : int
        Maximum number of calendar days to scan backwards (default 180).
    min_days_for_degraded : int
        Minimum days for a DEGRADED result (default 7).  If fewer days
        are found, the status is INSUFFICIENT_DAYS.

    Returns
    -------
    dict
        Keys: ``status``, ``selected_days``, ``selected_count``,
        ``skipped_days``, ``errors``, ``warnings``, ``latest_selected_day``,
        ``oldest_selected_day``, ``training_rows``, ``actual_rows``,
        ``required_days``, ``max_lookback_days``, ``min_days_for_degraded``.
    """
    # ── Build result scaffolding ──────────────────────────────────
    result: dict[str, Any] = {
        "status": "NO_VALID_DAYS",
        "selected_days": [],
        "selected_count": 0,
        "skipped_days": [],
        "errors": [],
        "warnings": [],
        "latest_selected_day": None,
        "oldest_selected_day": None,
        "training_rows": 0,
        "actual_rows": 0,
        "required_days": required_days,
        "max_lookback_days": max_lookback_days,
        "min_days_for_degraded": min_days_for_degraded,
    }

    # Guard: no trusted models
    if not trusted_models:
        result["errors"].append("trusted_models list is empty")
        result["warnings"].append(
            "No trusted models provided; cannot verify prediction completeness"
        )
        return result

    # ── Load ledgers ──────────────────────────────────────────────
    try:
        pred_df = pd.read_parquet(prediction_ledger_path)
    except FileNotFoundError:
        result["errors"].append(
            f"prediction ledger not found: {prediction_ledger_path}"
        )
        return result
    except Exception as exc:
        result["errors"].append(
            f"failed to read prediction ledger: {exc}"
        )
        return result

    try:
        act_df = pd.read_parquet(actual_ledger_path)
    except FileNotFoundError:
        result["errors"].append(
            f"actual ledger not found: {actual_ledger_path}"
        )
        return result
    except Exception as exc:
        result["errors"].append(
            f"failed to read actual ledger: {exc}"
        )
        return result

    if len(pred_df) == 0:
        result["errors"].append("prediction ledger is empty")
        return result

    if len(act_df) == 0:
        result["errors"].append("actual ledger is empty")
        return result

    # Filter to dayahead task (3.0 realtime is DA_ONLY/DRY_RUN)
    if "task" in pred_df.columns:
        pred_df = pred_df[pred_df["task"] == "dayahead"].copy()
    if "task" in act_df.columns:
        act_df = act_df[act_df["task"] == "dayahead"].copy()

    # Normalise date columns for consistent comparison
    for df in (pred_df, act_df):
        if "business_day" in df.columns:
            df["business_day"] = pd.to_datetime(df["business_day"])
        if "target_day" in df.columns:
            df["target_day"] = pd.to_datetime(df["target_day"])

    D = pd.Timestamp(target_date)

    # ── Scan backwards from D-1 ───────────────────────────────────
    selected: list[str] = []
    skipped: list[tuple[str, str]] = []
    n_models = len(trusted_models)

    for offset in range(1, max_lookback_days + 1):
        if len(selected) >= required_days:
            break

        day_dt = D - pd.Timedelta(days=offset)
        day_str = day_dt.strftime("%Y-%m-%d")

        # ---- Prediction check ------------------------------------
        day_pred = pred_df[pred_df["target_day"] == day_dt].copy()

        if len(day_pred) == 0:
            skipped.append((day_str, "prediction_ledger_no_data_for_day"))
            logger.debug("[adaptive_training_days] skip %s: no prediction data", day_str)
            continue

        # Check each trusted model
        models_incomplete: list[str] = []
        models_nan: list[str] = []
        models_dup: list[str] = []
        models_missing_hours: list[str] = []
        all_models_ok = True

        for model in trusted_models:
            model_pred = day_pred[day_pred["model_name"] == model].copy()

            if len(model_pred) == 0:
                models_incomplete.append(model)
                all_models_ok = False
                continue

            # Deduplicate by hour_business (keep last = latest append)
            if "hour_business" in model_pred.columns:
                model_pred = model_pred.drop_duplicates(
                    subset=["hour_business"], keep="last"
                )

            # Check hour_business completeness (must be exactly 1..24)
            if "hour_business" in model_pred.columns:
                actual_hours = set(model_pred["hour_business"].astype(int).tolist())
            else:
                actual_hours = set()

            missing_hours = _EXPECTED_HOURS - actual_hours
            extra_hours = actual_hours - _EXPECTED_HOURS

            if missing_hours or extra_hours:
                parts = []
                if missing_hours:
                    parts.append(f"missing={sorted(missing_hours)}")
                if extra_hours:
                    parts.append(f"extra={sorted(extra_hours)}")
                models_missing_hours.append(f"{model}({';'.join(parts)})")
                all_models_ok = False
                continue

            # Check NaN in y_pred
            if "y_pred" in model_pred.columns and model_pred["y_pred"].isna().any():
                models_nan.append(model)
                all_models_ok = False
                continue

            # Check duplicate keys on (task, model_name, target_day, business_day, hour_business)
            key_cols = ["task", "model_name", "target_day", "business_day", "hour_business"]
            available_keys = [c for c in key_cols if c in model_pred.columns]
            if available_keys:
                dups = model_pred.duplicated(subset=available_keys, keep=False)
                if dups.any():
                    models_dup.append(model)
                    all_models_ok = False
                    continue

        if not all_models_ok:
            parts = []
            if models_incomplete:
                parts.append(f"missing_models={models_incomplete}")
            if models_nan:
                parts.append(f"models_with_nan_y_pred={models_nan}")
            if models_dup:
                parts.append(f"models_with_duplicate_keys={models_dup}")
            if models_missing_hours:
                parts.append(f"models_with_bad_hours={models_missing_hours}")
            reason = "; ".join(parts) if parts else "prediction_incomplete"
            skipped.append((day_str, reason))
            logger.debug("[adaptive_training_days] skip %s: %s", day_str, reason)
            continue

        # ---- Actual check ----------------------------------------
        day_act = act_df[act_df["target_day"] == day_dt].copy()

        if len(day_act) == 0:
            skipped.append((day_str, "actual_ledger_no_data_for_day"))
            logger.debug("[adaptive_training_days] skip %s: no actual data", day_str)
            continue

        # Deduplicate by hour_business
        if "hour_business" in day_act.columns:
            day_act_dedup = day_act.drop_duplicates(
                subset=["hour_business"], keep="last"
            )
        else:
            day_act_dedup = day_act

        # Check hour_business completeness
        if "hour_business" in day_act_dedup.columns:
            actual_act_hours = set(day_act_dedup["hour_business"].astype(int).tolist())
        else:
            actual_act_hours = set()

        act_missing_hours = _EXPECTED_HOURS - actual_act_hours
        act_extra_hours = actual_act_hours - _EXPECTED_HOURS

        if act_missing_hours or act_extra_hours:
            parts = []
            if act_missing_hours:
                parts.append(f"missing_hours={sorted(act_missing_hours)}")
            if act_extra_hours:
                parts.append(f"extra_hours={sorted(act_extra_hours)}")
            reason = f"actual_incomplete_hours; {'; '.join(parts)}"
            skipped.append((day_str, reason))
            logger.debug("[adaptive_training_days] skip %s: %s", day_str, reason)
            continue

        # Check NaN in y_true
        if "y_true" in day_act_dedup.columns and day_act_dedup["y_true"].isna().any():
            n_nan = int(day_act_dedup["y_true"].isna().sum())
            skipped.append(
                (day_str, f"actual_nan_y_true; {n_nan} NaN values in y_true")
            )
            logger.debug(
                "[adaptive_training_days] skip %s: actual has %d NaN in y_true",
                day_str,
                n_nan,
            )
            continue

        # Check duplicate keys on actual ledger key
        act_key_cols = ["task", "target_day", "business_day", "hour_business"]
        act_available_keys = [c for c in act_key_cols if c in day_act_dedup.columns]
        if act_available_keys:
            act_dups = day_act_dedup.duplicated(subset=act_available_keys, keep=False)
            if act_dups.any():
                skipped.append((day_str, "actual_ledger_duplicate_keys"))
                logger.debug(
                    "[adaptive_training_days] skip %s: duplicate keys in actual ledger",
                    day_str,
                )
                continue

        # ---- Day is complete -------------------------------------
        selected.append(day_str)

    # ── Determine status ──────────────────────────────────────────
    selected_count = len(selected)

    if selected_count >= required_days:
        result["status"] = "COMPLETE_30D"
    elif selected_count >= min_days_for_degraded:
        result["status"] = "DEGRADED_MIN_DAYS"
    elif selected_count > 0:
        result["status"] = "INSUFFICIENT_DAYS"
    else:
        result["status"] = "NO_VALID_DAYS"

    # ── Populate result ───────────────────────────────────────────
    result["selected_days"] = selected
    result["selected_count"] = selected_count
    result["skipped_days"] = skipped
    result["latest_selected_day"] = selected[0] if selected else None
    result["oldest_selected_day"] = selected[-1] if selected else None
    result["training_rows"] = selected_count * n_models * 24
    result["actual_rows"] = selected_count * 24

    # Add warnings
    n_skipped = len(skipped)
    if n_skipped > 0:
        result["warnings"].append(
            f"Skipped {n_skipped} day(s) during scan"
        )

    if result["status"] == "COMPLETE_30D":
        if n_skipped > 0:
            result["warnings"].append(
                f"Required {required_days} days met; {n_skipped} day(s) within "
                f"lookback were incomplete"
            )
    elif result["status"] == "DEGRADED_MIN_DAYS":
        result["warnings"].append(
            f"Found {selected_count} complete training days (required {required_days}); "
            f"proceeding with degraded mode (minimum {min_days_for_degraded} met)"
        )
    elif result["status"] == "INSUFFICIENT_DAYS":
        result["warnings"].append(
            f"Found only {selected_count} complete training days; "
            f"below minimum {min_days_for_degraded}"
        )
    elif result["status"] == "NO_VALID_DAYS":
        if not result["errors"]:
            result["warnings"].append(
                "No valid training days found within lookback window"
            )

    logger.info(
        "[adaptive_training_days] status=%s selected=%d skipped=%d for %s",
        result["status"],
        selected_count,
        n_skipped,
        target_date,
    )

    return result
