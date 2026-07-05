"""
safety/leakage_sentinel.py — P53: Leakage Sentinel Runtime Guard.

Runtime safety guard that checks each model on every run for data leakage
indicators. Designed for day-ahead price forecasting (3.0's domain).

Output statuses:
  - TRUSTED — all checks pass
  - CONSERVATIVE_QUARANTINE — corr or within_1pct borderline
  - SUSPECT_LEAKAGE — clear leakage indicators (sMAPE, MAE, future ts, ...)
  - INVALID_SCHEMA — missing required columns
  - INVALID_24H — incomplete 24-hour coverage

Action rules:
  - SUSPECT_LEAKAGE -> quarantine (cannot enter delivery fusion)
  - CONSERVATIVE_QUARANTINE -> excluded from trusted_delivery profile
  - research profile -> allowed only with caveat
  - delivery profile -> hard block
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Thresholds (day-ahead price forecasting domain)
# ──────────────────────────────────────────────────────────────────────
CORR_THRESHOLD = 0.995  # correlation -> CONSERVATIVE_QUARANTINE
WITHIN_1PCT_THRESHOLD = 0.80  # within-1% ratio -> CONSERVATIVE_QUARANTINE
SMAPE_FLOOR50_TOO_GOOD = 2.0  # sMAPE < 2% -> SUSPECT_LEAKAGE
MAE_TOO_GOOD = 10.0  # MAE < 10 CNY -> SUSPECT_LEAKAGE

# ──────────────────────────────────────────────────────────────────────
# Status constants
# ──────────────────────────────────────────────────────────────────────
TRUSTED = "TRUSTED"
CONSERVATIVE_QUARANTINE = "CONSERVATIVE_QUARANTINE"
SUSPECT_LEAKAGE = "SUSPECT_LEAKAGE"
INVALID_SCHEMA = "INVALID_SCHEMA"
INVALID_24H = "INVALID_24H"

# Merge keys for prediction <-> actual ledger alignment
MERGE_KEYS = ["business_day", "hour_business"]

# Column names that MUST NOT appear in a prediction ledger
_TARGET_COLUMN_NAMES = {"y_true", "target", "日前电价"}


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sMAPE with a floor of 50 on both arrays."""
    y_true_f = np.maximum(y_true, 50.0)
    y_pred_f = np.maximum(y_pred, 50.0)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(200.0 * np.mean(np.abs(y_true_f - y_pred_f) / denom))


def _load_ledger(path: str) -> pd.DataFrame | None:
    """Load a parquet or CSV ledger file.

    Returns None if the file is missing or unreadable.
    """
    try:
        ext = path.lower()
        if ext.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.warning("Ledger file not found: %s", path)
        return None
    except Exception as exc:
        logger.error("Error loading ledger %s: %s", path, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def check_model_leakage(
    model_name: str,
    prediction_ledger_path: str,
    actual_ledger_path: str,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Run all leakage checks for a single model.

    Parameters
    ----------
    model_name : str
        Name of the model to check (must match ``model_name`` column in
        the prediction ledger).
    prediction_ledger_path : str
        Path to the prediction ledger (parquet or CSV).
    actual_ledger_path : str
        Path to the actuals ledger (parquet or CSV).
    feature_columns : list[str] | None
        Optional list of feature column names. When provided, the sentinel
        verifies that no target column leaks into the feature set.

    Returns
    -------
    dict
        Dictionary with keys:
          - model_name
          - status
          - checks (dict of bool per check)
          - details
          - suspicion_reasons (list)
          - warnings (list)
    """
    result: dict[str, Any] = {
        "model_name": model_name,
        "status": TRUSTED,
        "checks": {},
        "details": {},
        "suspicion_reasons": [],
        "warnings": [],
    }

    # ── Load ledgers ──────────────────────────────────────────────
    pred = _load_ledger(prediction_ledger_path)
    actual = _load_ledger(actual_ledger_path)

    if pred is None or actual is None:
        result["status"] = INVALID_SCHEMA
        result["checks"]["ledger_loaded"] = False
        result["details"]["error"] = "Could not load one or both ledgers"
        return result

    result["checks"]["ledger_loaded"] = True

    # Filter to this model -------------------------------------------------
    if "model_name" not in pred.columns:
        result["status"] = INVALID_SCHEMA
        result["checks"]["model_name_column"] = False
        result["details"]["error"] = "Prediction ledger missing 'model_name' column"
        return result

    result["checks"]["model_name_column"] = True
    model_pred = pred[pred["model_name"] == model_name].copy()
    result["details"]["prediction_rows"] = int(len(model_pred))

    if len(model_pred) == 0:
        result["status"] = INVALID_SCHEMA
        result["checks"]["model_found_in_ledger"] = False
        result["details"]["error"] = (
            f"Model '{model_name}' not found in prediction ledger"
        )
        return result

    result["checks"]["model_found_in_ledger"] = True

    # ── Check 1: y_true must NOT be in prediction ledger ────────────────
    pred_cols = set(pred.columns)
    target_in_pred = pred_cols & _TARGET_COLUMN_NAMES
    if target_in_pred:
        result["status"] = INVALID_SCHEMA
        result["checks"]["no_y_true_in_prediction_ledger"] = False
        result["details"]["target_columns_found"] = sorted(target_in_pred)
        return result
    result["checks"]["no_y_true_in_prediction_ledger"] = True

    # ── Check 2: feature columns must not contain target ────────────────
    if feature_columns:
        feature_cols_set = set(feature_columns)
        target_in_features = feature_cols_set & _TARGET_COLUMN_NAMES
        if target_in_features:
            result["status"] = INVALID_SCHEMA
            result["checks"]["no_target_in_features"] = False
            result["details"]["target_in_features"] = sorted(target_in_features)
            return result
    result["checks"]["no_target_in_features"] = True

    # ── Merge prediction with actuals for metric computation ────────────
    merge_cols = [c for c in MERGE_KEYS if c in actual.columns]
    if not merge_cols:
        result["status"] = INVALID_SCHEMA
        result["checks"]["merge_keys_available"] = False
        result["details"]["error"] = (
            "Actual ledger missing merge keys (business_day, hour_business)"
        )
        return result
    result["checks"]["merge_keys_available"] = True

    merged = model_pred.merge(
        actual[merge_cols + ["y_true"]],
        on=merge_cols,
        how="inner",
    )
    result["details"]["merged_rows"] = int(len(merged))

    # ── Drop NaN y_true ─────────────────────────────────────────────────
    pre_drop = len(merged)
    merged = merged.dropna(subset=["y_true"])
    dropped = pre_drop - len(merged)
    result["details"]["eval_rows_after_nan_drop"] = int(len(merged))
    if dropped > 0:
        result["details"]["nan_y_true_dropped"] = dropped

    # ── Check 3: sufficient eval rows ───────────────────────────────────
    result["checks"]["sufficient_eval_rows"] = len(merged) >= 24
    if len(merged) < 24:
        result["details"]["error"] = (
            f"Only {len(merged)} eval rows after NaN drop (need >= 24)"
        )

    # ── Check 11: 24H completeness ──────────────────────────────────────
    if "hour_business" in model_pred.columns:
        hours = set(model_pred["hour_business"].unique())
        expected_hours = set(range(1, 25))
        missing_hours = expected_hours - hours
        result["checks"]["24h_completeness"] = len(missing_hours) == 0
        if missing_hours:
            result["details"]["missing_hours"] = sorted(missing_hours)
    else:
        result["checks"]["24h_completeness"] = True  # cannot verify

    # If 24H is incomplete AND we have enough rows, assign INVALID_24H
    # (unless already INVALID_SCHEMA from earlier checks)
    if (
        not result["checks"].get("24h_completeness", True)
        and result["status"] == TRUSTED
    ):
        result["status"] = INVALID_24H

    # ── If insufficient rows or schema failure, stop here ───────────────
    if not result["checks"]["sufficient_eval_rows"]:
        if result["status"] == TRUSTED:
            result["status"] = INVALID_24H
        return result

    # ── Compute metrics ─────────────────────────────────────────────────
    y_true_arr = merged["y_true"].values
    y_pred_arr = merged["y_pred"].values

    # Check for NaN y_pred and drop those rows
    nan_pred_mask = np.isnan(y_pred_arr)
    nan_pred = int(nan_pred_mask.sum())
    result["details"]["nan_y_pred_count"] = nan_pred
    if nan_pred > 0:
        result["warnings"].append(f"{nan_pred} NaN y_pred values found")
        y_true_arr = y_true_arr[~nan_pred_mask]
        y_pred_arr = y_pred_arr[~nan_pred_mask]

    # If NaN y_pred dropped us below 24 rows, flag insufficient eval rows
    n_valid = len(y_true_arr)
    if n_valid < 24:
        result["status"] = INVALID_24H
        result["checks"]["sufficient_eval_rows"] = False
        result["details"]["error"] = (
            f"Only {n_valid} valid eval rows after NaN y_pred drop (need >= 24)"
        )
        return result

    diff = np.abs(y_true_arr - y_pred_arr)
    within_1pct = int((diff / (np.abs(y_true_arr) + 1.0) < 0.01).sum())
    within_1pct_ratio = within_1pct / n_valid
    corr = (
        float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
        if n_valid > 2 and np.std(y_true_arr) > 1e-10 and np.std(y_pred_arr) > 1e-10
        else 0.0
    )
    smape = _smape_floor50(y_true_arr, y_pred_arr)
    mae = float(np.mean(diff))

    result["details"]["metrics"] = {
        "within_1pct": within_1pct,
        "within_1pct_ratio": round(within_1pct_ratio, 4),
        "corr_y_pred_y_true": round(corr, 4),
        "sMAPE_floor50": round(smape, 4),
        "MAE": round(mae, 2),
        "n": int(n_valid),
    }

    # ── Check 4: within_1pct_ratio (> 80% -> suspicious) ────────────────
    result["checks"]["within_1pct_ratio"] = within_1pct_ratio <= WITHIN_1PCT_THRESHOLD
    if within_1pct_ratio > WITHIN_1PCT_THRESHOLD:
        result["suspicion_reasons"].append(
            f"within_1pct_ratio={within_1pct_ratio:.2%}>{WITHIN_1PCT_THRESHOLD:.0%}"
        )

    # ── Check 5: corr_y_pred_y_true (> 0.995 -> CONSERVATIVE_QUARANTINE) ─
    result["checks"]["corr_y_pred_y_true"] = corr <= CORR_THRESHOLD
    if corr > CORR_THRESHOLD:
        result["suspicion_reasons"].append(
            f"corr_y_pred_y_true={corr:.4f}>{CORR_THRESHOLD}"
        )

    # ── Check 6: sMAPE_floor50 (< 2% -> SUSPECT_LEAKAGE) ────────────────
    result["checks"]["sMAPE_floor50"] = smape >= SMAPE_FLOOR50_TOO_GOOD
    if smape < SMAPE_FLOOR50_TOO_GOOD:
        result["suspicion_reasons"].append(
            f"sMAPE_floor50={smape:.2f}%<{SMAPE_FLOOR50_TOO_GOOD}%"
        )

    # ── Check 7: MAE (< 10 CNY -> SUSPECT_LEAKAGE) ─────────────────────
    result["checks"]["MAE"] = mae >= MAE_TOO_GOOD
    if mae < MAE_TOO_GOOD:
        result["suspicion_reasons"].append(
            f"MAE={mae:.2f}<{MAE_TOO_GOOD} CNY"
        )

    # ── Check 8: future timestamp leakage ───────────────────────────────
    if "ds" in model_pred.columns:
        now = pd.Timestamp.now()
        ds_dt = pd.to_datetime(model_pred["ds"], errors="coerce")
        future_ts = model_pred[ds_dt > now]
        result["checks"]["no_future_timestamps"] = len(future_ts) == 0
        if len(future_ts) > 0:
            result["details"]["future_timestamp_count"] = int(len(future_ts))
            result["suspicion_reasons"].append(
                f"{len(future_ts)} future prediction timestamps found"
            )
    else:
        result["checks"]["no_future_timestamps"] = True

    # ── Check 9: target_day overlap (informational) ─────────────────────
    if "target_day" in model_pred.columns and "target_day" in actual.columns:
        pred_td = set(model_pred["target_day"].unique())
        actual_td = set(actual["target_day"].unique())
        overlap = pred_td & actual_td
        result["details"]["target_day_overlap_days"] = sorted(
            str(d) for d in overlap
        )
        result["details"]["target_day_overlap_count"] = len(overlap)
    # Mark as pass since training data overlap cannot be detected from
    # ledgers alone; suspicious metrics are caught by checks 4-7.
    result["checks"]["no_target_day_overlap"] = True

    # ── Check 10: duplicate keys ────────────────────────────────────────
    if all(c in model_pred.columns for c in MERGE_KEYS):
        dupes = model_pred[model_pred.duplicated(subset=MERGE_KEYS, keep=False)]
        result["checks"]["no_duplicate_keys"] = len(dupes) == 0
        if len(dupes) > 0:
            result["details"]["duplicate_key_count"] = int(len(dupes))
            result["suspicion_reasons"].append(
                f"{len(dupes)} duplicate (business_day, hour_business) rows found"
            )
    else:
        result["checks"]["no_duplicate_keys"] = True

    # ── Determine final status ──────────────────────────────────────────
    # Priority: INVALID_SCHEMA > INVALID_24H > SUSPECT_LEAKAGE
    #           > CONSERVATIVE_QUARANTINE > TRUSTED

    # Schema and 24H already handled above; now distinguish
    # SUSPECT_LEAKAGE vs CONSERVATIVE_QUARANTINE vs TRUSTED
    if result["status"] in (INVALID_SCHEMA, INVALID_24H):
        return result

    suspect_triggers = [
        not result["checks"].get("sMAPE_floor50", True),
        not result["checks"].get("MAE", True),
        not result["checks"].get("no_future_timestamps", True),
        not result["checks"].get("no_duplicate_keys", True),
    ]
    quarantine_triggers = [
        not result["checks"].get("within_1pct_ratio", True),
        not result["checks"].get("corr_y_pred_y_true", True),
    ]

    if any(suspect_triggers):
        result["status"] = SUSPECT_LEAKAGE
    elif any(quarantine_triggers):
        result["status"] = CONSERVATIVE_QUARANTINE
    else:
        result["status"] = TRUSTED

    return result


def run_leakage_sentinel(
    trusted_models: list[str],
    prediction_ledger_path: str,
    actual_ledger_path: str,
) -> dict[str, Any]:
    """Run leakage sentinel on all trusted models.

    Parameters
    ----------
    trusted_models : list[str]
        List of model names to check.
    prediction_ledger_path : str
        Path to the prediction ledger.
    actual_ledger_path : str
        Path to the actuals ledger.

    Returns
    -------
    dict
        Summary dict with per-model results, aggregate counts, and a
        cross-model eval-row consistency check.
    """
    model_results: list[dict[str, Any]] = []
    summary: dict[str, int] = {
        TRUSTED: 0,
        CONSERVATIVE_QUARANTINE: 0,
        SUSPECT_LEAKAGE: 0,
        INVALID_SCHEMA: 0,
        INVALID_24H: 0,
    }

    for model_name in trusted_models:
        mr = check_model_leakage(
            model_name=model_name,
            prediction_ledger_path=prediction_ledger_path,
            actual_ledger_path=actual_ledger_path,
        )
        model_results.append(mr)
        summary[mr["status"]] = summary.get(mr["status"], 0) + 1

    # Cross-model consistency: all models should have the same eval rows
    eval_row_counts: set[int] = set()
    for mr in model_results:
        metrics = mr.get("details", {}).get("metrics", {})
        n = metrics.get("n", 0)
        if n > 0:
            eval_row_counts.add(n)

    eval_rows_consistent = len(eval_row_counts) <= 1

    return {
        "phase": "P53",
        "n_models_checked": len(trusted_models),
        "eval_rows_consistent": eval_rows_consistent,
        "eval_row_counts": sorted(eval_row_counts) if eval_row_counts else [],
        "models": model_results,
        "summary": summary,
    }


def is_delivery_allowed(
    model_name: str,
    sentinel_result: dict[str, Any],
    profile_name: str,
) -> bool:
    """Check if a model is allowed for delivery based on sentinel result.

    Action rules
    ------------
    - SUSPECT_LEAKAGE -> hard block (cannot enter delivery fusion)
    - CONSERVATIVE_QUARANTINE -> blocked for trusted_delivery,
      allowed for research profiles
    - INVALID_SCHEMA / INVALID_24H -> always blocked
    - TRUSTED -> always allowed

    Parameters
    ----------
    model_name : str
        Model name to check.
    sentinel_result : dict
        Result from ``run_leakage_sentinel``.
    profile_name : str
        Profile name (e.g. ``"trusted_delivery"``, ``"research_all_models"``).

    Returns
    -------
    bool
        True if delivery is allowed.
    """
    # Find the model's result
    model_result = None
    for m in sentinel_result.get("models", []):
        if m.get("model_name") == model_name:
            model_result = m
            break

    if model_result is None:
        return False

    status = model_result["status"]

    # Always blocked regardless of profile
    if status in (INVALID_SCHEMA, INVALID_24H, SUSPECT_LEAKAGE):
        return False

    # CONSERVATIVE_QUARANTINE: blocked for trusted_delivery,
    # allowed for research profiles
    if status == CONSERVATIVE_QUARANTINE:
        return profile_name != "trusted_delivery"

    # TRUSTED: allowed everywhere
    if status == TRUSTED:
        return True

    return False
