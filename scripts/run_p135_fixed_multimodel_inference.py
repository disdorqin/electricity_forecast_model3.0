"""
scripts/run_p135_fixed_multimodel_inference.py — P135: Fixed Multi-Model Inference.

Fixes three critical bugs in P128:
  Fix 1: Each model gets its own feature matrix via build_features_for_model()
  Fix 2: Use enumerate() for prediction indexing, not DataFrame index
  Fix 3: NaN predictions don't get written to trusted ledger

Output: per_model_success_days, per_model_failed_days, per_model_rows, per_model_nan_rate
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.features.model_specific_features import (
    build_features_for_model,
    get_model_schema,
    list_supported_models,
)


# ── Model definitions (from P128, with fixes) ──────────────────────

MODEL_DEFS = {
    "lightgbm_cfg05_dayahead": {
        "path_rel": os.path.join("cfg05_dayahead_lgbm", "cfg05_model.txt"),
        "type": "lightgbm",
        "model_schema_name": "cfg05",
        "trusted": True,
        "quarantined": False,
    },
    "catboost_spike_residual": {
        "path_rel": os.path.join("catboost_spike_residual", "catboost_spike_residual.cbm"),
        "type": "catboost",
        "model_schema_name": "catboost_spike_residual",
        "trusted": True,
        "quarantined": False,
    },
    "catboost_sota": {
        "path_rel": os.path.join("catboost_sota", "catboost_sota_model.cbm"),
        "type": "catboost",
        "model_schema_name": "catboost_sota",
        "trusted": False,
        "quarantined": True,
    },
}


def _load_model(model_name: str, model_def: dict, model_dir: str):
    """Load a single model artifact. Returns (model_obj, error_string_or_None)."""
    artifact_path = os.path.join(model_dir, model_def["path_rel"])
    if not os.path.isfile(artifact_path):
        return None, f"FILE_MISSING:{artifact_path}"

    try:
        if model_def["type"] == "catboost":
            from catboost import CatBoost
            m = CatBoost()
            m.load_model(artifact_path)
        elif model_def["type"] == "lightgbm":
            import lightgbm as lgb
            m = lgb.Booster(model_file=artifact_path)
        else:
            return None, f"UNKNOWN_TYPE:{model_def['type']}"
        return m, None
    except Exception as e:
        return None, f"LOAD_FAILED:{e}"


def _predict_single_model(
    model_obj,
    model_name: str,
    model_def: dict,
    day_data: pd.DataFrame,
    full_raw: pd.DataFrame,
    precomputed_v3: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run inference for a single model on a single day.

    FIX 1: Each model gets its own feature matrix via build_features_for_model().
    FIX 2: Use enumerate() for prediction indexing, not DataFrame index.

    Returns
    -------
    predictions : np.ndarray
        Array of predictions (length = len(day_data)).
    info : dict
        Diagnostic info about the prediction.
    """
    info: dict[str, Any] = {
        "model_name": model_name,
        "n_rows": len(day_data),
        "feature_schema": model_def["model_schema_name"],
    }

    # FIX 1: Build model-specific features
    schema_name = model_def["model_schema_name"]
    try:
        X, report, score = build_features_for_model(
            full_raw,
            schema_name,
            precomputed_features=precomputed_v3,
        )
        info["schema_match_score"] = score
        info["feature_report"] = {
            "present_count": report.get("present_count", 0),
            "missing_count": report.get("missing_count", 0),
        }
    except Exception as e:
        info["error"] = f"FEATURE_BUILD_FAILED:{e}"
        return np.full(len(day_data), np.nan), info

    if score == 0.0:
        info["error"] = "SCHEMA_MATCH_ZERO"
        return np.full(len(day_data), np.nan), info

    # Filter to the target day's rows
    # We need to match day_data rows to the feature matrix rows
    # The feature matrix was built on full_raw, so we need to align
    day_ds_set = set(pd.to_datetime(day_data["ds"]).dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Try to find matching rows in X by ds
    if "ds" in X.columns:
        X_ds_str = pd.to_datetime(X["ds"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        mask = X_ds_str.isin(day_ds_set)
        X_day = X[mask]
    else:
        # Fallback: use last len(day_data) rows
        X_day = X.tail(len(day_data))

    if len(X_day) == 0:
        info["error"] = "NO_FEATURE_ROWS_MATCH_DAY"
        return np.full(len(day_data), np.nan), info

    # Extract only the feature columns (drop business columns)
    from data.features.model_specific_features import BUSINESS_COLUMNS
    feature_cols = [c for c in X_day.columns if c not in BUSINESS_COLUMNS]
    X_values = X_day[feature_cols].fillna(0).values

    # Run prediction
    try:
        pred_raw = model_obj.predict(X_values)
        if isinstance(pred_raw, np.ndarray):
            pred_flat = pred_raw.flatten()
        else:
            pred_flat = np.full(len(day_data), float(pred_raw))
    except Exception as e:
        info["error"] = f"PREDICT_FAILED:{e}"
        return np.full(len(day_data), np.nan), info

    # FIX 2: Ensure prediction length matches day_data length
    if len(pred_flat) != len(day_data):
        # Pad or truncate
        if len(pred_flat) < len(day_data):
            pred_flat = np.concatenate([
                pred_flat,
                np.full(len(day_data) - len(pred_flat), np.nan),
            ])
        else:
            pred_flat = pred_flat[:len(day_data)]

    info["n_predictions"] = len(pred_flat)
    info["n_nan"] = int(np.isnan(pred_flat).sum())

    return pred_flat, info


def run_fixed_multimodel_inference(
    raw_data_path: str = "",
    model_dir: str = "",
    output_dir: str = "",
    day_start: str = "2025-01-01",
    day_end: str = "2025-01-31",
    max_days: int = 0,
) -> dict[str, Any]:
    """Run fixed multi-model inference for 2025.

    Parameters
    ----------
    raw_data_path : str
        Path to raw CSV (GBK encoded).
    model_dir : str
        Directory containing model artifacts.
    output_dir : str
        Output directory.
    day_start : str
        First day to predict.
    day_end : str
        Last day to predict.
    max_days : int
        Maximum number of days to process (0 = unlimited).

    Returns
    -------
    dict
        Full result with per-model metrics and ledger.
    """
    if not raw_data_path:
        raw_data_path = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
    if not model_dir:
        model_dir = os.path.join(
            REPO_ROOT, ".local_artifacts", "p31_p40_multimodel_fusion", "models"
        )
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, ".local_artifacts", "p135_fixed_multimodel")
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    result: dict[str, Any] = {
        "phase": "P135",
        "title": "Fixed Multi-Model Inference",
        "status": "STARTED",
        "fixes_applied": [
            "FIX1: per-model feature matrices via build_features_for_model()",
            "FIX2: enumerate() for prediction indexing",
            "FIX3: NaN predictions filtered from trusted ledger",
        ],
        "models": {},
        "per_model_success_days": {},
        "per_model_failed_days": {},
        "per_model_rows": {},
        "per_model_nan_rate": {},
    }

    # Load raw data
    try:
        raw = pd.read_csv(raw_data_path, encoding="gbk")
        from data.features.model_specific_features import normalize_raw_columns
        raw = normalize_raw_columns(raw)
        raw["ds"] = pd.to_datetime(raw["ds"] if "ds" in raw.columns else raw.iloc[:, 0])
        from data.business_day import add_business_time_columns
        raw = add_business_time_columns(raw, timestamp_col="ds")
        raw = raw.sort_values("ds").reset_index(drop=True)
    except Exception as e:
        result["status"] = f"DATA_LOAD_FAILED:{e}"
        _save_result(result, output_dir)
        return result

    # Load models
    loaded_models = {}
    for name, mdef in MODEL_DEFS.items():
        model_obj, err = _load_model(name, mdef, model_dir)
        if model_obj is not None:
            loaded_models[name] = {"model": model_obj, "def": mdef}
            result["models"][name] = {
                "status": "LOADED",
                "trusted": mdef["trusted"],
                "quarantined": mdef["quarantined"],
            }
        else:
            result["models"][name] = {"status": err}

    if not loaded_models:
        result["status"] = "NO_MODELS_LOADED"
        _save_result(result, output_dir)
        return result

    # Pre-compute v3 features on the full dataset (expensive but done once)
    precomputed_v3 = None
    try:
        from data.features.model_specific_features import _build_v3_features_from_raw
        precomputed_v3 = _build_v3_features_from_raw(raw)
        logger.info(f"Pre-computed v3 features: {precomputed_v3.shape}")
    except Exception as e:
        logger.warning(f"Could not pre-compute v3 features: {e}")

    # Generate predictions day by day
    days = pd.date_range(day_start, day_end)
    if max_days > 0:
        days = days[:max_days]

    all_rows = []
    per_model_success = {name: 0 for name in loaded_models}
    per_model_failed = {name: 0 for name in loaded_models}
    per_model_nan_count = {name: 0 for name in loaded_models}
    per_model_total = {name: 0 for name in loaded_models}

    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        day_mask = raw["business_day"].astype(str) == day_str
        day_data = raw[day_mask].copy()

        if len(day_data) == 0:
            continue

        for name, lm in loaded_models.items():
            model_obj = lm["model"]
            mdef = lm["def"]

            pred_flat, info = _predict_single_model(
                model_obj, name, mdef, day_data, raw,
                precomputed_v3=precomputed_v3,
            )

            n_nan = int(np.isnan(pred_flat).sum())
            per_model_total[name] += len(day_data)
            per_model_nan_count[name] += n_nan

            # FIX 3: NaN predictions tracking
            if n_nan == len(pred_flat):
                per_model_failed[name] += 1
            else:
                per_model_success[name] += 1

            # Build output rows — only write non-NaN predictions for trusted models
            # For untrusted/quarantined models, write all (including NaN for transparency)
            for idx, (_, row) in enumerate(day_data.iterrows()):
                y_pred_val = float(pred_flat[idx]) if idx < len(pred_flat) else np.nan

                # FIX 3: Skip NaN for trusted models
                if np.isnan(y_pred_val) and mdef["trusted"]:
                    continue

                hb = row.get("hour_business", 0)
                all_rows.append({
                    "task": "dayahead",
                    "model_name": name,
                    "target_day": day_str,
                    "business_day": str(row.get("business_day", day_str)),
                    "ds": row["ds"],
                    "hour_business": int(hb) if not pd.isna(hb) else 0,
                    "period": str(row.get("period", "")),
                    "y_pred": y_pred_val,
                    "source_confidence": 1.0 if mdef["trusted"] else 0.5,
                    "model_version": "p135_v3_fixed",
                })

    # Build ledger
    ledger = pd.DataFrame(all_rows)
    result["total_rows"] = len(ledger)

    # Per-model metrics
    for name in loaded_models:
        total = per_model_total.get(name, 0)
        nan_count = per_model_nan_count.get(name, 0)
        result["per_model_success_days"][name] = per_model_success[name]
        result["per_model_failed_days"][name] = per_model_failed[name]
        result["per_model_rows"][name] = total
        result["per_model_nan_rate"][name] = round(
            nan_count / max(total, 1), 4
        )

    # Save ledger
    ledger_path = os.path.join(output_dir, "p135_fixed_prediction_ledger.csv")
    ledger.to_csv(ledger_path, index=False)
    result["ledger_path"] = ledger_path

    result["elapsed_seconds"] = round(time.time() - t_start, 2)
    result["day_count"] = ledger["target_day"].nunique() if len(ledger) > 0 else 0
    result["model_count"] = len(loaded_models)

    # Determine status
    any_success = any(v > 0 for v in per_model_success.values())
    if any_success and all(v > 0 for v in per_model_success.values()):
        result["status"] = "FIXED_MULTIMODEL_ALL_OK"
    elif any_success:
        result["status"] = "FIXED_MULTIMODEL_PARTIAL"
    else:
        result["status"] = "FIXED_MULTIMODEL_ALL_FAILED"

    _save_result(result, output_dir)
    return result


def _save_result(result: dict, output_dir: str) -> None:
    output_path = os.path.join(output_dir, "p135_fixed_multimodel_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    result["output_path"] = output_path


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="P135: Fixed Multi-Model Inference")
    parser.add_argument("--raw-data", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--day-start", default="2025-01-01")
    parser.add_argument("--day-end", default="2025-01-31")
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_fixed_multimodel_inference(
        raw_data_path=args.raw_data,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        day_start=args.day_start,
        day_end=args.day_end,
        max_days=args.max_days,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n=== P135: Fixed Multi-Model Inference ===")
        print(f"Status: {result['status']}")
        print(f"Models loaded: {result.get('model_count', 0)}")
        print(f"Total rows: {result.get('total_rows', 0)}")
        for name in result.get("per_model_success_days", {}):
            s = result["per_model_success_days"][name]
            f_ = result["per_model_failed_days"][name]
            r = result["per_model_rows"][name]
            nr = result["per_model_nan_rate"][name]
            print(f"  {name}: success={s}d fail={f_}d rows={r} nan_rate={nr}")
        print(f"Output: {result.get('output_path', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
