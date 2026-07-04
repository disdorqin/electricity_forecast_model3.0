"""
scripts/run_p15_cfg05_24h_smoke_and_eval.py — P15 orchestration.

Chains the fixed 24-hour pipeline and optional historical evaluation:

    raw CSV → contract check → train/export → REAL smoke
    → hour24 completeness (features + predictions)
    → optional historical eval → final status report

Usage::

    # Full pipeline for a single target day
    python -m scripts.run_p15_cfg05_24h_smoke_and_eval \\
        --raw-data /path/to/shandong_pmos_hourly.csv \\
        --target-day 2026-07-01

    # With historical eval over the past 7 complete business days
    python -m scripts.run_p15_cfg05_24h_smoke_and_eval \\
        --raw-data /path/to/shandong_pmos_hourly.csv \\
        --target-day 2026-07-01 --eval-days 7

    # JSON + strict
    python -m scripts.run_p15_cfg05_24h_smoke_and_eval \\
        --raw-data /path/to/shandong_pmos_hourly.csv \\
        --target-day 2026-06-30 --eval-days 5 --json --strict

    # No training, use existing artifacts
    python -m scripts.run_p15_cfg05_24h_smoke_and_eval \\
        --raw-data /path/to/shandong_pmos_hourly.csv \\
        --model .local_artifacts/p15_cfg05/cfg05_model.txt \\
        --features .local_artifacts/p15_cfg05/cfg05_full_features.csv

Options::

    --raw-data PATH             Path to raw Chinese CSV (required).
    --source-repo PATH          Path to epf-sota-experiment.
    --target-day YYYY-MM-DD     Target day for primary prediction (default: 2026-07-01).
    --train-window-days N       Training window in days (default: 90).
    --eval-days N               Number of historical days to evaluate (optional).
    --eval-start YYYY-MM-DD     Explicit eval range start (overrides --eval-days).
    --eval-end YYYY-MM-DD       Explicit eval range end (default: target-day).
    --model PATH                Path to existing model (skip train).
    --features PATH             Path to existing full feature CSV (skip train).
    --work-dir PATH             Local work dir (default: .local_artifacts/p15_cfg05).
    --force                     Overwrite existing output files.
    --json                      Output JSON report.
    --strict                    Exit non-zero on any blocker.
    --verbose, -v               Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

from artifacts.dayahead_window import (
    day_ahead_mask,
    filter_dayahead,
    get_business_day_info,
    get_dayahead_window,
)
from artifacts.readiness import LOADABLE, REAL_READY, SCHEMA_READY
from scripts.check_cfg05_hour24_completeness import (
    COMPLETE_24H,
    INCOMPLETE_23H,
    check_cfg05_hour24_completeness,
)
from scripts.check_cfg05_raw_data_contract import (
    RAW_DATA_MISSING,
    RAW_DATA_VALID,
    check_cfg05_raw_data_contract,
)
from scripts.inspect_cfg05_raw_csv_schema import inspect_cfg05_raw_csv_schema

logger = logging.getLogger(__name__)

# ── Safe paths ─────────────────────────────────────────────────────────────
_ALLOWED_WORK_DIRS = (".local_artifacts",)
_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "ledgers/", "reports/local/")

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p15_cfg05")
_DEFAULT_SOURCE_REPO = os.path.join(
    ".local_artifacts", "source_repos", "epf-sota-experiment",
)


def _path_is_safe(path: str) -> bool:
    """Check path is under an ignored, allowed directory."""
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


# ── Metric computation ─────────────────────────────────────────────────────


def compute_smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
) -> float:
    """Compute sMAPE with a floor of *floor* on both actuals and predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    floor : float
        Minimum value to floor (default 50.0).

    Returns
    -------
    float
        sMAPE in percent (e.g. 11.48 for 11.48%).
    """
    y_true_f = np.maximum(y_true, floor)
    y_pred_f = np.maximum(y_pred, floor)
    denominator = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denominator > 1e-10
    if not mask.any():
        return float("nan")
    return float(200 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denominator[mask]))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute sMAPE_floor50, MAE, RMSE.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict with metric keys.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "sMAPE_floor50": float("nan"),
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "n_observations": 0,
        }

    return {
        "sMAPE_floor50": compute_smape_floor50(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "n_observations": int(len(y_true)),
    }


def _load_raw_with_y_true(raw_data: str) -> pd.DataFrame:
    """Load raw Chinese CSV and extract y_true (日前电价) with ds alignment.

    Parameters
    ----------
    raw_data : str
        Path to raw Chinese CSV.

    Returns
    -------
    pd.DataFrame with ds, y_true columns.
    """
    # Try GBK first, then UTF-8
    try:
        raw_df = pd.read_csv(raw_data, encoding="gbk")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(raw_data, encoding="utf-8")

    raw_df["ds"] = pd.to_datetime(raw_df["时刻"])
    result = raw_df[["ds", "日前电价"]].copy()
    result.columns = ["ds", "y_true"]
    return result


# ── P15 orchestration ──────────────────────────────────────────────────────


def run_p15_cfg05_24h_smoke_and_eval(
    raw_data: Optional[str] = None,
    source_repo: Optional[str] = None,
    target_day: Optional[str] = None,
    train_window_days: int = 90,
    eval_days: Optional[int] = None,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
    model_path: Optional[str] = None,
    features_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full P15 pipeline: train/export → 24h smoke → eval.

    Parameters
    ----------
    raw_data : str, optional
        Path to raw Chinese CSV.
    source_repo : str, optional
        Path to epf-sota-experiment.
    target_day : str, optional
        Primary target day (YYYY-MM-DD).
    train_window_days : int
        Training window in days.
    eval_days : int, optional
        Number of historical days to evaluate.
    eval_start : str, optional
        Explicit eval range start (overrides eval_days).
    eval_end : str, optional
        Explicit eval range end.
    model_path : str, optional
        Path to existing model (skip training).
    features_path : str, optional
        Path to existing full feature CSV (skip training).
    work_dir : str, optional
        Local work dir.
    force : bool
        Overwrite existing output files.

    Returns
    -------
    dict with complete P15 summary.
    """
    source_repo = source_repo or _DEFAULT_SOURCE_REPO
    target_day = target_day or "2026-07-01"
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    result: dict[str, Any] = {
        "raw_data_status": "NOT_CHECKED",
        "source_repo_status": "NOT_CHECKED",
        "train_attempted": False,
        "train_done": False,
        "model_path": model_path or os.path.join(work_dir, "cfg05_model.txt"),
        "features_path": features_path or os.path.join(work_dir, "cfg05_full_features.csv"),
        "primary_prediction": {
            "target_day": target_day,
            "attempted": False,
            "prediction_rows": 0,
            "feature_hours_status": None,
            "prediction_hours_status": None,
            "validator_passed": False,
        },
        "eval_attempted": False,
        "eval_summary": None,
        "eval_metrics": None,
        "final_status": None,
        "hour24_fix_applied": True,
        "reason_codes": [],
    }

    # ── Step 1: Check raw data ──
    if not raw_data or not os.path.isfile(raw_data or ""):
        result["raw_data_status"] = RAW_DATA_MISSING
        result["final_status"] = "CFG05_RAW_DATA_MISSING"
        result["reason_codes"].append("RAW_DATA_MISSING_OR_NOT_FOUND")
        return result

    result["raw_data_status"] = RAW_DATA_VALID
    result["reason_codes"].append(f"RAW_DATA_FOUND: {raw_data}")

    # ── Step 2: Check source repo ──
    if not os.path.isdir(source_repo):
        result["source_repo_status"] = "MISSING"
        result["reason_codes"].append(f"SOURCE_REPO_MISSING: {source_repo}")
        result["final_status"] = "CFG05_SOURCE_REPO_MISSING"
        return result

    result["source_repo_status"] = "PRESENT"
    result["reason_codes"].append(f"SOURCE_REPO_FOUND: {source_repo}")

    # ── Step 3: Train/export (or use existing) ──
    model_exists = model_path and os.path.isfile(model_path or "")
    features_exist = features_path and os.path.isfile(features_path or "")

    if model_exists and features_exist and not force:
        result["reason_codes"].append("USING_EXISTING_MODEL_AND_FEATURES")
        result["train_attempted"] = False
        logger.info("Using existing model=%s features=%s", model_path, features_path)
    else:
        result["train_attempted"] = True
        logger.info("Training/exporting model for target_day=%s", target_day)

        # Import source modules
        try:
            import importlib

            if source_repo not in sys.path:
                sys.path.insert(0, source_repo)

            data_loader = _import_source_module(source_repo, "src.common.data_loader")
            feature_builder = _import_source_module(
                source_repo, "src.common.feature_builder_dayahead"
            )
            result["reason_codes"].append("SOURCE_MODULES_IMPORTED")
        except Exception as e:
            result["reason_codes"].append(f"SOURCE_MODULE_IMPORT_FAILED: {e}")
            result["final_status"] = "CFG05_MODULE_IMPORT_FAILED"
            return result

        # Load raw data
        try:
            df = data_loader.load_data(raw_data, target="dayahead")
            result["reason_codes"].append(f"RAW_DATA_LOADED: {len(df)} rows")
        except Exception as e:
            result["reason_codes"].append(f"RAW_DATA_LOAD_FAILED: {e}")
            result["final_status"] = "CFG05_DATA_LOAD_FAILED"
            return result

        # Build full features (all timestamps, not just target day)
        try:
            df_feat = feature_builder.build_features_dayahead(df, use_extended=True)
            result["reason_codes"].append(f"FULL_FEATURES_BUILT: {len(df_feat)} rows, {len(df_feat.columns)} cols")
        except Exception as e:
            result["reason_codes"].append(f"FULL_FEATURE_BUILD_FAILED: {e}")
            result["final_status"] = "CFG05_FEATURE_BUILD_FAILED"
            return result

        # Verify CFG05 feature columns present
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

        missing_feats = [c for c in CFG05_FEATURE_COLUMNS if c not in df_feat.columns]
        if missing_feats:
            result["reason_codes"].append(
                f"FEATURE_BUILD_MISSING_{len(missing_feats)}_CFG05_COLUMNS"
            )
            result["final_status"] = "CFG05_FEATURE_COLUMNS_MISSING"
            return result

        result["reason_codes"].append(f"ALL_{len(CFG05_FEATURE_COLUMNS)}_CFG05_FEATURES_PRESENT")

        # Save full features CSV (all timestamps for multi-day eval)
        out_cols = ["ds"] + list(CFG05_FEATURE_COLUMNS)
        df_feat[out_cols].to_csv(result["features_path"], index=False)
        result["reason_codes"].append(f"FULL_FEATURES_SAVED: {result['features_path']}")

        # ── Train model ──
        target_dt = pd.Timestamp(target_day)
        train_start = target_dt - pd.Timedelta(days=train_window_days)
        train_end = target_dt - pd.Timedelta(hours=1)

        train_mask = (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
        train_df = df_feat[train_mask].copy()
        result["train_rows"] = len(train_df)

        if len(train_df) < 100:
            result["reason_codes"].append(
                f"TRAIN_DATA_INSUFFICIENT: {len(train_df)} rows < 100"
            )
            result["final_status"] = "CFG05_TRAIN_DATA_INSUFFICIENT"
            return result

        result["reason_codes"].append(f"TRAIN_ROWS: {len(train_df)}")

        try:
            import lightgbm as lgb

            X_train = train_df[CFG05_FEATURE_COLUMNS].fillna(0).values
            y_train = train_df["y"].values

            from models.adapters.cfg05_dayahead_lgbm import CFG05_PARAMS

            params = dict(CFG05_PARAMS)
            params["verbosity"] = -1
            n_estimators = params.pop("n_estimators", 2000)

            booster = lgb.train(
                params,
                lgb.Dataset(X_train, y_train),
                num_boost_round=n_estimators,
                callbacks=[lgb.log_evaluation(0)],
            )
            booster.save_model(result["model_path"])
            result["train_done"] = True
            result["reason_codes"].append(
                f"TRAIN_DONE: {booster.best_iteration} iterations, model saved"
            )
        except Exception as e:
            result["reason_codes"].append(f"TRAIN_FAILED: {e}")
            result["final_status"] = "CFG05_TRAIN_FAILED"
            return result

    # ── Step 4: Primary prediction (target day) ──
    pp = result["primary_prediction"]
    pp["attempted"] = True

    try:
        # Run REAL smoke prediction via adapter
        from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter

        effective_model_dir = result["model_path"]
        if not os.path.isdir(effective_model_dir):
            effective_model_dir = os.path.dirname(effective_model_dir)

        adapter = CFG05DayaheadAdapter()
        adapter.load()

        pred_result = adapter.predict(
            data_path=result["features_path"],
            target_date=target_day,
            model_dir=effective_model_dir,
        )

        pp["prediction_rows"] = len(pred_result)
        pp["validator_passed"] = True  # validate_output is called inside predict()
        result["reason_codes"].append(f"PRIMARY_PREDICTION: {len(pred_result)} rows")

        # Save prediction output
        pred_out = os.path.join(work_dir, f"cfg05_prediction_{target_day}.csv")
        pred_result.to_csv(pred_out, index=False)
        result["primary_prediction"]["prediction_path"] = pred_out
        result["reason_codes"].append(f"PREDICTION_SAVED: {pred_out}")

        # ── Step 4a: Check hour-24 completeness on features ──
        feat_check = check_cfg05_hour24_completeness(
            input_path=result["features_path"],
            target_day=target_day,
        )
        pp["feature_hours_status"] = feat_check["completeness_status"]
        result["reason_codes"].append(
            f"FEATURE_HOUR24: {feat_check['completeness_status']} ({feat_check['row_count']} rows)"
        )

        # ── Step 4b: Check hour-24 completeness on predictions ──
        pred_check = check_cfg05_hour24_completeness(
            input_path=pred_out,
            target_day=target_day,
        )
        pp["prediction_hours_status"] = pred_check["completeness_status"]
        result["reason_codes"].append(
            f"PREDICTION_HOUR24: {pred_check['completeness_status']} ({pred_check['row_count']} rows)"
        )

        # Determine primary status
        if (
            pp["prediction_rows"] == 24
            and pp["prediction_hours_status"] == COMPLETE_24H
            and pp["feature_hours_status"] == COMPLETE_24H
        ):
            result["final_status"] = "CFG05_REAL_READY_24H_LOCAL"
            result["reason_codes"].append("CFG05_REAL_READY_24H_LOCAL_ACHIEVED")
        elif (
            pp["prediction_rows"] == 23
            and pp["prediction_hours_status"] == INCOMPLETE_23H
        ):
            result["final_status"] = "CFG05_REAL_READY_INCOMPLETE_23H"
            result["reason_codes"].append("CFG05_REAL_READY_23H_ONLY_NEEDS_24H_FIX")
        else:
            result["final_status"] = "CFG05_HOUR24_FIX_FAILED"
            result["reason_codes"].append(
                f"HOUR24_FIX_FAILED: feat={pp['feature_hours_status']} pred={pp['prediction_hours_status']}"
            )

    except Exception as e:
        pp["error"] = str(e)
        result["reason_codes"].append(f"PRIMARY_PREDICTION_FAILED: {e}")
        result["final_status"] = "CFG05_PREDICTION_FAILED"
        return result

    # ── Step 5: Optional historical evaluation ──
    if not result.get("train_rows"):
        # If using pre-existing model, we still need full features for eval
        pass

    eval_requested = eval_days is not None or eval_start is not None

    if eval_requested:
        result["eval_attempted"] = True
        try:
            eval_result = _run_historical_eval(
                raw_data=raw_data,
                features_path=result["features_path"],
                model_path=result["model_path"],
                target_day=target_day,
                eval_days=eval_days,
                eval_start=eval_start,
                eval_end=eval_end,
                work_dir=work_dir,
            )
            result["eval_summary"] = eval_result["summary"]
            result["eval_metrics"] = eval_result["metrics"]

            if eval_result["metrics"] and not np.isnan(eval_result["metrics"].get("sMAPE_floor50", float("nan"))):
                result["reason_codes"].append(
                    f"EVAL_DONE: {eval_result['summary']['complete_days_evaluated']} days, "
                    f"sMAPE_floor50={eval_result['metrics']['sMAPE_floor50']:.2f}%"
                )
            else:
                result["reason_codes"].append(
                    f"EVAL_NO_METRICS: {eval_result['summary'].get('reason', 'unknown')}"
                )

            if result["final_status"] in ("CFG05_REAL_READY_24H_LOCAL",):
                result["final_status"] = "CFG05_EVAL_COMPLETE"
                result["reason_codes"].append("CFG05_EVAL_COMPLETE")
            else:
                result["final_status"] = "CFG05_EVAL_READY_NO_METRIC"

        except Exception as e:
            result["reason_codes"].append(f"HISTORICAL_EVAL_FAILED: {e}")
            result["final_status"] = "CFG05_EVAL_FAILED"

    if result["final_status"] is None:
        result["final_status"] = "CFG05_EVAL_READY_NO_METRIC"

    return result


def _run_historical_eval(
    raw_data: str,
    features_path: str,
    model_path: str,
    target_day: str,
    eval_days: Optional[int] = None,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Run historical day-ahead evaluation.

    Parameters
    ----------
    raw_data : str
        Path to raw Chinese CSV.
    features_path : str
        Path to full feature CSV (all timestamps).
    model_path : str
        Path to trained LightGBM model.
    target_day : str
        Primary target day.
    eval_days : int, optional
        Number of historical days back from target_day.
    eval_start : str, optional
        Explicit start date (overrides eval_days).
    eval_end : str, optional
        Explicit end date.
    work_dir : str, optional
        Work directory.

    Returns
    -------
    dict with eval summary and metrics.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR

    # Determine eval range
    target_dt = pd.Timestamp(target_day)
    if eval_start:
        start_dt = pd.Timestamp(eval_start)
    elif eval_days:
        start_dt = target_dt - pd.Timedelta(days=eval_days)
    else:
        start_dt = target_dt - pd.Timedelta(days=7)  # default 7 days

    end_dt = pd.Timestamp(eval_end) if eval_end else target_dt

    # Generate list of business days to evaluate
    all_days = pd.date_range(start=start_dt, end=end_dt, freq="D")
    eval_day_strs = [d.strftime("%Y-%m-%d") for d in all_days]

    # Load full features
    df_full = pd.read_csv(features_path)
    df_full["ds"] = pd.to_datetime(df_full["ds"])

    # Load y_true from raw data
    df_ytrue = _load_raw_with_y_true(raw_data)

    # Load adapter and model
    from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter

    effective_model_dir = model_path
    if not os.path.isdir(effective_model_dir):
        effective_model_dir = os.path.dirname(effective_model_dir)

    adapter = CFG05DayaheadAdapter()
    adapter.load()

    all_predictions: list[pd.DataFrame] = []
    eval_day_results: list[dict[str, Any]] = []

    for eday in eval_day_strs:
        try:
            pred = adapter.predict(
                df=df_full,
                target_date=eday,
                model_dir=effective_model_dir,
            )

            if len(pred) == 24:
                all_predictions.append(pred)
                eval_day_results.append({
                    "target_day": eday,
                    "prediction_rows": len(pred),
                    "status": COMPLETE_24H,
                })
            else:
                eval_day_results.append({
                    "target_day": eday,
                    "prediction_rows": len(pred),
                    "status": f"INCOMPLETE_{len(pred)}H",
                })
        except Exception as e:
            eval_day_results.append({
                "target_day": eday,
                "prediction_rows": 0,
                "status": f"ERROR: {e}",
            })

    complete_days = [r for r in eval_day_results if r["status"] == COMPLETE_24H]

    # Merge predictions with y_true
    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        combined = pd.merge(
            combined,
            df_ytrue,
            on="ds",
            how="left",
        )

        # Drop rows without y_true
        eval_data = combined.dropna(subset=["y_true"]).copy()

        if len(eval_data) > 0:
            y_true = eval_data["y_true"].values
            y_pred = eval_data["y_pred"].values

            metrics = compute_metrics(y_true, y_pred)
            # Count unique complete days in eval data
            unique_eval_days = eval_data["business_day"].nunique()
            metrics["n_days_evaluated"] = int(unique_eval_days)
        else:
            metrics = {"sMAPE_floor50": float("nan"), "MAE": float("nan"),
                       "RMSE": float("nan"), "n_observations": 0,
                       "n_days_evaluated": 0}
    else:
        metrics = {"sMAPE_floor50": float("nan"), "MAE": float("nan"),
                   "RMSE": float("nan"), "n_observations": 0,
                   "n_days_evaluated": 0}

    summary = {
        "eval_start": str(start_dt.date()),
        "eval_end": str(end_dt.date()),
        "days_requested": len(eval_day_strs),
        "complete_days_evaluated": len(complete_days),
        "total_prediction_rows": sum(
            len(p) for p in all_predictions
        ),
    }

    return {
        "summary": summary,
        "metrics": metrics,
        "day_results": eval_day_results,
    }


def _import_source_module(source_repo: str, module_name: str):
    """Import a module from the source repo by name.

    Uses ``importlib.import_module`` with the source repo on ``sys.path``
    so that internal relative imports resolve correctly.
    """
    import importlib as _il

    full_path = os.path.join(source_repo, *module_name.split(".")) + ".py"
    if not os.path.isfile(full_path):
        raise ImportError(f"Source module not found: {full_path}")

    if source_repo not in sys.path:
        sys.path.insert(0, source_repo)

    mod = _il.import_module(module_name)
    return mod


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable P15 report."""
    print("=" * 60)
    print("P15 cfg05 24-Hour Smoke & Eval Report")
    print("=" * 60)
    print(f"  Raw data:           {result['raw_data_status']}")
    print(f"  Source repo:        {result['source_repo_status']}")
    print(f"  Train attempted:    {result['train_attempted']}")
    if result.get("train_done"):
        print(f"  Train rows:         {result.get('train_rows', 'N/A')}")
    print(f"  Model:              {result['model_path']}")
    print(f"  Features:           {result['features_path']}")
    print(f"  Hour24 fix applied: {result['hour24_fix_applied']}")
    print()

    pp = result["primary_prediction"]
    print(f"  Primary target day: {pp['target_day']}")
    print(f"  Prediction rows:    {pp['prediction_rows']}")
    print(f"  Validator passed:   {pp['validator_passed']}")
    print(f"  Feature hours:      {pp['feature_hours_status']}")
    print(f"  Prediction hours:   {pp['prediction_hours_status']}")
    print()

    if result["eval_attempted"] and result["eval_summary"]:
        es = result["eval_summary"]
        print("  Historical Eval:")
        print(f"    Range:            {es.get('eval_start', '?')} ~ {es.get('eval_end', '?')}")
        print(f"    Days requested:   {es.get('days_requested', 0)}")
        print(f"    Complete days:    {es.get('complete_days_evaluated', 0)}")
        if result["eval_metrics"]:
            em = result["eval_metrics"]
            print(f"    Observations:     {em.get('n_observations', 0)}")
            print(f"    sMAPE_floor50:    {em.get('sMAPE_floor50', 'N/A'):.2f}%")
            print(f"    MAE:              {em.get('MAE', 'N/A'):.2f}")
            print(f"    RMSE:             {em.get('RMSE', 'N/A'):.2f}")
            if em.get("n_days_evaluated"):
                print(f"    Days evaluated:   {em.get('n_days_evaluated')}")
                print("    *** NOTE: This evaluation is NOT walk-forward ***")
                print("    *** Metrics may be optimistic vs. true out-of-sample ***")
        print()

    print(f"  Final status:       {result['final_status']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P15: cfg05 24-hour smoke + optional historical evaluation.",
    )
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw Chinese CSV (required).")
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment.")
    parser.add_argument("--target-day", type=str, default="2026-07-01",
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--train-window-days", type=int, default=90,
                        help="Training window in days.")
    parser.add_argument("--eval-days", type=int, default=None,
                        help="Number of historical days to evaluate.")
    parser.add_argument("--eval-start", type=str, default=None,
                        help="Eval range start (YYYY-MM-DD).")
    parser.add_argument("--eval-end", type=str, default=None,
                        help="Eval range end (YYYY-MM-DD).")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to existing model (skip training).")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to existing full feature CSV (skip training).")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Local work dir.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite existing output files.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero on any blocker.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Validate path safety
    work_dir = args.work_dir or _DEFAULT_WORK_DIR
    if not _path_is_safe(work_dir):
        logger.error("Unsafe work-dir: %s", work_dir)
        return 1

    for pname, pval in [("--model", args.model), ("--features", args.features)]:
        if pval and not _path_is_safe(pval):
            logger.error("Unsafe %s path: %s", pname, pval)
            return 1

    result = run_p15_cfg05_24h_smoke_and_eval(
        raw_data=args.raw_data,
        source_repo=args.source_repo,
        target_day=args.target_day,
        train_window_days=args.train_window_days,
        eval_days=args.eval_days,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        model_path=args.model,
        features_path=args.features,
        work_dir=work_dir,
        force=args.force,
    )

    if args.json:
        # Convert any non-serializable types
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["final_status"] in (
            "CFG05_REAL_READY_24H_LOCAL",
            "CFG05_EVAL_COMPLETE",
        ):
            logger.info("P15 strict mode PASS: %s", result["final_status"])
            return 0
        else:
            logger.error("P15 strict mode FAIL: %s", result["final_status"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
