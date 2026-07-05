"""
scripts/run_p32_multimodel_30d_backtest.py — P32: Multi-model 30-day backtest.

Runs day-ahead predictions for each model across 2026-06-01 to 2026-06-30,
saves consolidated prediction CSVs, and reports per-model coverage.

Usage::

    python -m scripts.run_p32_multimodel_30d_backtest

    python -m scripts.run_p32_multimodel_30d_backtest --start-day 2026-06-01 \\
        --end-day 2026-06-30 --json --strict

Options::

    --start-day YYYY-MM-DD  Start date (default: 2026-06-01).
    --end-day YYYY-MM-DD    End date (default: 2026-06-30).
    --work-dir PATH         Model artifacts dir (default: .local_artifacts/...).
    --output-dir PATH       Output dir for prediction CSVs (default: work-dir/ledger).
    --force                 Re-run even if output exists.
    --json                  Output JSON report.
    --strict                Exit non-zero on failures.
    --verbose, -v           Increase verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from models.adapters.multimodel_pool import (
    ALL_CANDIDATE_MODELS,
    REAL_24H_READY,
    create_adapter,
)

logger = logging.getLogger(__name__)

_MODEL_NAMES = ["cfg05_dayahead_lgbm"] + ALL_CANDIDATE_MODELS
_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")
_RAW_DATA_PATH = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.1/data/shandong_pmos_hourly.csv"


def _import_source_modules(source_repo: str):
    """Import source data_loader and feature_builder."""
    import importlib as _il

    if source_repo not in sys.path:
        sys.path.insert(0, source_repo)

    data_loader = _il.import_module("src.common.data_loader")
    feature_builder = _il.import_module("src.common.feature_builder_dayahead")
    return data_loader, feature_builder


def _fill_v3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Same v3 fill as P31."""
    df = df.copy()
    for c in ["price_volatility_24h", "price_volatility_168h"]:
        if c not in df.columns:
            df[c] = df["y"].shift(1).rolling(24, min_periods=1).std().fillna(0) if "volatility_24" in c else df["y"].shift(1).rolling(168, min_periods=1).std().fillna(0)
    if "price_volatility_168h" not in df.columns:
        df["price_volatility_168h"] = df["y"].shift(1).rolling(168, min_periods=1).std().fillna(0)
    for c in ["renewable_penetration_rank_30d", "load_ramp_rank_30d"]:
        if c not in df.columns:
            df[c] = 0.5
    for c_src, c_dst in [("bidding_space", "bidding_space_change_24h"), ("net_load", "net_load_change_24h")]:
        if c_dst not in df.columns and c_src in df.columns:
            df[c_dst] = df[c_src].diff(24).fillna(0)
    if "renewable_change_24h" not in df.columns:
        renew = df.get("wind", 0) + df.get("solar", 0)
        df["renewable_change_24h"] = renew.diff(24).fillna(0)
    for c in ["is_spring_festival_exact", "days_to_spring_festival_exact", "days_after_spring_festival_exact"]:
        if c not in df.columns:
            src = c.replace("_exact", "").replace("is_", "is_")
            df[c] = df.get(src.replace("_exact", "_window") if "spring_festival_window" in src else src, 0)
    for c_src, c_dst in [("bidding_space", "hour_x_bidding_space"), ("net_load", "hour_x_net_load")]:
        if c_dst not in df.columns and c_src in df.columns:
            df[c_dst] = df["hour"] * df[c_src]
    for c in ["period_x_bidding_space", "period_x_renewable_penetration"]:
        if c not in df.columns:
            df[c] = 0
    for c in [
        "price_volatility_24h", "price_volatility_168h",
        "renewable_penetration_rank_30d", "load_ramp_rank_30d",
        "bidding_space_change_24h", "net_load_change_24h", "renewable_change_24h",
        "is_spring_festival_exact", "days_to_spring_festival_exact",
        "days_after_spring_festival_exact", "hour_x_bidding_space", "hour_x_net_load",
        "period_x_bidding_space", "period_x_renewable_penetration",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def run_30d_backtest(
    start_day: Optional[str] = None,
    end_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run 30-day backtest for all REAL_24H_READY models.

    Parameters
    ----------
    start_day : str, optional
        Start date YYYY-MM-DD (default: 2026-06-01).
    end_day : str, optional
        End date YYYY-MM-DD (default: 2026-06-30).
    work_dir : str, optional
        Model artifacts directory.
    output_dir : str, optional
        Output directory for prediction CSVs.
    force : bool
        Re-run if output exists.

    Returns
    -------
    dict with backtest summary.
    """
    start_day = start_day or "2026-06-01"
    end_day = end_day or "2026-06-30"
    work_dir = work_dir or _DEFAULT_WORK_DIR
    output_dir = output_dir or os.path.join(work_dir, "ledger")
    os.makedirs(output_dir, exist_ok=True)

    source_repo = os.path.join(".local_artifacts", "source_repos", "epf-sota-experiment")
    raw_data = _RAW_DATA_PATH

    result: dict[str, Any] = {
        "phase": "P32",
        "start_day": start_day,
        "end_day": end_day,
        "models": {},
        "summary": {
            "total_days": 0,
            "models_with_full_coverage": 0,
            "models_with_partial_coverage": 0,
            "total_prediction_rows": 0,
            "p32_status": "P32_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Build date list
    start_dt = pd.Timestamp(start_day)
    end_dt = pd.Timestamp(end_day)
    dates = pd.date_range(start_dt, end_dt, freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    result["summary"]["total_days"] = len(date_strs)
    result["reason_codes"].append(f"DATES:{len(date_strs)}_days_from_{start_day}_to_{end_day}")

    # Load data + build features
    try:
        data_loader, feature_builder = _import_source_modules(source_repo)
        df = data_loader.load_data(raw_data, target="dayahead")
        df_feat = feature_builder.build_features_dayahead(df, use_extended=True)
        df_feat = _fill_v3_columns(df_feat)
        result["reason_codes"].append(f"FEATURES_LOADED:{len(df_feat)}rows")
    except Exception as e:
        result["reason_codes"].append(f"DATA_FEATURE_FAILED:{e}")
        result["summary"]["p32_status"] = "P32_DATA_FAILED"
        return result

    # Per-model backtest
    for model_name in _MODEL_NAMES:
        model_dir = os.path.join(work_dir, "models", model_name)
        model_result = _backtest_single_model(
            model_name, model_dir, df_feat, date_strs, output_dir, force,
        )
        result["models"][model_name] = model_result
        result["summary"]["total_prediction_rows"] += model_result.get("total_rows", 0)

        if model_result.get("full_coverage", False):
            result["summary"]["models_with_full_coverage"] += 1
        elif model_result.get("days_with_predictions", 0) > 0:
            result["summary"]["models_with_partial_coverage"] += 1

        result["reason_codes"].append(
            f"{model_name}:{model_result.get('days_with_predictions', 0)}_{model_result.get('status', 'N/A')}"
        )

    # Final status
    if result["summary"]["models_with_full_coverage"] >= 2:
        result["summary"]["p32_status"] = "P32_BACKTEST_COMPLETE"
    elif result["summary"]["models_with_partial_coverage"] > 0:
        result["summary"]["p32_status"] = "P32_BACKTEST_PARTIAL"
    else:
        result["summary"]["p32_status"] = "P32_BACKTEST_FAILED"

    return result


def _get_model_feature_cols(model_name: str, model_dir: str) -> list[str] | None:
    """Determine the feature columns a model was trained on."""
    if model_name in ("cfg05_dayahead_lgbm", "best_two_average", "catboost_spike_residual"):
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        return list(CFG05_FEATURE_COLUMNS)
    if model_name == "stage3_business_fixed":
        # 42 source features
        return [
            "hour", "month", "day_of_week", "is_weekend",
            "lag_price_target", "lag_price_week",
            "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
            "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
            "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
            "morning_trend", "is_info_fresh",
            "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
            "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
            "same_hour_max_7d", "same_hour_min_7d",
            "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
            "is_spring_festival_window", "days_to_spring_festival",
            "days_after_spring_festival", "is_month_start", "is_month_end",
        ]
    if model_name == "catboost_sota":
        # 24 source CatBoost features
        return [
            "hour", "month", "day_of_week", "is_weekend",
            "lag_price_target", "lag_price_week",
            "load", "wind", "solar", "interconnect",
            "bidding_space", "space_ratio",
            "net_load", "solar_ratio", "net_load_sq",
            "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
            "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
        ]
    return None


def _backtest_single_model(
    model_name: str,
    model_dir: str,
    df_feat: pd.DataFrame,
    date_strs: list[str],
    output_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Run backtest for a single model over all dates."""
    result: dict[str, Any] = {
        "model_name": model_name,
        "status": "NOT_STARTED",
        "days_with_predictions": 0,
        "total_rows": 0,
        "full_coverage": False,
        "missing_dates": [],
        "output_path": None,
        "reason_codes": [],
    }

    output_path = os.path.join(output_dir, f"predictions_{model_name}_30d.csv")
    result["output_path"] = output_path

    # Check model artifacts exist
    if not os.path.isdir(model_dir):
        result["status"] = "MODEL_DIR_MISSING"
        result["reason_codes"].append(f"MODEL_DIR_NOT_FOUND:{model_dir}")
        return result

    # For cfg05, check model.txt; for others, check adapter format
    if model_name == "cfg05_dayahead_lgbm":
        model_file = os.path.join(model_dir, "cfg05_model.txt")
    elif model_name == "best_two_average":
        model_file = os.path.join(model_dir, "best_two_average_trial_02.txt")
    elif model_name == "stage3_business_fixed":
        model_file = os.path.join(model_dir, "stage3_model.txt")
    elif model_name == "catboost_sota":
        model_file = os.path.join(model_dir, "catboost_sota_model.cbm")
    elif model_name == "catboost_spike_residual":
        model_file = os.path.join(model_dir, "catboost_spike_residual.cbm")
    else:
        result["status"] = "UNKNOWN_MODEL"
        return result

    if not os.path.isfile(model_file):
        result["status"] = "MODEL_FILE_MISSING"
        result["reason_codes"].append(f"MODEL_FILE_NOT_FOUND:{model_file}")
        return result

    # Load adapter
    try:
        if model_name == "cfg05_dayahead_lgbm":
            from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter
            adapter = CFG05DayaheadAdapter()
            adapter.load()
            adapter._load_model(model_dir)
        else:
            adapter = create_adapter(model_name)
            adapter.load()
            adapter._load_artifacts(model_dir)

        # For models trained on a subset of features, restrict df_feat
        # to only the columns the model knows about (plus ds, y).
        model_known_cols = _get_model_feature_cols(model_name, model_dir)
        if model_known_cols:
            keep_cols = ["ds", "y"] + [c for c in model_known_cols if c in df_feat.columns]
            df_model = df_feat[keep_cols].copy()
            logger.info(
                "%s: restricted to %d/%d feature columns",
                model_name, len([c for c in model_known_cols if c in df_feat.columns]),
                len(model_known_cols),
            )
        else:
            df_model = df_feat
    except Exception as e:
        result["status"] = "MODEL_LOAD_FAILED"
        result["reason_codes"].append(f"LOAD_FAILED:{e}")
        return result

    # Predict each day
    all_preds = []
    for d in date_strs:
        try:
            pred_df = adapter.predict(df=df_model, target_date=d, model_dir=model_dir)
            if len(pred_df) > 0:
                all_preds.append(pred_df)
                result["days_with_predictions"] += 1
                result["total_rows"] += len(pred_df)
            else:
                result["missing_dates"].append(d)
        except Exception as e:
            result["missing_dates"].append(d)
            result["reason_codes"].append(f"PREDICT_FAILED_{d}:{e}")

    # Determine coverage
    total_days = len(date_strs)
    if result["days_with_predictions"] == total_days:
        result["full_coverage"] = True
        result["status"] = "FULL_COVERAGE"
        result["reason_codes"].append(f"ALL_{total_days}_DAYS_COVERED")
    elif result["days_with_predictions"] > 0:
        result["status"] = "PARTIAL_COVERAGE"
        result["reason_codes"].append(
            f"{result['days_with_predictions']}/{total_days}_DAYS_COVERED"
        )
    else:
        result["status"] = "NO_PREDICTIONS"
        result["reason_codes"].append("ZERO_DAYS_COVERED")
        return result

    # Save consolidated CSV
    if len(all_preds) > 0:
        try:
            consolidated = pd.concat(all_preds, ignore_index=True)
            consolidated = consolidated.sort_values(
                ["business_day", "hour_business"]
            ).reset_index(drop=True)
            # Add run_id, created_at for ledger format
            consolidated["run_id"] = "p32_backtest"
            consolidated["created_at"] = pd.Timestamp.now().isoformat()
            consolidated["updated_at"] = pd.Timestamp.now().isoformat()

            if not force and os.path.exists(output_path):
                result["reason_codes"].append(f"OUTPUT_EXISTS:{output_path}")
            else:
                consolidated.to_csv(output_path, index=False)
                result["reason_codes"].append(f"SAVED:{output_path}_({len(consolidated)}rows)")
        except Exception as e:
            result["reason_codes"].append(f"SAVE_FAILED:{e}")

    return result


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P32 — Multi-Model 30-Day Backtest")
    print("=" * 60)
    print(f"  Period:           {result['start_day']} ~ {result['end_day']}")
    print(f"  Total days:       {result['summary']['total_days']}")
    print()

    print("── Models ──")
    for model_name, mres in result.get("models", {}).items():
        status = mres.get("status", "N/A")
        days = mres.get("days_with_predictions", 0)
        rows = mres.get("total_rows", 0)
        coverage = "FULL" if mres.get("full_coverage") else "PARTIAL"
        print(f"  {model_name:<25} {status:<20} days={days:<2} rows={rows:<4} {coverage}")

    print()
    print("── Summary ──")
    s = result["summary"]
    print(f"  Full coverage:    {s['models_with_full_coverage']} models")
    print(f"  Partial coverage: {s['models_with_partial_coverage']} models")
    print(f"  Total rows:       {s['total_prediction_rows']}")
    print(f"  Status:           {s['p32_status']}")
    print()

    print("── Reason codes ──")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P32: Multi-model 30-day backtest.",
    )
    parser.add_argument("--start-day", type=str, default="2026-06-01")
    parser.add_argument("--end-day", type=str, default="2026-06-30")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = run_30d_backtest(
        start_day=args.start_day,
        end_day=args.end_day,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        force=args.force,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["summary"]["p32_status"] == "P32_BACKTEST_COMPLETE":
            logger.info("P32: PASS")
            return 0
        logger.error("P32: FAIL (%s)", result["summary"]["p32_status"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
