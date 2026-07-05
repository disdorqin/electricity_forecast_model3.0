"""
scripts/run_p31_train_dayahead_model_pool.py — P31: Train multi-model pool.

Trains all candidate day-ahead models (best_two_average, stage3_business_fixed,
catboost_sota, catboost_spike_residual) plus cfg05 baseline, saves artifacts,
runs hour-24 completeness checks, and reports readiness for each.

Usage::

    python -m scripts.run_p31_train_dayahead_model_pool \\
        --raw-data /path/to/raw.csv --target-day 2026-07-01

    python -m scripts.run_p31_train_dayahead_model_pool \\
        --raw-data /path/to/raw.csv --target-day 2026-07-01 \\
        --force --json --strict

Options::

    --source-repo PATH       Path to epf-sota-experiment.
    --raw-data PATH          Path to raw Chinese CSV (required).
    --target-day YYYY-MM-DD  Target day for prediction (default: 2026-07-01).
    --train-window-days N    Training window (default: 90).
    --work-dir PATH          Output dir (default: .local_artifacts/p31_p40_multimodel_fusion).
    --force                  Overwrite existing artifacts.
    --json                   Output JSON report.
    --strict                 Exit non-zero on any failure.
    --verbose, -v            Increase verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from models.adapters.multimodel_pool import (
    ALL_CANDIDATE_MODELS,
    BANNED_MODELS,
    REAL_24H_READY,
    TRAINED_BUT_NOT_24H,
    DEP_MISSING,
    SOURCE_SCRIPT_MISSING,
    MODEL_TRAIN_FAILED,
    INVALID_BANNED,
    create_adapter,
)
from scripts.check_cfg05_hour24_completeness import check_cfg05_hour24_completeness

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

_DEFAULT_SOURCE_REPO = os.path.join(
    ".local_artifacts", "source_repos", "epf-sota-experiment",
)
_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")
_RAW_DATA_PATH = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.1/data/shandong_pmos_hourly.csv"

# CFG05 v3 extra columns not produced by source feature_builder_dayahead
_V3_EXTRA_COLUMNS = [
    "price_volatility_24h", "price_volatility_168h",
    "renewable_penetration_rank_30d", "load_ramp_rank_30d",
    "bidding_space_change_24h", "net_load_change_24h", "renewable_change_24h",
    "is_spring_festival_exact", "days_to_spring_festival_exact",
    "days_after_spring_festival_exact",
    "hour_x_bidding_space", "hour_x_net_load",
    "period_x_bidding_space", "period_x_renewable_penetration",
]


def _import_source_module(source_repo: str, module_name: str):
    """Import a module from the source repo by name."""
    import importlib as _il

    full_path = os.path.join(source_repo, *module_name.split(".")) + ".py"
    if not os.path.isfile(full_path):
        raise ImportError(f"Source module not found: {full_path}")

    if source_repo not in sys.path:
        sys.path.insert(0, source_repo)

    return _il.import_module(module_name)


def _fill_v3_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing CFG05 v3 feature columns with sensible defaults.

    The source feature_builder_dayahead produces 42 columns. The cfg05
    3.0 adapter expects 54 (including 14 v3 columns). This function
    fills the gap so all adapters can train on the full feature set.
    """
    df = df.copy()

    # ── Volatility: use recent std as proxy ──
    if "price_volatility_24h" not in df.columns:
        df["price_volatility_24h"] = (
            df["y"].shift(1).rolling(24, min_periods=1).std().fillna(0)
        )
    if "price_volatility_168h" not in df.columns:
        df["price_volatility_168h"] = (
            df["y"].shift(1).rolling(168, min_periods=1).std().fillna(0)
        )

    # ── Additional ranks: use 0.5 as neutral ──
    for c in ["renewable_penetration_rank_30d", "load_ramp_rank_30d"]:
        if c not in df.columns:
            df[c] = 0.5

    # ── Change features: diff over 24h ──
    if "bidding_space_change_24h" not in df.columns and "bidding_space" in df.columns:
        df["bidding_space_change_24h"] = df["bidding_space"].diff(24).fillna(0)
    if "net_load_change_24h" not in df.columns and "net_load" in df.columns:
        df["net_load_change_24h"] = df["net_load"].diff(24).fillna(0)
    if "renewable_change_24h" not in df.columns:
        renew = df.get("wind", 0) + df.get("solar", 0)
        df["renewable_change_24h"] = renew.diff(24).fillna(0)

    # ── Exact spring festival: duplicate approximate versions ──
    if "is_spring_festival_exact" not in df.columns:
        df["is_spring_festival_exact"] = df.get("is_spring_festival_window", 0)
    if "days_to_spring_festival_exact" not in df.columns:
        df["days_to_spring_festival_exact"] = df.get("days_to_spring_festival", 0)
    if "days_after_spring_festival_exact" not in df.columns:
        df["days_after_spring_festival_exact"] = df.get("days_after_spring_festival", 0)

    # ── Interaction features ──
    if "hour_x_bidding_space" not in df.columns and "bidding_space" in df.columns:
        df["hour_x_bidding_space"] = df["hour"] * df["bidding_space"]
    elif "hour_x_bidding_space" not in df.columns:
        df["hour_x_bidding_space"] = 0

    if "hour_x_net_load" not in df.columns and "net_load" in df.columns:
        df["hour_x_net_load"] = df["hour"] * df["net_load"]
    elif "hour_x_net_load" not in df.columns:
        df["hour_x_net_load"] = 0

    if "period_x_bidding_space" not in df.columns and "bidding_space" in df.columns:
        # Map hour (1..24) to period 1,2,3 then multiply
        period_num = pd.cut(
            df["hour"], bins=[0, 8, 16, 24], labels=[1, 2, 3], right=True
        ).astype(int)
        df["period_x_bidding_space"] = period_num * df["bidding_space"]
    elif "period_x_bidding_space" not in df.columns:
        df["period_x_bidding_space"] = 0

    if "period_x_renewable_penetration" not in df.columns:
        renew_pen = df.get("renew_penetration", 0)
        period_num = pd.cut(
            df["hour"], bins=[0, 8, 16, 24], labels=[1, 2, 3], right=True
        ).astype(int)
        df["period_x_renewable_penetration"] = period_num * renew_pen
    elif "period_x_renewable_penetration" not in df.columns:
        df["period_x_renewable_penetration"] = 0

    # Final safety fill for any remaining NaN
    for c in _V3_EXTRA_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def train_model_pool(
    source_repo: Optional[str] = None,
    raw_data: Optional[str] = None,
    target_day: Optional[str] = None,
    train_window_days: int = 90,
    work_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Train all candidate models and report readiness.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment.
    raw_data : str, optional
        Path to raw Chinese CSV.
    target_day : str, optional
        Target day in YYYY-MM-DD.
    train_window_days : int
        Training window in days.
    work_dir : str, optional
        Output directory.
    force : bool
        Overwrite existing artifacts.

    Returns
    -------
    dict with per-model readiness summary.
    """
    source_repo = source_repo or _DEFAULT_SOURCE_REPO
    raw_data = raw_data or _RAW_DATA_PATH
    target_day = target_day or "2026-07-01"
    work_dir = work_dir or _DEFAULT_WORK_DIR

    result: dict[str, Any] = {
        "phase": "P31",
        "target_day": target_day,
        "train_window_days": train_window_days,
        "source_repo": source_repo,
        "raw_data": raw_data,
        "models": {},
        "summary": {
            "total_candidates": len(ALL_CANDIDATE_MODELS),
            "real_24h_ready": 0,
            "trained_but_not_24h": 0,
            "failed": 0,
            "skipped_banned": 0,
            "dep_missing": 0,
            "p31_status": "P31_NOT_STARTED",
        },
        "cfg05_baseline": None,
        "reason_codes": [],
    }

    os.makedirs(work_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # Step 1: Load raw data and build features
    # ════════════════════════════════════════════════════════════════════════
    try:
        data_loader = _import_source_module(source_repo, "src.common.data_loader")
        feature_builder = _import_source_module(
            source_repo, "src.common.feature_builder_dayahead"
        )

        df = data_loader.load_data(raw_data, target="dayahead")
        logger.info("Raw data loaded: %d rows", len(df))
        result["reason_codes"].append(f"RAW_DATA_LOADED:{len(df)}")

        df_feat = feature_builder.build_features_dayahead(df, use_extended=True)
        logger.info("Features built: %d rows, %d columns", len(df_feat), len(df_feat.columns))
        result["reason_codes"].append(f"FEATURES_BUILT:{len(df_feat)}rows_{len(df_feat.columns)}cols")

        # Fill v3 columns for CFG05 adapters
        df_feat = _fill_v3_columns(df_feat)
        logger.info("After v3 fill: %d columns", len(df_feat.columns))
    except Exception as e:
        result["reason_codes"].append(f"DATA_FEATURE_LOAD_FAILED:{e}")
        result["summary"]["p31_status"] = "P31_DATA_FAILED"
        return result

    # ════════════════════════════════════════════════════════════════════════
    # Step 2: Train/verify cfg05 baseline
    # ════════════════════════════════════════════════════════════════════════
    cfg05_result = _train_cfg05_baseline(
        source_repo, df_feat, target_day, train_window_days, work_dir, force,
    )
    result["cfg05_baseline"] = cfg05_result
    result["reason_codes"].extend(
        [f"CFG05:{rc}" for rc in cfg05_result.get("reason_codes", [])]
    )

    # ════════════════════════════════════════════════════════════════════════
    # Step 3: Train each candidate model
    # ════════════════════════════════════════════════════════════════════════
    for model_id in ALL_CANDIDATE_MODELS:
        model_result = _train_single_model(
            model_id, source_repo, df_feat, target_day,
            train_window_days, work_dir, force,
        )
        result["models"][model_id] = model_result

        # Update summary counts
        status = model_result.get("readiness", "UNKNOWN")
        if status == REAL_24H_READY:
            result["summary"]["real_24h_ready"] += 1
        elif status == TRAINED_BUT_NOT_24H:
            result["summary"]["trained_but_not_24h"] += 1
        elif status == INVALID_BANNED:
            result["summary"]["skipped_banned"] += 1
        elif status == DEP_MISSING:
            result["summary"]["dep_missing"] += 1
        else:
            result["summary"]["failed"] += 1

        result["reason_codes"].append(
            f"MODEL:{model_id}={status}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # Step 4: Final status
    # ════════════════════════════════════════════════════════════════════════
    ready = result["summary"]["real_24h_ready"]
    if ready >= 2:
        result["summary"]["p31_status"] = "P31_MULTIMODEL_POOL_READY"
    elif ready == 1:
        result["summary"]["p31_status"] = "P31_PARTIALLY_READY"
    else:
        result["summary"]["p31_status"] = "P31_INSUFFICIENT_READY"

    return result


def _train_cfg05_baseline(
    source_repo: str,
    df_feat: pd.DataFrame,
    target_day: str,
    train_window_days: int,
    work_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Retrain cfg05 and check 24h completeness."""
    result: dict[str, Any] = {
        "model_name": "cfg05_dayahead_lgbm",
        "readiness": "NOT_TRAINED",
        "train_done": False,
        "artifact_saved": False,
        "prediction_rows": 0,
        "hour24_status": None,
        "reason_codes": [],
    }

    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS, CFG05_PARAMS
    import lightgbm as lgb

    model_dir = os.path.join(work_dir, "models", "cfg05_dayahead_lgbm")
    model_path = os.path.join(model_dir, "cfg05_model.txt")
    pred_path = os.path.join(model_dir, f"predictions_{target_day}.csv")

    os.makedirs(model_dir, exist_ok=True)

    try:
        target_dt = pd.Timestamp(target_day)
        train_start = target_dt - pd.Timedelta(days=train_window_days)
        train_end = target_dt - pd.Timedelta(hours=1)

        mask = (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
        train_df = df_feat[mask].copy()

        available_cols = [c for c in CFG05_FEATURE_COLUMNS if c in train_df.columns]
        X_train = train_df[available_cols].fillna(0).values
        y_train = train_df["y"].values

        params = dict(CFG05_PARAMS)
        params["verbosity"] = -1
        n_estimators = params.pop("n_estimators", 2000)

        booster = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            num_boost_round=n_estimators,
            callbacks=[lgb.log_evaluation(0)],
        )
        result["train_done"] = True
        result["reason_codes"].append("TRAIN_OK")
        result["n_train"] = len(X_train)

        # Save model
        if force or not os.path.exists(model_path):
            booster.save_model(model_path)
            result["artifact_saved"] = True
            result["reason_codes"].append("MODEL_SAVED")

        # Run prediction
        from artifacts.dayahead_window import filter_dayahead
        day_df = filter_dayahead(df_feat, target_day)
        if len(day_df) > 0:
            X_pred = day_df[available_cols].fillna(0).values
            day_df["y_pred"] = booster.predict(X_pred)
            day_df[["ds"] + available_cols].to_csv(
                os.path.join(model_dir, f"features_{target_day}.csv"), index=False
            )
            day_df[["ds", "y_pred"]].to_csv(pred_path, index=False)

        # Check 24h completeness
        if os.path.exists(pred_path):
            ck = check_cfg05_hour24_completeness(
                input_path=pred_path, target_day=target_day,
            )
            result["hour24_status"] = ck["completeness_status"]
            result["prediction_rows"] = ck["row_count"]
            result["reason_codes"].append(f"HOUR24:{ck['completeness_status']}")

            if ck["completeness_status"] == "COMPLETE_24H":
                result["readiness"] = REAL_24H_READY
            else:
                result["readiness"] = TRAINED_BUT_NOT_24H
        else:
            result["reason_codes"].append("PREDICTION_NOT_GENERATED")
    except Exception as e:
        result["reason_codes"].append(f"ERROR:{e}")
        result["readiness"] = MODEL_TRAIN_FAILED

    return result


def _train_single_model(
    model_id: str,
    source_repo: str,
    df_feat: pd.DataFrame,
    target_day: str,
    train_window_days: int,
    work_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Train a single model and check readiness."""
    result: dict[str, Any] = {
        "model_name": model_id,
        "readiness": "NOT_STARTED",
        "train_done": False,
        "artifact_saved": False,
        "prediction_rows": 0,
        "hour24_status": None,
        "reason_codes": [],
    }

    # ── Check banned ──
    if model_id in BANNED_MODELS:
        result["readiness"] = INVALID_BANNED
        result["reason_codes"].append("MODEL_IS_BANNED")
        return result

    # ── Create adapter ──
    try:
        adapter = create_adapter(model_id)
    except Exception as e:
        result["readiness"] = MODEL_TRAIN_FAILED
        result["reason_codes"].append(f"ADAPTER_CREATE_FAILED:{e}")
        return result

    model_dir = os.path.join(work_dir, "models", model_id)
    os.makedirs(model_dir, exist_ok=True)

    # ── Check CatBoost dependency ──
    if model_id in ("catboost_sota", "catboost_spike_residual"):
        try:
            import catboost  # noqa: F401
        except ImportError:
            result["readiness"] = DEP_MISSING
            result["reason_codes"].append("CATBOOST_NOT_INSTALLED")
            return result

    # ── Train ──
    try:
        manifest = adapter.train(
            source_repo=source_repo,
            df_feat=df_feat,
            target_day=target_day,
            train_window_days=train_window_days,
        )
        result["train_done"] = True
        result["train_rows"] = manifest.get("train_rows", 0)
        result["reason_codes"].append("TRAIN_OK")
    except ImportError as e:
        if "catboost" in str(e).lower():
            result["readiness"] = DEP_MISSING
            result["reason_codes"].append(f"CATBOOST_IMPORT_ERROR:{e}")
            return result
        result["readiness"] = MODEL_TRAIN_FAILED
        result["reason_codes"].append(f"TRAIN_FAILED:{e}")
        return result
    except Exception as e:
        result["readiness"] = MODEL_TRAIN_FAILED
        result["reason_codes"].append(f"TRAIN_FAILED:{e}")
        return result

    # ── Save artifacts ──
    try:
        adapter.save_artifacts(model_dir)
        result["artifact_saved"] = True
        result["reason_codes"].append("ARTIFACTS_SAVED")
    except Exception as e:
        result["reason_codes"].append(f"SAVE_FAILED:{e}")

    # ── Run prediction and check 24h completeness ──
    try:
        pred_df = adapter.predict(
            df=df_feat,
            target_date=target_day,
            model_dir=model_dir,
        )
        result["prediction_rows"] = len(pred_df)

        # Save prediction CSV for completeness check
        pred_path = os.path.join(model_dir, f"predictions_{target_day}.csv")
        pred_df.to_csv(pred_path, index=False)

        # Check 24h
        ck = check_cfg05_hour24_completeness(
            input_path=pred_path, target_day=target_day,
        )
        result["hour24_status"] = ck["completeness_status"]
        result["reason_codes"].append(f"HOUR24:{ck['completeness_status']}")

        if ck["completeness_status"] == "COMPLETE_24H":
            result["readiness"] = REAL_24H_READY
        else:
            result["readiness"] = TRAINED_BUT_NOT_24H
    except Exception as e:
        result["reason_codes"].append(f"PREDICT_CHECK_FAILED:{e}")
        result["readiness"] = TRAINED_BUT_NOT_24H

    return result


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable P31 report."""
    print("=" * 60)
    print("P31 — Train Multi-Model Day-Ahead Pool")
    print("=" * 60)
    print(f"  Target day:       {result['target_day']}")
    print(f"  Train window:     {result['train_window_days']} days")
    print(f"  Source repo:      {result['source_repo']}")
    print(f"  Work dir:         {result.get('work_dir', result.get('cfg05_baseline', {}).get('artifact_saved', False) and '.local_artifacts/p31_p40_multimodel_fusion' or '')}")
    print()

    # Per-model
    print("── Models ──")
    print(f"  cfg05_baseline:     {result.get('cfg05_baseline', {}).get('readiness', 'N/A')}")
    for model_id, mres in result.get("models", {}).items():
        status = mres.get("readiness", "N/A")
        rows = mres.get("prediction_rows", 0)
        h24 = mres.get("hour24_status", "N/A")
        print(f"  {model_id:<20} {status:<20} rows={rows} h24={h24}")

    print()
    print("── Summary ──")
    s = result["summary"]
    print(f"  REAL_24H_READY:     {s['real_24h_ready']} / {s['total_candidates']}")
    print(f"  TRAINED_BUT_NOT_24H: {s['trained_but_not_24h']}")
    print(f"  Failed:             {s['failed']}")
    print(f"  Skipped (banned):   {s['skipped_banned']}")
    print(f"  Dep missing:        {s['dep_missing']}")
    print(f"  Status:             {s['p31_status']}")
    print()

    print("── Reason codes ──")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P31: Train multi-model day-ahead pool.",
    )
    parser.add_argument("--source-repo", type=str, default=None)
    parser.add_argument("--raw-data", type=str, default=None)
    parser.add_argument("--target-day", type=str, default="2026-07-01")
    parser.add_argument("--train-window-days", type=int, default=90)
    parser.add_argument("--work-dir", type=str, default=None)
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

    work_dir = args.work_dir or _DEFAULT_WORK_DIR

    result = train_model_pool(
        source_repo=args.source_repo,
        raw_data=args.raw_data,
        target_day=args.target_day,
        train_window_days=args.train_window_days,
        work_dir=work_dir,
        force=args.force,
    )
    result["work_dir"] = work_dir

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["summary"]["p31_status"] == "P31_MULTIMODEL_POOL_READY":
            logger.info("P31: PASS")
            return 0
        logger.error("P31: FAIL (%s)", result["summary"]["p31_status"])
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
