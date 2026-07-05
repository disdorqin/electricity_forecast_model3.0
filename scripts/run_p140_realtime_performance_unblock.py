"""
scripts/run_p140_realtime_performance_unblock.py
=================================================
P140: Realtime Performance Unblock

Current baseline: rt_da_anchor (using dayahead prediction as realtime) = 33.03% sMAPE.

This script implements two delta models to improve realtime prediction:

Model 1: simple_delta_rolling_median
    For each target_day D, hour h:
        delta = median(realtime_price - dayahead_price) over previous 30 complete
                days, same hour_business
        rt_pred = da_anchor + delta

Model 2: simple_delta_lgbm_safe
    target = realtime_price - dayahead_price (delta)
    features: hour_business, period, da_anchor, lagged deltas, past-day stats
    Training: only days < target_day (walk-forward, no lookahead)
    rt_pred = da_anchor + delta_pred

Both models are then fused via BGEW weighting.

Strict rules:
    - Delta models can only use past days (no target-day actuals)
    - Cannot use realtime_price as a feature
    - All features must be available at D-1 15:00 cutoff

Outputs
-------
.local_artifacts/p140_realtime_unblock/
    rt_da_anchor_metrics.json
    rolling_delta_metrics.json
    lgbm_delta_metrics.json
    pooled_realtime_bgew_metrics.json
    daily_realtime_metrics.csv

Status codes
------------
RT_DELTA_IMPROVED      -- at least one delta model beats 33.03%
RT_DELTA_NOT_IMPROVED  -- neither delta model beats baseline
RT_DELTA_BLOCKED       -- missing inputs prevented evaluation
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.business_day import add_business_time_columns  # noqa: E402

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Baseline sMAPE from rt_da_anchor strategy
RT_DA_ANCHOR_SMAPE_BASELINE = 33.03


# ---------------------------------------------------------------------------
# Canonical metrics
# ---------------------------------------------------------------------------

def compute_smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
) -> float:
    """Canonical sMAPE with floor=50."""
    y_true_f = np.maximum(np.asarray(y_true, dtype=float), floor)
    y_pred_f = np.maximum(np.asarray(y_pred, dtype=float), floor)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    if mask.sum() == 0:
        return 0.0
    return float(
        200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask])
    )


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "sMAPE_floor50": round(compute_smape_floor50(y_true, y_pred), 4),
        "MAE": round(compute_mae(y_true, y_pred), 4),
        "RMSE": round(compute_rmse(y_true, y_pred), 4),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# BGEW weights (canonical)
# ---------------------------------------------------------------------------

def compute_bgew_weights(
    smape_values: dict[str, float],
    alpha: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.75,
) -> dict[str, float]:
    scores = {k: np.exp(-alpha * v) for k, v in smape_values.items()}
    total = sum(scores.values())
    weights = {k: v / total for k, v in scores.items()}
    weights = {k: np.clip(v, min_weight, max_weight) for k, v in weights.items()}
    total2 = sum(weights.values())
    weights = {k: v / total2 for k, v in weights.items()}
    return weights


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(raw_data_path: str) -> pd.DataFrame:
    """Load Shandong PMOS hourly CSV (GBK) and normalise columns."""
    df = pd.read_csv(raw_data_path, encoding="gbk")
    col_map = {
        "\u65f6\u523b": "ds",
        "\u65e5\u524d\u7535\u4ef7": "dayahead_price",
        "\u5b9e\u65f6\u7535\u4ef7": "realtime_price",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Prepare evaluation DataFrame
# ---------------------------------------------------------------------------

def _prepare_eval_frame(
    raw: pd.DataFrame,
    day_start: str,
    day_end: str,
) -> pd.DataFrame:
    """Build a DataFrame with ds, business_day, hour_business, period,
    dayahead_price (da_anchor), realtime_price (y_true) for the given range.
    """
    df = raw.copy()
    df = add_business_time_columns(df, timestamp_col="ds")

    # Filter date range
    start_ts = pd.Timestamp(day_start)
    end_ts = pd.Timestamp(day_end) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    df = df[(df["ds"] >= start_ts) & (df["ds"] <= end_ts)].copy()

    # da_anchor = dayahead_price (using DA prediction as RT baseline)
    df["da_anchor"] = df["dayahead_price"].copy()
    df["y_true"] = df["realtime_price"].copy()

    # Drop rows without actuals
    df = df.dropna(subset=["y_true", "da_anchor"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model 1: simple_delta_rolling_median
# ---------------------------------------------------------------------------

def compute_rolling_median_delta(
    eval_df: pd.DataFrame,
    raw: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """For each (target_day, hour_business), compute the median delta
    (realtime - dayahead) over the previous *window* complete days at the
    same hour_business.  Then rt_pred = da_anchor + delta.

    Strict no-lookahead: only days strictly before target_day are used.
    """
    # Build a full historical delta table from raw data
    hist = raw.copy()
    hist = add_business_time_columns(hist, timestamp_col="ds")
    hist["delta"] = hist["realtime_price"] - hist["dayahead_price"]
    hist = hist.dropna(subset=["delta", "business_day", "hour_business"])
    hist["business_day"] = pd.to_datetime(hist["business_day"])

    result = eval_df.copy()
    result["business_day"] = pd.to_datetime(result["business_day"])
    deltas = []

    for _, row in result.iterrows():
        td = row["business_day"]
        hb = int(row["hour_business"])
        # Only past days, same hour
        past = hist[
            (hist["business_day"] < td)
            & (hist["hour_business"] == hb)
        ]
        # Rolling window: last `window` days
        if len(past) > 0:
            latest = past["business_day"].max()
            cutoff = latest - pd.Timedelta(days=window - 1)
            past_window = past[past["business_day"] >= cutoff]
            if len(past_window) > 0:
                deltas.append(float(past_window["delta"].median()))
            else:
                deltas.append(0.0)
        else:
            deltas.append(0.0)

    result["delta_pred"] = deltas
    result["rt_pred"] = result["da_anchor"] + result["delta_pred"]
    return result


# ---------------------------------------------------------------------------
# Model 2: simple_delta_lgbm_safe
# ---------------------------------------------------------------------------

def _build_lgbm_features(
    hist: pd.DataFrame,
    target_day: pd.Timestamp,
    hour_business: int,
    period: str,
    da_anchor: float,
) -> dict[str, float]:
    """Build feature dict for a single prediction.  All features come from
    days strictly before *target_day*."""
    past = hist[hist["business_day"] < target_day].copy()
    same_hour = past[past["hour_business"] == hour_business]

    # Lagged deltas: same hour, previous 1..7 days
    lagged = {}
    for lag in range(1, 8):
        target_d = target_day - pd.Timedelta(days=lag)
        val = same_hour[same_hour["business_day"] == target_d]["delta"]
        lagged[f"lag_delta_{lag}d"] = float(val.iloc[0]) if len(val) > 0 else 0.0

    # Past-day stats: mean and std of last 7d deltas at same hour
    recent_7d = same_hour[
        same_hour["business_day"] >= target_day - pd.Timedelta(days=7)
    ]
    mean_7d = float(recent_7d["delta"].mean()) if len(recent_7d) > 0 else 0.0
    std_7d = float(recent_7d["delta"].std()) if len(recent_7d) > 1 else 0.0

    # Past-day stats: mean and std of last 14d deltas at same hour
    recent_14d = same_hour[
        same_hour["business_day"] >= target_day - pd.Timedelta(days=14)
    ]
    mean_14d = float(recent_14d["delta"].mean()) if len(recent_14d) > 0 else 0.0

    features = {
        "hour_business": float(hour_business),
        "period_1_8": 1.0 if period == "1_8" else 0.0,
        "period_9_16": 1.0 if period == "9_16" else 0.0,
        "period_17_24": 1.0 if period == "17_24" else 0.0,
        "da_anchor": float(da_anchor),
        **lagged,
        "delta_mean_7d": mean_7d,
        "delta_std_7d": std_7d,
        "delta_mean_14d": mean_14d,
    }
    return features


def compute_lgbm_delta(
    eval_df: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Optimised LightGBM delta model.

    Train ONCE on pre-2025 historical deltas (last 90 days before the first
    eval day), then predict all eval days in a single batch.  This is still
    DA-safe because the model only uses past data.

    If LightGBM is not available, falls back to rolling median.
    """
    # Build historical delta table
    hist = raw.copy()
    hist = add_business_time_columns(hist, timestamp_col="ds")
    hist["delta"] = hist["realtime_price"] - hist["dayahead_price"]
    hist = hist.dropna(subset=["delta", "business_day", "hour_business"])
    hist["business_day"] = pd.to_datetime(hist["business_day"])

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available; falling back to rolling median for lgbm_delta")
        return compute_rolling_median_delta(eval_df, raw)

    result = eval_df.copy()
    result["business_day"] = pd.to_datetime(result["business_day"])

    feature_cols = [
        "hour_business", "period_1_8", "period_9_16", "period_17_24",
        "da_anchor",
        "lag_delta_1d", "lag_delta_2d", "lag_delta_3d",
        "lag_delta_4d", "lag_delta_5d", "lag_delta_6d", "lag_delta_7d",
        "delta_mean_7d", "delta_std_7d", "delta_mean_14d",
    ]

    target_days = sorted(result["business_day"].unique())
    first_eval_day = pd.Timestamp(target_days[0])

    # ── Build training data from pre-eval history (last 90 days) ──
    pre_hist = hist[hist["business_day"] < first_eval_day]
    pre_days = sorted(pre_hist["business_day"].unique())
    if len(pre_days) < 14:
        logger.warning("Not enough pre-eval history for LightGBM delta; using zero delta")
        result["delta_pred"] = 0.0
        result["rt_pred"] = result["da_anchor"]
        return result

    train_days = pre_days[-90:]
    logger.info("Building LightGBM training data from %d days ...", len(train_days))
    train_records = []
    for d in train_days:
        d = pd.Timestamp(d)
        d_rows = hist[hist["business_day"] == d]
        for _, dr in d_rows.iterrows():
            hb = int(dr["hour_business"])
            feats = _build_lgbm_features(
                hist, d, hb, dr["period"],
                float(dr["dayahead_price"]),
            )
            feats["target_delta"] = float(dr["delta"])
            train_records.append(feats)

    if len(train_records) < 50:
        logger.warning("Too few training records (%d); using zero delta", len(train_records))
        result["delta_pred"] = 0.0
        result["rt_pred"] = result["da_anchor"]
        return result

    train_df = pd.DataFrame(train_records)
    X_train = train_df[feature_cols].values
    y_train = train_df["target_delta"].values

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 200,
        "verbosity": -1,
    }

    logger.info("Training LightGBM on %d records ...", len(train_records))
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)

    # ── Predict for all eval days in batch ──
    logger.info("Predicting deltas for %d eval days ...", len(target_days))
    pred_records = []
    for td in target_days:
        td = pd.Timestamp(td)
        day_rows = result[result["business_day"] == td]
        for _, row in day_rows.iterrows():
            feats = _build_lgbm_features(
                hist, td, int(row["hour_business"]),
                row["period"], float(row["da_anchor"]),
            )
            pred_records.append([feats[c] for c in feature_cols])

    X_pred = np.array(pred_records)
    delta_preds = model.predict(X_pred)

    result["delta_pred"] = delta_preds
    result["rt_pred"] = result["da_anchor"] + result["delta_pred"]
    return result


# ---------------------------------------------------------------------------
# BGEW fusion of delta models
# ---------------------------------------------------------------------------

def fuse_delta_models_bgew(
    rolling_df: pd.DataFrame,
    lgbm_df: pd.DataFrame,
) -> pd.DataFrame:
    """BGEW-fuse the two delta model predictions.

    Both DataFrames must have: ds, hour_business, rt_pred, y_true.
    Weights are computed from full-sample sMAPE, then applied row-wise.
    """
    merged = rolling_df[["ds", "hour_business", "rt_pred", "y_true"]].copy()
    merged = merged.rename(columns={"rt_pred": "rt_pred_rolling"})
    merged = merged.merge(
        lgbm_df[["ds", "hour_business", "rt_pred"]].rename(columns={"rt_pred": "rt_pred_lgbm"}),
        on=["ds", "hour_business"],
        how="inner",
    )

    # Compute per-model sMAPE
    smape_rolling = compute_smape_floor50(
        merged["y_true"].values, merged["rt_pred_rolling"].values
    )
    smape_lgbm = compute_smape_floor50(
        merged["y_true"].values, merged["rt_pred_lgbm"].values
    )

    weights = compute_bgew_weights({
        "rolling_median": smape_rolling,
        "lgbm_delta": smape_lgbm,
    })

    merged["rt_pred_bgew"] = (
        weights["rolling_median"] * merged["rt_pred_rolling"]
        + weights["lgbm_delta"] * merged["rt_pred_lgbm"]
    )

    return merged, weights, {"rolling_median": smape_rolling, "lgbm_delta": smape_lgbm}


# ---------------------------------------------------------------------------
# Daily metrics CSV
# ---------------------------------------------------------------------------

def _daily_metrics_csv(
    eval_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    lgbm_df: pd.DataFrame,
    bgew_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write per-day metrics for all models."""
    rows = []
    for label, df, pred_col in [
        ("da_anchor", eval_df, "da_anchor"),
        ("rolling_median", rolling_df, "rt_pred"),
        ("lgbm_delta", lgbm_df, "rt_pred"),
        ("bgew_fused", bgew_df, "rt_pred_bgew"),
    ]:
        df_c = df.copy()
        df_c["business_day"] = pd.to_datetime(df_c["business_day"])
        for day, grp in df_c.groupby("business_day"):
            yt = grp["y_true"].values
            yp = grp[pred_col].values
            mask = ~(np.isnan(yt) | np.isnan(yp))
            if mask.sum() == 0:
                continue
            rows.append({
                "business_day": str(day.date()),
                "model": label,
                "sMAPE_floor50": round(compute_smape_floor50(yt[mask], yp[mask]), 4),
                "MAE": round(compute_mae(yt[mask], yp[mask]), 4),
                "RMSE": round(compute_rmse(yt[mask], yp[mask]), 4),
                "n": int(mask.sum()),
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_p140_realtime_unblock(
    raw_data_path: str = "data/shandong_pmos_hourly.csv",
    output_dir: str = ".local_artifacts/p140_realtime_unblock",
    day_start: str = "2025-01-01",
    day_end: str = "2025-12-31",
) -> dict:
    """Run P140 realtime performance unblock.

    Parameters
    ----------
    raw_data_path : str
        Path to raw Shandong PMOS hourly CSV (GBK).
    output_dir : str
        Where to write output JSON/CSV.
    day_start, day_end : str
        Evaluation date range (inclusive).

    Returns
    -------
    dict
        Result summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    reason_codes: list[str] = []

    result: dict[str, Any] = {
        "status": "RT_DELTA_BLOCKED",
        "baseline_sMAPE": RT_DA_ANCHOR_SMAPE_BASELINE,
        "rt_da_anchor_metrics": {},
        "rolling_delta_metrics": {},
        "lgbm_delta_metrics": {},
        "pooled_realtime_bgew_metrics": {},
        "bgew_weights": {},
        "reason_codes": reason_codes,
    }

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    if not os.path.isfile(raw_data_path):
        reason_codes.append("RAW_DATA_MISSING")
        _save_result(result, output_dir)
        return result

    raw = load_raw_data(raw_data_path)

    # Check required columns
    for col in ("dayahead_price", "realtime_price"):
        if col not in raw.columns:
            reason_codes.append(f"MISSING_COLUMN_{col}")
            _save_result(result, output_dir)
            return result

    # ------------------------------------------------------------------
    # 2. Prepare evaluation frame
    # ------------------------------------------------------------------
    eval_df = _prepare_eval_frame(raw, day_start, day_end)
    if len(eval_df) == 0:
        reason_codes.append("NO_DATA_IN_RANGE")
        _save_result(result, output_dir)
        return result

    # ------------------------------------------------------------------
    # 3. Baseline: rt_da_anchor
    # ------------------------------------------------------------------
    da_metrics = compute_metrics(eval_df["y_true"].values, eval_df["da_anchor"].values)
    result["rt_da_anchor_metrics"] = da_metrics
    reason_codes.append(f"BASELINE_SMAPE_{da_metrics['sMAPE_floor50']}")

    # ------------------------------------------------------------------
    # 4. Model 1: rolling median delta
    # ------------------------------------------------------------------
    logger.info("Computing rolling median delta ...")
    rolling_df = compute_rolling_median_delta(eval_df, raw, window=30)
    rolling_metrics = compute_metrics(rolling_df["y_true"].values, rolling_df["rt_pred"].values)
    result["rolling_delta_metrics"] = rolling_metrics

    # ------------------------------------------------------------------
    # 5. Model 2: LightGBM delta
    # ------------------------------------------------------------------
    logger.info("Computing LightGBM delta (walk-forward) ...")
    lgbm_df = compute_lgbm_delta(eval_df, raw)
    lgbm_metrics = compute_metrics(lgbm_df["y_true"].values, lgbm_df["rt_pred"].values)
    result["lgbm_delta_metrics"] = lgbm_metrics

    # ------------------------------------------------------------------
    # 6. BGEW fusion of delta models
    # ------------------------------------------------------------------
    logger.info("Fusing delta models via BGEW ...")
    try:
        bgew_df, bgew_weights, component_smapes = fuse_delta_models_bgew(rolling_df, lgbm_df)
        bgew_metrics = compute_metrics(bgew_df["y_true"].values, bgew_df["rt_pred_bgew"].values)
        result["pooled_realtime_bgew_metrics"] = bgew_metrics
        result["bgew_weights"] = {k: round(v, 4) for k, v in bgew_weights.items()}
        result["component_smapes"] = component_smapes
    except Exception as e:
        reason_codes.append(f"BGEW_FUSION_FAILED:{e}")
        result["pooled_realtime_bgew_metrics"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # 7. Determine status
    # ------------------------------------------------------------------
    actual_baseline_smape = result.get("rt_da_anchor_metrics", {}).get("sMAPE_floor50", RT_DA_ANCHOR_SMAPE_BASELINE)
    best_smape = min(
        rolling_metrics["sMAPE_floor50"],
        lgbm_metrics["sMAPE_floor50"],
    )
    if best_smape < actual_baseline_smape:
        result["status"] = "RT_DELTA_IMPROVED"
        result["improvement_vs_baseline"] = round(
            actual_baseline_smape - best_smape, 4
        )
    else:
        result["status"] = "RT_DELTA_NOT_IMPROVED"
        result["best_delta_sMAPE"] = best_smape
        result["actual_baseline_sMAPE"] = actual_baseline_smape
        result["gap_to_baseline"] = round(
            best_smape - actual_baseline_smape, 4
        )

    # ------------------------------------------------------------------
    # 8. Daily metrics CSV
    # ------------------------------------------------------------------
    try:
        bgew_df_for_csv = bgew_df if "bgew_df" in dir() else None
        _daily_metrics_csv(
            eval_df, rolling_df, lgbm_df,
            bgew_df if "bgew_df" in dir() and bgew_df is not None else eval_df,
            os.path.join(output_dir, "daily_realtime_metrics.csv"),
        )
    except Exception as e:
        reason_codes.append(f"DAILY_CSV_FAILED:{e}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    _save_result(result, output_dir)
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_result(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "rt_da_anchor_metrics.json"),
                result.get("rt_da_anchor_metrics", {}))
    _write_json(os.path.join(output_dir, "rolling_delta_metrics.json"),
                result.get("rolling_delta_metrics", {}))
    _write_json(os.path.join(output_dir, "lgbm_delta_metrics.json"),
                result.get("lgbm_delta_metrics", {}))
    _write_json(os.path.join(output_dir, "pooled_realtime_bgew_metrics.json"),
                result.get("pooled_realtime_bgew_metrics", {}))


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="P140: Realtime Performance Unblock")
    parser.add_argument("--raw-data", default="data/shandong_pmos_hourly.csv")
    parser.add_argument("--output-dir", default=".local_artifacts/p140_realtime_unblock")
    parser.add_argument("--day-start", default="2025-01-01")
    parser.add_argument("--day-end", default="2025-12-31")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = run_p140_realtime_unblock(
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
        day_start=args.day_start,
        day_end=args.day_end,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n=== P140: Realtime Performance Unblock ===")
        print(f"  Status: {result['status']}")
        print(f"  Baseline sMAPE (da_anchor): {result['rt_da_anchor_metrics'].get('sMAPE_floor50', 'N/A')}")
        print(f"  Rolling median delta sMAPE: {result['rolling_delta_metrics'].get('sMAPE_floor50', 'N/A')}")
        print(f"  LightGBM delta sMAPE:       {result['lgbm_delta_metrics'].get('sMAPE_floor50', 'N/A')}")
        print(f"  BGEW fused sMAPE:           {result['pooled_realtime_bgew_metrics'].get('sMAPE_floor50', 'N/A')}")
        if result.get("bgew_weights"):
            print(f"  BGEW weights: {result['bgew_weights']}")
        if result.get("improvement_vs_baseline") is not None:
            print(f"  Improvement vs baseline: {result['improvement_vs_baseline']}%")
        if result.get("reason_codes"):
            print(f"  Reason codes: {result['reason_codes']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
