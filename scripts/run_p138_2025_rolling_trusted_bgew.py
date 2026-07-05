"""
scripts/run_p138_2025_rolling_trusted_bgew.py — P138: 2025 Rolling Trusted BGEW.

Performs walk-forward (rolling) BGEW fusion over the full 2025 year using the
trusted prediction ledger from P137 and actuals from the raw Shandong PMOS data.

For each target_day D:
  1. Gather the previous 30 complete days of predictions + actuals.
  2. Compute per-model sMAPE_floor50 over those 30 days.
  3. Learn BGEW weights via ``compute_bgew_weights()``.
  4. Fuse predictions for day D using the learned weights.
  5. Evaluate the fused prediction against actuals.

Fallback: if fewer than 14 history days are available, use equal weights.

CANONICAL sMAPE_floor50:
    200 * mean(|y_f - yp_f| / (|y_f| + |yp_f|))
    where y_f = max(y, 50), yp_f = max(yp, 50)

Outputs:
  - .local_artifacts/p138_rolling_bgew/daily_metrics.csv
  - .local_artifacts/p138_rolling_bgew/weights.csv
  - .local_artifacts/p138_rolling_bgew/model_weight_summary.json
  - .local_artifacts/p138_rolling_bgew/bgew_2025_metrics.json

Status:
  BGEW_2025_IMPROVED        — bgew_smape < cfg05_smape
  BGEW_2025_NOT_IMPROVED    — bgew_smape >= cfg05_smape
  BGEW_2025_BLOCKED         — cannot run (single model, missing data, etc.)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fusion.unified_weight_learner import compute_bgew_weights

# ── Constants ─────────────────────────────────────────────────────────
LOOKBACK_DAYS = 30
MIN_HISTORY_DAYS = 14
SMAPE_FLOOR = 50.0

DEFAULT_LEDGER_PATH = os.path.join(
    REPO_ROOT, ".local_artifacts", "p137_trusted_2025", "ledger",
    "dayahead_prediction_ledger_2025_trusted.csv",
)
DEFAULT_RAW_DATA_PATH = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, ".local_artifacts", "p138_rolling_bgew")


# ── Canonical sMAPE_floor50 ──────────────────────────────────────────


def compute_smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the CANONICAL sMAPE with floor=50.

    Formula:
        200 * mean(|y_f - yp_f| / (|y_f| + |yp_f|))
    where y_f = max(y, 50), yp_f = max(yp, 50).

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        sMAPE_floor50 value (percentage scale, 0-200+).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_f = np.maximum(y_true, SMAPE_FLOOR)
    yp_f = np.maximum(y_pred, SMAPE_FLOOR)

    denom = np.abs(y_f) + np.abs(yp_f)
    mask = denom > 1e-10
    if not mask.any():
        return float("inf")

    return float(200.0 * np.mean(np.abs(y_f[mask] - yp_f[mask]) / denom[mask]))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ── Raw data loader ──────────────────────────────────────────────────


def _load_actuals(raw_data_path: str) -> pd.DataFrame:
    """Load actuals from the raw Shandong PMOS CSV (GBK encoded).

    Returns a DataFrame with columns: business_day, hour_business, y_true.
    """
    raw = pd.read_csv(raw_data_path, encoding="gbk")

    # Normalize Chinese column names
    from data.features.model_specific_features import normalize_raw_columns
    raw = normalize_raw_columns(raw)

    # Ensure ds column
    if "ds" not in raw.columns:
        # The first column is always the timestamp
        raw["ds"] = raw.iloc[:, 0]

    raw["ds"] = pd.to_datetime(raw["ds"], errors="coerce")

    # Add business time columns
    from data.business_day import add_business_time_columns
    raw = add_business_time_columns(raw, timestamp_col="ds")

    # Extract actuals: 日前电价 → y (which maps to da_anchor / y_true for dayahead)
    actuals = raw[["business_day", "hour_business", "y"]].copy()
    actuals = actuals.rename(columns={"y": "y_true"})
    actuals = actuals.dropna(subset=["y_true"])
    actuals["business_day"] = pd.to_datetime(actuals["business_day"]).dt.strftime("%Y-%m-%d")
    actuals["hour_business"] = actuals["hour_business"].astype(int)

    return actuals


# ── Core rolling BGEW logic ──────────────────────────────────────────


def _get_history_window(
    ledger: pd.DataFrame,
    actuals: pd.DataFrame,
    target_day: str,
    lookback: int = LOOKBACK_DAYS,
) -> Optional[pd.DataFrame]:
    """Get the history window of predictions + actuals for days < target_day.

    Returns a merged DataFrame with y_true and y_pred, or None if insufficient data.
    No-lookahead invariant: only days strictly before target_day are used.
    """
    target_dt = pd.Timestamp(target_day)

    # Filter predictions to history window
    pred_hist = ledger[
        pd.to_datetime(ledger["business_day"]) < target_dt
    ].copy()

    if len(pred_hist) == 0:
        return None

    # Keep only the last `lookback` unique days
    unique_days = sorted(pred_hist["business_day"].unique())
    if len(unique_days) > lookback:
        unique_days = unique_days[-lookback:]
    pred_hist = pred_hist[pred_hist["business_day"].isin(unique_days)]

    # Filter actuals to same window
    act_hist = actuals[
        pd.to_datetime(actuals["business_day"]) < target_dt
    ].copy()
    act_hist = act_hist[act_hist["business_day"].isin(unique_days)]

    if len(act_hist) == 0:
        return None

    # Merge on business_day + hour_business
    merge_keys = ["business_day", "hour_business"]
    merged = pd.merge(
        pred_hist,
        act_hist[merge_keys + ["y_true"]],
        on=merge_keys,
        how="inner",
    )

    return merged if len(merged) > 0 else None


def _compute_model_smape(
    merged: pd.DataFrame,
    models: List[str],
) -> Dict[str, float]:
    """Compute per-model sMAPE_floor50 over the merged history."""
    smape_dict: Dict[str, float] = {}
    for model in models:
        model_data = merged[merged["model_name"] == model]
        if len(model_data) == 0:
            continue
        y_true = model_data["y_true"].values
        y_pred = model_data["y_pred"].values
        min_len = min(len(y_true), len(y_pred))
        if min_len > 0:
            smape_dict[model] = compute_smape_floor50(y_true[:min_len], y_pred[:min_len])
    return smape_dict


def _fuse_predictions(
    day_preds: pd.DataFrame,
    weights: Dict[str, float],
    models: List[str],
) -> np.ndarray:
    """Fuse predictions for a single day using the given weights.

    Parameters
    ----------
    day_preds : DataFrame
        Predictions for one target_day, with model_name and y_pred columns.
    weights : dict
        {model_name: weight}.
    models : list
        Ordered list of model names.

    Returns
    -------
    np.ndarray
        Fused predictions, one per hour_business slot.
    """
    # Pivot: rows = hour_business, cols = model_name
    pivot = day_preds.pivot_table(
        index="hour_business",
        columns="model_name",
        values="y_pred",
        aggfunc="mean",
    )

    # Only use models that are both in weights and present in pivot
    available = [m for m in models if m in weights and m in pivot.columns]
    if not available:
        return pivot.mean(axis=1).values if len(pivot) > 0 else np.array([])

    # Weighted sum
    fused = np.zeros(len(pivot))
    total_weight = sum(weights[m] for m in available)
    for m in available:
        w = weights[m] / total_weight  # renormalize to available models
        fused += w * pivot[m].values

    return fused


# ── Main entry point ──────────────────────────────────────────────────


def run_p138_rolling_bgew(
    ledger_path: str = DEFAULT_LEDGER_PATH,
    raw_data_path: str = DEFAULT_RAW_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run 2025 Rolling Trusted BGEW.

    Parameters
    ----------
    ledger_path : str
        Path to the trusted prediction ledger from P137.
    raw_data_path : str
        Path to the raw Shandong PMOS CSV (GBK encoded).
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        Result dictionary with status and metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    result: dict[str, Any] = {
        "phase": "P138",
        "title": "2025 Rolling Trusted BGEW",
        "status": "STARTED",
        "reason_codes": [],
    }

    # ── Load inputs ───────────────────────────────────────────────────
    if not os.path.isfile(ledger_path):
        result["status"] = "BGEW_2025_BLOCKED"
        result["reason_codes"].append("LEDGER_FILE_MISSING")
        _save_outputs(result, output_dir)
        return result

    if not os.path.isfile(raw_data_path):
        result["status"] = "BGEW_2025_BLOCKED"
        result["reason_codes"].append("RAW_DATA_FILE_MISSING")
        _save_outputs(result, output_dir)
        return result

    ledger = pd.read_csv(ledger_path)
    ledger["business_day"] = pd.to_datetime(ledger["business_day"]).dt.strftime("%Y-%m-%d")
    ledger["hour_business"] = ledger["hour_business"].astype(int)

    actuals = _load_actuals(raw_data_path)

    # ── Check model count ─────────────────────────────────────────────
    models = sorted(ledger["model_name"].unique().tolist())
    result["models"] = models
    result["model_count"] = len(models)

    if len(models) < 2:
        result["status"] = "BGEW_2025_BLOCKED"
        result["reason_codes"].append(f"SINGLE_MODEL_COUNT:{len(models)}")
        _save_outputs(result, output_dir)
        return result

    logger.info("Models: %s", models)

    # ── Identify target days ──────────────────────────────────────────
    # In the dayahead context, business_day IS the target_day.
    target_days = sorted(ledger["business_day"].unique())
    result["total_target_days"] = len(target_days)
    logger.info("Target days: %d (%s .. %s)", len(target_days), target_days[0], target_days[-1])

    # ── Rolling walk-forward ──────────────────────────────────────────
    daily_metrics_rows: List[dict] = []
    weights_rows: List[dict] = []
    all_cfg05_errors: List[float] = []
    all_catboost_errors: List[float] = []
    all_bgew_errors: List[float] = []
    all_cfg05_preds: List[float] = []
    all_catboost_preds: List[float] = []
    all_bgew_preds: List[float] = []
    all_actuals: List[float] = []

    # Track cumulative weights for summary
    weight_accumulator: Dict[str, List[float]] = {m: [] for m in models}

    processed_days = 0
    skipped_days = 0

    for target_day in target_days:
        # Get history window
        merged = _get_history_window(ledger, actuals, target_day, LOOKBACK_DAYS)

        if merged is None:
            skipped_days += 1
            continue

        n_history_days = merged["business_day"].nunique()

        # Compute per-model sMAPE over history
        smape_dict = _compute_model_smape(merged, models)

        if len(smape_dict) < 2:
            skipped_days += 1
            continue

        # Learn BGEW weights
        if n_history_days < MIN_HISTORY_DAYS:
            # Fallback: equal weights
            weights = {m: 1.0 / len(smape_dict) for m in smape_dict}
            method = "equal_weights_fallback"
        else:
            weights = compute_bgew_weights(smape_dict)
            method = "bgew_rolling"

        # Record weights
        for m, w in weights.items():
            weights_rows.append({
                "target_day": target_day,
                "model_name": m,
                "weight": round(w, 6),
            })
            weight_accumulator[m].append(w)

        # Get predictions for target_day
        day_preds = ledger[ledger["business_day"] == target_day].copy()
        if len(day_preds) == 0:
            skipped_days += 1
            continue

        # Fuse predictions
        fused = _fuse_predictions(day_preds, weights, models)

        # Get actuals for target_day
        day_actuals = actuals[actuals["business_day"] == target_day].copy()
        if len(day_actuals) == 0:
            skipped_days += 1
            continue

        # Align by hour_business
        day_preds_pivot = day_preds.pivot_table(
            index="hour_business",
            columns="model_name",
            values="y_pred",
            aggfunc="mean",
        )
        day_actuals_aligned = day_actuals.set_index("hour_business")["y_true"]

        # Find common hours
        common_hours = day_preds_pivot.index.intersection(day_actuals_aligned.index)
        if len(common_hours) == 0:
            skipped_days += 1
            continue

        y_true_day = day_actuals_aligned.loc[common_hours].values

        # Per-model metrics for this day
        day_metrics: dict[str, Any] = {"target_day": target_day}

        for model in models:
            if model in day_preds_pivot.columns:
                y_pred_model = day_preds_pivot.loc[common_hours, model].values
                smape_m = compute_smape_floor50(y_true_day, y_pred_model)
                mae_m = compute_mae(y_true_day, y_pred_model)
                day_metrics[f"{model}_smape"] = round(smape_m, 4)
                day_metrics[f"{model}_mae"] = round(mae_m, 4)

                # Accumulate for overall metrics
                if "cfg05" in model:
                    all_cfg05_errors.extend(y_pred_model.tolist())
                    all_cfg05_preds.extend(y_pred_model.tolist())
                if "catboost" in model or "spike" in model:
                    all_catboost_errors.extend(y_pred_model.tolist())
                    all_catboost_preds.extend(y_pred_model.tolist())

        # BGEW fused metrics
        # Reconstruct fused aligned to common_hours
        fused_aligned = np.zeros(len(common_hours))
        total_w = sum(weights.get(m, 0) for m in models if m in day_preds_pivot.columns)
        for m in models:
            if m in day_preds_pivot.columns and m in weights:
                w = weights[m] / total_w if total_w > 0 else 0
                fused_aligned += w * day_preds_pivot.loc[common_hours, m].values

        bgew_smape = compute_smape_floor50(y_true_day, fused_aligned)
        bgew_mae = compute_mae(y_true_day, fused_aligned)

        # Use the first cfg05-like model name and first catboost-like model name
        cfg05_model = next((m for m in models if "cfg05" in m), models[0])
        catboost_model = next((m for m in models if "catboost" in m or "spike" in m), models[-1])

        day_metrics["bgew_smape"] = round(bgew_smape, 4)
        day_metrics["bgew_mae"] = round(bgew_mae, 4)
        day_metrics[f"{cfg05_model}_weight"] = round(weights.get(cfg05_model, 0), 6)
        day_metrics[f"{catboost_model}_weight"] = round(weights.get(catboost_model, 0), 6)

        daily_metrics_rows.append(day_metrics)

        # Accumulate for overall
        all_bgew_preds.extend(fused_aligned.tolist())
        all_actuals.extend(y_true_day.tolist())

        processed_days += 1
        if processed_days % 50 == 0:
            logger.info("Processed %d / %d days", processed_days, len(target_days))

    result["processed_days"] = processed_days
    result["skipped_days"] = skipped_days

    if processed_days == 0:
        result["status"] = "BGEW_2025_BLOCKED"
        result["reason_codes"].append("NO_DAYS_PROCESSED")
        _save_outputs(result, output_dir)
        return result

    # ── Build daily_metrics.csv ────────────────────────────────────────
    cfg05_model = next((m for m in models if "cfg05" in m), models[0])
    catboost_model = next((m for m in models if "catboost" in m or "spike" in m), models[-1])

    daily_metrics_df = pd.DataFrame(daily_metrics_rows)

    # Rename columns to match expected schema
    rename_map = {}
    if f"{cfg05_model}_smape" in daily_metrics_df.columns:
        rename_map[f"{cfg05_model}_smape"] = "cfg05_smape"
    if f"{catboost_model}_smape" in daily_metrics_df.columns:
        rename_map[f"{catboost_model}_smape"] = "catboost_smape"
    if f"{cfg05_model}_mae" in daily_metrics_df.columns:
        rename_map[f"{cfg05_model}_mae"] = "cfg05_mae"
    if f"{catboost_model}_mae" in daily_metrics_df.columns:
        rename_map[f"{catboost_model}_mae"] = "catboost_mae"
    if f"{cfg05_model}_weight" in daily_metrics_df.columns:
        rename_map[f"{cfg05_model}_weight"] = "cfg05_weight"
    if f"{catboost_model}_weight" in daily_metrics_df.columns:
        rename_map[f"{catboost_model}_weight"] = "catboost_weight"
    daily_metrics_df = daily_metrics_df.rename(columns=rename_map)

    daily_metrics_path = os.path.join(output_dir, "daily_metrics.csv")
    daily_metrics_df.to_csv(daily_metrics_path, index=False)
    result["daily_metrics_path"] = daily_metrics_path

    # ── Build weights.csv ─────────────────────────────────────────────
    weights_df = pd.DataFrame(weights_rows)
    weights_path = os.path.join(output_dir, "weights.csv")
    weights_df.to_csv(weights_path, index=False)
    result["weights_path"] = weights_path

    # ── Model weight summary ──────────────────────────────────────────
    weight_summary: Dict[str, float] = {}
    for m, w_list in weight_accumulator.items():
        if w_list:
            weight_summary[m] = round(float(np.mean(w_list)), 6)
        else:
            weight_summary[m] = 0.0

    weight_summary_path = os.path.join(output_dir, "model_weight_summary.json")
    with open(weight_summary_path, "w", encoding="utf-8") as f:
        json.dump(weight_summary, f, indent=2)
    result["weight_summary"] = weight_summary

    # ── Overall metrics ───────────────────────────────────────────────
    y_true_all = np.array(all_actuals)
    overall: Dict[str, Any] = {}

    # cfg05 overall
    if all_cfg05_preds:
        cfg05_preds_arr = np.array(all_cfg05_preds)
        min_len = min(len(y_true_all), len(cfg05_preds_arr))
        overall["cfg05_smape"] = round(compute_smape_floor50(y_true_all[:min_len], cfg05_preds_arr[:min_len]), 4)
        overall["cfg05_mae"] = round(compute_mae(y_true_all[:min_len], cfg05_preds_arr[:min_len]), 4)
    else:
        overall["cfg05_smape"] = None
        overall["cfg05_mae"] = None

    # catboost overall
    if all_catboost_preds:
        catboost_preds_arr = np.array(all_catboost_preds)
        min_len = min(len(y_true_all), len(catboost_preds_arr))
        overall["catboost_smape"] = round(compute_smape_floor50(y_true_all[:min_len], catboost_preds_arr[:min_len]), 4)
        overall["catboost_mae"] = round(compute_mae(y_true_all[:min_len], catboost_preds_arr[:min_len]), 4)
    else:
        overall["catboost_smape"] = None
        overall["catboost_mae"] = None

    # bgew overall
    if all_bgew_preds:
        bgew_preds_arr = np.array(all_bgew_preds)
        min_len = min(len(y_true_all), len(bgew_preds_arr))
        overall["bgew_smape"] = round(compute_smape_floor50(y_true_all[:min_len], bgew_preds_arr[:min_len]), 4)
        overall["bgew_mae"] = round(compute_mae(y_true_all[:min_len], bgew_preds_arr[:min_len]), 4)
    else:
        overall["bgew_smape"] = None
        overall["bgew_mae"] = None

    # Relative improvement
    if overall.get("cfg05_smape") is not None and overall.get("bgew_smape") is not None:
        cfg05_s = overall["cfg05_smape"]
        bgew_s = overall["bgew_smape"]
        if cfg05_s > 0:
            overall["relative_improvement_vs_cfg05"] = round(
                (cfg05_s - bgew_s) / cfg05_s * 100, 4
            )
        else:
            overall["relative_improvement_vs_cfg05"] = 0.0
    else:
        overall["relative_improvement_vs_cfg05"] = None

    overall["processed_days"] = processed_days
    overall["models"] = models

    overall_path = os.path.join(output_dir, "bgew_2025_metrics.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    result["overall_metrics"] = overall

    # ── Determine status ──────────────────────────────────────────────
    if overall.get("bgew_smape") is not None and overall.get("cfg05_smape") is not None:
        if overall["bgew_smape"] < overall["cfg05_smape"]:
            result["status"] = "BGEW_2025_IMPROVED"
        else:
            result["status"] = "BGEW_2025_NOT_IMPROVED"
    else:
        result["status"] = "BGEW_2025_BLOCKED"
        result["reason_codes"].append("METRICS_COMPUTE_FAILED")

    _save_outputs(result, output_dir)
    return result


def _save_outputs(result: dict, output_dir: str) -> None:
    """Save result manifest."""
    manifest_path = os.path.join(output_dir, "p138_rolling_bgew_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    result["manifest_path"] = manifest_path


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="P138: 2025 Rolling Trusted BGEW")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--raw-data", default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_p138_rolling_bgew(
        ledger_path=args.ledger,
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n=== P138: 2025 Rolling Trusted BGEW ===")
        print(f"  Status: {result['status']}")
        print(f"  Models: {result.get('models', [])}")
        print(f"  Processed days: {result.get('processed_days', 0)}")
        overall = result.get("overall_metrics", {})
        print(f"  cfg05 sMAPE:    {overall.get('cfg05_smape', 'N/A')}")
        print(f"  catboost sMAPE: {overall.get('catboost_smape', 'N/A')}")
        print(f"  BGEW sMAPE:     {overall.get('bgew_smape', 'N/A')}")
        print(f"  Improvement:    {overall.get('relative_improvement_vs_cfg05', 'N/A')}%")
        print(f"  Weight summary: {result.get('weight_summary', {})}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
