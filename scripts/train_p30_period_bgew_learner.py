"""
scripts/train_p30_period_bgew_learner.py — P30 period-based BGEW fusion learner.

Trains a period-based BGEW weight learner that assigns per-period weights
to each model in the fusion pool based on rolling sMAPE_floor50 performance.

Weight formula::

    score_m = exp(-alpha * smape_m)
    weight_m = score_m / sum(score)

Constraints::

    min_weight = 0.05
    max_weight = 0.75
    cfg05_min_prior = 0.30 (until more models prove better)
    renormalize after clipping

If only 1 real model is available, the learner refuses to train and reports
P30_LEARNER_BLOCKED_SINGLE_MODEL.

Usage::

    python -m scripts.train_p30_period_bgew_learner \\
        --actual-ledger .local_artifacts/p26_p30_fusion/ledgers/actual_ledger.csv \\
        --prediction-ledger .local_artifacts/p26_p30_fusion/ledgers/prediction_ledger.csv \\
        --work-dir .local_artifacts/p26_p30_fusion \\
        --rolling-days 30 \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from scripts.run_p16_cfg05_30d_walkforward_backtest import compute_smape_floor50

logger = logging.getLogger(__name__)

# ── Statuses ───────────────────────────────────────────────────────────────
P30_PERIOD_BGEW_TRAINED = "P30_PERIOD_BGEW_TRAINED"
P30_LEARNER_BLOCKED_SINGLE_MODEL = "P30_LEARNER_BLOCKED_SINGLE_MODEL"
P30_LEARNER_BLOCKED_NO_DATA = "P30_LEARNER_BLOCKED_NO_DATA"
P30_LEARNER_BLOCKED_NO_PREDICTIONS = "P30_LEARNER_BLOCKED_NO_PREDICTIONS"

# ── Constants ──────────────────────────────────────────────────────────────
VALID_PERIODS = ["1_8", "9_16", "17_24"]
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.75
CFG05_MIN_PRIOR = 0.30
DEFAULT_ALPHA = 0.05  # smoothing parameter for exp weighting
DEFAULT_ROLLING_DAYS = 30

# Labels that allow a model into the learner
ALLOWED_READINESS_LABELS = {
    "REAL_READY", "REAL_24H_READY", "BACKTESTED", "TRAINABLE_LOCAL",
}

# Labels that block a model from the learner
BLOCKED_READINESS_LABELS = {
    "DRY_RUN", "STUB", "DATA_MISSING", "INVALID_BANNED", "ARTIFACT_MISSING",
}

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p26_p30_fusion")


def _compute_period_smape(
    prediction_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    period: str,
    rolling_days: int,
) -> dict[str, float]:
    """Compute rolling sMAPE_floor50 per model for a given period.

    Returns dict {model_name: smape_floor50}.
    """
    # Merge predictions with actuals
    merged = pd.merge(
        prediction_df,
        actual_df[["task", "target_day", "business_day", "hour_business", "y_true"]],
        on=["task", "target_day", "business_day", "hour_business"],
        how="inner",
    )

    # Filter to period
    period_data = merged[merged["period"] == period].copy()
    if len(period_data) == 0:
        return {}

    # Drop null y_true
    period_data = period_data.dropna(subset=["y_true"])
    if len(period_data) == 0:
        return {}

    # Limit to rolling window
    if "business_day" in period_data.columns:
        max_bd = period_data["business_day"].max()
        cutoff = (pd.Timestamp(max_bd) - pd.Timedelta(days=rolling_days)).strftime("%Y-%m-%d")
        period_data = period_data[period_data["business_day"] >= cutoff]

    # Compute per-model sMAPE
    model_scores = {}
    for model_name, grp in period_data.groupby("model_name"):
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values
        if len(y_true) < 5:
            continue
        smape = compute_smape_floor50(y_true, y_pred)
        model_scores[model_name] = smape

    return model_scores


def _compute_weights(
    smape_scores: dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    min_weight: float = MIN_WEIGHT,
    max_weight: float = MAX_WEIGHT,
    cfg05_min_prior: float = CFG05_MIN_PRIOR,
) -> dict[str, float]:
    """Compute model weights from sMAPE scores with clipping and renormalization.

    Parameters
    ----------
    smape_scores : dict
        {model_name: sMAPE_floor50}
    alpha : float
        Smoothing parameter.
    min_weight : float
        Minimum weight floor.
    max_weight : float
        Maximum weight cap.
    cfg05_min_prior : float
        Minimum weight for cfg05 champion.

    Returns
    -------
    dict
        {model_name: weight} — sums to 1.0 after renormalization.
    """
    if not smape_scores:
        return {}

    # Raw scores: exp(-alpha * smape)
    raw_scores = {m: math.exp(-alpha * s) for m, s in smape_scores.items()}
    total_score = sum(raw_scores.values())
    if total_score < 1e-10:
        # Equal weight fallback
        n = len(smape_scores)
        return {m: 1.0 / n for m in smape_scores}

    weights = {m: s / total_score for m, s in raw_scores.items()}

    # Apply cfg05 minimum prior
    if "lightgbm_cfg05_dayahead" in weights:
        if weights["lightgbm_cfg05_dayahead"] < cfg05_min_prior:
            weights["lightgbm_cfg05_dayahead"] = cfg05_min_prior

    # Clip to [min_weight, max_weight]
    for m in weights:
        weights[m] = max(min_weight, min(max_weight, weights[m]))

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {m: w / total for m, w in weights.items()}

    return weights


def _compute_fusion(
    prediction_df: pd.DataFrame,
    weights: dict[str, float],
    fusion_method: str = "bgew_skeleton",
) -> pd.DataFrame:
    """Compute fused prices from predictions and weights.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Predictions from multiple models with columns:
        task, target_day, business_day, ds, hour_business, period, y_pred, model_name
    weights : dict
        {model_name: weight}
    fusion_method : str
        Fusion method label.

    Returns
    -------
    pd.DataFrame
        Fusion output with fused_price per (task, target_day, business_day, ds, hour_business).
    """
    if len(prediction_df) == 0:
        return pd.DataFrame()

    # Weight each model's prediction
    pred = prediction_df.copy()
    pred["weighted_pred"] = pred.apply(
        lambda row: row["y_pred"] * weights.get(row["model_name"], 0), axis=1
    )

    # Group by canonical key and sum weighted predictions
    group_keys = ["task", "target_day", "business_day", "ds", "hour_business", "period"]
    available_keys = [k for k in group_keys if k in pred.columns]
    fusion = pred.groupby(available_keys, as_index=False).agg(
        fused_price=("weighted_pred", "sum"),
    )

    # Add metadata
    weights_json = json.dumps(weights)
    included = ";".join(sorted(weights.keys()))
    fusion["weights_json"] = weights_json
    fusion["included_models"] = included
    fusion["excluded_models"] = ""
    fusion["fusion_method"] = fusion_method
    fusion["learner_version"] = "0.2.0-period-bgew"
    fusion["readiness_mode"] = "REAL"
    fusion["reason_codes"] = f"period_bgew_alpha_{DEFAULT_ALPHA}"

    return fusion


def train_p30_period_bgew_learner(
    actual_ledger_path: Optional[str] = None,
    prediction_ledger_path: Optional[str] = None,
    prediction_df: Optional[pd.DataFrame] = None,
    work_dir: Optional[str] = None,
    rolling_days: int = DEFAULT_ROLLING_DAYS,
    alpha: float = DEFAULT_ALPHA,
    model_pool_readiness: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Train the period-based BGEW fusion learner.

    Parameters
    ----------
    actual_ledger_path : str
        Path to actual ledger CSV.
    prediction_ledger_path : str
        Path to prediction ledger CSV.
    prediction_df : pd.DataFrame, optional
        Pre-loaded prediction DataFrame (overrides prediction_ledger_path).
    work_dir : str
        Output directory.
    rolling_days : int
        Rolling window for weight computation.
    alpha : float
        Smoothing parameter.
    model_pool_readiness : dict, optional
        {model_id: readiness_label} from P29 audit.

    Returns
    -------
    dict
        Complete learner training report.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    result: dict[str, Any] = {
        "actual_ledger_path": actual_ledger_path,
        "prediction_ledger_path": prediction_ledger_path,
        "work_dir": work_dir,
        "rolling_days": rolling_days,
        "alpha": alpha,
        "n_models_available": 0,
        "n_models_in_pool": 0,
        "period_weights": {},
        "weight_ledger_path": None,
        "fusion_ledger_path": None,
        "fusion_smape_floor50": None,
        "cfg05_alone_smape_floor50": None,
        "fusion_improvement_vs_cfg05": None,
        "negative_period_analysis": None,
        "final_status": None,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # ── Load actual ledger ──
    actual_df = None
    if actual_ledger_path and os.path.isfile(actual_ledger_path):
        try:
            actual_df = pd.read_csv(actual_ledger_path)
            result["reason_codes"].append(f"ACTUAL_LEDGER_LOADED:{len(actual_df)}_rows")
        except Exception as e:
            result["reason_codes"].append(f"ACTUAL_LEDGER_LOAD_FAILED:{e}")

    if actual_df is None or len(actual_df) == 0:
        result["final_status"] = P30_LEARNER_BLOCKED_NO_DATA
        result["reason_codes"].append("NO_ACTUAL_DATA")
        return result

    # ── Load prediction data ──
    pred_df = prediction_df
    if pred_df is None and prediction_ledger_path and os.path.isfile(prediction_ledger_path):
        try:
            pred_df = pd.read_csv(prediction_ledger_path)
            result["reason_codes"].append(f"PREDICTION_LEDGER_LOADED:{len(pred_df)}_rows")
        except Exception as e:
            result["reason_codes"].append(f"PREDICTION_LEDGER_LOAD_FAILED:{e}")

    if pred_df is None or len(pred_df) == 0:
        result["final_status"] = P30_LEARNER_BLOCKED_NO_PREDICTIONS
        result["reason_codes"].append("NO_PREDICTION_DATA")
        return result

    # ── Filter models by readiness ──
    available_models = pred_df["model_name"].unique().tolist() if "model_name" in pred_df.columns else []
    result["n_models_available"] = len(available_models)

    # If model_pool_readiness is given, filter out blocked models
    eligible_models = []
    for m in available_models:
        if model_pool_readiness:
            label = model_pool_readiness.get(m, "UNKNOWN")
            if label in BLOCKED_READINESS_LABELS:
                result["reason_codes"].append(f"MODEL_BLOCKED:{m}:{label}")
                continue
        eligible_models.append(m)

    result["n_models_in_pool"] = len(eligible_models)

    # ── Check if we have enough models ──
    if len(eligible_models) < 2:
        result["final_status"] = P30_LEARNER_BLOCKED_SINGLE_MODEL
        result["reason_codes"].append(
            f"ONLY_{len(eligible_models)}_MODEL(S):_{'_'.join(eligible_models) if eligible_models else 'NONE'}"
        )
        result["reason_codes"].append("CANNOT_FAKE_MULTI_MODEL_FUSION")
        return result

    # ── Compute per-period weights ──
    all_period_weights: dict[str, dict[str, float]] = {}
    all_weight_rows: list[dict[str, Any]] = []
    now_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for period in VALID_PERIODS:
        smape_scores = _compute_period_smape(pred_df, actual_df, period, rolling_days)
        if not smape_scores:
            result["reason_codes"].append(f"NO_SCORES_FOR_PERIOD:{period}")
            continue

        weights = _compute_weights(smape_scores, alpha=alpha)
        all_period_weights[period] = weights

        # Build weight ledger rows
        for model_name, weight in weights.items():
            all_weight_rows.append({
                "task": "dayahead",
                "target_day": "all",
                "business_day": "all",
                "ds": pd.Timestamp.now(),
                "hour_business": 0,
                "period": period,
                "model_name": model_name,
                "weight": round(weight, 6),
                "fusion_method": "bgew_skeleton",
                "learner_version": "0.2.0-period-bgew",
                "weight_source": f"rolling_{rolling_days}d_smape",
                "reason_codes": f"smape={smape_scores.get(model_name, 'N/A'):.4f}",
                "run_id": "p30_period_bgew",
                "created_at": now_str,
                "updated_at": now_str,
            })

    result["period_weights"] = all_period_weights

    if not all_period_weights:
        result["final_status"] = P30_LEARNER_BLOCKED_NO_DATA
        result["reason_codes"].append("NO_PERIOD_WEIGHTS_COMPUTED")
        return result

    # ── Save weight ledger ──
    ledger_dir = os.path.join(work_dir, "ledgers")
    os.makedirs(ledger_dir, exist_ok=True)

    if all_weight_rows:
        weight_df = pd.DataFrame(all_weight_rows)
        weight_path = os.path.join(ledger_dir, "weight_ledger.csv")
        weight_df.to_csv(weight_path, index=False)
        result["weight_ledger_path"] = weight_path

    # ── Compute fusion ──
    # Use average weights across periods for simplicity
    avg_weights: dict[str, float] = {}
    for period, weights in all_period_weights.items():
        for m, w in weights.items():
            avg_weights[m] = avg_weights.get(m, 0) + w
    n_periods = len(all_period_weights)
    if n_periods > 0:
        avg_weights = {m: w / n_periods for m, w in avg_weights.items()}
        # Renormalize
        total = sum(avg_weights.values())
        if total > 0:
            avg_weights = {m: w / total for m, w in avg_weights.items()}

    fusion_df = _compute_fusion(pred_df, avg_weights)
    if len(fusion_df) > 0:
        fusion_path = os.path.join(ledger_dir, "fusion_ledger.csv")
        fusion_df.to_csv(fusion_path, index=False)
        result["fusion_ledger_path"] = fusion_path

        # Compute fusion sMAPE
        fusion_merged = pd.merge(
            fusion_df,
            actual_df[["task", "target_day", "business_day", "hour_business", "y_true"]],
            on=["task", "target_day", "business_day", "hour_business"],
            how="inner",
        )
        fusion_merged = fusion_merged.dropna(subset=["y_true"])
        if len(fusion_merged) > 0:
            result["fusion_smape_floor50"] = round(
                compute_smape_floor50(fusion_merged["y_true"].values, fusion_merged["fused_price"].values), 4
            )

    # Compute cfg05-alone sMAPE for comparison
    cfg05_preds = pred_df[pred_df["model_name"] == "lightgbm_cfg05_dayahead"]
    if len(cfg05_preds) > 0:
        cfg05_merged = pd.merge(
            cfg05_preds,
            actual_df[["task", "target_day", "business_day", "hour_business", "y_true"]],
            on=["task", "target_day", "business_day", "hour_business"],
            how="inner",
        )
        cfg05_merged = cfg05_merged.dropna(subset=["y_true"])
        if len(cfg05_merged) > 0:
            result["cfg05_alone_smape_floor50"] = round(
                compute_smape_floor50(cfg05_merged["y_true"].values, cfg05_merged["y_pred"].values), 4
            )

    # Improvement
    if result["fusion_smape_floor50"] is not None and result["cfg05_alone_smape_floor50"] is not None:
        delta = result["cfg05_alone_smape_floor50"] - result["fusion_smape_floor50"]
        result["fusion_improvement_vs_cfg05"] = {
            "cfg05_smape": result["cfg05_alone_smape_floor50"],
            "fusion_smape": result["fusion_smape_floor50"],
            "delta_pp": round(delta, 4),
            "direction": "IMPROVED" if delta > 0 else "NO_IMPROVEMENT",
        }

    # ── Negative period analysis ──
    neg_analysis = {}
    for period in VALID_PERIODS:
        period_actual = actual_df[actual_df["period"] == period] if "period" in actual_df.columns else pd.DataFrame()
        if len(period_actual) > 0:
            neg_count = (period_actual["y_true"] < 0).sum()
            neg_analysis[period] = {
                "total_rows": len(period_actual),
                "negative_rows": int(neg_count),
                "negative_pct": round(100.0 * neg_count / len(period_actual), 2) if len(period_actual) > 0 else 0,
            }
    result["negative_period_analysis"] = neg_analysis

    # ── Final status ──
    result["final_status"] = P30_PERIOD_BGEW_TRAINED
    result["reason_codes"].append(f"MODELS_IN_POOL:{result['n_models_in_pool']}")
    result["reason_codes"].append(f"PERIODS_TRAINED:{len(all_period_weights)}")

    # Forbidden files check
    work_dir_norm = os.path.abspath(work_dir).replace("\\", "/")
    if not (any(a.lstrip(".") in work_dir_norm for a in (".local_artifacts",)) or os.path.isabs(work_dir)):
        result["forbidden_files_check"] = "FAIL"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P30 Period BGEW Learner Report")
    print("=" * 60)
    print(f"  Models available:   {result['n_models_available']}")
    print(f"  Models in pool:     {result['n_models_in_pool']}")
    print(f"  Rolling days:       {result['rolling_days']}")
    print(f"  Alpha:              {result['alpha']}")
    if result["period_weights"]:
        for period, weights in result["period_weights"].items():
            print(f"  Period {period}:")
            for m, w in sorted(weights.items()):
                print(f"    {m:30s} = {w:.4f}")
    if result["fusion_smape_floor50"] is not None:
        print(f"  Fusion sMAPE:       {result['fusion_smape_floor50']:.4f}%")
    if result["cfg05_alone_smape_floor50"] is not None:
        print(f"  cfg05 alone sMAPE:  {result['cfg05_alone_smape_floor50']:.4f}%")
    imp = result.get("fusion_improvement_vs_cfg05")
    if imp:
        print(f"  vs cfg05:           {imp['delta_pp']:+.4f}pp ({imp['direction']})")
    print(f"  Weight ledger:      {result['weight_ledger_path']}")
    print(f"  Fusion ledger:      {result['fusion_ledger_path']}")
    print(f"  Final status:       {result['final_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P30: period-based BGEW fusion learner.")
    p.add_argument("--actual-ledger", type=str, default=None)
    p.add_argument("--prediction-ledger", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--rolling-days", type=int, default=DEFAULT_ROLLING_DAYS)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--json", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    result = train_p30_period_bgew_learner(
        actual_ledger_path=args.actual_ledger,
        prediction_ledger_path=args.prediction_ledger,
        work_dir=args.work_dir,
        rolling_days=args.rolling_days,
        alpha=args.alpha,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
