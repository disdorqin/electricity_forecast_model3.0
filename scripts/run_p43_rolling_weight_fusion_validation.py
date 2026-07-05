"""
scripts/run_p43_rolling_weight_fusion_validation.py — P43: Rolling/no-lookahead weight validation.

Validates fusion weights with two approaches:
1. Train/test split (first 20 days / last 9 days)
2. Rolling expanding window (per day, weights use only days < D)

Usage::

    python -m scripts.run_p43_rolling_weight_fusion_validation --json

Options::

    --work-dir PATH     Model artifacts dir.
    --alpha FLOAT       BGEW alpha (default: 0.5).
    --warmup-days INT   Minimum days before rolling starts (default: 7).
    --json              Output JSON report.
    --strict            Exit non-zero on failure.
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

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")

# Trusted pool (excludes stage3)
_TRUSTED_MODELS = [
    "lightgbm_cfg05_dayahead",
    "best_two_average",
    "catboost_sota",
    "catboost_spike_residual",
]

_PERIODS = ["1_8", "9_16", "17_24"]


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_f = np.maximum(y_true, 50.0)
    y_pred_f = np.maximum(y_pred, 50.0)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return float(200.0 * np.mean(np.abs(y_true_f - y_pred_f) / denom))


def _compute_weights(
    df: pd.DataFrame,
    alpha: float,
    model_list: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-period BGEW weights from a DataFrame."""
    weights = {}
    for period in _PERIODS:
        pdata = df[df["period"] == period]
        scores = {}
        for m in model_list:
            md = pdata[pdata["model_name"] == m]
            if len(md) < 10:
                continue
            s = smape_floor50(md["y_true"].values, md["y_pred"].values)
            scores[m] = np.exp(-alpha * s)
        total = sum(scores.values())
        if total <= 0:
            continue
        raw_w = {m: s / total for m, s in scores.items()}
        constrained = {}
        for m, w in raw_w.items():
            w = max(w, 0.05)
            if m == "lightgbm_cfg05_dayahead":
                w = max(w, 0.30)
            constrained[m] = w
        total_c = sum(constrained.values())
        constrained = {m: w / total_c for m, w in constrained.items()}
        for m in constrained:
            if constrained[m] > 0.75:
                constrained[m] = 0.75
        total_f = sum(constrained.values())
        weights[period] = {m: w / total_f for m, w in constrained.items()}
    return weights


def _apply_weights(
    group_df: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    model_list: list[str],
) -> pd.DataFrame:
    """Apply per-period weights to a DataFrame and return fused rows."""
    rows = []
    for (td, bd, hb), group in group_df.groupby(["target_day", "business_day", "hour_business"]):
        period = group["period"].iloc[0]
        w = weights.get(period, {})
        if not w:
            continue
        fp, tw = 0.0, 0.0
        for _, row in group.iterrows():
            mname = row["model_name"]
            if mname not in model_list:
                continue
            mw = w.get(mname, 0)
            if mw > 0 and not np.isnan(row["y_pred"]):
                fp += mw * row["y_pred"]
                tw += mw
        if tw > 0:
            rows.append({
                "y_true": group["y_true"].iloc[0],
                "fused_price": round(fp / tw, 4),
            })
    return pd.DataFrame(rows)


def run_rolling_validation(
    work_dir: Optional[str] = None,
    alpha: float = 0.5,
    warmup_days: int = 7,
    trusted_models: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run rolling and split weight validation."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledger")
    model_list = trusted_models if trusted_models is not None else list(_TRUSTED_MODELS)

    result: dict[str, Any] = {
        "phase": "P43",
        "alpha": alpha,
        "warmup_days": warmup_days,
        "trusted_models": list(model_list),
        "full_period": {},
        "split": {},
        "rolling": {},
        "summary": {
            "p43_status": "P43_NOT_STARTED",
        },
        "reason_codes": [],
    }

    # Load data
    pred_path = os.path.join(ledger_dir, "prediction_ledger_30d.csv")
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")
    if not os.path.isfile(pred_path) or not os.path.isfile(actual_path):
        result["reason_codes"].append("DATA_MISSING")
        result["summary"]["p43_status"] = "P43_DATA_MISSING"
        return result

    pred = pd.read_csv(pred_path)
    actual = pd.read_csv(actual_path)

    merged = pred.merge(
        actual[["task", "target_day", "business_day", "hour_business", "y_true"]],
        on=["task", "target_day", "business_day", "hour_business"],
        how="inner",
    )
    merged = merged.dropna(subset=["y_true"])
    merged["period"] = merged["hour_business"].apply(
        lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
    )

    all_days = sorted(merged["target_day"].unique())
    result["all_days"] = list(all_days)
    result["total_days"] = len(all_days)
    result["reason_codes"].append(f"DAYS:{len(all_days)}")

    if len(all_days) < 20:
        result["summary"]["p43_status"] = "P43_INSUFFICIENT_DAYS"
        return result

    cfg05 = merged[merged["model_name"] == "lightgbm_cfg05_dayahead"]
    yt_cfg = cfg05["y_true"].values
    yp_cfg = cfg05["y_pred"].values
    cfg05_smape = smape_floor50(yt_cfg, yp_cfg)

    # 1. Full-period (current approach, trusted pool only)
    full_weights = _compute_weights(merged, alpha, model_list)
    full_fused = _apply_weights(merged, full_weights, model_list)
    if len(full_fused) > 0:
        fus_s = smape_floor50(full_fused["y_true"].values, full_fused["fused_price"].values)
        result["full_period"] = {
            "sMAPE": round(fus_s, 4),
            "n": len(full_fused),
            "improvement_vs_cfg05": round(((cfg05_smape - fus_s) / cfg05_smape) * 100, 2),
        }
        result["reason_codes"].append(f"FULL_PERIOD:sMAPE={fus_s:.4f}%")

    # 2. Train/test split (first 20 / last 9)
    split_idx = min(20, len(all_days) - 3)
    train_days = all_days[:split_idx]
    test_days = all_days[split_idx:]

    train = merged[merged["target_day"].isin(train_days)]
    test = merged[merged["target_day"].isin(test_days)]

    train_weights = _compute_weights(train, alpha, model_list)
    split_fused = _apply_weights(test, train_weights, model_list)

    cfg05_test = cfg05[cfg05["target_day"].isin(test_days)]
    if len(split_fused) > 0 and len(cfg05_test) > 0:
        yt_st = cfg05_test["y_true"].values
        yp_st = cfg05_test["y_pred"].values
        cfg05_test_smape = smape_floor50(yt_st, yp_st)
        fus_s = smape_floor50(split_fused["y_true"].values, split_fused["fused_price"].values)

        # Best single on test
        best_test_smape = float("inf")
        for m in model_list:
            md = test[test["model_name"] == m]
            if len(md) > 0:
                ms = smape_floor50(md["y_true"].values, md["y_pred"].values)
                best_test_smape = min(best_test_smape, ms)

        result["split"] = {
            "train_days": len(train_days),
            "test_days": len(test_days),
            "train_range": f"{train_days[0]}~{train_days[-1]}",
            "test_range": f"{test_days[0]}~{test_days[-1]}",
            "cfg05_sMAPE": round(cfg05_test_smape, 4),
            "best_single_sMAPE": round(best_test_smape, 4),
            "fusion_sMAPE": round(fus_s, 4),
            "fusion_vs_cfg05_delta": round(((cfg05_test_smape - fus_s) / cfg05_test_smape) * 100, 2),
            "fusion_vs_best_single_delta": round(((best_test_smape - fus_s) / best_test_smape) * 100, 2),
            "n": len(split_fused),
        }
        result["reason_codes"].append(
            f"SPLIT:train={len(train_days)}d/test={len(test_days)}d "
            f"fusion={fus_s:.4f}% vs cfg05={cfg05_test_smape:.4f}%"
        )

    # 3. Rolling expanding validation
    rolling_fused = []
    rolling_days_info = []
    for i, target_d in enumerate(all_days):
        if i < warmup_days:
            continue
        past_days = all_days[:i]  # days before this one
        if len(past_days) < warmup_days:
            continue
        past_df = merged[merged["target_day"].isin(past_days)]
        day_df = merged[merged["target_day"] == target_d]
        if len(day_df) == 0:
            continue

        w = _compute_weights(past_df, alpha, model_list)
        day_fused = _apply_weights(day_df, w, model_list)
        for _, row in day_fused.iterrows():
            rolling_fused.append({
                "target_day": target_d,
                "y_true": row["y_true"],
                "fused_price": row["fused_price"],
            })
        rolling_days_info.append({
            "target_day": target_d,
            "past_days": len(past_days),
            "past_range": f"{past_days[0]}~{past_days[-1]}",
            "n_fused": len(day_fused),
        })

    result["rolling"]["days_info"] = rolling_days_info

    if len(rolling_fused) >= 24:
        rdf = pd.DataFrame(rolling_fused)
        r_cfg05 = cfg05[cfg05["target_day"].isin(rdf["target_day"].unique())]
        r_yt_cfg = r_cfg05["y_true"].values
        r_yp_cfg = r_cfg05["y_pred"].values
        r_cfg05_smape = smape_floor50(r_yt_cfg, r_yp_cfg) if len(r_cfg05) > 0 else None
        r_fus_s = smape_floor50(rdf["y_true"].values, rdf["fused_price"].values)

        # Best single on rolling days
        r_best_smape = float("inf")
        for m in model_list:
            md = merged[(merged["model_name"] == m) & (merged["target_day"].isin(rdf["target_day"].unique()))]
            if len(md) > 0:
                ms = smape_floor50(md["y_true"].values, md["y_pred"].values)
                r_best_smape = min(r_best_smape, ms)

        result["rolling"]["metrics"] = {
            "n_days": len(rdf["target_day"].unique()),
            "n_rows": len(rdf),
            "cfg05_sMAPE": round(r_cfg05_smape, 4) if r_cfg05_smape else None,
            "best_single_sMAPE": round(r_best_smape, 4) if r_best_smape < float("inf") else None,
            "fusion_sMAPE": round(r_fus_s, 4),
            "fusion_vs_cfg05_delta": round(((r_cfg05_smape - r_fus_s) / r_cfg05_smape) * 100, 2) if r_cfg05_smape else None,
        }
        result["reason_codes"].append(
            f"ROLLING:{len(rdf['target_day'].unique())}days "
            f"fusion={r_fus_s:.4f}%"
        )

    # Status
    has_split = bool(result.get("split", {}).get("fusion_sMAPE"))
    has_rolling = bool(result.get("rolling", {}).get("metrics"))
    if has_split and has_rolling:
        result["summary"]["p43_status"] = "P43_VALIDATION_COMPLETE"
    elif has_split or has_rolling:
        result["summary"]["p43_status"] = "P43_VALIDATION_PARTIAL"
    else:
        result["summary"]["p43_status"] = "P43_VALIDATION_FAILED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P43 — Rolling/Split Weight Validation")
    print("=" * 60)
    print(f"  Alpha:            {result['alpha']}")
    print(f"  Warmup days:      {result['warmup_days']}")
    print(f"  Total days:       {result['total_days']}")
    print(f"  Trusted models:   {result['trusted_models']}")
    print()

    fp = result.get("full_period", {})
    if fp:
        print("── Full Period (Trusted) ──")
        print(f"  sMAPE: {fp['sMAPE']}%")
        print(f"  vs cfg05: {fp['improvement_vs_cfg05']}%")
        print(f"  n: {fp['n']}")

    sp = result.get("split", {})
    if sp:
        print()
        print("── Train/Test Split ──")
        print(f"  Train: {sp.get('train_range','')} ({sp['train_days']}d)")
        print(f"  Test:  {sp.get('test_range','')} ({sp['test_days']}d)")
        print(f"  cfg05: {sp['cfg05_sMAPE']}%")
        print(f"  best single: {sp['best_single_sMAPE']}%")
        print(f"  fusion: {sp['fusion_sMAPE']}%")
        print(f"  fusion vs cfg05: {sp['fusion_vs_cfg05_delta']}%")
        print(f"  fusion vs best single: {sp['fusion_vs_best_single_delta']}%")

    rl = result.get("rolling", {}).get("metrics", {})
    if rl:
        print()
        print("── Rolling ──")
        print(f"  Days: {rl.get('n_days', 'N/A')}")
        print(f"  Rows: {rl.get('n_rows', 'N/A')}")
        print(f"  cfg05: {rl.get('cfg05_sMAPE', 'N/A')}%")
        print(f"  best single: {rl.get('best_single_sMAPE', 'N/A')}%")
        print(f"  fusion: {rl['fusion_sMAPE']}%")
        print(f"  fusion vs cfg05: {rl.get('fusion_vs_cfg05_delta', 'N/A')}%")

    print()
    print(f"  Status: {result['summary']['p43_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P43: Rolling weight validation.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_rolling_validation(
        work_dir=args.work_dir, alpha=args.alpha, warmup_days=args.warmup_days,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict and result["summary"]["p43_status"] != "P43_VALIDATION_COMPLETE":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
