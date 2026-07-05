"""
scripts/run_p26_cfg05_per_day_retrain_backtest.py — P26 cfg05 per-day retrain walk-forward.

Converts the P21 train-once strategy into a true per-day retrain walk-forward:

    For each target_day D:
        train window = [D - 90d, D)
        prediction window = D 01:00 ~ D+1 00:00 (24 business hours)
        rows = 24

Every day gets a fresh model trained strictly on data before D.

Usage::

    python -m scripts.run_p26_cfg05_per_day_retrain_backtest \\
        --raw-data ../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv \\
        --source-repo .local_artifacts/source_repos/epf-sota-experiment \\
        --start-day 2026-06-01 --end-day 2026-06-30 \\
        --train-window-days 90 \\
        --work-dir .local_artifacts/p26_p30_fusion \\
        --json --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from scripts.run_p16_cfg05_30d_walkforward_backtest import (
    BACKTEST_BLOCKED,
    BACKTEST_COMPLETE,
    BACKTEST_INCOMPLETE,
    BACKTEST_NO_VALID_YTRUE,
    _path_is_safe,
    compute_metrics,
    compute_smape_floor50,
)

logger = logging.getLogger(__name__)

# ── Final statuses ─────────────────────────────────────────────────────────
P26_PER_DAY_RETRAIN_COMPLETE = "P26_PER_DAY_RETRAIN_COMPLETE"
P26_PER_DAY_RETRAIN_PARTIAL = "P26_PER_DAY_RETRAIN_PARTIAL"
P26_PER_DAY_RETRAIN_BLOCKED = "P26_PER_DAY_RETRAIN_BLOCKED"
P26_LOCAL_DATA_MISSING = "P26_LOCAL_DATA_MISSING"

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p26_p30_fusion")


def run_p26_cfg05_per_day_retrain_backtest(
    raw_data: Optional[str] = None,
    source_repo: Optional[str] = None,
    start_day: Optional[str] = None,
    end_day: Optional[str] = None,
    train_window_days: int = 90,
    work_dir: Optional[str] = None,
    device: str = "cpu",
    feature_version: str = "v2",
) -> dict[str, Any]:
    """Run cfg05 per-day retrain walk-forward backtest.

    Delegates to P16 with ``reuse_model=False`` to force retraining for each day.

    Parameters
    ----------
    feature_version : str
        Feature builder version: "v2" (40 cols) or "v3" (54 cols).

    Returns
    -------
    dict
        Complete backtest summary with per-day retrain metadata.
    """
    from scripts.run_p16_cfg05_30d_walkforward_backtest import (
        run_p16_cfg05_30d_walkforward_backtest,
    )

    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    result: dict[str, Any] = {
        "p26_mode": "per_day_retrain",
        "feature_version": feature_version,
        "raw_data": raw_data,
        "source_repo": source_repo,
        "eval_start": start_day,
        "eval_end": end_day,
        "train_window_days": train_window_days,
        "work_dir": work_dir,
        "attempted_days": 0,
        "complete_days": 0,
        "metric_days": 0,
        "eval_rows": 0,
        "sMAPE_floor50": None,
        "MAE": None,
        "RMSE": None,
        "per_day_metrics": None,
        "per_hour_metrics": None,
        "training_time_seconds": 0.0,
        "failed_days": 0,
        "failed_day_details": [],
        "reason_codes": [],
        "final_status": None,
        "improvement_vs_p21": None,
        "forbidden_files_check": "PASS",
    }

    # ── Validate inputs ──
    if not raw_data or not os.path.isfile(raw_data or ""):
        result["final_status"] = P26_LOCAL_DATA_MISSING
        result["reason_codes"].append("RAW_DATA_MISSING")
        return result

    # ── Delegate to P16 with reuse_model=False ──
    t0 = time.time()
    p16_result = run_p16_cfg05_30d_walkforward_backtest(
        raw_data=raw_data,
        source_repo=source_repo,
        start_day=start_day,
        end_day=end_day,
        train_window_days=train_window_days,
        work_dir=work_dir,
        reuse_model=False,  # KEY: per-day retrain
        device=device,
        feature_version=feature_version,
    )
    elapsed = time.time() - t0
    result["training_time_seconds"] = round(elapsed, 2)

    # ── Map P16 results to P26 output ──
    result["attempted_days"] = p16_result.get("attempted_days", 0)
    result["complete_days"] = p16_result.get("complete_days", 0)
    result["metric_days"] = p16_result.get("metric_days", 0)
    result["eval_rows"] = p16_result.get("eval_rows", 0)

    metrics = p16_result.get("metrics")
    if metrics:
        result["sMAPE_floor50"] = metrics.get("sMAPE_floor50")
        result["MAE"] = metrics.get("MAE")
        result["RMSE"] = metrics.get("RMSE")

    # Load per-day and per-hour metrics from files
    per_day_path = p16_result.get("per_day_metrics_path_local")
    if per_day_path and os.path.isfile(per_day_path):
        result["per_day_metrics"] = per_day_path

    per_hour_path = p16_result.get("per_hour_metrics_path_local")
    if per_hour_path and os.path.isfile(per_hour_path):
        result["per_hour_metrics"] = per_hour_path

    # Count failed days
    p16_reason = p16_result.get("reason_codes", [])
    result["reason_codes"].extend(p16_reason)
    result["failed_days"] = result["attempted_days"] - result["complete_days"]

    # Collect failed day details from per-day results
    pred_all_path = p16_result.get("predictions_path_local")
    if pred_all_path and os.path.isfile(pred_all_path):
        try:
            all_preds = pd.read_csv(pred_all_path)
            if "business_day" in all_preds.columns:
                complete_bdays = set(all_preds.dropna(subset=["y_true"])["business_day"].unique())
                # Check per-day metrics for incomplete days
                if per_day_path and os.path.isfile(per_day_path):
                    per_day_df = pd.read_csv(per_day_path)
                    all_target_days = set(per_day_df["target_day"].values) if "target_day" in per_day_df.columns else set()
        except Exception:
            pass

    # ── Determine final status ──
    p16_status = p16_result.get("final_status", "")
    if p16_status == BACKTEST_COMPLETE and result["complete_days"] == result["attempted_days"]:
        result["final_status"] = P26_PER_DAY_RETRAIN_COMPLETE
    elif p16_status == BACKTEST_COMPLETE:
        result["final_status"] = P26_PER_DAY_RETRAIN_PARTIAL
        result["reason_codes"].append(
            f"PARTIAL_COMPLETENESS:{result['complete_days']}/{result['attempted_days']}"
        )
    elif p16_status == BACKTEST_NO_VALID_YTRUE:
        result["final_status"] = P26_PER_DAY_RETRAIN_BLOCKED
        result["reason_codes"].append("NO_VALID_YTRUE")
    else:
        result["final_status"] = P26_PER_DAY_RETRAIN_BLOCKED
        result["reason_codes"].append(f"P16_STATUS:{p16_status}")

    # ── Compare with P21 baseline ──
    p21_smape = 20.71  # P21 baseline
    if result["sMAPE_floor50"] is not None:
        improvement = p21_smape - result["sMAPE_floor50"]
        result["improvement_vs_p21"] = {
            "p21_smape_floor50": p21_smape,
            "p26_smape_floor50": round(result["sMAPE_floor50"], 4),
            "delta_pp": round(improvement, 4),
            "direction": "IMPROVED" if improvement > 0 else "NO_IMPROVEMENT",
        }

    # Forbidden files check
    work_dir_norm = os.path.abspath(work_dir).replace("\\", "/")
    if not (any(a.lstrip(".") in work_dir_norm for a in (".local_artifacts",)) or os.path.isabs(work_dir)):
        result["forbidden_files_check"] = "FAIL"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P26 cfg05 Per-Day Retrain Walk-Forward Backtest Report")
    print("=" * 60)
    print(f"  Mode:               {result['p26_mode']}")
    print(f"  Eval range:         {result['eval_start']} ~ {result['eval_end']}")
    print(f"  Train window:       {result['train_window_days']}d")
    print(f"  Attempted days:     {result['attempted_days']}")
    print(f"  Complete days:      {result['complete_days']}")
    print(f"  Metric days:        {result['metric_days']}")
    print(f"  Eval rows:          {result['eval_rows']}")
    print(f"  Failed days:        {result['failed_days']}")
    print(f"  Training time:      {result['training_time_seconds']:.1f}s")
    if result["sMAPE_floor50"] is not None:
        print(f"  sMAPE_floor50:      {result['sMAPE_floor50']:.4f}%")
        print(f"  MAE:                {result['MAE']:.4f}")
        print(f"  RMSE:               {result['RMSE']:.4f}")
    imp = result.get("improvement_vs_p21")
    if imp:
        print(f"  vs P21 (20.71%):    {imp['delta_pp']:+.4f}pp ({imp['direction']})")
    print(f"  Final status:       {result['final_status']}")
    print(f"  Forbidden check:    {result['forbidden_files_check']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P26: cfg05 per-day retrain walk-forward backtest.")
    p.add_argument("--raw-data", type=str, default=None)
    p.add_argument("--source-repo", type=str, default=None)
    p.add_argument("--start-day", type=str, default=None)
    p.add_argument("--end-day", type=str, default=None)
    p.add_argument("--train-window-days", type=int, default=90)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu",
                   help="LightGBM device: cpu or gpu.")
    p.add_argument("--feature-version", type=str, default="v2", choices=["v2", "v3"],
                   help="Feature builder version: v2 (40 cols) or v3 (54 cols).")
    p.add_argument("--json", action="store_true", default=False)
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--verbose", "-v", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    work_dir = args.work_dir or _DEFAULT_WORK_DIR
    if not _path_is_safe(work_dir):
        logger.error("Unsafe work-dir: %s", work_dir)
        return 1

    result = run_p26_cfg05_per_day_retrain_backtest(
        raw_data=args.raw_data,
        source_repo=args.source_repo,
        start_day=args.start_day,
        end_day=args.end_day,
        train_window_days=args.train_window_days,
        work_dir=work_dir,
        device=args.device,
        feature_version=args.feature_version,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and result["final_status"] != P26_PER_DAY_RETRAIN_COMPLETE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
