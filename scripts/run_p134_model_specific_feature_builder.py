"""
scripts/run_p134_model_specific_feature_builder.py — P134: Per-Model Feature Builder.

Demonstrates and validates the per-model feature builder by building features
for each supported model and verifying schema alignment.

Output: .local_artifacts/p134_model_features/feature_builder_report.json
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


def run_p134_feature_builder(
    raw_data_path: str = "",
    output_dir: str = "",
    day_sample: str = "2025-01-15",
    history_days: int = 30,
) -> dict[str, Any]:
    """Run the per-model feature builder for all supported models.

    Parameters
    ----------
    raw_data_path : str
        Path to raw CSV data (GBK encoded).
    output_dir : str
        Output directory for reports.
    day_sample : str
        Sample day for feature building.
    history_days : int
        Number of history days to include for rolling features.

    Returns
    -------
    dict
        Feature builder report.
    """
    if not raw_data_path:
        raw_data_path = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, ".local_artifacts", "p134_model_features")
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    result: dict[str, Any] = {
        "phase": "P134",
        "title": "Per-Model Feature Builder",
        "status": "STARTED",
        "models": {},
        "summary": {},
    }

    # Load raw data
    try:
        raw = pd.read_csv(raw_data_path, encoding="gbk")
        raw["ds"] = pd.to_datetime(raw["时刻"])
        from data.business_day import add_business_time_columns
        raw = add_business_time_columns(raw, timestamp_col="ds")
        raw = raw.sort_values("ds").reset_index(drop=True)
    except Exception as e:
        result["status"] = f"DATA_LOAD_FAILED:{e}"
        _save_result(result, output_dir)
        return result

    # Get sample window: history_days before day_sample + the sample day
    day_ts = pd.Timestamp(day_sample)
    window_start = day_ts - pd.Timedelta(days=history_days)
    window_end = day_ts + pd.Timedelta(days=1)
    window_mask = (raw["ds"] >= window_start) & (raw["ds"] < window_end)
    window_df = raw[window_mask].copy()

    if len(window_df) == 0:
        result["status"] = "NO_DATA_IN_WINDOW"
        _save_result(result, output_dir)
        return result

    # Build features for each supported model
    supported = list_supported_models()
    # Deduplicate: only test unique schemas
    unique_models = ["cfg05", "catboost_spike_residual", "catboost_sota"]

    for model_name in unique_models:
        logger.info(f"Building features for: {model_name}")
        try:
            X, report, score = build_features_for_model(
                window_df, model_name
            )
            result["models"][model_name] = {
                "status": "OK",
                "feature_count": report["expected_count"],
                "present_count": report["present_count"],
                "missing_count": report["missing_count"],
                "schema_match_score": score,
                "reason_codes": report["reason_codes"],
                "output_shape": list(X.shape) if hasattr(X, "shape") else None,
            }
        except Exception as e:
            result["models"][model_name] = {
                "status": f"FAILED:{e}",
                "reason_codes": [str(e)],
            }

    result["elapsed_seconds"] = round(time.time() - t_start, 2)

    # Summary
    ok_count = sum(1 for m in result["models"].values() if m.get("status") == "OK")
    result["summary"] = {
        "total_models": len(unique_models),
        "successful": ok_count,
        "failed": len(unique_models) - ok_count,
    }

    if ok_count == len(unique_models):
        result["status"] = "FEATURE_BUILDER_ALL_OK"
    elif ok_count > 0:
        result["status"] = "FEATURE_BUILDER_PARTIAL"
    else:
        result["status"] = "FEATURE_BUILDER_ALL_FAILED"

    _save_result(result, output_dir)
    return result


def _save_result(result: dict, output_dir: str) -> None:
    """Save result to JSON."""
    output_path = os.path.join(output_dir, "feature_builder_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    result["output_path"] = output_path


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="P134: Per-Model Feature Builder")
    parser.add_argument("--raw-data", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--day-sample", default="2025-01-15")
    parser.add_argument("--history-days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_p134_feature_builder(
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
        day_sample=args.day_sample,
        history_days=args.history_days,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n=== P134: Per-Model Feature Builder ===")
        print(f"Status: {result['status']}")
        for name, info in result["models"].items():
            score = info.get("schema_match_score", "N/A")
            fc = info.get("feature_count", "?")
            pc = info.get("present_count", "?")
            print(f"  {name}: score={score} features={pc}/{fc}")
        print(f"Output: {result.get('output_path', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
