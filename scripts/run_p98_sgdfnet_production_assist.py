"""
scripts/run_p98_sgdfnet_production_assist.py — P98: Run SGDFNet production assist.

Usage::

    python -m scripts.run_p98_sgdfnet_production_assist \\
        --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \\
        --raw-data data/shandong_pmos_hourly.csv \\
        --dayahead-predictions .local_artifacts/p90_real_integrated_chain/dayahead/all_predictions.csv \\
        --target-start 2026-06-01 \\
        --target-end 2026-06-30 \\
        --work-dir .local_artifacts/p98_sgdfnet_production \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def run_production_assist(
    sgdfnet_root: str = "",
    raw_data: str = "",
    dayahead_predictions: str = "",
    target_start: str = "",
    target_end: str = "",
    work_dir: str = ".local_artifacts/p98_sgdfnet_production",
    strict: bool = False,
) -> dict:
    """Run SGDFNet production assist and export pack."""
    from adapters.sgdfnet_production_adapter import SGDFNetProductionAdapter

    os.makedirs(work_dir, exist_ok=True)

    adapter = SGDFNetProductionAdapter(sgdfnet_root=sgdfnet_root)
    adapter.load()

    # Load data
    data_df = None
    if raw_data and os.path.isfile(raw_data):
        try:
            data_df = pd.read_csv(raw_data, encoding="gbk")
        except Exception:
            data_df = pd.read_csv(raw_data)

    # Load day-ahead predictions
    da_df = None
    if dayahead_predictions and os.path.isfile(dayahead_predictions):
        da_df = pd.read_csv(dayahead_predictions)

    # If data has da_anchor or da_df merged
    result = adapter.export_assist_pack(
        output_dir=os.path.join(work_dir, "sgdfnet_assist_output"),
        data_path=raw_data,
        df=data_df,
        start=target_start,
        end=target_end,
    )

    return {
        "status": result["status"],
        "assist_available": result["assist_available"],
        "csv_path": result.get("csv_path", ""),
        "manifest_path": result.get("manifest_path", ""),
        "rows": result.get("rows", 0),
        "sgdfnet_root": sgdfnet_root,
        "adapter_status": adapter.status,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P98: Run SGDFNet production assist")
    p.add_argument("--sgdfnet-root", type=str, default="")
    p.add_argument("--raw-data", type=str, default="")
    p.add_argument("--dayahead-predictions", type=str, default="")
    p.add_argument("--target-start", type=str, default="")
    p.add_argument("--target-end", type=str, default="")
    p.add_argument("--work-dir", type=str, default=".local_artifacts/p98_sgdfnet_production")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_production_assist(
        sgdfnet_root=args.sgdfnet_root,
        raw_data=args.raw_data,
        dayahead_predictions=args.dayahead_predictions,
        target_start=args.target_start,
        target_end=args.target_end,
        work_dir=args.work_dir,
        strict=args.strict,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"P98 SGDFNet Production Assist: {result['status']}")
        print(f"{'='*60}")
        print(f"  Assist available: {result['assist_available']}")
        print(f"  Rows: {result['rows']}")
        print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
