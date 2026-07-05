"""
scripts/run_p92_sgdfnet_assist_adapter.py — Run SGDFNet assist adapter.

Usage::

    python -m scripts.run_p92_sgdfnet_assist_adapter \\
        --sgdfnet-root ../electricity_forecast_model2.0_exp/SGDFNet \\
        --raw-data ../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv \\
        --dayahead-predictions .local_artifacts/dayahead/predictions.csv \\
        --target-start 2026-06-01 \\
        --target-end 2026-06-30 \\
        --work-dir .local_artifacts/p92 \\
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


def run_sgdfnet_assist(
    sgdfnet_root: str = "",
    raw_data: str = "",
    dayahead_predictions: str = "",
    target_start: str = "",
    target_end: str = "",
    work_dir: str = ".local_artifacts/p92",
) -> dict:
    """Run SGDFNet assist adapter and export pack.

    Returns dict with status and output paths.
    """
    from models.adapters.sgdfnet_assist_adapter import SGDFNetAssistAdapter

    os.makedirs(work_dir, exist_ok=True)

    adapter = SGDFNetAssistAdapter(sgdfnet_root=sgdfnet_root)
    adapter.load()

    # Load dayahead predictions if available
    da_df = None
    if dayahead_predictions and os.path.isfile(dayahead_predictions):
        da_df = pd.read_csv(dayahead_predictions)

    # Load raw data
    data_df = None
    if raw_data and os.path.isfile(raw_data):
        try:
            data_df = pd.read_csv(raw_data, encoding="gbk")
        except Exception:
            data_df = pd.read_csv(raw_data)
        # Rename columns to standard schema
        col_map = {
            "时刻": "ds",
            "时间": "ds",
            "日前电价": "da_anchor",
            "da_price": "da_anchor",
        }
        data_df = data_df.rename(columns={c: col_map[c] for c in col_map if c in data_df.columns})

    # Export assist pack
    output_dir = os.path.join(work_dir, "sgdfnet_assist_output")
    result = adapter.export_assist_pack(
        output_dir=output_dir,
        data_path=None,
        df=data_df,
        da_predictions=da_df,
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
        "target_start": target_start,
        "target_end": target_end,
        "work_dir": work_dir,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P92: Run SGDFNet assist adapter")
    p.add_argument("--sgdfnet-root", type=str, default="")
    p.add_argument("--raw-data", type=str, default="")
    p.add_argument("--dayahead-predictions", type=str, default="")
    p.add_argument("--target-start", type=str, default="")
    p.add_argument("--target-end", type=str, default="")
    p.add_argument("--work-dir", type=str, default=".local_artifacts/p92")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = run_sgdfnet_assist(
        sgdfnet_root=args.sgdfnet_root,
        raw_data=args.raw_data,
        dayahead_predictions=args.dayahead_predictions,
        target_start=args.target_start,
        target_end=args.target_end,
        work_dir=args.work_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"P92 SGDFNet Assist Adapter: {result['status']}")
        print(f"{'='*60}")
        print(f"  Assist available: {result['assist_available']}")
        print(f"  Rows: {result['rows']}")
        print(f"  CSV: {result.get('csv_path', 'N/A')}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
