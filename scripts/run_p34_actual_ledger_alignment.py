"""
scripts/run_p34_actual_ledger_alignment.py — P34: Actual ledger alignment.

Extracts y_true from raw data for each target_day in the backtest period
and aligns with prediction ledger for eval.

Usage::

    python -m scripts.run_p34_actual_ledger_alignment

Options::

    --raw-data PATH      Raw Chinese CSV path.
    --start-day YYYY-MM-DD  Start date (default: 2026-06-01).
    --end-day YYYY-MM-DD    End date (default: 2026-06-30).
    --work-dir PATH      Model artifacts dir.
    --output-dir PATH    Output dir for ledger CSVs.
    --force              Overwrite existing.
    --json               Output JSON report.
    --strict             Exit non-zero on failures.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")
_RAW_DATA_PATH = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.1/data/shandong_pmos_hourly.csv"

ACTUAL_LEDGER_COLUMNS = [
    "task", "target_day", "business_day", "ds", "hour_business",
    "period", "y_true", "actual_source", "run_id", "created_at", "updated_at",
]


def build_actual_ledger(
    raw_data: Optional[str] = None,
    start_day: Optional[str] = None,
    end_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Extract y_true actuals from raw data and build actual ledger."""
    raw_data = raw_data or _RAW_DATA_PATH
    start_day = start_day or "2026-06-01"
    end_day = end_day or "2026-06-30"
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = output_dir or os.path.join(work_dir, "ledger")
    os.makedirs(ledger_dir, exist_ok=True)

    result: dict[str, Any] = {
        "phase": "P34",
        "start_day": start_day,
        "end_day": end_day,
        "total_rows": 0,
        "actual_ledger_path": os.path.join(ledger_dir, "actual_ledger_30d.csv"),
        "p34_status": "P34_NOT_STARTED",
        "reason_codes": [],
    }

    # Build date range
    dates = pd.date_range(start_day, end_day, freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Load raw data
    try:
        import importlib as _il
        source_repo = os.path.join(".local_artifacts", "source_repos", "epf-sota-experiment")
        if source_repo not in sys.path:
            sys.path.insert(0, source_repo)
        data_loader = _il.import_module("src.common.data_loader")
        df_raw = data_loader.load_data(raw_data, target="dayahead")
        result["reason_codes"].append(f"RAW_LOADED:{len(df_raw)}rows")
    except Exception as e:
        result["reason_codes"].append(f"RAW_LOAD_FAILED:{e}")
        result["p34_status"] = "P34_DATA_FAILED"
        return result

    from artifacts.dayahead_window import get_business_day_info

    all_rows = []
    for d in date_strs:
        target_dt = pd.Timestamp(d)
        start = target_dt + pd.Timedelta(hours=1)
        end = target_dt + pd.Timedelta(days=1, hours=1)
        mask = (df_raw["ds"] >= start) & (df_raw["ds"] < end)
        day_df = df_raw[mask].copy()

        if len(day_df) == 0:
            result["reason_codes"].append(f"NO_DATA:{d}")
            continue

        info = get_business_day_info(day_df["ds"])
        hour_business = info["hour_business"]

        from data.business_day import infer_period
        for idx, row in day_df.iterrows():
            hb = hour_business.loc[idx]
            all_rows.append({
                "task": "dayahead",
                "target_day": d,
                "business_day": info["business_day"].loc[idx],
                "ds": row["ds"],
                "hour_business": int(hb),
                "period": infer_period(int(hb)),
                "y_true": row["y"],
                "actual_source": "raw_csv",
                "run_id": "p34_actual_ledger",
                "created_at": pd.Timestamp.now().isoformat(),
                "updated_at": pd.Timestamp.now().isoformat(),
            })

    if len(all_rows) == 0:
        result["p34_status"] = "P34_NO_DATA"
        return result

    ledger = pd.DataFrame(all_rows)
    ledger = ledger.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
    result["total_rows"] = len(ledger)

    try:
        ledger.to_csv(result["actual_ledger_path"], index=False)
        result["reason_codes"].append(f"SAVED:{result['actual_ledger_path']}({len(ledger)}rows)")
        result["p34_status"] = "P34_ACTUAL_LEDGER_BUILT"
    except Exception as e:
        result["reason_codes"].append(f"SAVE_FAILED:{e}")
        result["p34_status"] = "P34_SAVE_FAILED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P34 — Actual Ledger Alignment")
    print("=" * 60)
    print(f"  Period:           {result['start_day']} ~ {result['end_day']}")
    print(f"  Total rows:       {result['total_rows']}")
    print(f"  Ledger path:      {result['actual_ledger_path']}")
    print(f"  Status:           {result['p34_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P34: Build actual ledger.")
    parser.add_argument("--raw-data", type=str, default=None)
    parser.add_argument("--start-day", type=str, default="2026-06-01")
    parser.add_argument("--end-day", type=str, default="2026-06-30")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    result = build_actual_ledger(
        raw_data=args.raw_data, start_day=args.start_day, end_day=args.end_day,
        work_dir=args.work_dir, output_dir=args.output_dir, force=args.force,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict and result["p34_status"] != "P34_ACTUAL_LEDGER_BUILT":
        logger.error("P34: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
