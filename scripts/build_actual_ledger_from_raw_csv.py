"""
scripts/build_actual_ledger_from_raw_csv.py — P28 actual ledger builder.

Builds an actual ledger from the raw Chinese CSV for use by BGEW / fusion learner.

Schema::

    task, target_day, business_day, ds, hour_business, period,
    y_true, actual_source, version

Usage::

    python -m scripts.build_actual_ledger_from_raw_csv \\
        --raw-data ../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv \\
        --start-day 2026-06-01 --end-day 2026-06-30 \\
        --work-dir .local_artifacts/p26_p30_fusion \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from artifacts.dayahead_window import filter_dayahead, get_business_day_info
from data.schema import ACTUAL_LEDGER_COLUMNS

logger = logging.getLogger(__name__)

# ── Statuses ───────────────────────────────────────────────────────────────
P28_ACTUAL_LEDGER_READY = "P28_ACTUAL_LEDGER_READY"
P28_ACTUAL_LEDGER_PARTIAL = "P28_ACTUAL_LEDGER_PARTIAL"
P28_ACTUAL_LEDGER_BLOCKED = "P28_ACTUAL_LEDGER_BLOCKED"
P28_RAW_DATA_MISSING = "P28_RAW_DATA_MISSING"

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p26_p30_fusion")


def _load_raw_csv(raw_data: str) -> pd.DataFrame:
    """Load raw Chinese CSV with ds column."""
    try:
        raw_df = pd.read_csv(raw_data, encoding="gbk")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(raw_data, encoding="utf-8")
    raw_df["ds"] = pd.to_datetime(raw_df["时刻"])
    return raw_df


def build_actual_ledger_from_raw_csv(
    raw_data: Optional[str] = None,
    start_day: Optional[str] = None,
    end_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Build actual ledger from raw CSV.

    Parameters
    ----------
    raw_data : str
        Path to raw Chinese CSV.
    start_day : str
        First target day YYYY-MM-DD.
    end_day : str
        Last target day YYYY-MM-DD.
    work_dir : str
        Output directory for actual_ledger.csv.
    version : str
        Version string for the ledger.

    Returns
    -------
    dict
        Complete actual ledger build report.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    result: dict[str, Any] = {
        "raw_data": raw_data,
        "eval_start": start_day,
        "eval_end": end_day,
        "work_dir": work_dir,
        "total_rows": 0,
        "target_days": 0,
        "complete_days": 0,
        "duplicate_keys": 0,
        "null_y_true_rows": 0,
        "null_y_true_days": [],
        "hour_business_range": None,
        "actual_ledger_path": None,
        "schema_valid": False,
        "final_status": None,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # ── Validate input ──
    if not raw_data or not os.path.isfile(raw_data or ""):
        result["final_status"] = P28_RAW_DATA_MISSING
        result["reason_codes"].append("RAW_DATA_MISSING")
        return result

    # ── Load raw CSV ──
    try:
        df_raw = _load_raw_csv(raw_data)
    except Exception as e:
        result["final_status"] = P28_ACTUAL_LEDGER_BLOCKED
        result["reason_codes"].append(f"RAW_LOAD_FAILED:{e}")
        return result

    # ── Extract y_true ──
    if "日前电价" not in df_raw.columns:
        result["final_status"] = P28_ACTUAL_LEDGER_BLOCKED
        result["reason_codes"].append("NO_日前电价_COLUMN")
        return result

    df_raw["y_true"] = df_raw["日前电价"]

    # ── Determine eval range ──
    if not start_day or not end_day:
        max_ds = df_raw["ds"].max()
        if not end_day:
            end_day = (max_ds - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if not start_day:
            start_day = (pd.Timestamp(end_day) - pd.Timedelta(days=29)).strftime("%Y-%m-%d")

    start_dt = pd.Timestamp(start_day)
    end_dt = pd.Timestamp(end_day)
    eval_days = pd.date_range(start=start_dt, end=end_dt, freq="D")
    result["eval_start"] = start_day
    result["eval_end"] = end_day

    # ── Build actual ledger row by row ──
    rows: list[dict[str, Any]] = []
    now_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for day in eval_days:
        day_str = day.strftime("%Y-%m-%d")
        day_df = filter_dayahead(df_raw, day_str)

        if len(day_df) == 0:
            result["reason_codes"].append(f"NO_DATA_FOR:{day_str}")
            continue

        # Add business time columns
        biz_info = get_business_day_info(day_df["ds"])
        day_df = day_df.copy()
        day_df["business_day"] = biz_info["business_day"]
        day_df["hour_business"] = biz_info["hour_business"]
        day_df["period"] = biz_info["period"]

        for _, row in day_df.iterrows():
            rows.append({
                "task": "dayahead",
                "target_day": day_str,
                "business_day": str(row["business_day"]),
                "ds": row["ds"],
                "hour_business": int(row["hour_business"]),
                "period": str(row["period"]),
                "y_true": row["y_true"],
                "actual_source": "shandong_pmos_hourly",
                "version": version,
                "run_id": f"p28_{day_str}",
                "created_at": now_str,
                "updated_at": now_str,
            })

    if not rows:
        result["final_status"] = P28_ACTUAL_LEDGER_BLOCKED
        result["reason_codes"].append("NO_ROWS_GENERATED")
        return result

    ledger_df = pd.DataFrame(rows)
    result["total_rows"] = len(ledger_df)
    result["target_days"] = ledger_df["target_day"].nunique()

    # ── Check completeness ──
    for day_str in eval_days:
        day_s = day_str.strftime("%Y-%m-%d")
        day_rows = ledger_df[ledger_df["target_day"] == day_s]
        if len(day_rows) == 24:
            result["complete_days"] += 1

    # ── Check duplicates ──
    key_cols = ["task", "target_day", "business_day", "hour_business"]
    dup_count = ledger_df.duplicated(subset=key_cols, keep=False).sum()
    result["duplicate_keys"] = int(dup_count)

    # ── Check null y_true ──
    null_mask = ledger_df["y_true"].isna()
    result["null_y_true_rows"] = int(null_mask.sum())
    if null_mask.any():
        null_days = ledger_df[null_mask]["target_day"].unique().tolist()
        result["null_y_true_days"] = null_days

    # ── Hour business range ──
    hb_min = int(ledger_df["hour_business"].min())
    hb_max = int(ledger_df["hour_business"].max())
    result["hour_business_range"] = [hb_min, hb_max]

    # ── Schema validation ──
    required_cols = [c for c in ACTUAL_LEDGER_COLUMNS if c in ledger_df.columns]
    missing_cols = [c for c in ACTUAL_LEDGER_COLUMNS if c not in ledger_df.columns]
    result["schema_valid"] = len(missing_cols) == 0
    if missing_cols:
        result["reason_codes"].append(f"SCHEMA_MISSING_COLS:{missing_cols}")
    else:
        result["reason_codes"].append("SCHEMA_VALID")

    # ── Save ──
    ledger_dir = os.path.join(work_dir, "ledgers")
    os.makedirs(ledger_dir, exist_ok=True)
    ledger_path = os.path.join(ledger_dir, "actual_ledger.csv")
    ledger_df.to_csv(ledger_path, index=False)
    result["actual_ledger_path"] = ledger_path
    result["reason_codes"].append(f"ACTUAL_LEDGER_SAVED:{len(ledger_df)}_rows")

    # ── Final status ──
    expected_rows = len(eval_days) * 24
    if result["total_rows"] == expected_rows and result["duplicate_keys"] == 0:
        result["final_status"] = P28_ACTUAL_LEDGER_READY
    elif result["total_rows"] > 0:
        result["final_status"] = P28_ACTUAL_LEDGER_PARTIAL
        result["reason_codes"].append(
            f"PARTIAL:{result['total_rows']}/{expected_rows}_rows"
        )
    else:
        result["final_status"] = P28_ACTUAL_LEDGER_BLOCKED

    # Forbidden files check
    work_dir_norm = os.path.abspath(work_dir).replace("\\", "/")
    if not (any(a.lstrip(".") in work_dir_norm for a in (".local_artifacts",)) or os.path.isabs(work_dir)):
        result["forbidden_files_check"] = "FAIL"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P28 Actual Ledger Builder Report")
    print("=" * 60)
    print(f"  Eval range:         {result['eval_start']} ~ {result['eval_end']}")
    print(f"  Total rows:         {result['total_rows']}")
    print(f"  Target days:        {result['target_days']}")
    print(f"  Complete days:      {result['complete_days']}")
    print(f"  Duplicate keys:     {result['duplicate_keys']}")
    print(f"  Null y_true rows:   {result['null_y_true_rows']}")
    print(f"  Hour range:         {result['hour_business_range']}")
    print(f"  Schema valid:       {result['schema_valid']}")
    print(f"  Ledger path:        {result['actual_ledger_path']}")
    print(f"  Final status:       {result['final_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P28: actual ledger builder from raw CSV.")
    p.add_argument("--raw-data", type=str, default=None)
    p.add_argument("--start-day", type=str, default=None)
    p.add_argument("--end-day", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--json", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    result = build_actual_ledger_from_raw_csv(
        raw_data=args.raw_data,
        start_day=args.start_day,
        end_day=args.end_day,
        work_dir=args.work_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
