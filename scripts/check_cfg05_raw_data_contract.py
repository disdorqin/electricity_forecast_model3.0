"""
scripts/check_cfg05_raw_data_contract.py — Raw Chinese CSV contract checker.

Validates that a raw Chinese CSV has all required columns, parseable timestamps,
and numeric price/load/energy columns required for cfg05 training.

Usage::

    python -m scripts.check_cfg05_raw_data_contract --raw-data /path/to/data.csv
    python -m scripts.check_cfg05_raw_data_contract --raw-data /path/to/data.csv --json
    python -m scripts.check_cfg05_raw_data_contract --strict

Options::

    --raw-data PATH     Path to raw Chinese CSV.
    --json              Output JSON report.
    --strict            Exit non-zero on invalid/missing.
    --verbose, -v       Increase log verbosity.
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

# ── Required Chinese column names ──────────────────────────────────────────

REQUIRED_CHINESE_COLUMNS = [
    "时刻",
    "日前电价",
    "实时电价",
    "直调负荷预测值",
    "风电总加预测值",
    "光伏总加预测值",
    "联络线受电负荷预测值",
    "竞价空间预测值",
]

# ── Status constants ───────────────────────────────────────────────────────

RAW_DATA_MISSING = "CFG05_RAW_DATA_MISSING"
RAW_DATA_INVALID = "CFG05_RAW_DATA_INVALID"
RAW_DATA_VALID = "CFG05_RAW_DATA_VALID"


def check_cfg05_raw_data_contract(
    raw_data: Optional[str] = None,
) -> dict[str, Any]:
    """Check the raw Chinese CSV data contract.

    Parameters
    ----------
    raw_data : str, optional
        Path to raw Chinese CSV.

    Returns
    -------
    dict with raw data contract summary.
    """
    result: dict[str, Any] = {
        "raw_data_status": RAW_DATA_MISSING,
        "raw_data_path": raw_data,
        "rows": 0,
        "columns_present": [],
        "missing_columns": [],
        "time_min": None,
        "time_max": None,
        "null_counts": {},
        "reason_codes": [],
    }

    # 1. File must be provided and exist
    if not raw_data:
        result["reason_codes"].append("NO_RAW_DATA_PATH_PROVIDED")
        return result

    if not os.path.isfile(raw_data):
        result["reason_codes"].append(f"RAW_DATA_FILE_NOT_FOUND: {raw_data}")
        return result

    # 2. Try reading CSV (GBK first, then UTF-8)
    try:
        try:
            df = pd.read_csv(raw_data, encoding="gbk")
            logger.info("Read CSV with GBK encoding: %d rows, %d cols", len(df), len(df.columns))
        except UnicodeDecodeError:
            df = pd.read_csv(raw_data, encoding="utf-8")
            logger.info("Read CSV with UTF-8 encoding: %d rows, %d cols", len(df), len(df.columns))
    except Exception as e:
        result["raw_data_status"] = RAW_DATA_INVALID
        result["reason_codes"].append(f"RAW_DATA_READ_FAILED: {e}")
        return result

    result["rows"] = len(df)
    result["columns_present"] = list(df.columns)

    # 3. Validate required Chinese columns
    missing = [c for c in REQUIRED_CHINESE_COLUMNS if c not in df.columns]
    result["missing_columns"] = missing
    if missing:
        result["raw_data_status"] = RAW_DATA_INVALID
        result["reason_codes"].append(
            f"RAW_DATA_MISSING_{len(missing)}_REQUIRED_COLUMNS: {missing}"
        )
        # Still check what we can for detailed diagnostics
    else:
        result["reason_codes"].append("ALL_REQUIRED_CHINESE_COLUMNS_PRESENT")

    # 4. Validate timestamp column
    if "时刻" in df.columns:
        times = pd.to_datetime(df["时刻"], errors="coerce")
        null_times = times.isna().sum()
        result["null_counts"]["时刻"] = int(null_times)
        if null_times > 0:
            result["reason_codes"].append(f"RAW_DATA_{null_times}_UNPARSEABLE_TIMESTAMPS")
        if times.notna().any():
            result["time_min"] = str(times.min())
            result["time_max"] = str(times.max())
    else:
        result["null_counts"]["时刻"] = len(df)

    # 5. Validate numeric columns
    numeric_checks = [
        "日前电价", "实时电价", "直调负荷预测值", "风电总加预测值",
        "光伏总加预测值", "联络线受电负荷预测值", "竞价空间预测值",
    ]
    for col in numeric_checks:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            null_count = int(numeric.isna().sum() - df[col].isna().sum())
            null_count = max(0, null_count)
            if null_count > 0:
                result["null_counts"][col] = null_count
                result["reason_codes"].append(
                    f"RAW_DATA_{null_count}_NON_NUMERIC_IN_{col}"
                )
            # Check all values missing
            if df[col].isna().all():
                result["reason_codes"].append(f"RAW_DATA_COLUMN_ALL_NAN: {col}")
        # else: already tracked in missing_columns

    # 6. Determine overall status
    if missing:
        result["raw_data_status"] = RAW_DATA_INVALID
    else:
        # Check if there are any data quality issues that warrant INVALID
        data_issues = [
            rc for rc in result.get("reason_codes", [])
            if "UNPARSEABLE_TIMESTAMPS" in rc
            or "NON_NUMERIC" in rc
            or "ALL_NAN" in rc
        ]
        if data_issues:
            result["raw_data_status"] = RAW_DATA_INVALID
        else:
            result["raw_data_status"] = RAW_DATA_VALID

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable contract check report."""
    print("=" * 60)
    print("cfg05 Raw Data Contract Check")
    print("=" * 60)
    print(f"  Status:           {result['raw_data_status']}")
    print(f"  Path:             {result['raw_data_path'] or 'N/A'}")
    print(f"  Rows:             {result['rows']}")
    print(f"  Columns present:  {len(result['columns_present'])}")
    print(f"  Missing columns:  {result['missing_columns']}")
    print(f"  Time range:       {result['time_min']} ~ {result['time_max']}")
    if result["null_counts"]:
        print(f"  Null counts:      {result['null_counts']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check cfg05 raw Chinese CSV data contract.",
    )
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw Chinese CSV.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON report.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero on invalid/missing.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = check_cfg05_raw_data_contract(raw_data=args.raw_data)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["raw_data_status"] == RAW_DATA_VALID:
            logger.info("Raw data contract: PASS")
            return 0
        else:
            logger.error("Raw data contract: FAIL (%s)", result["raw_data_status"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
