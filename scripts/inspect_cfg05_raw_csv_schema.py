"""
scripts/inspect_cfg05_raw_csv_schema.py — cfg05 raw CSV schema inspector.

Reads a raw Chinese CSV (GBK/UTF-8), validates columns/dtypes/timestamps,
and outputs metadata WITHOUT exposing actual data values by default.

Usage::

    python -m scripts.inspect_cfg05_raw_csv_schema \\
        --raw-data /path/to/shandong_pmos_hourly.csv

    python -m scripts.inspect_cfg05_raw_csv_schema \\
        --raw-data /path/to/data.csv --sample-rows 5 --no-redact-values --json

Options::

    --raw-data PATH         Path to raw Chinese CSV.
    --sample-rows N         Show N sample rows (default: 0, requires --no-redact-values).
    --redact-values         Redact actual data values (default: True, use --no-redact-values to show).
    --json                  Output JSON report.
    --strict                Exit non-zero on invalid/missing.
    --verbose, -v           Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from scripts.check_cfg05_raw_data_contract import (
    check_cfg05_raw_data_contract,
    RAW_DATA_MISSING,
    RAW_DATA_INVALID,
    RAW_DATA_VALID,
    REQUIRED_CHINESE_COLUMNS,
)

logger = logging.getLogger(__name__)

_REDACTED = "***REDACTED***"


def inspect_cfg05_raw_csv_schema(
    raw_data: Optional[str] = None,
    redact_values: bool = True,
    sample_rows: int = 0,
) -> dict[str, Any]:
    """Inspect raw Chinese CSV schema and metadata.

    Parameters
    ----------
    raw_data : str, optional
        Path to raw Chinese CSV.
    redact_values : bool
        Whether to redact actual data values in output.
    sample_rows : int
        Number of sample rows to show (0 = none).

    Returns
    -------
    dict with schema inspection summary.
    """
    # Run the contract checker first
    contract = check_cfg05_raw_data_contract(raw_data=raw_data)

    result: dict[str, Any] = {
        "raw_data_status": contract["raw_data_status"],
        "raw_data_path": raw_data,
        "rows": contract["rows"],
        "columns": list(contract["columns_present"]),
        "dtypes": {},
        "null_counts": dict(contract.get("null_counts", {})),
        "time_min": contract["time_min"],
        "time_max": contract["time_max"],
        "missing_columns": list(contract.get("missing_columns", [])),
        "redacted": redact_values,
        "sample_rows": [],
        "reason_codes": list(contract.get("reason_codes", [])),
    }

    # If file is missing or unreadable, return early
    if contract["raw_data_status"] != RAW_DATA_VALID or not raw_data or not os.path.isfile(raw_data):
        return result

    # Read CSV for detailed inspection (same encoding logic)
    try:
        try:
            df = pd.read_csv(raw_data, encoding="gbk")
        except UnicodeDecodeError:
            df = pd.read_csv(raw_data, encoding="utf-8")
    except Exception as e:
        result["reason_codes"].append(f"INSPECT_READ_FAILED: {e}")
        return result

    # Collect dtypes
    for col in df.columns:
        result["dtypes"][str(col)] = str(df[col].dtype)

    # Collect null counts per column
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            result["null_counts"][str(col)] = null_count

    # Sample rows (redacted by default)
    if sample_rows > 0 and len(df) > 0:
        sample = df.head(min(sample_rows, len(df)))
        if redact_values:
            # Show column names only, no values
            for _, row in sample.iterrows():
                result["sample_rows"].append(
                    {str(col): _REDACTED for col in df.columns}
                )
        else:
            for _, row in sample.iterrows():
                row_dict = {}
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, pd.Timestamp):
                        val = str(val)
                    row_dict[str(col)] = val
                result["sample_rows"].append(row_dict)

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable schema inspection report."""
    print("=" * 60)
    print("cfg05 Raw CSV Schema Inspection")
    print("=" * 60)
    print(f"  Status:           {result['raw_data_status']}")
    print(f"  Path:             {result['raw_data_path'] or 'N/A'}")
    print(f"  Rows:             {result['rows']}")
    print(f"  Columns:          {len(result['columns'])}")
    print(f"  Missing required: {result['missing_columns'] or 'None'}")
    print(f"  Time range:       {result['time_min']} ~ {result['time_max']}")
    print(f"  Redacted:         {result['redacted']}")
    print()
    if result["dtypes"]:
        print("  Column dtypes:")
        for col, dtype in result["dtypes"].items():
            print(f"    {col}: {dtype}")
    print()
    if result["null_counts"]:
        print("  Null counts:")
        for col, count in result["null_counts"].items():
            if count > 0:
                print(f"    {col}: {count}")
    print()
    if result["sample_rows"]:
        print(f"  Sample rows ({len(result['sample_rows'])}):")
        for i, row in enumerate(result["sample_rows"]):
            print(f"    Row {i}: {row}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cfg05 raw CSV schema inspection (values redacted by default).",
    )
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw Chinese CSV.")
    parser.add_argument("--sample-rows", type=int, default=0,
                        help="Show N sample rows (needs --no-redact-values).")
    parser.add_argument("--redact-values", action="store_true", default=True,
                        help="Redact actual data values (default).")
    parser.add_argument("--no-redact-values", action="store_false", dest="redact_values",
                        help="Show actual data values (use with care).")
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

    result = inspect_cfg05_raw_csv_schema(
        raw_data=args.raw_data,
        redact_values=args.redact_values,
        sample_rows=args.sample_rows,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["raw_data_status"] == RAW_DATA_VALID:
            logger.info("Schema inspection: PASS")
            return 0
        else:
            logger.error("Schema inspection: FAIL (%s)", result["raw_data_status"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
