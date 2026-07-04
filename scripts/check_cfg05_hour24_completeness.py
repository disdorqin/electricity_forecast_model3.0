"""
scripts/check_cfg05_hour24_completeness.py — Hour-24 completeness checker.

Verifies that a day-ahead prediction or feature CSV contains exactly 24 rows
for a target day, covering hours 1 through 24 (where hour 24 = D+1 00:00).

Usage::

    python -m scripts.check_cfg05_hour24_completeness \\
        --input /path/to/predictions.csv --target-day 2026-06-30

    python -m scripts.check_cfg05_hour24_completeness \\
        --input /path/to/features.csv --target-day 2026-06-30 --json --strict

Options::

    --input PATH            Path to prediction or feature CSV.
    --target-day YYYY-MM-DD Target day to verify.
    --json                  Output JSON report.
    --strict                Exit non-zero if not COMPLETE_24H.
    --verbose, -v           Increase verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from artifacts.dayahead_window import get_dayahead_window, get_business_day_info

logger = logging.getLogger(__name__)

# ── Status constants ───────────────────────────────────────────────────────

COMPLETE_24H = "COMPLETE_24H"
INCOMPLETE_23H = "INCOMPLETE_23H"
MISSING_HOURS = "MISSING_HOURS"
DUPLICATE_HOURS = "DUPLICATE_HOURS"
INVALID = "INVALID"


def check_cfg05_hour24_completeness(
    input_path: Optional[str] = None,
    target_day: Optional[str] = None,
) -> dict[str, Any]:
    """Check that a CSV has complete 24-hour coverage for a target day.

    Parameters
    ----------
    input_path : str, optional
        Path to CSV with at least a ``ds`` column.
    target_day : str, optional
        Target day in ``YYYY-MM-DD``.

    Returns
    -------
    dict with completeness summary.
    """
    result: dict[str, Any] = {
        "completeness_status": INVALID,
        "row_count": 0,
        "expected_hours": list(range(1, 25)),
        "present_hours": [],
        "missing_hours": [],
        "duplicate_hours": [],
        "ds_min": None,
        "ds_max": None,
        "reason_codes": [],
    }

    if not input_path or not os.path.isfile(input_path):
        result["reason_codes"].append("INPUT_FILE_MISSING")
        return result

    if not target_day:
        result["reason_codes"].append("TARGET_DAY_NOT_PROVIDED")
        return result

    # Read CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        result["reason_codes"].append(f"INPUT_READ_FAILED: {e}")
        return result

    if "ds" not in df.columns:
        result["reason_codes"].append("INPUT_MISSING_DS_COLUMN")
        return result

    result["row_count"] = len(df)

    ds = pd.to_datetime(df["ds"])
    result["ds_min"] = str(ds.min()) if len(df) > 0 else None
    result["ds_max"] = str(ds.max()) if len(df) > 0 else None

    # Compute canonical 24-hour window
    expected_start, expected_end = get_dayahead_window(target_day)
    window_mask = (ds >= expected_start) & (ds < expected_end)
    window_rows = df[window_mask].copy()

    if len(window_rows) == 0:
        result["reason_codes"].append(f"NO_ROWS_IN_TARGET_WINDOW: {expected_start}~{expected_end}")
        result["completeness_status"] = MISSING_HOURS
        return result

    # Derive hour_business if not present
    if "hour_business" in window_rows.columns:
        hours = window_rows["hour_business"].astype(int).values
    else:
        info = get_business_day_info(window_rows["ds"])
        hours = info["hour_business"].values

    result["present_hours"] = sorted(set(int(h) for h in hours))
    result["row_count"] = len(window_rows)

    # Check missing hours
    result["missing_hours"] = [h for h in result["expected_hours"] if h not in result["present_hours"]]

    # Check duplicate hours
    from collections import Counter
    hour_counts = Counter(hours)
    result["duplicate_hours"] = [int(h) for h, c in hour_counts.items() if c > 1]

    # Determine status
    if result["missing_hours"] and result["duplicate_hours"]:
        result["completeness_status"] = DUPLICATE_HOURS
        result["reason_codes"].append(
            f"DUPLICATE_HOURS: {result['duplicate_hours']}"
        )
        result["reason_codes"].append(
            f"MISSING_HOURS: {result['missing_hours']}"
        )
    elif result["duplicate_hours"]:
        result["completeness_status"] = DUPLICATE_HOURS
        result["reason_codes"].append(
            f"DUPLICATE_HOURS: {result['duplicate_hours']}"
        )
    elif result["missing_hours"]:
        if set(result["missing_hours"]) == {24} and result["row_count"] == 23:
            result["completeness_status"] = INCOMPLETE_23H
            result["reason_codes"].append(
                "MISSING_HOUR_24: D+1 00:00 excluded by old exclusive-end filter"
            )
        else:
            result["completeness_status"] = MISSING_HOURS
            result["reason_codes"].append(
                f"MISSING_HOURS: {result['missing_hours']}"
            )
    else:
        result["completeness_status"] = COMPLETE_24H
        result["reason_codes"].append("ALL_24_HOURS_PRESENT")

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable completeness report."""
    print("=" * 60)
    print("cfg05 Hour-24 Completeness Check")
    print("=" * 60)
    print(f"  Status:            {result['completeness_status']}")
    print(f"  Rows in window:    {result['row_count']}")
    print(f"  Expected hours:    {result['expected_hours'][0]}..{result['expected_hours'][-1]}")
    print(f"  Present hours:     {result['present_hours']}")
    if result["missing_hours"]:
        print(f"  Missing hours:     {result['missing_hours']}")
    if result["duplicate_hours"]:
        print(f"  Duplicate hours:   {result['duplicate_hours']}")
    print(f"  ds range:          {result['ds_min']} ~ {result['ds_max']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check cfg05 prediction/feature CSV for 24-hour completeness.",
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to prediction or feature CSV.")
    parser.add_argument("--target-day", type=str, required=True,
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero if not COMPLETE_24H.")
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

    result = check_cfg05_hour24_completeness(
        input_path=args.input,
        target_day=args.target_day,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["completeness_status"] == COMPLETE_24H:
            logger.info("Hour24 completeness: PASS")
            return 0
        else:
            logger.error("Hour24 completeness: FAIL (%s)", result["completeness_status"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
