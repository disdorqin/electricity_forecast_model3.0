"""
scripts/prepare_cfg05_real_input.py — Validate/prepare cfg05 real input CSV.

Checks an input CSV for cfg05 feature compatibility and optionally
writes a prepared (filtered/reordered) copy.

Usage::

    # Check input CSV compatibility
    python -m scripts.prepare_cfg05_real_input --input /path/to/data.csv

    # With target day filter and output
    python -m scripts.prepare_cfg05_real_input \\
        --input /path/to/data.csv --target-day 2026-07-01 \\
        --out /tmp/cfg05_prepared.csv

    # JSON output
    python -m scripts.prepare_cfg05_real_input \\
        --input /path/to/data.csv --json

Options::

    --input PATH                Path to candidate input CSV (required).
    --target-day YYYY-MM-DD     Target day for 24-row filter.
    --out PATH                  Optional output path for prepared CSV.
    --json                      Output JSON report.
    --verbose, -v               Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from artifacts.readiness import (
    check_cfg05_input, SCHEMA_READY, INVALID, MISSING,
)

logger = logging.getLogger(__name__)


def prepare_cfg05_real_input(
    input_path: Optional[str] = None,
    target_day: Optional[str] = None,
    out_path: Optional[str] = None,
) -> dict[str, Any]:
    """Validate and optionally prepare cfg05 real input CSV.

    Parameters
    ----------
    input_path : str, optional
        Path to candidate input CSV.
    target_day : str, optional
        Target day in YYYY-MM-DD format (for row count validation).
    out_path : str, optional
        If set, write prepared CSV to this path.

    Returns
    -------
    dict with keys:
        input_path, exists, feature_count, columns_present, columns_missing,
        has_ds, ds_parsable, target_day_rows, status, reason_codes, out_written
    """
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

    result: dict[str, Any] = {
        "input_path": input_path,
        "exists": False,
        "feature_count": len(CFG05_FEATURE_COLUMNS),
        "columns_present": 0,
        "columns_missing": [],
        "has_ds": False,
        "ds_parsable": True,
        "target_day_rows": None,
        "status": MISSING,
        "reason_codes": [],
        "out_written": False,
    }

    if not input_path or not os.path.isfile(input_path):
        result["reason_codes"].append("CFG05_INPUT_MISSING")
        return result

    result["exists"] = True

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        result["status"] = INVALID
        result["reason_codes"].append(f"CFG05_INPUT_LOAD_FAILED: {e}")
        return result

    if len(df) == 0:
        result["status"] = INVALID
        result["reason_codes"].append("CFG05_INPUT_EMPTY")
        return result

    # Check feature columns
    present = [c for c in CFG05_FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in CFG05_FEATURE_COLUMNS if c not in df.columns]

    result["columns_present"] = len(present)
    result["columns_missing"] = missing

    if missing:
        result["status"] = INVALID
        result["reason_codes"].append(
            f"CFG05_INPUT_MISSING_{len(missing)}_FEATURE_COLUMNS: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
        # Still continue to provide full diagnostics

    # Check ds column
    result["has_ds"] = "ds" in df.columns
    if not result["has_ds"]:
        result["reason_codes"].append("CFG05_INPUT_MISSING_DS_COLUMN")
        if result["status"] == MISSING:
            result["status"] = INVALID
        return result

    # Check ds is parsable
    if result["has_ds"]:
        try:
            pd.to_datetime(df["ds"])
            result["ds_parsable"] = True
        except Exception:
            result["ds_parsable"] = False
            result["reason_codes"].append("CFG05_INPUT_DS_NOT_PARSABLE")
            result["status"] = INVALID
            return result

    # Check target day row count
    if target_day:
        target_dt = pd.Timestamp(target_day)
        start = target_dt + pd.Timedelta(hours=1)
        end = target_dt + pd.Timedelta(days=1)
        ds_parsed = pd.to_datetime(df["ds"])
        mask = (ds_parsed >= start) & (ds_parsed < end)
        target_rows = int(mask.sum())
        result["target_day_rows"] = target_rows
        result["reason_codes"].append(f"CFG05_INPUT_TARGET_DAY_ROWS: {target_rows}")
        if target_rows == 0:
            result["reason_codes"].append(
                f"CFG05_INPUT_NO_DATA_FOR_TARGET_DAY: {target_day}"
            )

    # Determine final status
    if missing:
        result["status"] = INVALID
    elif result["has_ds"] and result["ds_parsable"]:
        result["status"] = SCHEMA_READY
        result["reason_codes"].append("CFG05_INPUT_SCHEMA_READY")
    else:
        result["status"] = INVALID

    # Optionally write prepared CSV
    if out_path and result["status"] == SCHEMA_READY:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Write with canonical column order
        out_cols = ["ds"] + list(CFG05_FEATURE_COLUMNS)
        out_df = df[[c for c in out_cols if c in df.columns]]
        out_df.to_csv(out_path, index=False)
        result["out_written"] = True
        logger.info("Wrote prepared input to %s", out_path)

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable prepare report."""
    print("=" * 60)
    print("cfg05 Input Prepare Report")
    print("=" * 60)
    print(f"  Input path:     {result['input_path'] or 'N/A'}")
    print(f"  Exists:         {result['exists']}")
    print(f"  Feature count:  {result['feature_count']} (from CFG05_FEATURE_COLUMNS)")
    print(f"  Present:        {result['columns_present']}")
    print(f"  Missing:        {len(result['columns_missing'])} cols: {result['columns_missing'][:8]}{'...' if len(result['columns_missing']) > 8 else ''}")
    print(f"  Has ds:         {result['has_ds']}")
    print(f"  ds parsable:    {result['ds_parsable']}")
    print(f"  Target rows:    {result['target_day_rows']}")
    print(f"  Status:         {result['status']}")
    print(f"  Out written:    {result['out_written']}")
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = prepare_cfg05_real_input(
        input_path=args.input,
        target_day=args.target_day,
        out_path=args.out,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate/prepare cfg05 real input CSV.",
    )
    parser.add_argument("--input", type=str, default=None, required=True,
                        help="Path to candidate input CSV.")
    parser.add_argument("--target-day", type=str, default=None,
                        help="Target day in YYYY-MM-DD for row filter.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output path for prepared CSV.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON report.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
