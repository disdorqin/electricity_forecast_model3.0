"""
scripts/ledger_update_actuals.py — Append actuals to the actual ledger.

Usage:
    python scripts/ledger_update_actuals.py \\
        --actuals-input actuals.csv \\
        --ledger-dir ./ledger_data \\
        --run-id "run_20260704_001"
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from pipelines.ledger_backfill import run_ledger_backfill

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append actuals to the actual ledger."
    )
    parser.add_argument(
        "--actuals-input", type=str, required=True,
        help="Path to actuals CSV.",
    )
    parser.add_argument(
        "--ledger-dir", type=str, required=True,
        help="Directory to read/write ledger files.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier.",
    )
    args = parser.parse_args(argv)

    actuals_df = pd.read_csv(args.actuals_input)
    logger.info("Loaded %d actual rows from %s", len(actuals_df), args.actuals_input)

    summary = run_ledger_backfill(
        actuals_df=actuals_df,
        ledger_dir=args.ledger_dir,
        run_id=args.run_id,
    )

    print(f"Run ID: {summary['run_id']}")
    print(f"Actual rows appended: {summary['actual_rows']}")
    print(f"Actual ledger size: {summary['actual_ledger_size']}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
