"""
scripts/ledger_append_predictions.py — Append P2/P3 predictions to ledgers.

Usage:
    python scripts/ledger_append_predictions.py \\
        --prediction-input predictions.csv \\
        --corrected-input corrected.csv \\
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
        description="Append P2/P3 predictions to ledgers."
    )
    parser.add_argument(
        "--prediction-input", type=str, default=None,
        help="Path to P2 standard prediction output CSV.",
    )
    parser.add_argument(
        "--corrected-input", type=str, default=None,
        help="Path to P3 corrected prediction output CSV.",
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

    prediction_df = None
    corrected_df = None

    if args.prediction_input:
        prediction_df = pd.read_csv(args.prediction_input)
        logger.info("Loaded %d prediction rows from %s", len(prediction_df), args.prediction_input)

    if args.corrected_input:
        corrected_df = pd.read_csv(args.corrected_input)
        logger.info("Loaded %d corrected rows from %s", len(corrected_df), args.corrected_input)

    summary = run_ledger_backfill(
        prediction_df=prediction_df,
        corrected_df=corrected_df,
        ledger_dir=args.ledger_dir,
        run_id=args.run_id,
    )

    print(f"Run ID: {summary['run_id']}")
    print(f"Prediction rows appended: {summary['prediction_rows']}")
    print(f"Corrected rows appended: {summary['corrected_rows']}")
    print(f"Prediction ledger size: {summary['prediction_ledger_size']}")
    print(f"Corrected ledger size: {summary['corrected_ledger_size']}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
