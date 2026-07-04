"""
scripts/validate_ledgers.py — Validate all ledgers in a directory.

Checks each ledger file for schema compliance:
- prediction_ledger.csv
- corrected_ledger.csv
- actual_ledger.csv
- fusion_ledger.csv
- weight_ledger.csv

Usage:
    python scripts/validate_ledgers.py --ledger-dir ./ledger_data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    CORRECTED_LEDGER_COLUMNS,
    ACTUAL_LEDGER_COLUMNS,
    FUSION_LEDGER_COLUMNS,
    WEIGHT_LEDGER_COLUMNS,
)
from ledgers.store import load_ledger
from ledgers.prediction_ledger import validate_prediction_ledger, validate_corrected_ledger
from ledgers.actual_ledger import validate_actual_ledger
from ledgers.fusion_ledger import validate_fusion_ledger
from ledgers.weight_ledger import validate_weight_ledger

logger = logging.getLogger(__name__)

_LEDGER_CHECKS = [
    ("prediction_ledger.csv", PREDICTION_LEDGER_COLUMNS, validate_prediction_ledger),
    ("corrected_ledger.csv", CORRECTED_LEDGER_COLUMNS, validate_corrected_ledger),
    ("actual_ledger.csv", ACTUAL_LEDGER_COLUMNS, validate_actual_ledger),
    ("fusion_ledger.csv", FUSION_LEDGER_COLUMNS, validate_fusion_ledger),
    ("weight_ledger.csv", WEIGHT_LEDGER_COLUMNS, validate_weight_ledger),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate all ledgers in a directory."
    )
    parser.add_argument(
        "--ledger-dir", type=str, required=True,
        help="Directory containing ledger CSV files.",
    )
    args = parser.parse_args(argv)

    all_ok = True

    for filename, columns, validator in _LEDGER_CHECKS:
        path = os.path.join(args.ledger_dir, filename)
        df = load_ledger(path, columns=columns)

        if len(df) == 0 and not os.path.isfile(path):
            print(f"[MISSING] {filename} — file not found")
            all_ok = False
            continue

        is_valid, issues = validator(df)

        if is_valid:
            print(f"[  OK  ] {filename} — {len(df)} rows")
        else:
            print(f"[FAILED] {filename} — {len(df)} rows")
            for issue in issues:
                print(f"        - {issue}")
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
