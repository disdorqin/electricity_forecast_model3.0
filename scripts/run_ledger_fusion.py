"""
scripts/run_ledger_fusion.py — Ledger-based fusion runner CLI.

Reads from corrected and actual ledgers (stored as CSV files), runs fusion,
and writes to fusion and weight ledgers.

Usage:
    python scripts/run_ledger_fusion.py \\
        --ledger-dir ./ledger_data \\
        --method equal_weight \\
        --allow-dry-run \\
        --run-id "run_20260704_001"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from data.schema import CORRECTED_LEDGER_COLUMNS, ACTUAL_LEDGER_COLUMNS
from ledgers.store import load_ledger
from pipelines.ledger_fusion import run_ledger_fusion
from ledgers.store import save_ledger
from data.schema import FUSION_LEDGER_COLUMNS, WEIGHT_LEDGER_COLUMNS
from ledgers.fusion_ledger import append_fusion_to_ledger
from ledgers.weight_ledger import extract_weight_rows, append_weights_to_ledger

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ledger-based fusion."
    )
    parser.add_argument(
        "--ledger-dir", type=str, required=True,
        help="Directory containing ledger CSV files.",
    )
    parser.add_argument(
        "--method", type=str, default="equal_weight",
        choices=["equal_weight", "prior_weight", "bgew_skeleton"],
        help="Fusion method (default: equal_weight).",
    )
    parser.add_argument(
        "--allow-dry-run", action="store_true", default=False,
        help="Include READY_DRY_RUN models in fusion.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier.",
    )
    args = parser.parse_args(argv)

    # Load corrected ledger
    corrected_path = os.path.join(args.ledger_dir, "corrected_ledger.csv")
    corrected_ledger = load_ledger(corrected_path, columns=CORRECTED_LEDGER_COLUMNS)
    logger.info("Loaded corrected ledger with %d rows", len(corrected_ledger))

    if len(corrected_ledger) == 0:
        print("ERROR: Corrected ledger is empty. Cannot run fusion.")
        return 1

    # Load actual ledger (optional)
    actual_path = os.path.join(args.ledger_dir, "actual_ledger.csv")
    actual_ledger = load_ledger(actual_path, columns=ACTUAL_LEDGER_COLUMNS)
    logger.info("Loaded actual ledger with %d rows", len(actual_ledger))

    # Load existing fusion and weight ledgers (if any)
    fusion_path = os.path.join(args.ledger_dir, "fusion_ledger.csv")
    existing_fusion = load_ledger(fusion_path, columns=FUSION_LEDGER_COLUMNS)

    weight_path = os.path.join(args.ledger_dir, "weight_ledger.csv")
    existing_weights = load_ledger(weight_path, columns=WEIGHT_LEDGER_COLUMNS)

    # Run fusion pipeline
    summary = run_ledger_fusion(
        corrected_ledger_df=corrected_ledger,
        actual_ledger_df=actual_ledger if len(actual_ledger) > 0 else None,
        method=args.method,
        allow_dry_run=args.allow_dry_run,
        run_id=args.run_id,
        existing_fusion_ledger=existing_fusion if len(existing_fusion) > 0 else None,
        existing_weight_ledger=existing_weights if len(existing_weights) > 0 else None,
    )

    # Save fusion and weight ledgers
    # Re-load the full ledger after append for saving
    fusion_ledger = load_ledger(fusion_path, columns=FUSION_LEDGER_COLUMNS)
    weight_ledger = load_ledger(weight_path, columns=WEIGHT_LEDGER_COLUMNS)

    # We need to get the actual ledger DataFrames back. Let's re-construct them.
    # Since run_ledger_fusion returns a summary, we need to re-compute or use a different approach.
    # Simpler: just re-read after the pipeline has done its work.
    # Actually, the pipeline appends in-memory but doesn't save. Let me handle differently.

    # For simplicity, re-run the append step and save
    fusion_result = append_fusion_to_ledger(
        corrected_ledger,  # placeholder - we need the actual fusion_df
    )

    # Simpler approach: just read fusion_df from the call
    # We need to modify run_ledger_fusion to return fusion_df
    # OR just do the steps inline here.

    # Inline fusion for CLI:
    from fusion.engine import run_fusion as do_fusion
    from ledgers.actual_ledger import filter_actuals_for_training

    actuals_for_fusion = None
    if len(actual_ledger) > 0:
        target_days = sorted(corrected_ledger["target_day"].unique())
        if len(target_days) > 0:
            actuals_for_fusion = filter_actuals_for_training(
                actual_ledger, target_day=str(target_days[0]), window=90,
            )

    fusion_df = do_fusion(
        corrected_ledger,
        method=args.method,
        actuals_df=actuals_for_fusion,
        allow_dry_run=args.allow_dry_run,
    )
    logger.info("Fusion produced %d rows", len(fusion_df))

    # Append to fusion ledger and save
    fusion_ledger_new = append_fusion_to_ledger(
        fusion_df,
        ledger_df=existing_fusion if len(existing_fusion) > 0 else None,
        run_id=args.run_id,
    )
    save_ledger(fusion_ledger_new, fusion_path)
    logger.info("Saved fusion ledger (%d rows) to %s", len(fusion_ledger_new), fusion_path)

    # Extract weight rows and save
    weight_df = extract_weight_rows(fusion_df)
    weight_ledger_new = append_weights_to_ledger(
        weight_df,
        ledger_df=existing_weights if len(existing_weights) > 0 else None,
        run_id=args.run_id,
    )
    save_ledger(weight_ledger_new, weight_path)
    logger.info("Saved weight ledger (%d rows) to %s", len(weight_ledger_new), weight_path)

    print(f"Run ID: {args.run_id}")
    print(f"Fusion method: {args.method}")
    print(f"Fusion rows: {len(fusion_df)}")
    print(f"Fusion ledger size: {len(fusion_ledger_new)}")
    print(f"Weight ledger size: {len(weight_ledger_new)}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
