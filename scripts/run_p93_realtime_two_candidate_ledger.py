"""
scripts/run_p93_realtime_two_candidate_ledger.py — Build realtime two-candidate ledger.

Usage::

    python -m scripts.run_p93_realtime_two_candidate_ledger \\
        --da-anchor-predictions .local_artifacts/realtime/online_pack/realtime_online_pack.csv \\
        --sgdfnet-predictions .local_artifacts/p92/sgdfnet_assist_output/sgdfnet_realtime_assist_pack.csv \\
        --output-dir .local_artifacts/p93 \\
        --run-id p93_demo \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def run_realtime_ledger(
    da_anchor_predictions: str = "",
    sgdfnet_predictions: str = "",
    output_dir: str = ".local_artifacts/p93",
    run_id: str = "p93",
) -> dict:
    """Build realtime two-candidate prediction ledger."""
    from ledgers.realtime_prediction_ledger import (
        build_realtime_ledger,
        validate_realtime_ledger,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Load DA anchor predictions
    if not da_anchor_predictions or not os.path.isfile(da_anchor_predictions):
        return {"status": "BLOCKED", "reason": f"DA anchor not found: {da_anchor_predictions}"}

    da_df = pd.read_csv(da_anchor_predictions)

    # Load SGDFNet predictions (optional)
    sg_df = None
    if sgdfnet_predictions and os.path.isfile(sgdfnet_predictions):
        sg_df = pd.read_csv(sgdfnet_predictions)
        logger.info("Loaded SGDFNet predictions: %d rows", len(sg_df))
    else:
        logger.info("SGDFNet predictions not available — DA-Safe Baseline only")

    # Build ledger
    ledger = build_realtime_ledger(
        da_anchor_predictions=da_df,
        sgdfnet_predictions=sg_df,
        run_id=run_id,
    )

    # Validate
    valid, issues = validate_realtime_ledger(ledger)

    # Save
    csv_path = os.path.join(output_dir, "realtime_prediction_ledger.csv")
    ledger.to_csv(csv_path, index=False)

    models_present = ledger["model_name"].unique().tolist() if "model_name" in ledger.columns else []

    result = {
        "status": "BUILT" if valid else "BUILT_WITH_ISSUES",
        "rows": len(ledger),
        "models": models_present,
        "csv_path": csv_path,
        "valid": valid,
        "issues": issues,
        "run_id": run_id,
    }

    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P93: Build realtime two-candidate ledger")
    p.add_argument("--da-anchor-predictions", type=str, required=True)
    p.add_argument("--sgdfnet-predictions", type=str, default="")
    p.add_argument("--output-dir", type=str, default=".local_artifacts/p93")
    p.add_argument("--run-id", type=str, default="p93")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = run_realtime_ledger(
        da_anchor_predictions=args.da_anchor_predictions,
        sgdfnet_predictions=args.sgdfnet_predictions,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"P93 Realtime Two-Candidate Ledger: {result['status']}")
        print(f"{'='*60}")
        print(f"  Rows: {result['rows']}")
        print(f"  Models: {result['models']}")
        print(f"  Valid: {result['valid']}")
        print(f"  CSV: {result.get('csv_path', 'N/A')}")
        if result.get("issues"):
            for issue in result["issues"]:
                print(f"  Issue: {issue}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
