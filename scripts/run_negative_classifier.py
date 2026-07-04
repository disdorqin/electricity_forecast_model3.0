"""
scripts/run_negative_classifier.py — CLI for negative-price classifier.

Usage::

    # Dry-run with synthetic data
    python -m scripts.run_negative_classifier --dry-run

    # Run on fusion output / ledger
    python -m scripts.run_negative_classifier \\
        --input /path/to/fusion_output.csv \\
        --out /path/to/final_output.csv

    # Disable rule fallback
    python -m scripts.run_negative_classifier \\
        --input fusion.csv --out final.csv --no-rule-fallback

Options:

    --input PATH        Fusion output or ledger CSV path
    --out PATH          Output path for final output CSV (default: stdout)
    --model-dir PATH    Directory containing classifier artifacts
    --rule-fallback     Apply rule-based fallback (default: on)
    --no-rule-fallback  Disable rule-based fallback
    --dry-run           Generate synthetic fusion data and run
    --production        Production mode (default: True)
    --verbose, -v       Increase log verbosity
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from data.schema import FUSION_OUTPUT_COLUMNS
from pipelines.classifier_pipeline import (
    run_negative_classifier,
    _build_synthetic_fusion,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run negative-price classifier on fusion output.",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Fusion output CSV path")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: stdout)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing classifier artifacts")
    parser.add_argument("--rule-fallback", dest="rule_fallback",
                        action="store_true", default=True,
                        help="Apply rule-based fallback")
    parser.add_argument("--no-rule-fallback", dest="rule_fallback",
                        action="store_false",
                        help="Disable rule-based fallback")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Generate synthetic fusion data and run")
    parser.add_argument("--production", action="store_true", default=True,
                        help="Production mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        default=False, help="Increase verbosity")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # ── Load or generate fusion data ────────────────────────────────────
    if args.dry_run:
        logger.info("Dry-run mode: generating synthetic fusion data")
        fusion_df = _build_synthetic_fusion(24, include_negative=True)
    elif args.input:
        logger.info("Loading fusion input from: %s", args.input)
        fusion_df = pd.read_csv(args.input)
        # Ensure datetime columns
        for c in ["business_day", "ds"]:
            if c in fusion_df.columns:
                fusion_df[c] = pd.to_datetime(fusion_df[c])
    else:
        logger.error("Either --input or --dry-run is required")
        return 1

    # ── Run classifier ──────────────────────────────────────────────────
    try:
        result = run_negative_classifier(
            fusion_df=fusion_df,
            model_dir=args.model_dir,
            rule_fallback=args.rule_fallback,
            production=args.production,
        )
    except ValueError as e:
        logger.error("Classifier pipeline failed: %s", e)
        return 1

    # ── Output ──────────────────────────────────────────────────────────
    if args.out:
        result.to_csv(args.out, index=False)
        logger.info("Final output written to: %s", args.out)
    else:
        result.to_csv(sys.stdout, index=False)

    n_flagged = result["negative_flag"].sum()
    logger.info(
        "Done: %d rows, %d negative-flagged, "
        "classifier=%s",
        len(result),
        n_flagged,
        result["classifier_applied"].iloc[0] if len(result) > 0 else "N/A",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
