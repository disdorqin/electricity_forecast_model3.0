"""
scripts/run_residual_correction.py — Residual correction CLI runner.

Applies residual correction (P5M / negative / low-valley) to standard
prediction output.  Default behaviour: DATA-MISSING no-op.

Usage:
    # Dry-run with synthetic predictions
    python scripts/run_residual_correction.py --dry-run --out corrected.csv

    # Real run from prediction output
    python scripts/run_residual_correction.py --input predictions.csv --out corrected.csv

    # With risk data
    python scripts/run_residual_correction.py --input predictions.csv \\
        --risk-path risk_data.csv --profile aggressive --out corrected.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_residual_correction")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply residual correction to prediction output.",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to prediction output CSV (standard schema).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path for corrected predictions.",
    )
    parser.add_argument(
        "--risk-path", type=str, default=None,
        help="Path to risk data CSV (e.g. with negative_prob column).",
    )
    parser.add_argument(
        "--canonical-pack", type=str, default=None,
        help="Path to canonical prediction pack (for P5M adapter).",
    )
    parser.add_argument(
        "--profile", type=str, default="conservative",
        choices=["conservative", "moderate", "aggressive"],
        help="Correction profile (default: conservative).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic prediction data (no real input needed).",
    )
    parser.add_argument(
        "--production", action="store_true", default=True,
        help="Production mode: y_true is forbidden (default).",
    )
    parser.add_argument(
        "--no-production",
        action="store_false",
        dest="production",
        help="Eval mode: allow y_true column.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _synthetic_predictions(n_hours: int = 24) -> pd.DataFrame:
    """Generate synthetic prediction output for dry-run mode."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-03-05 01:00", periods=n_hours, freq="h")
    return pd.DataFrame({
        "task": "dayahead",
        "model_name": "cfg05",
        "target_day": "2026-03-05",
        "ds": timestamps,
        "y_pred": rng.uniform(80, 200, n_hours),
    })


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from pipelines.residual_correction import apply_residual_correction

    # Load or generate predictions
    if args.input:
        logger.info(f"Loading predictions from {args.input}")
        df = pd.read_csv(args.input)
    elif args.dry_run:
        df = _synthetic_predictions(24)
        logger.info("Dry-run mode: generated synthetic 24h predictions")
    else:
        logger.error("Either --input or --dry-run is required")
        return 1

    # Load risk data if provided
    risk_df = None
    if args.risk_path:
        logger.info(f"Loading risk data from {args.risk_path}")
        risk_df = pd.read_csv(args.risk_path)

    # Apply correction
    result = apply_residual_correction(
        predictions_df=df,
        correction_profile=args.profile,
        risk_df=risk_df,
        canonical_pack_path=args.canonical_pack,
        production=args.production,
    )

    # Count corrections
    n_corrected = result["correction_applied"].sum()
    logger.info(
        f"Residual correction complete: {len(result)} rows, "
        f"{n_corrected} corrected ({result['correction_module'].iloc[0]}), "
        f"risk_source={result['risk_source'].iloc[0]}"
    )

    # Write output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info(f"Corrected output written to {out_path}")
    else:
        print(f"\n=== Residual Correction Summary ===")
        print(f"Total rows: {len(result)}")
        print(f"Correction module: {result['correction_module'].iloc[0]}")
        print(f"Correction applied: {n_corrected} / {len(result)} rows")
        print(f"Risk source: {result['risk_source'].iloc[0]}")
        print(f"Columns: {list(result.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
