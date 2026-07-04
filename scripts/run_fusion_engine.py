"""
scripts/run_fusion_engine.py — Fusion engine CLI runner.

Applies fusion to corrected prediction output.

Usage:
    # Dry-run with synthetic corrected predictions
    python scripts/run_fusion_engine.py --dry-run --out fused.csv

    # Real run from corrected output
    python scripts/run_fusion_engine.py --input corrected.csv --out fused.csv

    # With prior weights
    python scripts/run_fusion_engine.py --input corrected.csv \\
        --method prior_weight --prior-weights-json '{"cfg05":0.6,"best_two_average":0.4}' \\
        --out fused.csv

    # BGEW skeleton
    python scripts/run_fusion_engine.py --input corrected.csv \\
        --method bgew_skeleton --actuals actuals.csv --out fused.csv
"""

from __future__ import annotations

import argparse
import json
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
logger = logging.getLogger("run_fusion_engine")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply fusion to corrected prediction output.",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to corrected prediction CSV.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path for fused predictions.",
    )
    parser.add_argument(
        "--method", type=str, default="equal_weight",
        choices=["equal_weight", "prior_weight", "bgew_skeleton"],
        help="Fusion method (default: equal_weight).",
    )
    parser.add_argument(
        "--prior-weights-json", type=str, default=None,
        help="JSON-encoded prior weights, e.g. "
             "'{\"cfg05\":0.6,\"best_two_average\":0.4}'.",
    )
    parser.add_argument(
        "--actuals", type=str, default=None,
        help="Path to actuals CSV (for bgew_skeleton).",
    )
    parser.add_argument(
        "--allow-dry-run", action="store_true",
        help="Include READY_DRY_RUN models in fusion.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic corrected predictions (no real input needed).",
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


def _synthetic_corrected_predictions(n_hours: int = 24) -> pd.DataFrame:
    """Generate synthetic corrected prediction output for dry-run mode."""
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    models = ["cfg05", "best_two_average"]
    rows: list[pd.DataFrame] = []
    for model in models:
        y_pred_raw = rng.uniform(80, 200, n_hours)
        df = pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "model_name": [model] * n_hours,
            "target_day": ["2026-07-04"] * n_hours,
            "ds": timestamps,
            "y_pred_raw": y_pred_raw,
            "y_pred_corrected": y_pred_raw.copy(),
            "residual_delta": np.zeros(n_hours),
            "correction_applied": [False] * n_hours,
            "correction_module": ["p5m_residual_noop"] * n_hours,
            "risk_source": ["DATA_MISSING"] * n_hours,
            "reason_codes": ["DATA_MISSING_NO_OP"] * n_hours,
            "correction_version": ["0.0.0"] * n_hours,
            "source_confidence": [0.5] * n_hours,
            "model_version": ["1.0.0"] * n_hours,
        })
        df = add_business_time_columns(df, timestamp_col="ds")
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse prior weights
    prior_weights = None
    if args.prior_weights_json:
        prior_weights = json.loads(args.prior_weights_json)

    # Load actuals
    actuals_df = None
    if args.actuals:
        logger.info(f"Loading actuals from {args.actuals}")
        actuals_df = pd.read_csv(args.actuals)

    from fusion.engine import run_fusion

    # Load or generate corrected predictions
    if args.input:
        logger.info(f"Loading corrected predictions from {args.input}")
        corrected_df = pd.read_csv(args.input)
    elif args.dry_run:
        corrected_df = _synthetic_corrected_predictions(24)
        logger.info("Dry-run mode: generated synthetic corrected predictions")
    else:
        logger.error("Either --input or --dry-run is required")
        return 1

    # Run fusion
    result = run_fusion(
        corrected_df=corrected_df,
        method=args.method,
        actuals_df=actuals_df,
        prior_weights=prior_weights,
        allow_dry_run=args.allow_dry_run,
        production=args.production,
    )

    # Summary
    n_included = len(result)
    if n_included > 0:
        included = result["included_models"].iloc[0] if "included_models" in result.columns else ""
        method_used = result["fusion_method"].iloc[0] if "fusion_method" in result.columns else args.method
        logger.info(
            f"Fusion complete: {n_included} rows, method={method_used}, "
            f"included_models={included}"
        )
    else:
        logger.warning("Fusion returned empty output — no models passed the readiness gate")

    # Write output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info(f"Fusion output written to {out_path}")
    else:
        print(f"\n=== Fusion Summary ===")
        print(f"Total rows: {len(result)}")
        if len(result) > 0:
            print(f"Method: {result['fusion_method'].iloc[0]}")
            print(f"Included models: {result['included_models'].iloc[0]}")
            print(f"Excluded models: {result['excluded_models'].iloc[0]}")
            print(f"Readiness mode: {result['readiness_mode'].iloc[0]}")
            print(f"Columns: {list(result.columns)}")
        else:
            print("(empty output)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
