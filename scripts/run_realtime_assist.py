"""
scripts/run_realtime_assist.py — DA-Safe Realtime Assist runner.

CLI to run the DA-Safe Realtime Assist model. Default behaviour:
``rt_pred = da_anchor`` (most stable). Optional safe correction
can be enabled with ``--enable-safe-correction`` and a model pack.

Usage:
    # Dry run with synthetic data
    python scripts/run_realtime_assist.py --dry-run

    # Real run with input CSV
    python scripts/run_realtime_assist.py --input data.csv --out predictions.csv

    # With safe correction enabled
    python scripts/run_realtime_assist.py --input data.csv \\
        --enable-safe-correction --model-dir ./rt_assist_pack \\
        --start 2026-03-01 --end 2026-03-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data.schema import PREDICTION_OUTPUT_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_realtime_assist")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DA-Safe Realtime Assist prediction runner",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV with hourly data (must have da_anchor or da_price).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to exported rt_assist_pack directory (for safe correction).",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date inclusive (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date inclusive (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use synthetic da_anchor if input not provided.",
    )
    parser.add_argument(
        "--enable-safe-correction", action="store_true",
        help="Enable residual correction (requires --model-dir).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _synthetic_input(n_hours: int = 48) -> pd.DataFrame:
    """Generate synthetic input for dry-run mode."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-03-05 01:00", periods=n_hours, freq="h")
    return pd.DataFrame({
        "ds": timestamps,
        "da_anchor": rng.uniform(80, 200, n_hours),
    })


def run_realtime_assist(
    *,
    input_df: pd.DataFrame,
    dry_run: bool = False,
    model_dir: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    enable_safe_correction: bool = False,
) -> pd.DataFrame:
    """Execute realtime assist prediction.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input with at minimum ``da_anchor`` (or alias) and ``ds`` columns.
    dry_run : bool
        If True, use adapter dry-run mode.
    model_dir : str, optional
        Path to rt_assist_pack for safe correction.
    start : str, optional
        Start date filter.
    end : str, optional
        End date filter.
    enable_safe_correction : bool
        If True and model_dir provided, enable residual correction.

    Returns
    -------
    pd.DataFrame
        Standard schema prediction output.
    """
    from models.adapters.realtime_da_safe_assist import DASafeRealtimeAssistAdapter

    adapter = DASafeRealtimeAssistAdapter(
        enable_safe_correction=enable_safe_correction,
    )

    if model_dir and not dry_run:
        adapter.load_model_pack(model_dir)
    else:
        adapter.load()

    # Use adapter predict
    kwargs = dict(df=input_df)
    if start:
        kwargs["start"] = start
    if end:
        kwargs["end"] = end
    if model_dir and not dry_run:
        kwargs["model_dir"] = model_dir

    result = adapter.predict(**kwargs)

    # If dry-run and result is empty (no da_anchor), fill synthetic
    if dry_run and len(result) > 0 and "da_anchor" in input_df.columns:
        # adapter already set rt_pred = da_anchor
        pass

    if len(result) > 0:
        logger.info(
            f"Realtime assist: {len(result)} rows, "
            f"date range: {result['ds'].min()} ~ {result['ds'].max()}"
        )

    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load or generate input
    if args.input:
        logger.info(f"Loading input from {args.input}")
        df = pd.read_csv(args.input)
    elif args.dry_run:
        df = _synthetic_input(48)
        logger.info("Dry-run mode: generated synthetic 48h input")
    else:
        logger.error("Either --input or --dry-run is required")
        return 1

    # Run
    result = run_realtime_assist(
        input_df=df,
        dry_run=args.dry_run,
        model_dir=args.model_dir,
        start=args.start,
        end=args.end,
        enable_safe_correction=args.enable_safe_correction,
    )

    if len(result) == 0:
        logger.warning("No predictions produced.")
        return 0

    # Write output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info(f"Predictions written to {out_path}")
    else:
        print(f"\n=== Realtime Assist Prediction Summary ===")
        print(f"Total rows: {len(result)}")
        print(f"Date range: {result['ds'].min()} ~ {result['ds'].max()}")
        print(f"rt_pred = da_anchor: {(result['y_pred'] == df['da_anchor']).all() if 'da_anchor' in df.columns else 'N/A'}")
        print(f"Safe correction: {'ON' if args.enable_safe_correction else 'OFF'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
