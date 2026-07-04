"""
scripts/run_dayahead_model_zoo.py — Day-ahead model zoo prediction runner.

CLI to run predictions from the day-ahead model zoo. Supports
dry-run mode (synthetic predictions) and real adapter-based execution.

Usage:
    # Dry-run with default fusion pool
    python scripts/run_dayahead_model_zoo.py --dry-run --out predictions.csv

    # Specific models (comma-separated)
    python scripts/run_dayahead_model_zoo.py --models cfg05,best_two_average --dry-run

    # Real run with model weights
    python scripts/run_dayahead_model_zoo.py --input data.csv --models cfg05 --model-dir ./weights
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import PREDICTION_OUTPUT_COLUMNS
from src.registry.dayahead_models import (
    CHAMPION_MODEL_ID,
    DEFAULT_FUSION_POOL,
    get_model_config,
    is_invalid_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_dayahead_model_zoo")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day-ahead model zoo prediction runner",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="default",
        help=(
            "Comma-separated model IDs, or 'default' for DEFAULT_FUSION_POOL. "
            "Example: --models cfg05,best_two_average"
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV with raw hourly data.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic predictions (no real model weights needed).",
    )
    parser.add_argument(
        "--allow-missing-model-artifacts",
        action="store_true",
        help="Skip models whose artifacts are missing instead of raising.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model weight files (used by adapters).",
    )
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Target date for prediction (YYYY-MM-DD). Used by cfg05 adapter.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def resolve_model_list(models_arg: str) -> list[dict[str, Any]]:
    """Resolve the --models argument to a list of model configs."""
    if models_arg == "default":
        return list(DEFAULT_FUSION_POOL)

    model_ids = [m.strip() for m in models_arg.split(",") if m.strip()]
    configs = []
    for mid in model_ids:
        if is_invalid_model(mid):
            raise ValueError(
                f"Model '{mid}' is INVALID and cannot be used. "
                f"Remove it from --models."
            )
        cfg = get_model_config(mid)
        configs.append(cfg)
    return configs


def _validate_input(df: pd.DataFrame) -> None:
    """Check that input has minimum required columns."""
    required = ["ds"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")


def _synthetic_prediction(
    df: pd.DataFrame,
    model_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Generate synthetic predictions for dry-run mode.

    Uses ``da_anchor`` or ``y_pred`` from input if available, otherwise
    generates random values.  All rows are tagged with reason DRY_RUN.
    """
    model_id = model_cfg["model_id"]
    formal_name = model_cfg.get("formal_name", model_id)
    n = len(df)

    # Seed for reproducibility per model
    rng = np.random.default_rng(abs(hash(model_id)) % (2**31))

    if "y_pred" in df.columns:
        base = df["y_pred"].values.astype(float)
        noise = rng.normal(0, base.std() * 0.02 if base.std() > 0 else 0.5, n)
        y_pred = base + noise
    elif "da_anchor" in df.columns:
        base = df["da_anchor"].values.astype(float)
        noise = rng.normal(0, base.std() * 0.02 if base.std() > 0 else 0.5, n)
        y_pred = base + noise
    else:
        y_pred = rng.uniform(80, 200, n)

    # Build output
    out = pd.DataFrame({
        "task": "dayahead",
        "model_name": formal_name,
        "model_version": "dry_run",
        "ds": pd.to_datetime(df["ds"]),
        "y_pred": y_pred,
        "source_confidence": np.full(n, 0.0),
    })

    # Add business-time columns
    from data.business_day import add_business_time_columns
    out = add_business_time_columns(out, timestamp_col="ds")

    out["target_day"] = out["ds"].dt.date.astype(str)
    out["period"] = out["hour_business"].apply(
        lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
    )

    return out[PREDICTION_OUTPUT_COLUMNS]


def run_dayahead_zoo(
    *,
    model_configs: list[dict[str, Any]],
    input_df: pd.DataFrame,
    dry_run: bool = False,
    model_dir: Optional[str] = None,
    target_date: Optional[str] = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Execute day-ahead model zoo predictions.

    Parameters
    ----------
    model_configs : list[dict]
        List of model config dicts from the registry.
    input_df : pd.DataFrame
        Input data with at minimum a ``ds`` column.
    dry_run : bool
        If True, use synthetic predictions.
    model_dir : str, optional
        Path to model weight directory.
    target_date : str, optional
        Target date forwarded to adapter predict().
    allow_missing : bool
        If True, skip models that raise (e.g. missing weights).

    Returns
    -------
    pd.DataFrame
        Concatenated predictions in standard schema.
    """
    _validate_input(input_df)
    df = input_df.copy()

    all_preds: list[pd.DataFrame] = []

    for cfg in model_configs:
        model_id = cfg["model_id"]
        logger.info(f"Running model: {model_id}")

        if dry_run:
            pred = _synthetic_prediction(df, cfg)
            pred["model_version"] = "dry_run"
            all_preds.append(pred)
            continue

        # Real adapter execution
        try:
            from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter
            if model_id == "cfg05":
                adapter = CFG05DayaheadAdapter()
                adapter.load()
                pred = adapter.predict(
                    df=df,
                    target_date=target_date,
                    model_dir=model_dir,
                )
            else:
                if allow_missing:
                    logger.warning(
                        f"Model '{model_id}' adapter not wired for real execution. "
                        "Use --allow-missing-model-artifacts to skip."
                    )
                    continue
                raise NotImplementedError(
                    f"Real execution for model '{model_id}' not yet wired. "
                    "Use --dry-run or implement the adapter call."
                )
            all_preds.append(pred)
        except (FileNotFoundError, NotImplementedError) as e:
            if allow_missing:
                logger.warning(f"Skipping model '{model_id}': {e}")
                continue
            raise

    if not all_preds:
        logger.warning("No predictions generated.")
        return pd.DataFrame(columns=PREDICTION_OUTPUT_COLUMNS)

    result = pd.concat(all_preds, ignore_index=True)
    result = result.sort_values(
        ["business_day", "hour_business", "model_name"]
    ).reset_index(drop=True)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve model list
    model_configs = resolve_model_list(args.models)
    logger.info(
        f"Models to run ({len(model_configs)}): "
        f"{[c['model_id'] for c in model_configs]}"
    )

    # Load input
    if args.dry_run:
        # Create synthetic input for dry-run
        if args.input:
            df = pd.read_csv(args.input)
        else:
            # Create 24-hour synthetic input
            from datetime import datetime, timedelta
            today = datetime.now().strftime("%Y-%m-%d")
            timestamps = pd.date_range(f"{today} 01:00", periods=24, freq="h")
            df = pd.DataFrame({"ds": timestamps})
            df["da_anchor"] = np.random.default_rng(42).uniform(80, 200, 24)
            logger.info("No --input provided; using synthetic 24h data for dry-run")
    else:
        if not args.input:
            logger.error("--input is required for non-dry-run mode")
            return 1
        df = pd.read_csv(args.input)

    # Run
    result = run_dayahead_zoo(
        model_configs=model_configs,
        input_df=df,
        dry_run=args.dry_run,
        model_dir=args.model_dir,
        target_date=args.target_date,
        allow_missing=args.allow_missing_model_artifacts,
    )

    if len(result) == 0:
        logger.warning("No predictions produced. Check model configurations.")
        return 0

    logger.info(
        f"Generated {len(result)} prediction rows "
        f"for {result['model_name'].nunique()} models."
    )

    # Write output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        logger.info(f"Predictions written to {out_path}")
    else:
        # Print summary
        print(f"\n=== Day-ahead Zoo Prediction Summary ===")
        print(f"Total rows: {len(result)}")
        print(f"Models: {result['model_name'].unique().tolist()}")
        print(f"Date range: {result['ds'].min()} ~ {result['ds'].max()}")
        print(f"Columns: {list(result.columns)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
