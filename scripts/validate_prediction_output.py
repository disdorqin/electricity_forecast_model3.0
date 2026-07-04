"""
scripts/validate_prediction_output.py — Prediction output schema validator.

Validates that a prediction CSV or DataFrame conforms to the
3.0 standard output schema.

Checks:
    1. Required columns exist.
    2. No eval-only columns (y_true).
    3. hour_business in 1..24.
    4. period in 1_8 / 9_16 / 17_24.
    5. No duplicate keys (task, model_name, business_day, hour_business).
    6. y_pred has no NaN values.
    7. Each (task, model_name, business_day) has exactly 24 rows (optional).
    8. No NaN in key columns.

Usage:
    python scripts/validate_prediction_output.py predictions.csv
    python scripts/validate_prediction_output.py --no-require-24h predictions.csv
    python scripts/validate_prediction_output.py predictions.csv --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from data.schema import (
    PREDICTION_OUTPUT_COLUMNS,
    EVAL_ONLY_COLUMNS,
    VALID_PERIODS,
)

EXIT_PASS = 0
EXIT_FAIL = 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate prediction output against the 3.0 standard schema.",
    )
    parser.add_argument(
        "input", type=str,
        help="Path to prediction CSV or parquet file.",
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
        "--require-24h", action="store_true", default=False,
        help="Check that each (task, model, business_day) has 24 rows.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed validation output.",
    )
    return parser.parse_args(argv)


def validate_prediction_file(
    path: str,
    *,
    production: bool = True,
    require_24h: bool = False,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a prediction file against the standard schema.

    Parameters
    ----------
    path : str
        Path to the prediction file (.csv).
    production : bool
        If True, y_true is forbidden. Default: True.
    require_24h : bool
        If True, validate each (task, model, business_day) has 24 rows.
    verbose : bool
        If True, print details.

    Returns
    -------
    tuple[bool, list[str]]
        (passed, list of error/warning messages).
    """
    p = Path(path)
    if not p.exists():
        return False, [f"File not found: {path}"]

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        return False, [f"Unsupported file format: {p.suffix}"]

    return validate_prediction_dataframe(
        df,
        production=production,
        require_24h=require_24h,
        verbose=verbose,
    )


def validate_prediction_dataframe(
    df: pd.DataFrame,
    *,
    production: bool = True,
    require_24h: bool = False,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a prediction DataFrame against the standard schema.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction output to validate.
    production : bool
        If True, y_true is forbidden.
    require_24h : bool
        If True, require 24 rows per (task, model, business_day).
    verbose : bool
        If True, print details.

    Returns
    -------
    tuple[bool, list[str]]
        (passed, list of error/warning messages).
    """
    errors: list[str] = []

    # ── 1. Check input is not empty ────────────
    if len(df) == 0:
        errors.append("DataFrame is empty")

    # ── 2. Check required columns exist ─────────
    missing_cols = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # ── 3. Check for eval-only columns ─────────
    if production:
        leaked = [c for c in EVAL_ONLY_COLUMNS if c in df.columns]
        if leaked:
            errors.append(
                f"Production output must NOT contain eval-only columns: {leaked}"
            )

    if not errors and verbose:
        print(f"Columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")

    # Skip further checks if basic schema is broken
    if errors:
        return False, errors

    # ── 4. Check hour_business range ────────────
    if "hour_business" in df.columns:
        hb = df["hour_business"]
        if hb.min() < 1 or hb.max() > 24:
            errors.append(
                f"hour_business out of range [1,24]: "
                f"min={hb.min()}, max={hb.max()}"
            )
        if verbose:
            print(f"hour_business range: {hb.min()} ~ {hb.max()}")

    # ── 5. Check period values ──────────────────
    if "period" in df.columns:
        invalid_periods = df[~df["period"].isin(VALID_PERIODS)]["period"].unique()
        if len(invalid_periods) > 0:
            errors.append(f"Invalid period values: {list(invalid_periods)}")

    # ── 6. Check y_pred no NaN ──────────────────
    if "y_pred" in df.columns:
        nan_count = df["y_pred"].isna().sum()
        if nan_count > 0:
            errors.append(f"y_pred has {nan_count} NaN values ({(nan_count / len(df)) * 100:.1f}%)")

    # ── 7. Check for NaN in key columns ─────────
    key_cols = ["task", "model_name", "business_day", "hour_business"]
    for col in key_cols:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                errors.append(f"Key column '{col}' has {n_null} null values")

    # ── 8. Check duplicate keys ─────────────────
    if all(c in df.columns for c in ["task", "model_name", "business_day", "hour_business"]):
        dup_mask = df.duplicated(
            subset=["task", "model_name", "business_day", "hour_business"],
            keep=False,
        )
        n_dup = dup_mask.sum()
        if n_dup > 0:
            dup_examples = df[dup_mask].head(10)
            errors.append(
                f"Found {n_dup} rows with duplicate keys "
                f"(task, model_name, business_day, hour_business). "
                f"Examples: {dup_examples[['task', 'model_name', 'business_day', 'hour_business']].values.tolist()}"
            )

    # ── 9. Check 24 rows per day per model ──────
    if require_24h and all(
        c in df.columns for c in ["task", "model_name", "business_day", "hour_business"]
    ):
        group_sizes = df.groupby(["task", "model_name", "business_day"]).size()
        bad_groups = group_sizes[group_sizes != 24]
        if len(bad_groups) > 0:
            for (task, model, bd), size in bad_groups.items():
                errors.append(
                    f"Expected 24 rows for ({task}, {model}, {bd}), got {size}"
                )

    # ── 10. Check task values ──────────────────
    if "task" in df.columns:
        invalid_tasks = df[~df["task"].isin(["dayahead", "realtime"])]["task"].unique()
        if len(invalid_tasks) > 0:
            errors.append(f"Invalid task values: {list(invalid_tasks)}")

    passed = len(errors) == 0
    return passed, errors


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    passed, errors = validate_prediction_file(
        args.input,
        production=args.production,
        require_24h=args.require_24h,
        verbose=args.verbose,
    )

    if passed:
        print(f"PASS: {args.input} — output schema is valid")
        return EXIT_PASS
    else:
        print(f"FAIL: {args.input} — {len(errors)} validation error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
