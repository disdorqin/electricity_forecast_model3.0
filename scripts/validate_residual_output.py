"""
scripts/validate_residual_output.py — Corrected prediction output validator.

Validates that a corrected prediction DataFrame conforms to the
P3 corrected prediction schema.

Checks:
    1. Required columns exist.
    2. hour_business in 1..24.
    3. period in 1_8 / 9_16 / 17_24.
    4. No duplicate keys (task, model_name, target_day, business_day, hour_business).
    5. y_pred_raw has no NaN.
    6. y_pred_corrected has no NaN.
    7. residual_delta has no NaN.
    8. residual_delta == y_pred_corrected - y_pred_raw.
    9. correction_applied is boolean.
    10. production mode: no y_true.

Usage:
    python scripts/validate_residual_output.py corrected.csv
    python scripts/validate_residual_output.py corrected.csv --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    CORRECTED_UNIQUE_KEY,
    EVAL_ONLY_COLUMNS,
    VALID_PERIODS,
)

EXIT_PASS = 0
EXIT_FAIL = 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate corrected prediction output (P3 schema).",
    )
    parser.add_argument(
        "input", type=str,
        help="Path to corrected prediction CSV.",
    )
    parser.add_argument(
        "--production", action="store_true", default=True,
        help="Production mode: y_true forbidden (default).",
    )
    parser.add_argument(
        "--no-production",
        action="store_false",
        dest="production",
        help="Eval mode: allow y_true column.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed validation output.",
    )
    return parser.parse_args(argv)


def validate_residual_file(
    path: str,
    *,
    production: bool = True,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a corrected prediction file against the P3 schema.

    Parameters
    ----------
    path : str
        Path to corrected prediction CSV.
    production : bool
        If True, y_true is forbidden.
    verbose : bool
        If True, print details.

    Returns
    -------
    tuple[bool, list[str]]
        (passed, list of errors).
    """
    p = Path(path)
    if not p.exists():
        return False, [f"File not found: {path}"]

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        return False, [f"Unsupported file format: {p.suffix}"]

    return validate_residual_dataframe(df, production=production, verbose=verbose)


def validate_residual_dataframe(
    df: pd.DataFrame,
    *,
    production: bool = True,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a corrected prediction DataFrame against P3 schema.

    Parameters
    ----------
    df : pd.DataFrame
        Corrected prediction DataFrame.
    production : bool
        If True, y_true is forbidden.
    verbose : bool
        If True, print details.

    Returns
    -------
    tuple[bool, list[str]]
        (passed, list of errors).
    """
    errors: list[str] = []

    # ── 1. Check input is not empty ────────────
    if len(df) == 0:
        errors.append("DataFrame is empty")

    # ── 2. Check required columns exist ─────────
    missing_cols = [c for c in CORRECTED_PREDICTION_COLUMNS if c not in df.columns]
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

    # ── 6. Check y_pred_raw no NaN ──────────────
    if "y_pred_raw" in df.columns:
        nan_count = df["y_pred_raw"].isna().sum()
        if nan_count > 0:
            errors.append(
                f"y_pred_raw has {nan_count} NaN values "
                f"({(nan_count / len(df)) * 100:.1f}%)"
            )

    # ── 7. Check y_pred_corrected no NaN ────────
    if "y_pred_corrected" in df.columns:
        nan_count = df["y_pred_corrected"].isna().sum()
        if nan_count > 0:
            errors.append(
                f"y_pred_corrected has {nan_count} NaN values "
                f"({(nan_count / len(df)) * 100:.1f}%)"
            )

    # ── 8. Check residual_delta no NaN ──────────
    if "residual_delta" in df.columns:
        nan_count = df["residual_delta"].isna().sum()
        if nan_count > 0:
            errors.append(
                f"residual_delta has {nan_count} NaN values "
                f"({(nan_count / len(df)) * 100:.1f}%)"
            )

    # ── 9. Check residual_delta arithmetic ──────
    if all(c in df.columns for c in ["y_pred_corrected", "y_pred_raw", "residual_delta"]):
        # Compare with tolerance for float precision
        expected_delta = df["y_pred_corrected"].values.astype(float) - df["y_pred_raw"].values.astype(float)
        actual_delta = df["residual_delta"].values.astype(float)
        mismatches = ~(abs(expected_delta - actual_delta) < 1e-6)
        n_mismatch = mismatches.sum()
        if n_mismatch > 0:
            errors.append(
                f"residual_delta != y_pred_corrected - y_pred_raw for "
                f"{n_mismatch} rows ({(n_mismatch / len(df)) * 100:.1f}%)"
            )

    # ── 10. Check correction_applied is boolean ─
    if "correction_applied" in df.columns:
        try:
            bool_vals = df["correction_applied"].astype(bool)
            # Check that original values are actually boolean-like
            non_bool = df[~df["correction_applied"].isin([True, False, 1, 0, 1.0, 0.0])]
            if len(non_bool) > 0 and not df["correction_applied"].dtype in (bool,):
                # Check string representations
                unique_vals = df["correction_applied"].dropna().unique()
                valid_bool = {True, False, 1, 0, 1.0, 0.0, "True", "False", "true", "false"}
                invalid = [v for v in unique_vals if v not in valid_bool]
                if invalid:
                    errors.append(
                        f"correction_applied has non-boolean values: {invalid}"
                    )
        except Exception as e:
            errors.append(f"correction_applied type check failed: {e}")

    # ── 11. Check NaN in key columns ────────────
    for col in CORRECTED_UNIQUE_KEY:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                errors.append(f"Key column '{col}' has {n_null} null values")

    # ── 12. Check duplicate keys ────────────────
    if all(c in df.columns for c in CORRECTED_UNIQUE_KEY):
        dup_mask = df.duplicated(
            subset=CORRECTED_UNIQUE_KEY,
            keep=False,
        )
        n_dup = dup_mask.sum()
        if n_dup > 0:
            dup_examples = df[dup_mask].head(10)
            errors.append(
                f"Found {n_dup} rows with duplicate keys "
                f"{CORRECTED_UNIQUE_KEY}. "
                f"Examples: {dup_examples[CORRECTED_UNIQUE_KEY].values.tolist()}"
            )

    # ── 13. Check task values ──────────────────
    if "task" in df.columns:
        invalid_tasks = df[~df["task"].isin(["dayahead", "realtime"])]["task"].unique()
        if len(invalid_tasks) > 0:
            errors.append(f"Invalid task values: {list(invalid_tasks)}")

    passed = len(errors) == 0
    return passed, errors


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    passed, errors = validate_residual_file(
        args.input,
        production=args.production,
        verbose=args.verbose,
    )

    if passed:
        print(f"PASS: {args.input} — corrected output schema is valid")
        return EXIT_PASS
    else:
        print(f"FAIL: {args.input} — {len(errors)} validation error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
