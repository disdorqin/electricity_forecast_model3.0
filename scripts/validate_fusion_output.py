"""
scripts/validate_fusion_output.py — Fusion output validator.

Checks:
    1. Required columns present.
    2. fused_price no NaN.
    3. hour_business in 1..24.
    4. period in 1_8 / 9_16 / 17_24.
    5. No duplicate fusion key.
    6. weights_json is valid JSON.
    7. Weights sum to 1 within tolerance (1e-4).
    8. included_models non-empty (unless allow_empty=True).
    9. Task in dayahead / realtime.
    10. Production mode: no y_true.

Usage:
    python scripts/validate_fusion_output.py fusion.csv
    python scripts/validate_fusion_output.py fusion.csv --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from data.schema import (
    FUSION_OUTPUT_COLUMNS,
    FUSION_UNIQUE_KEY,
    EVAL_ONLY_COLUMNS,
    VALID_PERIODS,
    VALID_TASKS,
)

EXIT_PASS = 0
EXIT_FAIL = 1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fusion output (P4 schema).",
    )
    parser.add_argument(
        "input", type=str,
        help="Path to fusion output CSV.",
    )
    parser.add_argument(
        "--allow-empty", action="store_true",
        help="Allow empty included_models (no models passed gate).",
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


def validate_fusion_dataframe(
    df: pd.DataFrame,
    *,
    allow_empty: bool = False,
    production: bool = True,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a fusion output DataFrame against P4 schema.

    Parameters
    ----------
    df : pd.DataFrame
        Fusion output DataFrame.
    allow_empty : bool
        If True, empty included_models is allowed.
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
    missing_cols = [c for c in FUSION_OUTPUT_COLUMNS if c not in df.columns]
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

    # ── 4. Check fused_price no NaN ─────────────
    if "fused_price" in df.columns:
        nan_count = df["fused_price"].isna().sum()
        if nan_count > 0:
            errors.append(
                f"fused_price has {nan_count} NaN values "
                f"({(nan_count / len(df)) * 100:.1f}%)"
            )

    # ── 5. Check hour_business range ────────────
    if "hour_business" in df.columns:
        hb = df["hour_business"]
        if hb.min() < 1 or hb.max() > 24:
            errors.append(
                f"hour_business out of range [1,24]: "
                f"min={hb.min()}, max={hb.max()}"
            )
        if verbose:
            print(f"hour_business range: {hb.min()} ~ {hb.max()}")

    # ── 6. Check period values ──────────────────
    if "period" in df.columns:
        invalid_periods = df[~df["period"].isin(VALID_PERIODS)]["period"].unique()
        if len(invalid_periods) > 0:
            errors.append(f"Invalid period values: {list(invalid_periods)}")

    # ── 7. Check weights_json is valid JSON ──────
    if "weights_json" in df.columns:
        for idx, raw in enumerate(df["weights_json"]):
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(parsed, dict):
                    errors.append(f"Row {idx}: weights_json is not a JSON object")
            except (json.JSONDecodeError, TypeError) as e:
                errors.append(f"Row {idx}: weights_json is not valid JSON: {e}")

    # ── 8. Check weights sum to 1 ────────────────
    if "weights_json" in df.columns:
        for idx, raw in enumerate(df["weights_json"]):
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, dict):
                    total = sum(float(v) for v in parsed.values())
                    if abs(total - 1.0) > 1e-4:
                        errors.append(
                            f"Row {idx}: weights sum to {total:.6f}, expected 1.0"
                        )
            except (json.JSONDecodeError, TypeError):
                pass  # already reported above

    # ── 9. Check included_models non-empty ───────
    if "included_models" in df.columns and not allow_empty:
        empty_mask = df["included_models"].isna() | (df["included_models"] == "")
        n_empty = empty_mask.sum()
        if n_empty > 0:
            errors.append(
                f"included_models is empty for {n_empty} rows "
                f"(use --allow-empty to allow)"
            )

    # ── 10. Check task values ────────────────────
    if "task" in df.columns:
        invalid_tasks = df[~df["task"].isin(VALID_TASKS)]["task"].unique()
        if len(invalid_tasks) > 0:
            errors.append(f"Invalid task values: {list(invalid_tasks)}")

    # ── 11. Check duplicate fusion keys ─────────
    if all(c in df.columns for c in FUSION_UNIQUE_KEY):
        dup_mask = df.duplicated(
            subset=FUSION_UNIQUE_KEY,
            keep=False,
        )
        n_dup = dup_mask.sum()
        if n_dup > 0:
            dup_examples = df[dup_mask].head(10)
            errors.append(
                f"Found {n_dup} rows with duplicate fusion keys "
                f"{FUSION_UNIQUE_KEY}. "
                f"Examples: {dup_examples[FUSION_UNIQUE_KEY].values.tolist()}"
            )

    # ── 12. Check NaN in key columns ────────────
    for col in FUSION_UNIQUE_KEY:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                errors.append(f"Key column '{col}' has {n_null} null values")

    # ── 13. Check readiness_mode values ─────────
    if "readiness_mode" in df.columns:
        invalid_modes = df[~df["readiness_mode"].isin(["REAL", "DRY_RUN"])]["readiness_mode"].unique()
        if len(invalid_modes) > 0:
            errors.append(f"Invalid readiness_mode values: {list(invalid_modes)}")

    passed = len(errors) == 0
    return passed, errors


def validate_fusion_file(
    path: str,
    *,
    allow_empty: bool = False,
    production: bool = True,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a fusion output file against P4 schema."""
    p = Path(path)
    if not p.exists():
        return False, [f"File not found: {path}"]

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        return False, [f"Unsupported file format: {p.suffix}"]

    return validate_fusion_dataframe(
        df,
        allow_empty=allow_empty,
        production=production,
        verbose=verbose,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    passed, errors = validate_fusion_file(
        args.input,
        allow_empty=args.allow_empty,
        production=args.production,
        verbose=args.verbose,
    )

    if passed:
        print(f"PASS: {args.input} — fusion output schema is valid")
        return EXIT_PASS
    else:
        print(f"FAIL: {args.input} — {len(errors)} validation error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
