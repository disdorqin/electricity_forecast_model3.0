"""
scripts/validate_final_output.py — Final output validator for P6.

Checks:

1. Required columns present (FINAL_OUTPUT_COLUMNS)
2. final_price no NaN
3. fused_price no NaN
4. negative_prob in [0, 1] or NaN (policy-dependent)
5. negative_flag is boolean-like
6. classifier_applied is boolean-like
7. hour_business in 1..24
8. period in 1_8 / 9_16 / 17_24
9. No duplicate final key (task, target_day, business_day, hour_business)
10. model_lineage_json can json.loads
11. Production mode: no y_true required
12. task in dayahead / realtime

Usage::

    python -m scripts.validate_final_output /path/to/final_output.csv
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FINAL_UNIQUE_KEY,
    VALID_TASKS,
    VALID_PERIODS,
)


def validate_final_dataframe(
    df: pd.DataFrame,
    allow_empty: bool = False,
    production: bool = True,
) -> tuple[bool, list[str]]:
    """Validate a final output DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Final output to validate.
    allow_empty : bool
        If True, empty DataFrame is considered valid.
    production : bool
        If True, y_true must not be present.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages).
    """
    errors: list[str] = []

    # ── 1. Required columns ─────────────────────────────────────────────
    missing = [c for c in FINAL_OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return False, errors

    if len(df) == 0:
        if allow_empty:
            return True, ["Empty DataFrame"]
        errors.append("Empty DataFrame")
        return False, errors

    # ── 2. final_price no NaN ───────────────────────────────────────────
    null_final = df["final_price"].isna().sum()
    if null_final > 0:
        errors.append(f"{null_final} rows with NaN final_price")

    # ── 3. fused_price no NaN ───────────────────────────────────────────
    null_fused = df["fused_price"].isna().sum()
    if null_fused > 0:
        errors.append(f"{null_fused} rows with NaN fused_price")

    # ── 4. negative_prob in [0, 1] or NaN ──────────────────────────────
    if "negative_prob" in df.columns:
        prob = df["negative_prob"]
        non_null = prob.dropna()
        if len(non_null) > 0:
            out_of_range = non_null[(non_null < 0) | (non_null > 1)]
            if len(out_of_range) > 0:
                errors.append(
                    f"{len(out_of_range)} rows with negative_prob outside [0, 1]"
                )

    # ── 5. negative_flag is boolean-like ────────────────────────────────
    if "negative_flag" in df.columns:
        unique_vals = set(df["negative_flag"].dropna().unique())
        allowed = {True, False, 1, 0, 1.0, 0.0}
        if not unique_vals.issubset(allowed):
            bad = unique_vals - allowed
            errors.append(f"negative_flag has non-boolean values: {bad}")

    # ── 6. classifier_applied is boolean-like ───────────────────────────
    if "classifier_applied" in df.columns:
        unique_vals = set(df["classifier_applied"].dropna().unique())
        allowed = {True, False, 1, 0, 1.0, 0.0}
        if not unique_vals.issubset(allowed):
            bad = unique_vals - allowed
            errors.append(f"classifier_applied has non-boolean values: {bad}")

    # ── 7. hour_business in 1..24 ──────────────────────────────────────
    if "hour_business" in df.columns:
        invalid_hours = df[~df["hour_business"].between(1, 24)].index.tolist()
        if invalid_hours:
            errors.append(f"{len(invalid_hours)} rows with hour_business outside 1..24")

    # ── 8. period validity ─────────────────────────────────────────────
    if "period" in df.columns:
        invalid_periods = df[~df["period"].isin(VALID_PERIODS)].index.tolist()
        if invalid_periods:
            errors.append(f"{len(invalid_periods)} rows with invalid period")

    # ── 9. No duplicate final key ──────────────────────────────────────
    dups = df.duplicated(subset=FINAL_UNIQUE_KEY, keep=False)
    if dups.any():
        n_dup = dups.sum()
        errors.append(f"{n_dup} duplicate rows on final key {FINAL_UNIQUE_KEY}")

    # ── 10. model_lineage_json is valid JSON ────────────────────────────
    if "model_lineage_json" in df.columns:
        for idx, raw in df["model_lineage_json"].items():
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    errors.append(
                        f"Row {idx}: model_lineage_json not a dict"
                    )
            except (json.JSONDecodeError, TypeError) as e:
                errors.append(
                    f"Row {idx}: model_lineage_json invalid JSON: {e}"
                )

    # ── 11. Production mode: no y_true ──────────────────────────────────
    if production and "y_true" in df.columns:
        errors.append("Production output must not contain y_true")

    # ── 12. task validity ──────────────────────────────────────────────
    if "task" in df.columns:
        invalid_tasks = df[~df["task"].isin(VALID_TASKS)].index.tolist()
        if invalid_tasks:
            errors.append(f"{len(invalid_tasks)} rows with invalid task")

    return len(errors) == 0, errors


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.validate_final_output <file.csv>",
              file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    valid, errors = validate_final_dataframe(df, production=True)

    if valid:
        print(f"VALID: {len(df)} rows, all checks passed.")
        sys.exit(0)
    else:
        print(f"INVALID: {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    _main()
