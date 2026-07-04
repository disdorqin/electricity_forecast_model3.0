"""
pipelines/classifier_pipeline.py — Negative classifier pipeline.

Orchestrates the negative-price classification step:

1. Validates fusion input
2. Calls ``NegativeClassifierAdapter``
3. Returns final output conforming to ``FINAL_OUTPUT_COLUMNS``

Usage::

    from pipelines.classifier_pipeline import run_negative_classifier

    final = run_negative_classifier(
        fusion_df,
        model_dir="/path/to/models",
        rule_fallback=True,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FINAL_UNIQUE_KEY,
    FUSION_OUTPUT_COLUMNS,
    VALID_TASKS,
    VALID_PERIODS,
)
from extreme.negative_classifier import NegativeClassifierAdapter

logger = logging.getLogger(__name__)


def validate_fusion_input(df: pd.DataFrame) -> list[str]:
    """Validate fusion input for classifier consumption.

    Returns list of error messages (empty = valid).
    """
    issues: list[str] = []

    if len(df) == 0:
        issues.append("Empty fusion input DataFrame")
        return issues

    # Check required columns
    required = ["task", "target_day", "business_day", "ds",
                 "hour_business", "period", "fused_price"]
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    if issues:
        return issues

    # Check fused_price NaN
    null_price = df["fused_price"].isna().sum()
    if null_price > 0:
        issues.append(f"{null_price} rows with NaN fused_price")

    # Check hour_business range
    invalid_hours = df[~df["hour_business"].between(1, 24)].index.tolist()
    if invalid_hours:
        issues.append(f"{len(invalid_hours)} rows with hour_business outside 1..24")

    # Check period validity
    invalid_periods = df[~df["period"].isin(VALID_PERIODS)].index.tolist()
    if invalid_periods:
        issues.append(f"{len(invalid_periods)} rows with invalid period")

    # Check task validity
    invalid_tasks = df[~df["task"].isin(VALID_TASKS)].index.tolist()
    if invalid_tasks:
        issues.append(f"{len(invalid_tasks)} rows with invalid task")

    return issues


def run_negative_classifier(
    fusion_df: pd.DataFrame,
    model_dir: Optional[str] = None,
    rule_fallback: bool = True,
    production: bool = True,
) -> pd.DataFrame:
    """Run negative-price classifier on fusion output.

    Parameters
    ----------
    fusion_df : pd.DataFrame
        Fusion output (must contain ``FUSION_OUTPUT_COLUMNS`` fields).
    model_dir : str, optional
        Directory containing classifier artifacts.  If None, no-op.
    rule_fallback : bool
        Apply rule-based fallback for fused_price < 0 (default True).
    production : bool
        Production mode flag (default True).

    Returns
    -------
    pd.DataFrame
        Final output in ``FINAL_OUTPUT_COLUMNS`` schema.

    Raises
    ------
    ValueError
        If fusion input fails validation.
    """
    issues = validate_fusion_input(fusion_df)
    if issues:
        raise ValueError(
            f"Fusion input validation failed: {'; '.join(issues)}"
        )

    adapter = NegativeClassifierAdapter(
        rule_fallback=rule_fallback,
        production=production,
    )
    adapter.load(model_dir=model_dir)
    result = adapter.predict(fusion_df, rule_fallback=rule_fallback)

    # Final: sort, reorder, check no duplicate keys
    result = result.sort_values(
        ["business_day", "hour_business"]
    ).reset_index(drop=True)

    result = result[FINAL_OUTPUT_COLUMNS]

    # Verify no duplicate final keys
    dups = result.duplicated(subset=FINAL_UNIQUE_KEY, keep=False)
    if dups.any():
        n = dups.sum()
        logger.warning("Final output has %d duplicate rows on final key", n)

    n_flagged = result["negative_flag"].sum() if "negative_flag" in result.columns else 0
    logger.info(
        "Negative classifier complete: rows=%d, flagged=%d, "
        "classifier_applied=%s",
        len(result),
        n_flagged,
        result["classifier_applied"].iloc[0] if len(result) > 0 else "N/A",
    )

    return result


def _build_synthetic_fusion(
    n_hours: int = 24,
    include_negative: bool = False,
) -> pd.DataFrame:
    """Build a synthetic fusion DataFrame for dry-run / testing.

    Parameters
    ----------
    n_hours : int
        Number of hourly rows (default 24).
    include_negative : bool
        If True, inject some negative fused_price values.

    Returns
    -------
    pd.DataFrame
        Synthetic fusion output in ``FUSION_OUTPUT_COLUMNS`` schema.
    """
    import json
    import numpy as np
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    base_price = -20.0 if include_negative else 120.0
    prices = base_price + rng.uniform(-10, 30, n_hours)

    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    rows: list[dict] = []
    for i in range(n_hours):
        w1 = round(rng.uniform(0.3, 0.7), 4)
        w2 = round(1.0 - w1, 4)
        rows.append({
            "task": "dayahead",
            "target_day": "2026-07-04",
            "ds": timestamps[i],
            "fused_price": float(prices[i]),
            "weights_json": json.dumps({"cfg05": w1, "best_two_average": w2}),
            "included_models": "cfg05;best_two_average",
            "excluded_models": "",
            "fusion_method": "equal_weight",
            "learner_version": "0.1.0-skeleton",
            "readiness_mode": "DRY_RUN",
            "reason_codes": "FUSION_EQUAL_WEIGHT",
        })

    df = pd.DataFrame(rows)
    df = add_business_time_columns(df, timestamp_col="ds")

    # Ensure all FUSION_OUTPUT_COLUMNS are present
    for c in FUSION_OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = None

    return df[FUSION_OUTPUT_COLUMNS]
