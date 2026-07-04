"""
data/schema.py — Unified schema definitions for 3.0.

Defines the canonical column names, ledger schemas, and output contracts
that every pipeline stage and adapter must conform to.

Join key across all tables:
    (task, business_day, hour_business)

Canonical timestamp rule:
    timestamp D 00:00:00 → business_day D-1, hour_business 24
    timestamp D 01:00~23:00 → business_day D, hour_business 1~23
"""

from __future__ import annotations

from typing import Final

# ──────────────────────────────────────────────
# Standard production prediction output columns
# Every prediction adapter must produce these.
# y_true is EVAL-ONLY — production adapters must not depend on it.
# ──────────────────────────────────────────────

PREDICTION_OUTPUT_COLUMNS: Final[list[str]] = [
    "task",                # "dayahead" | "realtime"
    "model_name",          # e.g. "cfg05_dayahead_lgbm"
    "target_day",          # the calendar day being predicted (YYYY-MM-DD)
    "business_day",        # business day (YYYY-MM-DD), hour 24 rule applies
    "ds",                  # full timestamp (YYYY-MM-DD HH:MM:SS)
    "hour_business",       # 1..24
    "period",              # "1_8" | "9_16" | "17_24"
    "y_pred",              # predicted price (float)
    "source_confidence",   # confidence score [0, 1] or NaN
    "model_version",       # version string
]

# Columns allowed ONLY in eval mode (never in production predict path)
EVAL_ONLY_COLUMNS: Final[list[str]] = [
    "y_true",  # actual price — must NOT be available at prediction time
]

# Full output columns for eval/analysis
FULL_OUTPUT_COLUMNS: Final[list[str]] = PREDICTION_OUTPUT_COLUMNS + EVAL_ONLY_COLUMNS

# ──────────────────────────────────────────────
# Prediction ledger schema
# Accumulates model predictions day by day.
# ──────────────────────────────────────────────

PREDICTION_LEDGER_COLUMNS: Final[list[str]] = [
    "task",
    "model_name",
    "forecast_date",       # date the prediction was made (YYYY-MM-DD)
    "target_day",
    "business_day",
    "ds",
    "hour_business",
    "period",
    "y_pred",
    "source_confidence",
    "model_version",
    "created_at",
]

PREDICTION_UNIQUE_KEY: Final[list[str]] = [
    "task", "model_name", "forecast_date", "target_day", "business_day", "hour_business",
]

# ──────────────────────────────────────────────
# Actual ledger schema
# Accumulates real price actuals day by day.
# ──────────────────────────────────────────────

ACTUAL_LEDGER_COLUMNS: Final[list[str]] = [
    "task",
    "target_day",
    "business_day",
    "ds",
    "hour_business",
    "period",
    "y_true",
    "actual_source",
    "created_at",
]

ACTUAL_UNIQUE_KEY: Final[list[str]] = [
    "task", "target_day", "business_day", "hour_business",
]

# ──────────────────────────────────────────────
# Training table schema (used by weight learner)
# Merged prediction + actual for weight learning.
# ──────────────────────────────────────────────

TRAINING_TABLE_COLUMNS: Final[list[str]] = [
    "task",
    "model_name",
    "target_day",
    "business_day",
    "ds",
    "hour_business",
    "period",
    "y_pred",
    "y_true",
    "source_confidence",
    "model_version",
]

# ──────────────────────────────────────────────
# Assist / risk ledger schema
# Accumulates per-hour assist signals.
# ──────────────────────────────────────────────

ASSIST_LEDGER_COLUMNS: Final[list[str]] = [
    "task",
    "business_day",
    "hour_business",
    "ds",
    "da_error_prob",
    "residual_direction_prob",
    "uncertainty_score",
    "correction_permission",
    "spike_prob",
    "negative_prob",
    "regime_cluster_id",
    "risk_source",
    "reason_codes",
    "model_version",
]

# ──────────────────────────────────────────────
# Validators
# ──────────────────────────────────────────────

VALID_TASKS: Final[list[str]] = ["dayahead", "realtime"]
VALID_PERIODS: Final[list[str]] = ["1_8", "9_16", "17_24"]


# ──────────────────────────────────────────────
# Corrected prediction schema (residual correction output)
# P3: after residual correction, the prediction output is augmented
# with pre/post correction fields and correction metadata.
# ──────────────────────────────────────────────

CORRECTED_PREDICTION_COLUMNS: Final[list[str]] = [
    "task",                 # "dayahead" | "realtime"
    "model_name",           # original model name
    "target_day",           # calendar day being predicted
    "business_day",         # business day
    "ds",                   # full timestamp
    "hour_business",        # 1..24
    "period",               # "1_8" | "9_16" | "17_24"
    "y_pred_raw",           # prediction value before correction
    "y_pred_corrected",     # prediction value after correction
    "residual_delta",       # y_pred_corrected - y_pred_raw
    "correction_applied",   # boolean — was a real correction applied?
    "correction_module",    # module identifier (e.g. "p5m_residual_noop")
    "risk_source",          # risk data source (e.g. "DATA_MISSING", "NONE", "NEGATIVE_RISK")
    "reason_codes",         # semicolon-delimited codes describing transformations
    "correction_version",   # version of the correction module
    "source_confidence",    # confidence score or NaN
    "model_version",        # original model version
]

CORRECTED_UNIQUE_KEY: Final[list[str]] = [
    "task", "model_name", "target_day", "business_day", "hour_business",
]

# Merge key used for joining risk data onto predictions for residual correction.
# The full 6-column key is preferred.  Fewer columns are accepted when the
# risk DataFrame does not carry the full set (degraded merge).
CORRECTED_MERGE_KEY: Final[list[str]] = [
    "task", "model_name", "target_day", "business_day", "ds", "hour_business",
]

CORRECTED_REQUIRED_KEYS: Final[list[str]] = [
    "task", "model_name", "target_day", "business_day", "ds",
    "hour_business", "period",
]


def validate_output_columns(df_columns: list[str]) -> list[str]:
    """Check which required production columns are missing from a DataFrame.

    Returns a list of missing column names. Empty list = fully compliant.
    """
    return [c for c in PREDICTION_OUTPUT_COLUMNS if c not in df_columns]


def validate_no_eval_columns(df_columns: list[str]) -> list[str]:
    """Check if any eval-only columns are present in a production DataFrame.

    Returns a list of leaked eval column names. Empty list = safe.
    """
    return [c for c in EVAL_ONLY_COLUMNS if c in df_columns]


def ensure_output_schema(df, *, production: bool = True):
    """Validate and reorder a DataFrame to match the output schema.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least PREDICTION_OUTPUT_COLUMNS.
    production : bool
        If True (default), raises if y_true is present.

    Returns
    -------
    pandas.DataFrame
        Reordered with canonical column order.
    """
    import pandas as pd

    missing = validate_output_columns(list(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if production:
        leaked = validate_no_eval_columns(list(df.columns))
        if leaked:
            raise ValueError(
                f"Production output must not contain eval-only columns: {leaked}. "
                f"Strip them before delivery."
            )

    cols = [c for c in PREDICTION_OUTPUT_COLUMNS if c in df.columns]
    if not production:
        cols += [c for c in EVAL_ONLY_COLUMNS if c in df.columns]

    return df[cols].copy()
