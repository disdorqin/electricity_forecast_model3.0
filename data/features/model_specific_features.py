"""
data/features/model_specific_features.py — P134: Per-Model Feature Builders.

Provides ``build_features_for_model()`` which builds a feature matrix that
exactly matches the schema a given model was trained on.

Key rules:
  - cfg05 and catboost_spike_residual: full 56 v3 features (same schema)
  - catboost_sota: 24 base features (subset of v3)
  - Output column order matches training artifact
  - Missing columns filled with NaN + reason_codes
  - Extra columns removed
  - No y_true leakage
  - Returns (feature_matrix, feature_report, schema_match_score)

Usage:
    from data.features.model_specific_features import build_features_for_model

    X, report, score = build_features_for_model(raw_day_df, "cfg05")
    X, report, score = build_features_for_model(raw_day_df, "catboost_sota")
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Deny list (target leakage) ──────────────────────────────────────
DENY_LIST: set[str] = {"y_true", "residual", "error", "abs_error"}

# ── Business columns to preserve ────────────────────────────────────
BUSINESS_COLUMNS: list[str] = ["business_day", "ds", "hour_business", "period"]

# ── Chinese column name mapping for raw Shandong PMOS data ──────────
# Maps Chinese column names to the English names expected by the v3 feature builder.
CN_TO_EN_RAW: dict[str, str] = {
    "时刻": "ds",
    "日前电价": "y",
    "实时电价": "rt_actual",
    "风电总加实际值": "wind",
    "光伏总加实际值": "solar",
    "联络线受电负荷实际值": "interconnect",
    "直调负荷实际值": "load",
    "竞价空间实际值": "bidding_space",
    "新能源总加实际值": "renewable_total",
    "直调负荷预测值": "load_forecast",
    "风电总加预测值": "wind_forecast",
    "光伏总加预测值": "solar_forecast",
    "联络线受电负荷预测值": "interconnect_forecast",
    "竞价空间预测值": "bidding_space_forecast",
}


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Chinese column names to English names expected by the feature builder.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with Chinese column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    rename_map = {}
    for col in df.columns:
        if col in CN_TO_EN_RAW:
            rename_map[col] = CN_TO_EN_RAW[col]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ── Model → feature schema mapping ──────────────────────────────────
# These are the ACTUAL feature names extracted from the model artifacts.
# catboost_spike_residual: 56 features = identical to cfg05 v3
# catboost_sota: 24 features = base subset
# cfg05: 56 features = full v3

# Import the canonical 56-feature list from the adapter
from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS as _CFG05_56

# The 24 base features for catboost_sota
CATBOOST_SOTA_24_FEATURES: list[str] = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
    "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
    "morning_trend", "is_info_fresh",
]

# Map model names to their expected feature schemas
MODEL_FEATURE_SCHEMAS: dict[str, list[str]] = {
    "cfg05": list(_CFG05_56),
    "lightgbm_cfg05_dayahead": list(_CFG05_56),
    "catboost_spike_residual": list(_CFG05_56),  # Same 56 features as cfg05
    "catboost_sota": list(CATBOOST_SOTA_24_FEATURES),
}


def _build_v3_features_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build all 56 v3 features from raw data using the source repo's feature builder.

    This calls the source repo's feature_builder_dayahead_v3.py which computes
    all features from the raw CSV columns (ds, y, load, wind, solar, interconnect, etc.).

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw hourly data with columns: ds, y, load, wind, solar, interconnect, etc.
        May have Chinese column names (时刻, 日前电价, etc.) which will be auto-mapped.
        Must have sufficient history for rolling features (at least 14 days).

    Returns
    -------
    pd.DataFrame
        DataFrame with all v3 feature columns computed.
    """
    # Normalize Chinese column names to English
    df = normalize_raw_columns(raw_df.copy())

    source_repo = os.path.join(
        REPO_ROOT, ".local_artifacts", "source_repos", "epf-sota-experiment"
    )
    src_common = os.path.join(source_repo, "src", "common")

    if src_common not in sys.path:
        sys.path.insert(0, src_common)

    # Also need the parent for relative imports
    src_dir = os.path.join(source_repo, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from feature_builder_dayahead_v3 import build_features_dayahead_v3
        result = build_features_dayahead_v3(df)
        return result
    except ImportError:
        logger.warning("Could not import feature_builder_dayahead_v3, trying alternative path")

    # Fallback: try importing via the package path
    try:
        # Add the source repo root for package imports
        if source_repo not in sys.path:
            sys.path.insert(0, source_repo)
        from src.common.feature_builder_dayahead_v3 import build_features_dayahead_v3
        result = build_features_dayahead_v3(df)
        return result
    except Exception as e:
        logger.error(f"Failed to build v3 features: {e}")
        raise


def _ensure_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure business columns exist on the DataFrame."""
    df = df.copy()
    if "ds" not in df.columns:
        raise ValueError("Input DataFrame must have a 'ds' column")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if "hour_business" not in df.columns or "business_day" not in df.columns:
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")

    return df


def _strip_deny_list(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove any denied columns from the DataFrame. Returns (cleaned_df, removed_cols)."""
    removed = []
    for col in list(df.columns):
        col_lower = col.lower()
        for deny_term in DENY_LIST:
            if deny_term in col_lower:
                removed.append(col)
                break
    if removed:
        df = df.drop(columns=removed)
    return df, removed


def build_features_for_model(
    raw_day_df: pd.DataFrame,
    model_name: str,
    schema_manifest: Optional[dict[str, list[str]]] = None,
    precomputed_features: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict[str, Any], float]:
    """Build a feature matrix exactly matching a model's training schema.

    Parameters
    ----------
    raw_day_df : pd.DataFrame
        Raw hourly data for the target day(s). Must have 'ds' column.
        For proper feature computation, should include history (14+ days).
    model_name : str
        Model identifier (e.g., "cfg05", "catboost_spike_residual", "catboost_sota").
    schema_manifest : dict, optional
        Override schema mapping. Keys are model names, values are feature lists.
        If None, uses built-in MODEL_FEATURE_SCHEMAS.
    precomputed_features : pd.DataFrame, optional
        Pre-computed feature DataFrame (from v3 builder). If provided, skips
        the expensive feature computation step.

    Returns
    -------
    feature_matrix : pd.DataFrame
        Feature matrix with columns in exact training order.
        Business columns (business_day, ds, hour_business, period) are preserved.
    feature_report : dict
        Report with keys: model_name, expected_features, present_features,
        missing_features, extra_removed, reason_codes, schema_match_score.
    schema_match_score : float
        1.0 = exact match, <1.0 = degraded (some features missing/filled),
        0.0 = cannot infer (no features available).
    """
    # Determine expected schema
    schemas = schema_manifest or MODEL_FEATURE_SCHEMAS
    expected_features = schemas.get(model_name)

    if expected_features is None:
        # Try to find by partial match
        for key, feats in schemas.items():
            if model_name in key or key in model_name:
                expected_features = feats
                break

    if expected_features is None:
        report = {
            "model_name": model_name,
            "expected_features": [],
            "present_features": [],
            "missing_features": [],
            "extra_removed": [],
            "reason_codes": [f"UNKNOWN_MODEL_SCHEMA:{model_name}"],
            "schema_match_score": 0.0,
        }
        return pd.DataFrame(), report, 0.0

    reason_codes: list[str] = []

    # Step 1: Ensure business columns
    df = _ensure_business_columns(raw_day_df)

    # Step 2: Compute features if not precomputed
    if precomputed_features is not None:
        enriched = precomputed_features.copy()
    else:
        try:
            enriched = _build_v3_features_from_raw(df)
        except Exception as e:
            reason_codes.append(f"FEATURE_COMPUTE_FAILED:{e}")
            # Fall back to whatever columns are already in the raw df
            enriched = df.copy()

    # Step 3: Strip deny-list columns
    enriched, removed_deny = _strip_deny_list(enriched)
    if removed_deny:
        reason_codes.append(f"DENY_LIST_REMOVED:{removed_deny}")

    # Step 4: Select and order features
    present = [c for c in expected_features if c in enriched.columns]
    missing = [c for c in expected_features if c not in enriched.columns]
    extra = [c for c in enriched.columns if c not in expected_features and c not in BUSINESS_COLUMNS]

    # Build feature matrix with exact column order
    feature_data = {}
    for col in expected_features:
        if col in enriched.columns:
            feature_data[col] = enriched[col].values
        else:
            feature_data[col] = np.nan
            reason_codes.append(f"MISSING_FEATURE_FILLED_NAN:{col}")

    feature_matrix = pd.DataFrame(feature_data)

    # Add business columns
    for bc in BUSINESS_COLUMNS:
        if bc in enriched.columns:
            feature_matrix[bc] = enriched[bc].values[:len(feature_matrix)]

    if missing:
        reason_codes.append(f"MISSING_FEATURES_COUNT:{len(missing)}")

    # Step 5: Compute schema match score
    n_expected = len(expected_features)
    n_present = len(present)
    if n_expected == 0:
        schema_match_score = 0.0
    else:
        schema_match_score = n_present / n_expected

    if schema_match_score == 1.0:
        reason_codes.append("SCHEMA_EXACT_MATCH")
    elif schema_match_score > 0:
        reason_codes.append(f"SCHEMA_DEGRADED:{n_present}/{n_expected}")
    else:
        reason_codes.append("SCHEMA_NO_MATCH")

    feature_report = {
        "model_name": model_name,
        "expected_features": expected_features,
        "expected_count": n_expected,
        "present_features": present,
        "present_count": n_present,
        "missing_features": missing,
        "missing_count": len(missing),
        "extra_removed": extra[:10],  # Truncate for readability
        "extra_removed_count": len(extra),
        "reason_codes": reason_codes,
        "schema_match_score": round(schema_match_score, 4),
    }

    return feature_matrix, feature_report, schema_match_score


def get_model_schema(model_name: str) -> list[str]:
    """Get the expected feature schema for a model.

    Parameters
    ----------
    model_name : str
        Model identifier.

    Returns
    -------
    list[str]
        Feature column names in training order.

    Raises
    ------
    KeyError
        If model_name is not recognized.
    """
    schema = MODEL_FEATURE_SCHEMAS.get(model_name)
    if schema is None:
        for key, feats in MODEL_FEATURE_SCHEMAS.items():
            if model_name in key or key in model_name:
                return list(feats)
        raise KeyError(f"Unknown model schema: {model_name}")
    return list(schema)


def list_supported_models() -> list[str]:
    """Return list of model names with known feature schemas."""
    return list(MODEL_FEATURE_SCHEMAS.keys())
