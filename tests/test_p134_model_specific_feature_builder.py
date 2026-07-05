"""
tests/test_p134_model_specific_feature_builder.py — Tests for P134 Per-Model Feature Builders.
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.features.model_specific_features import (
    BUSINESS_COLUMNS,
    CATBOOST_SOTA_24_FEATURES,
    DENY_LIST,
    MODEL_FEATURE_SCHEMAS,
    _ensure_business_columns,
    _strip_deny_list,
    build_features_for_model,
    get_model_schema,
    list_supported_models,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_raw_df():
    """Create a sample raw DataFrame with enough history for rolling features."""
    np.random.seed(42)
    n_hours = 24 * 35  # 35 days of hourly data
    dates = pd.date_range("2024-12-01", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "ds": dates,
        "y": np.random.uniform(100, 500, n_hours),
        "load": np.random.uniform(20000, 40000, n_hours),
        "wind": np.random.uniform(1000, 8000, n_hours),
        "solar": np.random.uniform(0, 3000, n_hours),
        "interconnect": np.random.uniform(500, 2000, n_hours),
    })
    return df


@pytest.fixture
def sample_with_deny_cols():
    """DataFrame that includes deny-list columns."""
    np.random.seed(42)
    n = 48
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "ds": dates,
        "y": np.random.uniform(100, 500, n),
        "load": np.random.uniform(20000, 40000, n),
        "wind": np.random.uniform(1000, 8000, n),
        "solar": np.random.uniform(0, 3000, n),
        "interconnect": np.random.uniform(500, 2000, n),
        "y_true": np.random.uniform(100, 500, n),
        "residual": np.random.uniform(-50, 50, n),
    })
    return df


# ── Tests: Schema definitions ─────────────────────────────────────


class TestModelSchemas:
    """Tests for schema constants and lookups."""

    def test_cfg05_has_56_features(self):
        schema = MODEL_FEATURE_SCHEMAS["cfg05"]
        assert len(schema) == 56

    def test_catboost_spike_same_as_cfg05(self):
        """catboost_spike_residual should have identical schema to cfg05."""
        assert MODEL_FEATURE_SCHEMAS["catboost_spike_residual"] == MODEL_FEATURE_SCHEMAS["cfg05"]

    def test_catboost_sota_has_24_features(self):
        schema = MODEL_FEATURE_SCHEMAS["catboost_sota"]
        assert len(schema) == 24

    def test_catboost_sota_is_subset_of_cfg05(self):
        """catboost_sota features should be a subset of cfg05 features."""
        sota_set = set(MODEL_FEATURE_SCHEMAS["catboost_sota"])
        cfg05_set = set(MODEL_FEATURE_SCHEMAS["cfg05"])
        assert sota_set.issubset(cfg05_set)

    def test_get_model_schema_known(self):
        schema = get_model_schema("cfg05")
        assert len(schema) == 56

    def test_get_model_schema_unknown_raises(self):
        with pytest.raises(KeyError):
            get_model_schema("nonexistent_model_xyz")

    def test_list_supported_models(self):
        models = list_supported_models()
        assert "cfg05" in models
        assert "catboost_spike_residual" in models
        assert "catboost_sota" in models


# ── Tests: Business columns ───────────────────────────────────────


class TestEnsureBusinessColumns:
    """Tests for _ensure_business_columns()."""

    def test_adds_business_columns(self, sample_raw_df):
        result = _ensure_business_columns(sample_raw_df)
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert "period" in result.columns

    def test_hour_business_range(self, sample_raw_df):
        result = _ensure_business_columns(sample_raw_df)
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_missing_ds_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="ds"):
            _ensure_business_columns(df)


# ── Tests: Deny list enforcement ──────────────────────────────────


class TestStripDenyList:
    """Tests for _strip_deny_list()."""

    def test_removes_y_true(self, sample_with_deny_cols):
        cleaned, removed = _strip_deny_list(sample_with_deny_cols)
        assert "y_true" in removed
        assert "y_true" not in cleaned.columns

    def test_removes_residual(self, sample_with_deny_cols):
        cleaned, removed = _strip_deny_list(sample_with_deny_cols)
        assert "residual" in removed
        assert "residual" not in cleaned.columns

    def test_no_deny_cols_passes_through(self, sample_raw_df):
        cleaned, removed = _strip_deny_list(sample_raw_df)
        assert len(removed) == 0
        assert list(cleaned.columns) == list(sample_raw_df.columns)


# ── Tests: build_features_for_model ───────────────────────────────


class TestBuildFeaturesForModel:
    """Tests for build_features_for_model()."""

    def test_unknown_model_returns_zero_score(self, sample_raw_df):
        X, report, score = build_features_for_model(sample_raw_df, "unknown_model_xyz")
        assert score == 0.0
        assert len(X) == 0
        assert any("UNKNOWN_MODEL_SCHEMA" in rc for rc in report["reason_codes"])

    def test_cfg05_output_has_correct_columns(self, sample_raw_df):
        """cfg05 should produce features matching the 56-feature schema."""
        X, report, score = build_features_for_model(sample_raw_df, "cfg05")
        expected = MODEL_FEATURE_SCHEMAS["cfg05"]
        # All expected feature columns should be present
        for col in expected:
            assert col in X.columns, f"Missing column: {col}"

    def test_catboost_sota_output_has_24_features(self, sample_raw_df):
        """catboost_sota should produce exactly 24 feature columns."""
        X, report, score = build_features_for_model(sample_raw_df, "catboost_sota")
        feature_cols = [c for c in X.columns if c not in BUSINESS_COLUMNS]
        assert len(feature_cols) == 24

    def test_catboost_spike_same_schema_as_cfg05(self, sample_raw_df):
        """catboost_spike_residual should produce same features as cfg05."""
        X_spike, report_spike, score_spike = build_features_for_model(
            sample_raw_df, "catboost_spike_residual"
        )
        X_cfg05, report_cfg05, score_cfg05 = build_features_for_model(
            sample_raw_df, "cfg05"
        )
        assert score_spike == score_cfg05
        spike_feats = set(c for c in X_spike.columns if c not in BUSINESS_COLUMNS)
        cfg05_feats = set(c for c in X_cfg05.columns if c not in BUSINESS_COLUMNS)
        assert spike_feats == cfg05_feats

    def test_no_deny_list_in_output(self, sample_with_deny_cols):
        """Output must never contain y_true, residual, etc."""
        X, report, score = build_features_for_model(sample_with_deny_cols, "cfg05")
        for col in X.columns:
            for deny in DENY_LIST:
                assert deny not in col.lower(), f"Deny-list violation: {col}"

    def test_schema_match_score_range(self, sample_raw_df):
        """Score should be between 0 and 1."""
        for model_name in ["cfg05", "catboost_sota"]:
            _, _, score = build_features_for_model(sample_raw_df, model_name)
            assert 0.0 <= score <= 1.0

    def test_custom_schema_manifest(self, sample_raw_df):
        """Custom schema manifest should override defaults."""
        custom = {"my_model": ["hour", "month", "load"]}
        X, report, score = build_features_for_model(
            sample_raw_df, "my_model", schema_manifest=custom
        )
        assert "hour" in X.columns
        assert "month" in X.columns
        assert "load" in X.columns

    def test_report_has_required_keys(self, sample_raw_df):
        _, report, _ = build_features_for_model(sample_raw_df, "cfg05")
        required_keys = [
            "model_name", "expected_features", "expected_count",
            "present_features", "present_count", "missing_features",
            "missing_count", "reason_codes", "schema_match_score",
        ]
        for key in required_keys:
            assert key in report, f"Missing report key: {key}"

    def test_precomputed_features_used(self, sample_raw_df):
        """When precomputed_features is provided, it should be used directly."""
        # Build a fake precomputed frame
        precomputed = pd.DataFrame({
            "hour": [1, 2, 3],
            "month": [1, 1, 1],
            "load": [100, 200, 300],
            "ds": pd.date_range("2025-01-01", periods=3, freq="h"),
        })
        from data.business_day import add_business_time_columns
        precomputed = add_business_time_columns(precomputed, timestamp_col="ds")

        custom = {"test_model": ["hour", "month", "load"]}
        X, report, score = build_features_for_model(
            sample_raw_df, "test_model",
            schema_manifest=custom,
            precomputed_features=precomputed,
        )
        assert score == 1.0
        assert len(X) == 3
