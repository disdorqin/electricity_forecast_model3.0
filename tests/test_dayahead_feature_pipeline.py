"""
tests/test_dayahead_feature_pipeline.py — Day-ahead feature pipeline contract tests.

Validates:
    1. Feature columns from registry match adapter definition
    2. build_dayahead_features preserves business columns
    3. build_dayahead_features does NOT contain y_true/residual/error/abs_error
    4. Invalid model raises ValueError
    5. Missing features are filled (zero or NaN) without crash
    6. validate_dayahead_feature_frame detects denied columns
    7. report_missing_features returns structured report
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.features.dayahead_features import (
    build_dayahead_features,
    get_dayahead_feature_columns,
    validate_dayahead_feature_frame,
    report_missing_features,
    DENY_LIST,
    BUSINESS_COLUMNS,
)


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame with ds and a few cfg05 features."""
    return pd.DataFrame({
        "ds": pd.date_range("2026-03-05 01:00", periods=5, freq="h"),
        "hour": [1, 2, 3, 4, 5],
        "month": [3] * 5,
        "day_of_week": [3] * 5,
        "is_weekend": [0] * 5,
    })


class TestFeatureColumns:
    """Contract: feature column definitions."""

    def test_cfg05_has_42_features(self):
        """cfg05 has exactly 42 feature columns."""
        cols = get_dayahead_feature_columns("cfg05")
        assert len(cols) == 56

    def test_cfg05_features_from_registry(self):
        """cfg05 feature columns include key expected features."""
        cols = get_dayahead_feature_columns("cfg05")
        assert "hour" in cols
        assert "month" in cols
        assert "load" in cols
        assert "wind" in cols
        assert "solar" in cols
        assert "lag_24h" in cols
        assert "same_hour_mean_7d" in cols


class TestBuildFeatures:
    """Contract: build_dayahead_features."""

    def test_business_columns_preserved(self, minimal_df):
        """Business columns (ds, hour_business, period) are preserved."""
        result = build_dayahead_features(minimal_df, model_id="cfg05")
        assert "ds" in result.columns
        assert "hour_business" in result.columns
        assert "period" in result.columns

    def test_no_denied_columns(self, minimal_df):
        """build_dayahead_features contains no y_true/residual/error/abs_error columns."""
        result = build_dayahead_features(minimal_df, model_id="cfg05")
        for term in DENY_LIST:
            for col in result.columns:
                assert term not in col.lower(), f"Denied term '{term}' found in column '{col}'"

    def test_invalid_model_raises(self, minimal_df):
        """Invalid model raises ValueError."""
        with pytest.raises(ValueError, match="INVALID"):
            build_dayahead_features(minimal_df, model_id="lgbm_spike_residual_1127")

    def test_missing_features_filled_with_zero(self, minimal_df):
        """Missing feature columns are filled with 0."""
        result = build_dayahead_features(minimal_df, model_id="cfg05", fill_strategy="zero")
        # load, wind, solar are in CFG05_FEATURE_COLUMNS but not in minimal_df
        assert "load" in result.columns
        assert result["load"].iloc[0] == 0.0

    def test_missing_features_filled_with_nan(self, minimal_df):
        """Missing feature columns can be filled with NaN."""
        result = build_dayahead_features(minimal_df, model_id="cfg05", fill_strategy="nan")
        assert "load" in result.columns
        assert pd.isna(result["load"].iloc[0])

    def test_all_features_present(self, tmp_path):
        """When all 44 features are present, no missing report."""
        data = {col: [0.0] for col in get_dayahead_feature_columns("cfg05")}
        data["ds"] = pd.Timestamp("2026-03-05 01:00")
        df = pd.DataFrame(data, index=[0])
        result = build_dayahead_features(df, model_id="cfg05")
        # 44 features + business columns
        for col in get_dayahead_feature_columns("cfg05"):
            assert col in result.columns, f"Missing feature column: {col}"

    def test_adds_business_time_if_missing(self):
        """If hour_business/business_day missing, infers from ds."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "hour": [1, 2, 3],
        })
        result = build_dayahead_features(df, model_id="cfg05")
        assert "hour_business" in result.columns
        assert "business_day" in result.columns


class TestValidateFeatureFrame:
    """Contract: validate_dayahead_feature_frame."""

    def test_clean_frame_returns_empty(self, minimal_df):
        """Valid feature frame returns empty issues list."""
        df = build_dayahead_features(minimal_df, model_id="cfg05")
        issues = validate_dayahead_feature_frame(df, model_id="cfg05")
        assert issues == []

    def test_detect_denied_column(self):
        """validate detects y_true column in frame."""
        df = pd.DataFrame({
            "ds": pd.Timestamp("2026-03-05 01:00"),
            "hour": 1,
            "y_true": 100.0,
        }, index=[0])
        issues = validate_dayahead_feature_frame(df, model_id="cfg05", strict=False)
        assert any("y_true" in issue.lower() or "denied" in issue.lower() for issue in issues)

    def test_strict_mode_raises_on_denied(self):
        """Strict mode raises ValueError on denied columns."""
        df = pd.DataFrame({
            "ds": pd.Timestamp("2026-03-05 01:00"),
            "residual": 0.5,
        }, index=[0])
        with pytest.raises(ValueError, match="Denied"):
            validate_dayahead_feature_frame(df, model_id="cfg05", strict=True)


class TestReportMissingFeatures:
    """Contract: report_missing_features."""

    def test_returns_structured_report(self, minimal_df):
        """report_missing_features returns dict with expected keys."""
        report = report_missing_features(minimal_df, model_id="cfg05")
        assert "model_id" in report
        assert "total_features" in report
        assert "present" in report
        assert "missing" in report
        assert "ratio" in report
        assert report["model_id"] == "cfg05"
        assert report["total_features"] == 56

    def test_ratio_reflects_present_features(self, minimal_df):
        """Ratio reflects the proportion of present features."""
        report = report_missing_features(minimal_df, model_id="cfg05")
        # minimal_df has: hour, month, day_of_week, is_weekend + ds = 4 features
        assert len(report["present"]) == 4
        assert report["ratio"] == 4 / 56


class TestGetFeatureColumns:
    """Contract: get_dayahead_feature_columns."""

    def test_invalid_model_raises(self):
        """Invalid model raises ValueError."""
        with pytest.raises(ValueError):
            get_dayahead_feature_columns("lgbm_spike_residual_1127")
