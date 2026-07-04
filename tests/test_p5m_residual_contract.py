"""
tests/test_p5m_residual_contract.py — P5M Residual Plugin adapter contract tests.

Validates:
    1. No-op (output == input) when DATA-MISSING (no negative_prob)
    2. No crash when risk data is absent
    3. Standard schema output
    4. Profile validation
    5. Load / predict lifecycle
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.adapters.p5m_residual_plugin import P5MResidualPluginAdapter


def _make_fused_prediction_df(n_hours: int = 24, start: str = "2026-03-05 01:00") -> pd.DataFrame:
    """Build a synthetic fused prediction DataFrame."""
    timestamps = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({
        "task": "dayahead",
        "business_day": timestamps.normalize(),
        "hour_business": [(t.hour if t.hour != 0 else 24) for t in timestamps],
        "ds": timestamps,
        "y_pred": np.random.default_rng(42).uniform(80, 200, n_hours),
    })


class TestDataMissingBehavior:
    """Contract: no-op when DATA-MISSING (no risk data)."""

    def test_noop_when_no_risk_data(self):
        """Without negative_prob, output y_pred == input y_pred."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        np.testing.assert_array_almost_equal(result["y_pred"].values, df["y_pred"].values)

    def test_does_not_crash_without_risk_data(self):
        """Adapter does not crash when risk data is absent."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(48)
        # Should not raise any exception
        result = adapter.predict(df=df)
        assert len(result) == 48

    def test_noop_with_minimal_input(self):
        """Adapter handles minimal input (only required columns) without crashing."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = pd.DataFrame({
            "task": ["dayahead"] * 3,
            "business_day": [pd.Timestamp("2026-03-05")] * 3,
            "hour_business": [1, 2, 3],
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "y_pred": [100.0, 110.0, 120.0],
        })
        result = adapter.predict(df=df)
        np.testing.assert_array_almost_equal(result["y_pred"], [100.0, 110.0, 120.0])


class TestStandardSchema:
    """Contract: output conforms to standard prediction schema."""

    def test_output_has_standard_schema(self):
        """Output contains all PREDICTION_OUTPUT_COLUMNS."""
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        for col in PREDICTION_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_in_y_pred(self):
        """y_pred contains no NaN values."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(48)
        result = adapter.predict(df=df)
        assert not result["y_pred"].isna().any()

    def test_task_is_dayahead(self):
        """Output task column is 'dayahead'."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        assert (result["task"] == "dayahead").all()

    def test_model_name_is_p5m_residual_plugin(self):
        """Output model_name is correct."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        assert (result["model_name"] == "p5m_residual_plugin").all()

    def test_hour_business_in_range(self):
        """Output hour_business in 1..24."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(72, start="2026-03-05 00:00")
        result = adapter.predict(df=df)
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_no_y_true_in_output(self):
        """Output must NOT contain y_true."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        assert "y_true" not in result.columns


class TestProfileValidation:
    """Contract: profile parameter validation."""

    def test_conservative_profile_is_valid(self):
        """'conservative' profile is accepted."""
        adapter = P5MResidualPluginAdapter(profile="conservative")
        assert adapter.profile == "conservative"

    def test_moderate_profile_is_valid(self):
        """'moderate' profile is accepted."""
        adapter = P5MResidualPluginAdapter(profile="moderate")
        assert adapter.profile == "moderate"

    def test_aggressive_profile_is_valid(self):
        """'aggressive' profile is accepted."""
        adapter = P5MResidualPluginAdapter(profile="aggressive")
        assert adapter.profile == "aggressive"

    def test_invalid_profile_raises(self):
        """Invalid profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            P5MResidualPluginAdapter(profile="extreme")

    def test_default_profile_is_conservative(self):
        """Default profile is 'conservative'."""
        adapter = P5MResidualPluginAdapter()
        assert adapter.profile == "conservative"


class TestLifecycle:
    """Contract: load/predict lifecycle."""

    def test_load_then_predict_works(self):
        """Calling load() then predict() succeeds."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        assert adapter._loaded is True
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        assert len(result) > 0

    def test_predict_calls_load_automatically(self):
        """predict() calls load() if not already loaded."""
        adapter = P5MResidualPluginAdapter()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        assert len(result) > 0


class TestDataLoading:
    """Contract: data loading from df= and data_path=."""

    def test_accepts_dataframe(self):
        """Adapter accepts pre-loaded DataFrame."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = _make_fused_prediction_df(24)
        result = adapter.predict(df=df)
        assert len(result) > 0

    def test_raises_on_no_data(self):
        """Adapter raises ValueError if neither df nor data_path."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        with pytest.raises(ValueError):
            adapter.predict()

    def test_empty_dataframe_returns_empty(self):
        """Empty input returns empty output with correct columns."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = pd.DataFrame(columns=["task", "business_day", "hour_business", "ds", "y_pred"])
        result = adapter.predict(df=df)
        assert len(result) == 0


class TestMissingColumnHandling:
    """Contract: missing column handling."""

    def test_missing_y_pred_does_not_crash(self):
        """If y_pred is missing, fills with 0 (no-op)."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = pd.DataFrame({
            "task": ["dayahead"] * 3,
            "business_day": [pd.Timestamp("2026-03-05")] * 3,
            "hour_business": [1, 2, 3],
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
        })
        result = adapter.predict(df=df)
        assert not result["y_pred"].isna().any()

    def test_missing_required_column_raises(self):
        """Missing 'task' column raises ValueError."""
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")],
            "hour_business": [1],
            "ds": [pd.Timestamp("2026-03-05 01:00")],
        })
        with pytest.raises(ValueError):
            adapter.predict(df=df)
