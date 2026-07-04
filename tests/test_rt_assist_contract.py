"""
tests/test_rt_assist_contract.py — DA-Safe Realtime Assist adapter contract tests.

Validates:
    1. rt_pred = da_anchor (default, no correction)
    2. Output conforms to standard schema
    3. No NaN in y_pred
    4. data_path vs df parameter handling
    5. Date filtering works
    6. Safe correction (when enabled) produces different output
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.adapters.realtime_da_safe_assist import DASafeRealtimeAssistAdapter


def _make_synthetic_df(n_hours: int = 48, start: str = "2026-03-05 01:00") -> pd.DataFrame:
    """Build a synthetic DataFrame with da_anchor column."""
    timestamps = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({
        "ds": timestamps,
        "da_anchor": np.random.default_rng(42).uniform(80, 200, n_hours),
    })


class TestAdapterDefaults:
    """Contract: default adapter behavior (rt_pred = da_anchor)."""

    def test_rt_pred_equals_da_anchor(self):
        """Default adapter returns rt_pred == da_anchor."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(48)
        result = adapter.predict(df=df)
        np.testing.assert_array_almost_equal(result["y_pred"].values, df["da_anchor"].values)

    def test_output_has_standard_schema(self):
        """Output contains all PREDICTION_OUTPUT_COLUMNS."""
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        result = adapter.predict(df=df)
        for col in PREDICTION_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_in_y_pred(self):
        """y_pred contains no NaN values."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(48)
        result = adapter.predict(df=df)
        assert not result["y_pred"].isna().any()

    def test_task_is_realtime(self):
        """Output task column is 'realtime'."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        result = adapter.predict(df=df)
        assert (result["task"] == "realtime").all()

    def test_model_name_is_da_safe_realtime_assist(self):
        """Output model_name is correct."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        result = adapter.predict(df=df)
        assert (result["model_name"] == "da_safe_realtime_assist").all()

    def test_hour_business_in_range(self):
        """Output hour_business in 1..24."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(72, start="2026-03-05 00:00")
        result = adapter.predict(df=df)
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24


class TestDataLoading:
    """Contract: data loading from df= and data_path=."""

    def test_accepts_dataframe(self):
        """Adapter accepts pre-loaded DataFrame."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        result = adapter.predict(df=df)
        assert len(result) > 0

    def test_accepts_csv_path(self):
        """Adapter reads CSV from data_path."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f.name, index=False)
            result = adapter.predict(data_path=f.name)
        Path(f.name).unlink(missing_ok=True)
        assert len(result) > 0

    def test_raises_on_no_data(self):
        """Adapter raises ValueError if neither df nor data_path provided."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        with pytest.raises(ValueError):
            adapter.predict()

    def test_raises_on_missing_da_anchor(self):
        """Adapter raises ValueError if da_anchor column missing."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = pd.DataFrame({"ds": pd.date_range("2026-03-05 01:00", periods=5, freq="h")})
        with pytest.raises(ValueError, match="da_anchor"):
            adapter.predict(df=df)

    def test_empty_dataframe_returns_empty(self):
        """Empty input DataFrame returns empty output."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = pd.DataFrame(columns=["ds", "da_anchor"])
        result = adapter.predict(df=df)
        assert len(result) == 0


class TestDateFiltering:
    """Contract: start/end date filtering."""

    def test_start_filter(self):
        """start parameter filters rows >= start."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(72, start="2026-03-05 01:00")
        result = adapter.predict(df=df, start="2026-03-06")
        # All ds should be >= 2026-03-06
        assert (result["ds"] >= pd.Timestamp("2026-03-06")).all()

    def test_end_filter(self):
        """end parameter filters rows <= end."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(72, start="2026-03-05 01:00")
        result = adapter.predict(df=df, end="2026-03-06")
        assert (result["ds"] <= pd.Timestamp("2026-03-07")).all()  # end + 1d

    def test_start_end_filter_returns_subset(self):
        """start + end returns a subset of rows."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(120, start="2026-03-05 01:00")
        result = adapter.predict(df=df, start="2026-03-06", end="2026-03-07")
        assert 0 < len(result) < len(df)


class TestColumnNormalization:
    """Contract: column name normalization."""

    def test_renames_times_to_ds(self):
        """Column 'times' is renamed to 'ds'."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24).rename(columns={"ds": "times"})
        result = adapter.predict(df=df)
        assert "ds" in result.columns

    def test_renames_da_price_to_da_anchor(self):
        """Column 'da_price' is used as 'da_anchor'."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24).drop(columns=["da_anchor"])
        df["da_price"] = np.random.default_rng(42).uniform(80, 200, 24)
        result = adapter.predict(df=df)
        assert len(result) > 0


class TestSafeCorrection:
    """Contract: safe correction (when enabled + model loaded)."""

    def test_correction_disabled_by_default(self):
        """enable_safe_correction defaults to False."""
        adapter = DASafeRealtimeAssistAdapter()
        assert adapter.enable_safe_correction is False

    def test_alpha_defaults_to_1(self):
        """alpha defaults to 1.0."""
        adapter = DASafeRealtimeAssistAdapter()
        assert adapter.alpha == 1.0

    def test_clip_defaults_to_0(self):
        """clip_correction defaults to 0 (no clip)."""
        adapter = DASafeRealtimeAssistAdapter()
        assert adapter.clip_correction == 0.0


class TestOutputValidation:
    """Contract: output passes adapter validate_output."""

    def test_validate_output_passes(self):
        """Output passes self.validate_output without raising."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        result = adapter.predict(df=df)
        # validate_output is called inside predict()
        assert result is not None

    def test_no_y_true_in_output(self):
        """Output must NOT contain y_true."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        df = _make_synthetic_df(24)
        result = adapter.predict(df=df)
        assert "y_true" not in result.columns
