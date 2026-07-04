"""
tests/test_realtime_feature_pipeline.py — Realtime feature pipeline contract tests.

Validates:
    1. normalize_realtime_columns: da_price → da_anchor
    2. normalize_realtime_columns: forecast_price → da_anchor (with reason)
    3. normalize_realtime_columns: rt_price → rt_actual
    4. normalize_realtime_columns: times → ds
    5. build_realtime_assist_input adds business-time columns
    6. Production mode: missing rt_actual does not crash
    7. Reason codes recorded for all transformations
    8. validate_realtime_assist_input detects missing da_anchor
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.features.realtime_features import (
    normalize_realtime_columns,
    build_realtime_assist_input,
    validate_realtime_assist_input,
)


class TestNormalizeColumns:
    """Contract: normalize_realtime_columns column mapping."""

    def test_da_price_maps_to_da_anchor(self):
        """da_price column is renamed to da_anchor."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_price": [100.0, 110.0, 120.0],
        })
        result, reasons = normalize_realtime_columns(df)
        assert "da_anchor" in result.columns
        assert "da_price" not in result.columns
        assert any("DA_ANCHOR_FROM_DA_PRICE" in r for r in reasons)

    def test_forecast_price_fallback_to_da_anchor(self):
        """forecast_price is used as da_anchor only when no da_price present."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "forecast_price": [100.0, 110.0, 120.0],
        })
        result, reasons = normalize_realtime_columns(df)
        assert "da_anchor" in result.columns
        assert any("DA_ANCHOR_FROM_FORECAST_PRICE" in r for r in reasons)

    def test_rt_price_maps_to_rt_actual(self):
        """rt_price is renamed to rt_actual."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "rt_price": [95.0, 105.0, 115.0],
        })
        result, reasons = normalize_realtime_columns(df)
        assert "rt_actual" in result.columns
        assert "rt_price" not in result.columns
        assert any("RT_ACTUAL_FROM_RT_PRICE" in r for r in reasons)

    def test_times_maps_to_ds(self):
        """'times' column is renamed to 'ds'."""
        df = pd.DataFrame({
            "times": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
        })
        result, reasons = normalize_realtime_columns(df)
        assert "ds" in result.columns
        assert "times" not in result.columns
        assert any("TIMESTAMP_FROM_TIMES" in r for r in reasons)

    def test_timestamp_maps_to_ds(self):
        """'timestamp' column is renamed to 'ds'."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
        })
        result, reasons = normalize_realtime_columns(df)
        assert "ds" in result.columns
        assert "timestamp" not in result.columns

    def test_da_anchor_already_present(self):
        """If da_anchor already exists, no da_price mapping needed."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
            "da_price": [101.0] * 3,
        })
        result, reasons = normalize_realtime_columns(df)
        assert "da_anchor" in result.columns
        # da_price might also be mapped, but da_anchor already existed
        assert "da_anchor" in result.columns


class TestBuildRealtimeAssistInput:
    """Contract: build_realtime_assist_input."""

    def test_adds_business_time_columns(self):
        """build_realtime_assist_input adds business_day, hour_business, period."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
        })
        result, meta = build_realtime_assist_input(df)
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert "period" in result.columns
        assert any("BUSINESS_TIME_ADDED" in r for r in meta["reason_codes"])

    def test_production_missing_rt_actual_does_not_crash(self):
        """Production mode: missing rt_actual is not an error."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
        })
        result, meta = build_realtime_assist_input(df, production=True)
        assert "RT_ACTUAL_MISSING_FOR_PRODUCTION" in meta["reason_codes"]
        assert len(meta["errors"]) == 0  # non-fatal

    def test_eval_mode_missing_rt_actual_reports_error(self):
        """Eval mode: missing rt_actual adds to errors list."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
        })
        result, meta = build_realtime_assist_input(df, production=False)
        assert len(meta["errors"]) > 0
        assert any("RT_ACTUAL_MISSING" in e for e in meta["errors"])

    def test_da_anchor_source_tracked(self):
        """Metadata tracks how da_anchor was derived."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_price": [100.0] * 3,
        })
        result, meta = build_realtime_assist_input(df)
        assert meta["da_anchor_source"] == "da_price"

    def test_has_rt_actual_flag(self):
        """Metadata correctly reports whether rt_actual is present."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
            "rt_actual": [95.0] * 3,
        })
        result, meta = build_realtime_assist_input(df)
        assert meta["has_rt_actual"] is True

    def test_has_rt_actual_flag_false_when_missing(self):
        """Metadata correctly reports False when rt_actual missing."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
        })
        result, meta = build_realtime_assist_input(df)
        assert meta["has_rt_actual"] is False


class TestValidateRealtimeInput:
    """Contract: validate_realtime_assist_input."""

    def test_valid_production_input_returns_empty(self):
        """Production input with ds and da_anchor passes."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "da_anchor": [100.0] * 3,
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": [1, 2, 3],
        })
        errors = validate_realtime_assist_input(df, production=True)
        assert errors == []

    def test_missing_da_anchor_reported(self):
        """Missing da_anchor is detected."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": [1, 2, 3],
        })
        errors = validate_realtime_assist_input(df, production=True)
        assert any("da_anchor" in e for e in errors)

    def test_missing_ds_reported(self):
        """Missing ds column is detected."""
        df = pd.DataFrame({
            "da_anchor": [100.0] * 3,
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": [1, 2, 3],
        })
        errors = validate_realtime_assist_input(df, production=True)
        assert any("ds" in e for e in errors)
