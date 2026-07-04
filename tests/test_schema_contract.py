"""
tests/test_schema_contract.py — Schema and business_day contract tests.

Validates:
    1. business_day / hour_business mapping from timestamps
    2. PREDICTION_OUTPUT_COLUMNS completeness
    3. hour_business in 1..24
    4. No duplicate unique keys
    5. y_pred no NaN in adapter output
    6. EVAL_ONLY columns rejected in production mode
    7. ensure_output_schema validation
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.business_day import (
    business_day_from_timestamp,
    hour_business_from_timestamp,
    hour_business_to_timestamp,
    infer_period,
    add_business_time_columns,
    standardize_business_columns,
    validate_daily_predictions,
)
from data.schema import (
    PREDICTION_OUTPUT_COLUMNS,
    EVAL_ONLY_COLUMNS,
    PREDICTION_LEDGER_KEY,
    ACTUAL_LEDGER_KEY,
    validate_output_columns,
    validate_no_eval_columns,
    ensure_output_schema,
    VALID_TASKS,
    VALID_PERIODS,
)


class TestBusinessDayMapping:
    """Contract: timestamp→business_day / hour_business mapping correctness."""

    def test_midnight_maps_to_previous_day_hour_24(self):
        """Timestamp D 00:00 → business_day D-1, hour 24."""
        ts = pd.Timestamp("2026-03-05 00:00:00")
        bd = business_day_from_timestamp(ts)
        hb = hour_business_from_timestamp(ts)
        assert bd == pd.Timestamp("2026-03-04")
        assert hb == 24

    def test_afternoon_maps_to_same_day(self):
        """Timestamp D HH:00 → business_day D, hour HH (HH=1..23)."""
        ts = pd.Timestamp("2026-03-05 14:00:00")
        bd = business_day_from_timestamp(ts)
        hb = hour_business_from_timestamp(ts)
        assert bd == pd.Timestamp("2026-03-05")
        assert hb == 14

    def test_hour_1_maps_correctly(self):
        """Timestamp D 01:00 → business_day D, hour 1."""
        ts = pd.Timestamp("2026-06-01 01:00:00")
        assert business_day_from_timestamp(ts) == pd.Timestamp("2026-06-01")
        assert hour_business_from_timestamp(ts) == 1

    def test_hour_24_round_trip(self):
        """hour_business_to_timestamp(D, 24) → D+1 00:00 → hour 24."""
        bd = pd.Timestamp("2026-03-05")
        ts = hour_business_to_timestamp(bd, 24)
        assert ts == pd.Timestamp("2026-03-06 00:00:00")
        # Round-trip
        assert business_day_from_timestamp(ts) == bd
        assert hour_business_from_timestamp(ts) == 24

    def test_hour_1_round_trip(self):
        """hour_business_to_timestamp(D, 1) → D 01:00 → hour 1."""
        bd = pd.Timestamp("2026-03-05")
        ts = hour_business_to_timestamp(bd, 1)
        assert ts == pd.Timestamp("2026-03-05 01:00:00")
        assert business_day_from_timestamp(ts) == bd
        assert hour_business_from_timestamp(ts) == 1

    def test_all_24_hours_round_trip(self):
        """Every hour_business 1..24 round-trips correctly."""
        bd = pd.Timestamp("2026-03-10")
        for hb in range(1, 25):
            ts = hour_business_to_timestamp(bd, hb)
            assert hour_business_from_timestamp(ts) == hb
            assert business_day_from_timestamp(ts) == bd

    def test_infer_period_mapping(self):
        """hour_business→period: 1-8→1_8, 9-16→9_16, 17-24→17_24."""
        for hb in range(1, 9):
            assert infer_period(hb) == "1_8", f"hb={hb} should be 1_8"
        for hb in range(9, 17):
            assert infer_period(hb) == "9_16", f"hb={hb} should be 9_16"
        for hb in range(17, 25):
            assert infer_period(hb) == "17_24", f"hb={hb} should be 17_24"

    def test_infer_period_invalid_raises(self):
        """infer_period with value outside 1..24 raises ValueError."""
        with pytest.raises(ValueError):
            infer_period(0)
        with pytest.raises(ValueError):
            infer_period(25)


class TestAddBusinessTimeColumns:
    """Contract: add_business_time_columns produces correct columns."""

    def test_adds_all_three_columns(self):
        """add_business_time_columns adds business_day, hour_business, period."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
        })
        result = add_business_time_columns(df)
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert "period" in result.columns
        assert len(result) == 24

    def test_hour_business_range(self):
        """All 24 generated rows have hour_business in 1..24."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 00:00", periods=25, freq="h"),
        })
        result = add_business_time_columns(df)
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_midnight_gets_hour_24(self):
        """00:00 timestamp gets hour_business=24."""
        df = pd.DataFrame({"ds": [pd.Timestamp("2026-03-05 00:00")]})
        result = add_business_time_columns(df)
        assert result["hour_business"].iloc[0] == 24

    def test_period_values(self):
        """period column only contains valid period values."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 00:00", periods=48, freq="h"),
        })
        result = add_business_time_columns(df)
        assert result["period"].isin(VALID_PERIODS).all()


class TestStandardizeBusinessColumns:
    """Contract: standardize_business_columns validation."""

    def test_validates_existing_columns(self):
        """Existing business_day and hour_business are validated."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=5, freq="h"),
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": [1, 2, 3, 4, 5],
        })
        result = standardize_business_columns(df, inplace=False)
        assert result["hour_business"].tolist() == [1, 2, 3, 4, 5]

    def test_raises_on_invalid_hour(self):
        """hour_business outside 1..24 raises ValueError."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=3, freq="h"),
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": [1, 99, 3],
        })
        with pytest.raises(ValueError, match="hour_business out of range"):
            standardize_business_columns(df)


class TestValidateDailyPredictions:
    """Contract: validate_daily_predictions detects errors."""

    def test_valid_24h_passes(self):
        """A full 24-hour set with no NaN y_pred passes."""
        df = pd.DataFrame({
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": list(range(1, 25)),
            "y_pred": [100.0 + i for i in range(24)],
        })
        errors = validate_daily_predictions(df)
        assert errors == []

    def test_missing_hours_reported(self):
        """Only 23 hours → error for missing hour."""
        df = pd.DataFrame({
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": list(range(1, 24)),
            "y_pred": [100.0] * 23,
        })
        errors = validate_daily_predictions(df)
        assert any("Missing hour_business" in e for e in errors)

    def test_duplicate_hours_reported(self):
        """Duplicate hour_business → error."""
        df = pd.DataFrame({
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": [1, 1, 2, 3],
            "y_pred": [100.0] * 4,
        })
        errors = validate_daily_predictions(df)
        assert any("Duplicate" in e for e in errors)

    def test_nan_y_pred_reported(self):
        """NaN in y_pred → error."""
        df = pd.DataFrame({
            "business_day": pd.Timestamp("2026-03-05"),
            "hour_business": list(range(1, 25)),
            "y_pred": [float("nan")] * 24,
        })
        errors = validate_daily_predictions(df)
        assert any("NaN" in e for e in errors)


class TestSchemaCompleteness:
    """Contract: schema constants are complete and consistent."""

    def test_prediction_output_has_10_columns(self):
        """PREDICTION_OUTPUT_COLUMNS has exactly 10 columns."""
        assert len(PREDICTION_OUTPUT_COLUMNS) == 10

    def test_prediction_unique_key_is_subset_of_ledger(self):
        """PREDICTION_LEDGER_KEY columns all exist in PREDICTION_LEDGER_COLUMNS."""
        from data.schema import PREDICTION_LEDGER_COLUMNS
        for col in PREDICTION_LEDGER_KEY:
            assert col in PREDICTION_LEDGER_COLUMNS, f"{col} missing from ledger columns"

    def test_actual_unique_key_columns_exist(self):
        """ACTUAL_LEDGER_KEY columns exist in ACTUAL_LEDGER_COLUMNS."""
        from data.schema import ACTUAL_LEDGER_COLUMNS
        for col in ACTUAL_LEDGER_KEY:
            assert col in ACTUAL_LEDGER_COLUMNS, f"{col} missing from actual ledger"

    def test_eval_only_not_in_output(self):
        """No EVAL_ONLY column is in PREDICTION_OUTPUT_COLUMNS."""
        for col in EVAL_ONLY_COLUMNS:
            assert col not in PREDICTION_OUTPUT_COLUMNS

    def test_valid_tasks(self):
        """VALID_TASKS contains dayahead and realtime."""
        assert "dayahead" in VALID_TASKS
        assert "realtime" in VALID_TASKS

    def test_valid_periods(self):
        """VALID_PERIODS contains all three periods."""
        assert set(VALID_PERIODS) == {"1_8", "9_16", "17_24"}


class TestSchemaValidators:
    """Contract: schema validation functions."""

    def test_validate_output_columns_missing(self):
        """validate_output_columns reports missing columns."""
        missing = validate_output_columns(["task", "y_pred"])
        assert "model_name" in missing
        assert "business_day" in missing

    def test_validate_output_columns_full(self):
        """validate_output_columns returns empty for complete columns."""
        missing = validate_output_columns(PREDICTION_OUTPUT_COLUMNS)
        assert missing == []

    def test_validate_no_eval_columns_detects_leak(self):
        """validate_no_eval_columns detects y_true."""
        leaked = validate_no_eval_columns(PREDICTION_OUTPUT_COLUMNS + ["y_true"])
        assert "y_true" in leaked

    def test_validate_no_eval_columns_clean(self):
        """validate_no_eval_columns returns empty when no eval columns."""
        leaked = validate_no_eval_columns(PREDICTION_OUTPUT_COLUMNS)
        assert leaked == []

    def test_ensure_output_schema_production_rejects_y_true(self):
        """ensure_output_schema(production=True) raises if y_true present."""
        df = pd.DataFrame({c: [0] for c in PREDICTION_OUTPUT_COLUMNS})
        df["y_true"] = 100.0
        with pytest.raises(ValueError, match="eval-only"):
            ensure_output_schema(df, production=True)

    def test_ensure_output_schema_eval_allows_y_true(self):
        """ensure_output_schema(production=False) allows y_true."""
        df = pd.DataFrame({c: [0] for c in PREDICTION_OUTPUT_COLUMNS})
        df["y_true"] = 100.0
        result = ensure_output_schema(df, production=False)
        assert "y_true" in result.columns

    def test_ensure_output_schema_reorders(self):
        """ensure_output_schema reorders columns to canonical order."""
        df = pd.DataFrame({c: [0] for c in reversed(PREDICTION_OUTPUT_COLUMNS)})
        result = ensure_output_schema(df)
        assert list(result.columns) == PREDICTION_OUTPUT_COLUMNS
