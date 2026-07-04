"""
tests/test_residual_correction_runner.py — Residual correction pipeline contract tests.

Validates:
    1. No-op: y_pred_corrected == y_pred_raw
    2. No-op: residual_delta == 0
    3. No-op: correction_applied == False
    4. No-op: reason_codes contains DATA_MISSING_NO_OP
    5. Dayahead task supported
    6. Realtime task supported
    7. Invalid profile raises ValueError
    8. Production mode: no y_true dependency
    9. Empty input returns empty output
    10. Risk data integration (no crash)
    11. CLI dry-run produces corrected output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.residual_correction import (
    apply_residual_correction,
    is_data_missing_noop,
    get_corrected_schema_columns,
)


@pytest.fixture
def dayahead_predictions() -> pd.DataFrame:
    """24-hour day-ahead prediction output."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")
    return pd.DataFrame({
        "task": "dayahead",
        "model_name": "cfg05",
        "target_day": "2026-03-05",
        "ds": timestamps,
        "y_pred": rng.uniform(80, 200, 24),
        "source_confidence": 0.5,
        "model_version": "1.0.0",
    })


@pytest.fixture
def realtime_predictions() -> pd.DataFrame:
    """24-hour realtime prediction output."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")
    return pd.DataFrame({
        "task": "realtime",
        "model_name": "da_safe_realtime_assist",
        "target_day": "2026-03-05",
        "ds": timestamps,
        "y_pred": rng.uniform(80, 200, 24),
        "source_confidence": 0.5,
        "model_version": "1.0.0",
    })


class TestNoOpBehavior:
    """Contract: DATA-MISSING no-op behavior."""

    def test_noop_y_pred_corrected_equals_raw(self, dayahead_predictions):
        """No-op: y_pred_corrected == y_pred_raw."""
        result = apply_residual_correction(dayahead_predictions)
        np.testing.assert_array_almost_equal(
            result["y_pred_corrected"].values,
            result["y_pred_raw"].values,
        )

    def test_noop_residual_delta_zero(self, dayahead_predictions):
        """No-op: residual_delta == 0."""
        result = apply_residual_correction(dayahead_predictions)
        assert (result["residual_delta"] == 0.0).all()

    def test_noop_correction_applied_false(self, dayahead_predictions):
        """No-op: correction_applied == False."""
        result = apply_residual_correction(dayahead_predictions)
        assert (~result["correction_applied"]).all()

    def test_noop_reason_contains_data_missing(self, dayahead_predictions):
        """No-op: reason_codes contains DATA_MISSING_NO_OP."""
        result = apply_residual_correction(dayahead_predictions)
        assert all("DATA_MISSING_NO_OP" in rc for rc in result["reason_codes"])

    def test_is_data_missing_noop_returns_true(self, dayahead_predictions):
        """is_data_missing_noop returns True for no-op output."""
        result = apply_residual_correction(dayahead_predictions)
        assert is_data_missing_noop(result) == True

    def test_noop_risk_source_is_data_missing(self, dayahead_predictions):
        """No-op: risk_source == DATA_MISSING."""
        result = apply_residual_correction(dayahead_predictions)
        assert (result["risk_source"] == "DATA_MISSING").all()

    def test_noop_correction_module(self, dayahead_predictions):
        """No-op: correction_module == p5m_residual_noop."""
        result = apply_residual_correction(dayahead_predictions)
        assert (result["correction_module"] == "p5m_residual_noop").all()


class TestTaskSupport:
    """Contract: supports dayahead and realtime tasks."""

    def test_supports_dayahead(self, dayahead_predictions):
        """Dayahead task is supported."""
        result = apply_residual_correction(dayahead_predictions)
        assert (result["task"] == "dayahead").all()

    def test_supports_realtime(self, realtime_predictions):
        """Realtime task is supported."""
        result = apply_residual_correction(realtime_predictions)
        assert (result["task"] == "realtime").all()

    def test_task_preserved(self, dayahead_predictions):
        """Task column is preserved from input."""
        result = apply_residual_correction(dayahead_predictions)
        assert (result["task"] == dayahead_predictions["task"]).all()


class TestSchemaCompliance:
    """Contract: output conforms to corrected schema."""

    def test_output_has_all_corrected_columns(self, dayahead_predictions):
        """Output contains all CORRECTED_PREDICTION_COLUMNS."""
        from data.schema import CORRECTED_PREDICTION_COLUMNS
        result = apply_residual_correction(dayahead_predictions)
        for col in CORRECTED_PREDICTION_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_in_y_pred_raw(self, dayahead_predictions):
        """y_pred_raw has no NaN values."""
        result = apply_residual_correction(dayahead_predictions)
        assert not result["y_pred_raw"].isna().any()

    def test_no_nan_in_y_pred_corrected(self, dayahead_predictions):
        """y_pred_corrected has no NaN values."""
        result = apply_residual_correction(dayahead_predictions)
        assert not result["y_pred_corrected"].isna().any()

    def test_no_nan_in_residual_delta(self, dayahead_predictions):
        """residual_delta has no NaN values."""
        result = apply_residual_correction(dayahead_predictions)
        assert not result["residual_delta"].isna().any()

    def test_hour_business_in_range(self, dayahead_predictions):
        """hour_business values in 1..24."""
        result = apply_residual_correction(dayahead_predictions)
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_no_y_true_in_production(self, dayahead_predictions):
        """Production output must not contain y_true."""
        result = apply_residual_correction(dayahead_predictions, production=True)
        assert "y_true" not in result.columns


class TestInputValidation:
    """Contract: input validation."""

    def test_invalid_profile_raises(self, dayahead_predictions):
        """Invalid profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown correction profile"):
            apply_residual_correction(dayahead_predictions, correction_profile="extreme")

    def test_missing_required_column_raises(self):
        """Missing required key column raises ValueError."""
        df = pd.DataFrame({"some_column": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required columns"):
            apply_residual_correction(df)

    def test_empty_input_returns_empty(self):
        """Empty input returns empty DataFrame with correct columns."""
        df = pd.DataFrame(columns=["task", "model_name", "ds", "y_pred"])
        result = apply_residual_correction(df)
        assert len(result) == 0
        from data.schema import CORRECTED_PREDICTION_COLUMNS
        for col in CORRECTED_PREDICTION_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"


class TestRiskDataIntegration:
    """Contract: risk data integration (no crash, no-op fallback)."""

    def test_risk_df_present_does_not_crash(self, dayahead_predictions):
        """Providing risk_df does not crash (falls back to no-op)."""
        risk_df = pd.DataFrame({
            "negative_prob": np.random.default_rng(42).uniform(0, 1, 24),
            "risk_source": ["SYNTHETIC"] * 24,
        })
        result = apply_residual_correction(
            dayahead_predictions, risk_df=risk_df
        )
        # Without negative risk model, P5M adapter will still no-op
        assert not result["y_pred_corrected"].isna().any()

    def test_canonical_pack_nonexistent_does_not_crash(self, dayahead_predictions):
        """Non-existent canonical pack path is silently ignored."""
        result = apply_residual_correction(
            dayahead_predictions, canonical_pack_path="/nonexistent/pack.csv"
        )
        # Falls back to no-op because path doesn't exist
        assert is_data_missing_noop(result)


class TestGetCorrectedSchema:
    """Contract: get_corrected_schema_columns."""

    def test_returns_list_of_17_strings(self):
        """get_corrected_schema_columns returns list of 17 strings."""
        cols = get_corrected_schema_columns()
        assert len(cols) == 17
        assert all(isinstance(c, str) for c in cols)


class TestCLIDryRun:
    """Contract: CLI dry-run produces corrected output."""

    def test_dry_run_produces_output(self):
        """CLI dry-run mode produces corrected output."""
        from scripts.run_residual_correction import main
        exit_code = main(["--dry-run"])
        assert exit_code == 0

    def test_dry_run_noop_behavior(self):
        """CLI dry-run produces no-op (no risk data available)."""
        # Run the pipeline directly simulating dry-run
        from pipelines.residual_correction import apply_residual_correction
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")
        df = pd.DataFrame({
            "task": "dayahead",
            "model_name": "cfg05",
            "target_day": "2026-03-05",
            "ds": timestamps,
            "y_pred": rng.uniform(80, 200, 24),
        })
        result = apply_residual_correction(df)
        assert is_data_missing_noop(result)
        np.testing.assert_array_almost_equal(
            result["y_pred_corrected"].values, result["y_pred_raw"].values
        )
