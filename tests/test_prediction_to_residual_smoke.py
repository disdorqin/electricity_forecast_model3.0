"""
tests/test_prediction_to_residual_smoke.py — Synthetic end-to-end smoke test.

Validates the full prediction-to-residual pipeline with synthetic tiny data:
    1. Day-ahead dry-run prediction → validate_prediction_output
    2. apply_residual_correction → validate_residual_output
    3. Output is corrected schema
    4. DATA-MISSING no-op is stable (y_pred_corrected == y_pred_raw)

Does NOT depend on real data.
Does NOT write CSV to the repository (uses pytest tmp_path when needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
)
from scripts.validate_prediction_output import validate_prediction_dataframe
from scripts.validate_residual_output import validate_residual_dataframe
from pipelines.residual_correction import (
    apply_residual_correction,
    is_data_missing_noop,
)


def _synthetic_dayahead_predictions(n_hours: int = 24) -> pd.DataFrame:
    """Synthetic day-ahead prediction output (standard schema)."""
    from data.business_day import add_business_time_columns
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-03-05 01:00", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "task": ["dayahead"] * n_hours,
        "model_name": ["cfg05"] * n_hours,
        "target_day": ["2026-03-05"] * n_hours,
        "ds": timestamps,
        "y_pred": rng.uniform(80, 200, n_hours),
        "source_confidence": [0.5] * n_hours,
        "model_version": ["1.0.0"] * n_hours,
    })
    # Add business-time columns for full schema compliance
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


class TestPredictionToResidualSmoke:
    """Contract: synthetic prediction-to-residual smoke test."""

    def test_synthetic_predictions_pass_validation(self):
        """Synthetic day-ahead predictions pass prediction output validation."""
        df = _synthetic_dayahead_predictions(24)
        passed, errors = validate_prediction_dataframe(df)
        assert passed, f"Prediction validation failed: {errors}"

    def test_prediction_to_residual_noop(self):
        """Full pipeline: prediction → residual, DATA-MISSING no-op."""
        # Step 1: Generate synthetic predictions
        preds = _synthetic_dayahead_predictions(24)

        # Step 2: Validate predictions
        passed, errors = validate_prediction_dataframe(preds)
        assert passed, f"Prediction validation failed: {errors}"

        # Step 3: Apply residual correction (no risk data → DATA-MISSING no-op)
        corrected = apply_residual_correction(preds)

        # Step 4: Validate corrected output
        passed, errors = validate_residual_dataframe(corrected)
        assert passed, f"Residual validation failed: {errors}"

        # Step 5: Verify no-op
        assert is_data_missing_noop(corrected), "Expected DATA-MISSING no-op"

    def test_corrected_output_has_17_columns(self):
        """Corrected output has all 17 corrected schema columns."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        for col in CORRECTED_PREDICTION_COLUMNS:
            assert col in corrected.columns, f"Missing column: {col}"
        assert len(corrected.columns) == len(CORRECTED_PREDICTION_COLUMNS)

    def test_y_pred_corrected_equals_raw_noop(self):
        """No-op: y_pred_corrected == y_pred_raw."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        np.testing.assert_array_almost_equal(
            corrected["y_pred_corrected"].values,
            corrected["y_pred_raw"].values,
        )

    def test_residual_delta_zero_noop(self):
        """No-op: residual_delta == 0."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        assert (corrected["residual_delta"] == 0.0).all()

    def test_correction_applied_false(self):
        """No-op: correction_applied is False."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        assert (~corrected["correction_applied"]).all()

    def test_risk_source_is_data_missing(self):
        """No-op: risk_source is DATA_MISSING."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        assert (corrected["risk_source"] == "DATA_MISSING").all()

    def test_hour_business_in_range(self):
        """hour_business in [1, 24]."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        assert corrected["hour_business"].min() >= 1
        assert corrected["hour_business"].max() <= 24

    def test_no_nan_in_output(self):
        """No NaN in y_pred_raw, y_pred_corrected, or residual_delta."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds)
        assert not corrected["y_pred_raw"].isna().any()
        assert not corrected["y_pred_corrected"].isna().any()
        assert not corrected["residual_delta"].isna().any()

    def test_no_y_true_in_production_mode(self):
        """Production mode output has no y_true."""
        preds = _synthetic_dayahead_predictions(24)
        corrected = apply_residual_correction(preds, production=True)
        assert "y_true" not in corrected.columns

    def test_full_smoke_with_tmp_csv(self, tmp_path):
        """Full smoke: write predictions CSV, validate, correct, re-validate."""
        preds = _synthetic_dayahead_predictions(24)

        # Write predictions to temp CSV (simulating real file flow)
        csv_path = tmp_path / "predictions.csv"
        preds.to_csv(str(csv_path), index=False)

        # Read back and validate
        df_loaded = pd.read_csv(str(csv_path))
        passed, errors = validate_prediction_dataframe(df_loaded)
        assert passed, f"Loaded prediction validation failed: {errors}"

        # Apply correction
        corrected = apply_residual_correction(df_loaded)
        passed, errors = validate_residual_dataframe(corrected)
        assert passed, f"Residual validation failed: {errors}"

        # Write corrected to temp CSV
        out_path = tmp_path / "corrected.csv"
        corrected.to_csv(str(out_path), index=False)

        # Read back and validate again
        corrected_loaded = pd.read_csv(str(out_path))
        passed, errors = validate_residual_dataframe(corrected_loaded)
        assert passed, f"Round-trip residual validation failed: {errors}"

        # Verify no-op in round-trip
        assert is_data_missing_noop(corrected_loaded)

    def test_multiple_models_smoke(self):
        """Pipeline works with multiple models in single DataFrame."""
        from data.business_day import add_business_time_columns
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")

        # Two models, 24 hours each = 48 rows
        df_list = []
        for model in ["cfg05", "best_two_average"]:
            df_list.append(pd.DataFrame({
                "task": ["dayahead"] * 24,
                "model_name": [model] * 24,
                "target_day": ["2026-03-05"] * 24,
                "ds": timestamps,
                "y_pred": rng.uniform(80, 200, 24),
                "source_confidence": [0.5] * 24,
                "model_version": ["1.0.0"] * 24,
            }))
        preds = pd.concat(df_list, ignore_index=True)
        preds = add_business_time_columns(preds, timestamp_col="ds")

        # Validate
        passed, errors = validate_prediction_dataframe(preds)
        assert passed, f"Multi-model prediction validation failed: {errors}"

        # Correct
        corrected = apply_residual_correction(preds)

        # Validate corrected
        passed, errors = validate_residual_dataframe(corrected)
        assert passed, f"Multi-model residual validation failed: {errors}"

        # Should be no-op
        assert is_data_missing_noop(corrected)
        # 48 rows (2 models × 24 hours)
        assert len(corrected) == 48
