"""
tests/test_residual_output_validator.py — Residual output validator contract tests.

Validates:
    1. Valid corrected DataFrame passes all checks
    2. Duplicate key detection
    3. y_pred_raw NaN detection
    4. y_pred_corrected NaN detection
    5. residual_delta NaN detection
    6. residual_delta arithmetic mismatch detection
    7. correction_applied non-boolean detection
    8. hour_business out of range detection
    9. y_true in production mode detection
    10. Missing required column detection
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import CORRECTED_PREDICTION_COLUMNS


def _valid_corrected_df(n_rows: int = 24) -> pd.DataFrame:
    """Create a valid corrected prediction DataFrame."""
    import numpy as np
    bd = pd.Timestamp("2026-03-05")
    rng = np.random.default_rng(42)
    y_pred_raw = rng.uniform(80, 200, n_rows)
    return pd.DataFrame({
        "task": ["dayahead"] * n_rows,
        "model_name": ["cfg05"] * n_rows,
        "target_day": ["2026-03-05"] * n_rows,
        "business_day": [bd] * n_rows,
        "ds": pd.date_range("2026-03-05 01:00", periods=n_rows, freq="h"),
        "hour_business": list(range(1, n_rows + 1)),
        "period": ["1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24") for h in range(1, n_rows + 1)],
        "y_pred_raw": y_pred_raw,
        "y_pred_corrected": y_pred_raw.copy(),
        "residual_delta": np.zeros(n_rows),
        "correction_applied": [False] * n_rows,
        "correction_module": ["p5m_residual_noop"] * n_rows,
        "risk_source": ["DATA_MISSING"] * n_rows,
        "reason_codes": ["DATA_MISSING_NO_OP"] * n_rows,
        "correction_version": ["0.0.0"] * n_rows,
        "source_confidence": [0.5] * n_rows,
        "model_version": ["1.0.0"] * n_rows,
    })


class TestValidOutput:
    """Contract: valid corrected output passes."""

    def test_valid_df_passes(self):
        """Valid corrected DataFrame passes all checks."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        passed, errors = validate_residual_dataframe(df)
        assert passed, f"Expected pass, got errors: {errors}"
        assert errors == []

    def test_valid_corrected_df_passes(self):
        """Corrected DataFrame with non-zero deltas passes."""
        from scripts.validate_residual_output import validate_residual_dataframe
        import numpy as np
        df = _valid_corrected_df(24)
        # Apply a small correction to some rows
        df.loc[0:4, "y_pred_corrected"] = df.loc[0:4, "y_pred_raw"].values * 0.95
        df.loc[0:4, "residual_delta"] = df.loc[0:4, "y_pred_corrected"].values - df.loc[0:4, "y_pred_raw"].values
        df.loc[0:4, "correction_applied"] = True
        df.loc[0:4, "correction_module"] = "p5m_residual_plugin"
        df.loc[0:4, "risk_source"] = "NEGATIVE_RISK"
        passed, errors = validate_residual_dataframe(df)
        assert passed, f"Expected pass, got errors: {errors}"


class TestDuplicateKeys:
    """Contract: duplicate key detection."""

    def test_detects_duplicate_keys(self):
        """Duplicate (task, model_name, business_day, hour_business) is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        dup_row = df.iloc[[0]].copy()
        df = pd.concat([df, dup_row], ignore_index=True)
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("duplicate" in e.lower() for e in errors)


class TestNaNChecks:
    """Contract: NaN detection."""

    def test_detects_nan_in_y_pred_raw(self):
        """NaN in y_pred_raw is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df.loc[0, "y_pred_raw"] = float("nan")
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("y_pred_raw" in e and "NaN" in e for e in errors)

    def test_detects_nan_in_y_pred_corrected(self):
        """NaN in y_pred_corrected is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df.loc[0, "y_pred_corrected"] = float("nan")
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("y_pred_corrected" in e and "NaN" in e for e in errors)

    def test_detects_nan_in_residual_delta(self):
        """NaN in residual_delta is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df.loc[0, "residual_delta"] = float("nan")
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("residual_delta" in e and "NaN" in e for e in errors)


class TestResidualDeltaArithmetic:
    """Contract: residual_delta == y_pred_corrected - y_pred_raw."""

    def test_detects_delta_mismatch(self):
        """residual_delta != y_pred_corrected - y_pred_raw is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        import numpy as np
        df = _valid_corrected_df(24)
        # Set y_pred_corrected != y_pred_raw but keep residual_delta = 0
        df.loc[0, "y_pred_corrected"] = df.loc[0, "y_pred_raw"] - 10.0
        # residual_delta is still 0.0 → mismatch
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("residual_delta" in e and "!=" in e for e in errors)

    def test_correct_delta_passes(self):
        """Correct residual_delta passes validation."""
        from scripts.validate_residual_output import validate_residual_dataframe
        import numpy as np
        df = _valid_corrected_df(24)
        df.loc[0, "y_pred_corrected"] = df.loc[0, "y_pred_raw"] - 10.0
        df.loc[0, "residual_delta"] = -10.0
        passed, errors = validate_residual_dataframe(df)
        assert passed, f"Expected pass, got errors: {errors}"


class TestCorrectionAppliedBoolean:
    """Contract: correction_applied is boolean."""

    def test_non_boolean_correction_applied_detected(self):
        """Non-boolean correction_applied is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        # Convert bool column to object so we can inject non-boolean values
        df["correction_applied"] = df["correction_applied"].astype(object)
        df.loc[0, "correction_applied"] = "maybe"
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("correction_applied" in e for e in errors)

    def test_integer_01_passes(self):
        """Integer 0/1 values for correction_applied pass."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df["correction_applied"] = [1 if i < 5 else 0 for i in range(24)]
        passed, errors = validate_residual_dataframe(df)
        assert passed, f"Expected pass, got errors: {errors}"


class TestHourBusiness:
    """Contract: hour_business range check."""

    def test_out_of_range_hour_detected(self):
        """hour_business outside 1..24 is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df.loc[0, "hour_business"] = 0
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("out of range" in e.lower() for e in errors)


class TestProductionMode:
    """Contract: production mode."""

    def test_y_true_in_production_detected(self):
        """y_true in production mode is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df["y_true"] = 100.0
        passed, errors = validate_residual_dataframe(df, production=True)
        assert not passed
        assert any("y_true" in e for e in errors)

    def test_y_true_in_eval_allowed(self):
        """y_true in eval mode (production=False) is allowed."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df["y_true"] = 100.0
        passed, errors = validate_residual_dataframe(df, production=False)
        assert passed


class TestMissingColumns:
    """Contract: missing column detection."""

    def test_missing_required_column_detected(self):
        """Missing required column is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df = df.drop(columns=["residual_delta"])
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("residual_delta" in e for e in errors)

    def test_empty_df_detected(self):
        """Empty DataFrame is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(0)
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("empty" in e.lower() for e in errors)


class TestPeriodValidation:
    """Contract: period value check."""

    def test_invalid_period_detected(self):
        """Invalid period value is detected."""
        from scripts.validate_residual_output import validate_residual_dataframe
        df = _valid_corrected_df(24)
        df.loc[0, "period"] = "invalid"
        passed, errors = validate_residual_dataframe(df)
        assert not passed
        assert any("period" in e.lower() for e in errors)


class TestFileValidation:
    """Contract: file-based validation."""

    def test_missing_file_returns_error(self):
        """Missing file returns error, not crash."""
        from scripts.validate_residual_output import validate_residual_file
        passed, errors = validate_residual_file("/nonexistent/file.csv")
        assert not passed
        assert any("not found" in e.lower() for e in errors)

    def test_valid_csv_file_passes(self, tmp_path):
        """Valid CSV file passes validation."""
        from scripts.validate_residual_output import validate_residual_file
        csv_path = tmp_path / "corrected.csv"
        df = _valid_corrected_df(24)
        df.to_csv(str(csv_path), index=False)
        passed, errors = validate_residual_file(str(csv_path))
        assert passed, f"Expected pass, got errors: {errors}"
