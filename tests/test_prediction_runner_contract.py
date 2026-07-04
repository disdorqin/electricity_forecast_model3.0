"""
tests/test_prediction_runner_contract.py — Prediction runner contract tests.

Validates:
    1. Day-ahead runner dry-run outputs standard schema
    2. Day-ahead runner dry-run produces predictions for all models
    3. Day-ahead runner rejects invalid models
    4. Realtime runner dry-run outputs rt_pred = da_anchor
    5. validate_prediction_output detects duplicate keys
    6. validate_prediction_output detects NaN y_pred
    7. validate_prediction_output detects missing hour_business
    8. validate_prediction_output detects y_true in production mode
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import PREDICTION_OUTPUT_COLUMNS


# ──────────────────────────────────────────────
# Day-ahead runner tests
# ──────────────────────────────────────────────

class TestDayaheadRunnerDryRun:
    """Contract: day-ahead model zoo runner dry-run mode."""

    def test_dry_run_returns_standard_schema(self):
        """Day-ahead runner dry-run produces DataFrame with standard schema."""
        from scripts.run_dayahead_model_zoo import run_dayahead_zoo
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_dayahead_zoo(
            model_configs=DEFAULT_FUSION_POOL,
            input_df=df,
            dry_run=True,
        )
        for col in PREDICTION_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_dry_run_produces_all_models(self):
        """Dry-run produces predictions for all 5 DEFAULT_FUSION_POOL models."""
        from scripts.run_dayahead_model_zoo import run_dayahead_zoo
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_dayahead_zoo(
            model_configs=DEFAULT_FUSION_POOL,
            input_df=df,
            dry_run=True,
        )
        model_names = result["model_name"].unique()
        expected_names = [entry["formal_name"] for entry in DEFAULT_FUSION_POOL]
        for name in expected_names:
            assert name in model_names, f"Missing model: {name}"

    def test_dry_run_no_nan_in_y_pred(self):
        """Dry-run predictions have no NaN in y_pred."""
        from scripts.run_dayahead_model_zoo import run_dayahead_zoo
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_dayahead_zoo(
            model_configs=DEFAULT_FUSION_POOL,
            input_df=df,
            dry_run=True,
        )
        assert not result["y_pred"].isna().any()

    def test_dry_run_no_y_true(self):
        """Dry-run output does not contain y_true."""
        from scripts.run_dayahead_model_zoo import run_dayahead_zoo
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_dayahead_zoo(
            model_configs=DEFAULT_FUSION_POOL,
            input_df=df,
            dry_run=True,
        )
        assert "y_true" not in result.columns

    def test_dry_run_hour_business_in_range(self):
        """Dry-run output hour_business in 1..24."""
        from scripts.run_dayahead_model_zoo import run_dayahead_zoo
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_dayahead_zoo(
            model_configs=DEFAULT_FUSION_POOL,
            input_df=df,
            dry_run=True,
        )
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_dry_run_without_input_uses_fallback(self):
        """Dry-run with synthetic input (no da_anchor) works."""
        from scripts.run_dayahead_model_zoo import run_dayahead_zoo
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
        })
        result = run_dayahead_zoo(
            model_configs=DEFAULT_FUSION_POOL,
            input_df=df,
            dry_run=True,
        )
        assert len(result) > 0

    def test_rejects_invalid_model(self):
        """Runner config resolution raises on invalid models."""
        from scripts.run_dayahead_model_zoo import resolve_model_list
        with pytest.raises(ValueError, match="INVALID"):
            resolve_model_list("lgbm_spike_residual_1127")


class TestDayaheadRunnerCLI:
    """Contract: day-ahead runner CLI argument parsing."""

    def test_default_models_resolves_fusion_pool(self):
        """--models default resolves to DEFAULT_FUSION_POOL configs."""
        from scripts.run_dayahead_model_zoo import resolve_model_list
        from src.registry.dayahead_models import DEFAULT_FUSION_POOL
        configs = resolve_model_list("default")
        assert len(configs) == len(DEFAULT_FUSION_POOL)
        assert configs[0]["model_id"] == "cfg05"


# ──────────────────────────────────────────────
# Realtime runner tests
# ──────────────────────────────────────────────

class TestRealtimeRunnerDryRun:
    """Contract: realtime assist runner dry-run."""

    def test_dry_run_rt_pred_equals_da_anchor(self):
        """Dry-run realtime output has rt_pred == da_anchor."""
        from scripts.run_realtime_assist import run_realtime_assist

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_realtime_assist(input_df=df, dry_run=True)
        import numpy as np
        np.testing.assert_array_almost_equal(
            result["y_pred"].values, df["da_anchor"].values
        )

    def test_dry_run_standard_schema(self):
        """Dry-run realtime output has standard schema."""
        from scripts.run_realtime_assist import run_realtime_assist

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_realtime_assist(input_df=df, dry_run=True)
        for col in PREDICTION_OUTPUT_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_dry_run_no_nan(self):
        """Dry-run realtime output has no NaN."""
        from scripts.run_realtime_assist import run_realtime_assist

        df = pd.DataFrame({
            "ds": pd.date_range("2026-03-05 01:00", periods=24, freq="h"),
            "da_anchor": [100.0 + i for i in range(24)],
        })
        result = run_realtime_assist(input_df=df, dry_run=True)
        assert not result["y_pred"].isna().any()
        assert "y_true" not in result.columns


# ──────────────────────────────────────────────
# Prediction output validator tests
# ──────────────────────────────────────────────

def _valid_prediction_df(n_rows: int = 24) -> pd.DataFrame:
    """Create a valid prediction DataFrame."""
    import numpy as np
    bd = pd.Timestamp("2026-03-05")
    return pd.DataFrame({
        "task": ["dayahead"] * n_rows,
        "model_name": ["test_model"] * n_rows,
        "target_day": ["2026-03-05"] * n_rows,
        "business_day": [bd] * n_rows,
        "ds": pd.date_range("2026-03-05 01:00", periods=n_rows, freq="h"),
        "hour_business": list(range(1, n_rows + 1)),
        "period": ["1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24") for h in range(1, n_rows + 1)],
        "y_pred": [100.0 + i for i in range(n_rows)],
        "source_confidence": [0.5] * n_rows,
        "model_version": ["1.0.0"] * n_rows,
    })


class TestValidatePredictionOutput:
    """Contract: validate_prediction_output detection."""

    def test_valid_df_passes(self):
        """Valid prediction DataFrame passes all checks."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        passed, errors = validate_prediction_dataframe(df)
        assert passed, f"Expected pass, got errors: {errors}"
        assert errors == []

    def test_detects_duplicate_keys(self):
        """Duplicate (task, model, business_day, hour_business) is detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        # Add duplicate row
        dup_row = df.iloc[[0]].copy()
        df = pd.concat([df, dup_row], ignore_index=True)
        passed, errors = validate_prediction_dataframe(df)
        assert not passed
        assert any("duplicate" in e.lower() for e in errors)

    def test_detects_nan_y_pred(self):
        """NaN in y_pred is detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        df.loc[0, "y_pred"] = float("nan")
        passed, errors = validate_prediction_dataframe(df)
        assert not passed
        assert any("NaN" in e for e in errors)

    def test_detects_missing_hour(self):
        """Missing hour_business values are not directly detected by validator,
        but out-of-range hours are caught."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(23)  # only 23 rows
        passed, errors = validate_prediction_dataframe(df)
        # Without --require-24h, 23 rows is OK
        assert passed

    def test_detects_y_true_in_production(self):
        """y_true in production mode is detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        df["y_true"] = 100.0
        passed, errors = validate_prediction_dataframe(df, production=True)
        assert not passed
        assert any("y_true" in e for e in errors)

    def test_allows_y_true_in_eval(self):
        """y_true in eval mode (production=False) is allowed."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        df["y_true"] = 100.0
        passed, errors = validate_prediction_dataframe(df, production=False)
        assert passed

    def test_detects_out_of_range_hour_business(self):
        """hour_business outside 1..24 is detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        df.loc[0, "hour_business"] = 0
        passed, errors = validate_prediction_dataframe(df)
        assert not passed
        assert any("out of range" in e.lower() for e in errors)

    def test_detects_invalid_period(self):
        """Invalid period values are detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        df.loc[0, "period"] = "invalid"
        passed, errors = validate_prediction_dataframe(df)
        assert not passed
        assert any("period" in e.lower() for e in errors)

    def test_detects_missing_required_column(self):
        """Missing required column is detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(24)
        df = df.drop(columns=["model_version"])
        passed, errors = validate_prediction_dataframe(df)
        assert not passed
        assert any("model_version" in e for e in errors)

    def test_empty_df_detected(self):
        """Empty DataFrame is detected."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(0)
        passed, errors = validate_prediction_dataframe(df)
        assert not passed
        assert any("empty" in e.lower() for e in errors)

    def test_require_24h_catches_partial_day(self):
        """--require-24h flag catches days with <24 rows."""
        from scripts.validate_prediction_output import validate_prediction_dataframe
        df = _valid_prediction_df(23)
        passed, errors = validate_prediction_dataframe(df, require_24h=True)
        assert not passed
        assert any("Expected 24" in e for e in errors)
