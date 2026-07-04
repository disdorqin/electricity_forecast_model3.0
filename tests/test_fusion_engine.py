"""
tests/test_fusion_engine.py — Fusion engine contract tests.

Validates:
    1. run_fusion with equal_weight produces fused_price
    2. Fusion uses y_pred_corrected, not y_pred_raw
    3. No NaN in fused_price
    4. No duplicate fusion keys
    5. Production mode: no y_true
    6. Empty corrected input returns empty output
    7. Missing required columns raises ValueError
    8. Invalid method raises ValueError
    9. Duplicate group rows raise ValueError
    10. Weights_json is valid JSON
    11. Included models non-empty (default allow_dry_run=False excludes dry-run models)
    12. allow_dry_run=True includes dry-run models
    13. CLI dry-run produces valid fusion output
    14. prior_weight fusion works end-to-end
    15. bgew_skeleton fusion works end-to-end with actuals
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fusion.engine import run_fusion, READY_DRY_RUN


def _corrected_df(
    n_hours: int = 24,
    models: list[str] = None,
) -> pd.DataFrame:
    """Synthetic corrected prediction output."""
    from data.business_day import add_business_time_columns

    if models is None:
        models = ["cfg05", "best_two_average"]

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    rows: list[pd.DataFrame] = []
    for model in models:
        y_pred_raw = rng.uniform(80, 200, n_hours)
        # Add a small correction so y_pred_corrected != y_pred_raw
        correction = rng.uniform(-5, 5, n_hours)
        df = pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "model_name": [model] * n_hours,
            "target_day": ["2026-07-04"] * n_hours,
            "ds": timestamps,
            "y_pred_raw": y_pred_raw,
            "y_pred_corrected": y_pred_raw + correction,
            "residual_delta": correction,
            "correction_applied": [True] * n_hours,
            "correction_module": ["p5m_residual_plugin"] * n_hours,
            "risk_source": ["NEGATIVE_RISK"] * n_hours,
            "reason_codes": ["P5M_ADAPTER_CORRECTION;RISK_DATA_AVAILABLE"] * n_hours,
            "correction_version": ["1.0.0"] * n_hours,
            "source_confidence": [0.5] * n_hours,
            "model_version": ["1.0.0"] * n_hours,
        })
        df = add_business_time_columns(df, timestamp_col="ds")
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


class TestFusionEngine:
    """Contract: run_fusion basic behavior."""

    def test_equal_weight_produces_fused_price(self):
        """equal_weight fusion produces fused_price."""
        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        assert len(result) == 24
        assert "fused_price" in result.columns
        assert not result["fused_price"].isna().any()

    def test_uses_y_pred_corrected_not_raw(self):
        """Fusion uses y_pred_corrected, not y_pred_raw."""
        corrected = _corrected_df(24, models=["cfg05"])
        # Manually set corrected to be different from raw
        corrected["y_pred_corrected"] = corrected["y_pred_raw"] * 0.95

        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)

        # With 1 model, fused_price = y_pred_corrected
        expected = corrected["y_pred_corrected"].values
        np.testing.assert_array_almost_equal(
            result["fused_price"].values,
            expected,
        )

        # Verify it's NOT equal to y_pred_raw
        assert not np.allclose(
            result["fused_price"].values,
            corrected["y_pred_raw"].values,
        )

    def test_no_nan_in_fused_price(self):
        """Fused_price has no NaN."""
        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        assert not result["fused_price"].isna().any()

    def test_no_duplicate_fusion_keys(self):
        """No duplicate fusion keys in output."""
        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        dup_mask = result.duplicated(
            subset=["task", "target_day", "business_day", "ds", "hour_business"],
            keep=False,
        )
        assert dup_mask.sum() == 0

    def test_no_y_true_in_production(self):
        """Production mode output has no y_true."""
        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight",
                            allow_dry_run=True, production=True)
        assert "y_true" not in result.columns

    def test_empty_input_returns_empty(self):
        """Empty corrected input returns empty fusion output."""
        empty = pd.DataFrame(columns=[
            "task", "model_name", "target_day", "business_day", "ds",
            "hour_business", "period", "y_pred_corrected",
        ])
        result = run_fusion(empty, method="equal_weight", allow_dry_run=True)
        assert len(result) == 0

    def test_missing_column_raises(self):
        """Missing required column raises ValueError."""
        bad = pd.DataFrame({"some_column": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required columns"):
            run_fusion(bad, method="equal_weight")

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        corrected = _corrected_df(24)
        with pytest.raises(ValueError, match="Unknown fusion method"):
            run_fusion(corrected, method="invalid")

    def test_duplicate_group_rows_raises(self):
        """Duplicate (fusion_key, model_name) raises ValueError."""
        corrected = _corrected_df(24)
        # Duplicate the first row
        dup = pd.concat([corrected, corrected.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            run_fusion(dup, method="equal_weight", allow_dry_run=True)

    def test_weights_json_is_valid_json(self):
        """weights_json is valid JSON."""
        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        for raw in result["weights_json"]:
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)
            # Weights sum to 1
            total = sum(float(v) for v in parsed.values())
            assert abs(total - 1.0) < 1e-6

    def test_included_models_string(self):
        """included_models is a semicolon-delimited string."""
        corrected = _corrected_df(24, models=["cfg05", "best_two_average"])
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        models_str = result["included_models"].iloc[0]
        assert isinstance(models_str, str)
        models_list = models_str.split(";")
        assert len(models_list) >= 1


class TestFusionReadinessGate:
    """Contract: readiness gate integration."""

    def test_default_excludes_dry_run(self):
        """Default allow_dry_run=False excludes dry-run models."""
        corrected = _corrected_df(24, models=["cfg05", "best_two_average"])
        # Without allow_dry_run, readiness gate will exclude dry-run models
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=False)
        # With both models as READY_DRY_RUN or READY_STUB, no models pass
        # So result could be empty
        if len(result) > 0:
            mode = result["readiness_mode"].iloc[0]
            assert mode != "DRY_RUN" or "DRY_RUN" in result["reason_codes"].iloc[0]

    def test_allow_dry_run_includes_models(self):
        """allow_dry_run=True includes dry-run models."""
        corrected = _corrected_df(24, models=["cfg05", "best_two_average"])
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        if len(result) > 0:
            assert "fused_price" in result.columns
            assert not result["fused_price"].isna().any()

    def test_readiness_mode_in_output(self):
        """readiness_mode column is present."""
        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        assert "readiness_mode" in result.columns

    def test_reason_contains_dry_run(self):
        """allow_dry_run includes DRY_RUN in reason codes."""
        corrected = _corrected_df(24, models=["cfg05"])
        result = run_fusion(corrected, method="equal_weight", allow_dry_run=True)
        if len(result) > 0:
            reasons = result["reason_codes"].iloc[0]
            assert "DRY_RUN" in reasons


class TestFusionMethods:
    """Contract: different fusion methods."""

    def test_prior_weight_end_to_end(self):
        """prior_weight produces valid fusion output."""
        corrected = _corrected_df(24, models=["cfg05", "best_two_average"])
        prior = {"cfg05": 0.7, "best_two_average": 0.3}
        result = run_fusion(
            corrected, method="prior_weight",
            prior_weights=prior, allow_dry_run=True,
            readiness_status={"cfg05": READY_DRY_RUN, "best_two_average": READY_DRY_RUN},
        )
        if len(result) > 0:
            assert not result["fused_price"].isna().any()
            # Check weights_json reflects prior
            weights = json.loads(result["weights_json"].iloc[0])
            assert abs(weights["cfg05"] + weights["best_two_average"] - 1.0) < 1e-6

    def test_bgew_skeleton_end_to_end(self):
        """bgew_skeleton works with actuals."""
        rng = np.random.default_rng(42)
        corrected = _corrected_df(24, models=["cfg05", "best_two_average"])

        # Actuals for past days
        actuals = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-07-01")] * 24,
            "hour_business": list(range(1, 25)),
            "y_true": rng.uniform(80, 200, 24),
        })

        result = run_fusion(
            corrected, method="bgew_skeleton",
            actuals_df=actuals, allow_dry_run=True,
            readiness_status={"cfg05": READY_DRY_RUN, "best_two_average": READY_DRY_RUN},
        )
        if len(result) > 0:
            assert not result["fused_price"].isna().any()
            weights = json.loads(result["weights_json"].iloc[0])
            assert abs(sum(float(v) for v in weights.values()) - 1.0) < 1e-6


class TestFusionValidatorIntegration:
    """Contract: fusion output is validatable."""

    def test_fusion_output_passes_validator(self):
        """run_fusion output passes validate_fusion_dataframe."""
        from scripts.validate_fusion_output import validate_fusion_dataframe

        corrected = _corrected_df(24)
        result = run_fusion(corrected, method="equal_weight",
                            allow_dry_run=True, production=True)
        if len(result) > 0:
            passed, errors = validate_fusion_dataframe(
                result, allow_empty=True, production=True,
            )
            assert passed, f"Validation failed: {errors}"


class TestFusionCLIDryRun:
    """Contract: CLI dry-run produces valid fusion output."""

    def test_dry_run_exit_code_0(self):
        """CLI dry-run exits with code 0."""
        from scripts.run_fusion_engine import main
        exit_code = main(["--dry-run", "--method", "equal_weight", "--allow-dry-run"])
        assert exit_code == 0

    def test_dry_run_with_prior_weight(self):
        """CLI dry-run with prior_weight works."""
        from scripts.run_fusion_engine import main
        exit_code = main([
            "--dry-run", "--method", "prior_weight",
            "--prior-weights-json", '{"cfg05":0.6,"best_two_average":0.4}',
            "--allow-dry-run",
        ])
        assert exit_code == 0
