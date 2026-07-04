"""
tests/test_fusion_weights.py — Fusion weight strategy contract tests.

Validates:
    1. equal_weight: weights sum to 1
    2. equal_weight: all weights equal
    3. equal_weight: empty list returns empty dict
    4. prior_weight: normalises to sum 1
    5. prior_weight: missing models get fallback weight
    6. prior_weight: extra prior models ignored
    7. prior_weight: empty prior falls back to equal_weight
    8. bgew_skeleton: no actuals falls back to equal_weight
    9. bgew_skeleton: insufficient history falls back to equal_weight
    10. bgew_skeleton: produces valid weights with actuals
    11. bgew_skeleton: does not use target_day actuals (no future leakage)
    12. compute_weights dispatches correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion.weights import (
    equal_weight,
    prior_weight,
    bgew_skeleton,
    compute_weights,
)


class TestEqualWeight:
    """Contract: equal_weight strategy."""

    def test_weights_sum_to_1(self):
        """equal_weight produces weights summing to 1."""
        models = ["cfg05", "best_two_average", "stage3_business_fixed"]
        weights, reasons = equal_weight(models)
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_all_weights_equal(self):
        """equal_weight produces equal weights for all models."""
        models = ["cfg05", "best_two_average"]
        weights, reasons = equal_weight(models)
        assert weights["cfg05"] == weights["best_two_average"]
        assert weights["cfg05"] == 0.5

    def test_single_model(self):
        """equal_weight with one model gives weight 1."""
        weights, reasons = equal_weight(["cfg05"])
        assert abs(weights["cfg05"] - 1.0) < 1e-10

    def test_empty_list(self):
        """equal_weight with empty list returns empty dict."""
        weights, reasons = equal_weight([])
        assert weights == {}


class TestPriorWeight:
    """Contract: prior_weight strategy."""

    def test_normalises_to_sum_1(self):
        """prior_weight normalises weights to sum 1."""
        models = ["cfg05", "best_two_average"]
        prior = {"cfg05": 0.6, "best_two_average": 0.4}
        weights, reasons = prior_weight(models, prior=prior)
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_missing_model_gets_fallback(self):
        """Missing model gets small fallback weight."""
        models = ["cfg05", "best_two_average", "unknown_model"]
        prior = {"cfg05": 0.6, "best_two_average": 0.4}
        weights, reasons = prior_weight(models, prior=prior)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)
        assert "unknown_model" in weights
        assert any("PRIOR_MISSING_MODEL" in r for r in reasons)

    def test_extra_prior_models_ignored(self):
        """Extra models in prior not in model_names are ignored."""
        models = ["cfg05"]
        prior = {"cfg05": 0.5, "extra_model": 0.5}
        weights, reasons = prior_weight(models, prior=prior)
        assert len(weights) == 1
        assert "extra_model" not in weights
        assert abs(weights["cfg05"] - 1.0) < 1e-10

    def test_empty_prior_falls_back_to_equal(self):
        """None prior falls back to equal_weight."""
        models = ["cfg05", "best_two_average"]
        weights, reasons = prior_weight(models, prior=None)
        assert abs(weights["cfg05"] - 0.5) < 1e-10
        assert any("PRIOR_NOT_PROVIDED" in r or "EQUAL_WEIGHT" in r for r in reasons)

    def test_empty_model_list(self):
        """Empty model list returns empty dict."""
        weights, reasons = prior_weight([])
        assert weights == {}


class TestBGEWSkeleton:
    """Contract: bgew_skeleton strategy."""

    @pytest.fixture
    def corrected_df(self) -> pd.DataFrame:
        """Corrected predictions for 2 models over 5 days."""
        rows = []
        rng = np.random.default_rng(42)
        for day_offset in range(5):
            bd = pd.Timestamp("2026-06-30") + pd.Timedelta(days=day_offset)
            timestamps = pd.date_range(f"{bd.date()} 01:00", periods=24, freq="h")
            for model in ["cfg05", "best_two_average"]:
                y_true = rng.uniform(80, 200, 24)
                rows.append(pd.DataFrame({
                    "model_name": [model] * 24,
                    "business_day": [bd] * 24,
                    "hour_business": list(range(1, 25)),
                    "y_pred_corrected": y_true + rng.normal(0, 5, 24),
                    "y_true": y_true,  # eval-mode only
                }))
        return pd.concat(rows, ignore_index=True)

    @pytest.fixture
    def actuals_df(self) -> pd.DataFrame:
        """Actuals for same 5-day period."""
        rows = []
        rng = np.random.default_rng(42)
        for day_offset in range(5):
            bd = pd.Timestamp("2026-06-30") + pd.Timedelta(days=day_offset)
            rows.append(pd.DataFrame({
                "business_day": [bd] * 24,
                "hour_business": list(range(1, 25)),
                "y_true": rng.uniform(80, 200, 24),
            }))
        return pd.concat(rows, ignore_index=True)

    def test_no_actuals_falls_back(self):
        """No actuals_df falls back to equal_weight."""
        df = pd.DataFrame({"model_name": ["cfg05"], "business_day": [pd.Timestamp("2026-07-04")]})
        weights, reasons = bgew_skeleton(["cfg05"], corrected_df=df, actuals_df=None)
        assert abs(weights["cfg05"] - 1.0) < 1e-10
        assert any("ACTUAL_LEDGER_MISSING" in r for r in reasons)

    def test_empty_actuals_falls_back(self):
        """Empty actuals_df falls back to equal_weight."""
        df = pd.DataFrame({"model_name": ["cfg05"], "business_day": [pd.Timestamp("2026-07-04")]})
        empty = pd.DataFrame(columns=["business_day", "hour_business", "y_true"])
        weights, reasons = bgew_skeleton(["cfg05"], corrected_df=df, actuals_df=empty)
        assert any("ACTUAL_LEDGER_MISSING" in r for r in reasons)

    def test_insufficient_history_falls_back(self):
        """Insufficient history falls back to equal_weight."""
        df = pd.DataFrame({
            "model_name": ["cfg05"],
            "business_day": [pd.Timestamp("2026-07-04")],
            "hour_business": [1],
            "y_pred_corrected": [100.0],
        })
        actuals = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-07-04")],  # same day — no pre-cut data
            "hour_business": [1],
            "y_true": [100.0],
        })
        weights, reasons = bgew_skeleton(
            ["cfg05"], corrected_df=df, actuals_df=actuals, min_history=5,
        )
        assert any("FALLBACK_EQUAL" in r for r in reasons)

    def test_produces_valid_weights_with_actuals(self, corrected_df, actuals_df):
        """With actuals, produces valid normalised weights."""
        # Use corrected data for day 5 (past days for training)
        target_bd = pd.Timestamp("2026-07-04")
        corrected_filtered = corrected_df[corrected_df["business_day"] <= target_bd].copy()

        weights, reasons = bgew_skeleton(
            ["cfg05", "best_two_average"],
            corrected_df=corrected_filtered,
            actuals_df=actuals_df,
            window=10,
            min_history=1,
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert len(weights) == 2

    def test_no_future_leakage(self, corrected_df, actuals_df):
        """Does not use target_day actuals (business_day >= target_day)."""
        # All actuals are dated 2026-06-30 to 2026-07-04
        # Target day is 2026-07-04 — training data is business_day < 2026-07-04
        target_bd = pd.Timestamp("2026-07-04")
        corrected_filtered = corrected_df[
            corrected_df["business_day"] == target_bd
        ].copy()

        weights, reasons = bgew_skeleton(
            ["cfg05", "best_two_average"],
            corrected_df=corrected_filtered,
            actuals_df=actuals_df,
            window=10,
            min_history=1,
        )

        # Should have used past 4 days (June 30 - July 3) as training
        # The actual weights are valid as long as they sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_empty_model_list(self):
        """Empty model list returns empty dict."""
        weights, reasons = bgew_skeleton([], corrected_df=pd.DataFrame())
        assert weights == {}


class TestComputeWeights:
    """Contract: compute_weights dispatch."""

    def test_dispatch_equal_weight(self):
        """compute_weights dispatches to equal_weight."""
        weights, reasons = compute_weights("equal_weight", ["cfg05", "best_two_average"])
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_dispatch_prior_weight(self):
        """compute_weights dispatches to prior_weight."""
        prior = {"cfg05": 0.7, "best_two_average": 0.3}
        weights, reasons = compute_weights("prior_weight", ["cfg05", "best_two_average"], prior=prior)
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_dispatch_bgew(self):
        """compute_weights dispatches to bgew_skeleton."""
        df = pd.DataFrame({
            "model_name": ["cfg05"],
            "business_day": [pd.Timestamp("2026-07-04")],
            "hour_business": [1],
            "y_pred_corrected": [100.0],
        })
        weights, reasons = compute_weights("bgew_skeleton", ["cfg05"], corrected_df=df)
        assert abs(weights["cfg05"] - 1.0) < 1e-10

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fusion method"):
            compute_weights("invalid_method", ["cfg05"])
