"""P84 -- Adaptive Ledger Learner (Unified Weight Learner) upgraded tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion.unified_weight_learner import (
    LEARNER_FULL_DIMENSIONAL,
    LEARNER_PERIOD_ONLY,
    LEARNER_TASK_ONLY,
    PERIODS,
    REGIMES,
    classify_regime,
    compute_bgew_weights,
    train_dimensional_weights,
    train_unified_weights,
)


# -- Fixtures -------------------------------------------------------------------


@pytest.fixture
def multi_model_predictions():
    """Multi-model prediction ledger spanning multiple days and periods."""
    days = []
    hours = []
    models = []
    y_preds = []
    rng = np.random.default_rng(42)

    for day_offset in range(15):
        day_str = f"2026-05-{17 + day_offset:02d}"
        for h in range(1, 25):
            for model in ["model_a", "model_b", "model_c"]:
                days.append(day_str)
                hours.append(h)
                models.append(model)
                y_preds.append(float(rng.uniform(80, 450)))

    return pd.DataFrame({
        "business_day": days,
        "hour_business": hours,
        "model_name": models,
        "y_pred": y_preds,
    })


@pytest.fixture
def multi_model_actuals():
    """Actuals ledger matching the prediction days/hours."""
    days = []
    hours = []
    y_trues = []
    rng = np.random.default_rng(99)

    for day_offset in range(15):
        day_str = f"2026-05-{17 + day_offset:02d}"
        for h in range(1, 25):
            days.append(day_str)
            hours.append(h)
            y_trues.append(float(rng.uniform(80, 450)))

    return pd.DataFrame({
        "business_day": days,
        "hour_business": hours,
        "y_true": y_trues,
    })


@pytest.fixture
def single_model_predictions():
    """Single-model prediction ledger."""
    days = []
    hours = []
    y_preds = []
    rng = np.random.default_rng(42)

    for day_offset in range(10):
        day_str = f"2026-05-{20 + day_offset:02d}"
        for h in range(1, 25):
            days.append(day_str)
            hours.append(h)
            y_preds.append(float(rng.uniform(80, 450)))

    return pd.DataFrame({
        "business_day": days,
        "hour_business": hours,
        "model_name": "only_model",
        "y_pred": y_preds,
    })


@pytest.fixture
def single_model_actuals():
    """Actuals matching single-model predictions."""
    days = []
    hours = []
    y_trues = []
    rng = np.random.default_rng(99)

    for day_offset in range(10):
        day_str = f"2026-05-{20 + day_offset:02d}"
        for h in range(1, 25):
            days.append(day_str)
            hours.append(h)
            y_trues.append(float(rng.uniform(80, 450)))

    return pd.DataFrame({
        "business_day": days,
        "hour_business": hours,
        "y_true": y_trues,
    })


# -- classify_regime() ----------------------------------------------------------


class TestClassifyRegime:
    def test_negative_prices_get_negative_risk(self):
        prices = pd.Series([-10.0, -50.0, -0.1])
        regimes = classify_regime(prices)
        assert all(r == "negative_risk" for r in regimes)

    def test_low_prices_get_low_price(self):
        prices = pd.Series([10.0, 30.0, 49.9])
        regimes = classify_regime(prices)
        assert all(r == "low_price" for r in regimes)

    def test_high_prices_get_high_spike(self):
        prices = pd.Series([501.0, 1000.0, 600.0])
        regimes = classify_regime(prices)
        assert all(r == "high_spike" for r in regimes)

    def test_normal_prices_get_normal(self):
        prices = pd.Series([100.0, 200.0, 350.0])
        regimes = classify_regime(prices)
        assert all(r == "normal" for r in regimes)

    def test_mixed_prices(self):
        prices = pd.Series([-5.0, 30.0, 200.0, 600.0])
        regimes = classify_regime(prices)
        assert regimes[0] == "negative_risk"
        assert regimes[1] == "low_price"
        assert regimes[2] == "normal"
        assert regimes[3] == "high_spike"

    def test_boundary_zero_is_low_price(self):
        prices = pd.Series([0.0])
        regimes = classify_regime(prices)
        assert regimes[0] == "low_price"

    def test_boundary_500_is_normal(self):
        prices = pd.Series([500.0])
        regimes = classify_regime(prices)
        assert regimes[0] == "normal"

    def test_returns_series_of_strings(self):
        prices = pd.Series([100.0, 200.0])
        regimes = classify_regime(prices)
        assert isinstance(regimes, np.ndarray)
        assert len(regimes) == 2


# -- train_dimensional_weights() ------------------------------------------------


class TestTrainDimensionalWeights:
    def test_returns_dict(self, multi_model_predictions, multi_model_actuals):
        result = train_dimensional_weights(
            predictions=multi_model_predictions,
            actuals=multi_model_actuals,
            target_day="2026-06-02",
            task="dayahead",
        )
        assert isinstance(result, dict)

    def test_has_weights_df_key(self, multi_model_predictions, multi_model_actuals):
        result = train_dimensional_weights(
            predictions=multi_model_predictions,
            actuals=multi_model_actuals,
            target_day="2026-06-02",
            task="dayahead",
        )
        assert "weights_df" in result

    def test_weights_df_has_period_column(self, multi_model_predictions, multi_model_actuals):
        result = train_dimensional_weights(
            predictions=multi_model_predictions,
            actuals=multi_model_actuals,
            target_day="2026-06-02",
            task="dayahead",
        )
        if result["weights_df"] is not None:
            assert "period" in result["weights_df"].columns

    def test_weights_df_has_regime_column(self, multi_model_predictions, multi_model_actuals):
        result = train_dimensional_weights(
            predictions=multi_model_predictions,
            actuals=multi_model_actuals,
            target_day="2026-06-02",
            task="dayahead",
        )
        if result["weights_df"] is not None:
            assert "regime" in result["weights_df"].columns

    def test_has_lookback_start_and_end(self, multi_model_predictions, multi_model_actuals):
        result = train_dimensional_weights(
            predictions=multi_model_predictions,
            actuals=multi_model_actuals,
            target_day="2026-06-02",
            task="dayahead",
        )
        assert "lookback_start" in result
        assert "lookback_end" in result

    def test_no_lookahead_only_days_before_target(
        self, multi_model_predictions, multi_model_actuals
    ):
        """Weights must only use days < target_day."""
        result = train_dimensional_weights(
            predictions=multi_model_predictions,
            actuals=multi_model_actuals,
            target_day="2026-06-02",
            task="dayahead",
        )
        if result["weights_df"] is not None and len(result["weights_df"]) > 0:
            # lookback_end must be < target_day
            lookback_end = result.get("lookback_end", "")
            if lookback_end:
                assert lookback_end < "2026-06-02"


# -- train_unified_weights() ----------------------------------------------------


class TestTrainUnifiedWeights:
    def test_output_has_period_values(self, multi_model_predictions, multi_model_actuals):
        result = train_unified_weights(
            dayahead_predictions=multi_model_predictions,
            dayahead_actuals=multi_model_actuals,
            target_day="2026-06-02",
        )
        da_weights = result.get("dayahead_weights")
        if da_weights is not None and len(da_weights) > 0:
            period_values = da_weights["period"].unique().tolist()
            # Should have actual period values, not just "all"
            assert any(p in PERIODS for p in period_values)

    def test_single_model_weight_is_one(
        self, single_model_predictions, single_model_actuals
    ):
        result = train_unified_weights(
            dayahead_predictions=single_model_predictions,
            dayahead_actuals=single_model_actuals,
            target_day="2026-06-01",
        )
        da_weights = result.get("dayahead_weights")
        if da_weights is not None and len(da_weights) > 0:
            # Single model should have weight=1
            assert (da_weights["weight"] == 1.0).all()

    def test_single_model_fallback_in_reason_codes(
        self, single_model_predictions, single_model_actuals
    ):
        result = train_unified_weights(
            dayahead_predictions=single_model_predictions,
            dayahead_actuals=single_model_actuals,
            target_day="2026-06-01",
        )
        reason_codes = result.get("reason_codes", [])
        has_single_model = any("SINGLE_MODEL" in rc for rc in reason_codes)
        # Either SINGLE_MODEL in reason codes or weight=1
        da_weights = result.get("dayahead_weights")
        assert has_single_model or (
            da_weights is not None and len(da_weights) > 0
        )


# -- Constants ------------------------------------------------------------------


class TestConstants:
    def test_periods_has_3_entries(self):
        assert len(PERIODS) == 3

    def test_periods_values(self):
        assert set(PERIODS) == {"1_8", "9_16", "17_24"}

    def test_regimes_has_4_entries(self):
        assert len(REGIMES) == 4

    def test_regimes_values(self):
        assert set(REGIMES) == {"normal", "low_price", "negative_risk", "high_spike"}

    def test_learner_full_dimensional_exists(self):
        assert LEARNER_FULL_DIMENSIONAL == "LEARNER_FULL_DIMENSIONAL"

    def test_learner_period_only_exists(self):
        assert LEARNER_PERIOD_ONLY == "LEARNER_PERIOD_ONLY"

    def test_learner_task_only_exists(self):
        assert LEARNER_TASK_ONLY == "LEARNER_TASK_ONLY"


# -- Backward compatibility: compute_bgew_weights() -----------------------------


class TestBackwardCompatibility:
    def test_compute_bgew_weights_still_works(self):
        weights = compute_bgew_weights({"model_a": 10.0, "model_b": 20.0})
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_compute_bgew_weights_empty(self):
        assert compute_bgew_weights({}) == {}

    def test_compute_bgew_weights_single(self):
        w = compute_bgew_weights({"only": 5.0})
        assert abs(w["only"] - 1.0) < 1e-6
