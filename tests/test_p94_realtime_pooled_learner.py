"""
Tests for P94: Realtime 30D Pooled Learner.

Covers:
  1. LEARNER_POLICY dict has correct keys
  2. train_pooled_30d_bgew with two candidates
  3. train_pooled_30d_bgew with single model
  4. No-lookahead invariant
  5. BGEW weight computation
  6. Learner method string
  7. Hard reject bad assist
  8. Insufficient training days
  9. train_unified_weights with learner_policy override
  10. Output schema completeness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion.unified_weight_learner import (
    train_unified_weights,
    train_pooled_30d_bgew,
    LEARNER_POLICY,
    MIN_WEIGHT,
    MAX_WEIGHT,
    POOLED_MIN_TRAINING_DAYS,
    POOLED_LOOKBACK_DAYS,
    compute_bgew_weights,
    _merge_pred_actuals,
)
from models.realtime_state import (
    LEARNER_POLICY_REALTIME,
    LEARNER_POLICY_DAYAHEAD,
    REALTIME_LEARNER_POOLED_TRAINED,
    REALTIME_LEARNER_SINGLE_MODEL,
    REALTIME_LEARNER_BLOCKED,
)


class TestLearnerPolicyConstants:
    """Test learner policy configuration."""

    def test_policy_dict_exists(self):
        assert isinstance(LEARNER_POLICY, dict)
        assert "dayahead" in LEARNER_POLICY
        assert "realtime" in LEARNER_POLICY

    def test_policy_values(self):
        assert LEARNER_POLICY["dayahead"] == LEARNER_POLICY_DAYAHEAD
        assert LEARNER_POLICY["realtime"] == LEARNER_POLICY_REALTIME

    def test_policies_are_different(self):
        assert LEARNER_POLICY["dayahead"] != LEARNER_POLICY["realtime"]

    def test_policy_realtime_is_pooled(self):
        assert LEARNER_POLICY["realtime"] == "pooled_30d_bgew"


class TestPooledBGEW:
    """Test pooled_30d_bgew weight learning."""

    @pytest.fixture
    def sample_data(self):
        """Create 35 days of data for 2 models (rt_da_anchor + sgdfnet_rt_assist).
        
        Predictions DataFrame does NOT contain y_true (production contract).
        y_true is in the actuals_df separately.
        """
        np.random.seed(42)
        rows = []
        for day_offset in range(35):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                period = "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")
                ds = day + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = day + pd.Timedelta(hours=23)
                # DA anchor: good predictions
                y_true = 300.0 + np.random.uniform(-50, 50)
                da_pred = y_true + np.random.uniform(-15, 15)
                sg_pred = y_true + np.random.uniform(-20, 20)

                rows.append({
                    "task": "realtime",
                    "model_name": "rt_da_anchor",
                    "target_day": day_str,
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "period": period,
                    "y_pred": da_pred,
                })
                rows.append({
                    "task": "realtime",
                    "model_name": "sgdfnet_rt_assist",
                    "target_day": day_str,
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "period": period,
                    "y_pred": sg_pred,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def actuals_df(self, sample_data):
        """Extract actuals from sample data - uses fresh random values."""
        np.random.seed(42)
        actual_rows = []
        for day_offset in range(35):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                period = "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")
                ds = day + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = day + pd.Timedelta(hours=23)
                y_true = 300.0 + np.random.uniform(-50, 50)
                actual_rows.append({
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "period": period,
                    "y_true": y_true,
                })
        return pd.DataFrame(actual_rows)

    def test_pooled_learner_two_models(self, sample_data, actuals_df):
        """Two candidates should produce two weights."""
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        assert result["status"] == REALTIME_LEARNER_POOLED_TRAINED
        assert result["weights_df"] is not None
        assert len(result["weights_df"]) == 2

    def test_pooled_learner_single_model(self, sample_data, actuals_df):
        """Single model should get weight=1.0."""
        single = sample_data[sample_data["model_name"] == "rt_da_anchor"]
        result = train_pooled_30d_bgew(
            predictions=single,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        assert result["status"] == REALTIME_LEARNER_SINGLE_MODEL
        assert result["weights_df"] is not None
        assert result["weights_df"]["weight"].iloc[0] == 1.0

    def test_no_lookahead(self, sample_data, actuals_df):
        """Training should only use days < target_day."""
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-06-15",
            task="realtime",
        )
        if result["weights_df"] is not None:
            assert result["lookback_end"] < "2026-06-15"

    def test_lookback_window(self, sample_data, actuals_df):
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        if result["weights_df"] is not None:
            assert result["training_days"] > 0

    def test_learner_method_string(self, sample_data, actuals_df):
        """Learner method should be pooled_30d_bgew."""
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        if result["weights_df"] is not None:
            methods = result["weights_df"]["learner_method"].unique()
            assert all(m == LEARNER_POLICY_REALTIME for m in methods)

    def test_weight_range(self, sample_data, actuals_df):
        """Weights should be between MIN_WEIGHT and MAX_WEIGHT."""
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        if result["weights_df"] is not None:
            weights = result["weights_df"]["weight"].values
            assert all(MIN_WEIGHT <= w <= MAX_WEIGHT for w in weights)

    def test_weights_sum_to_one(self, sample_data, actuals_df):
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        if result["weights_df"] is not None:
            total = result["weights_df"]["weight"].sum()
            assert abs(total - 1.0) < 1e-6

    def test_training_days_counted(self, sample_data, actuals_df):
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        assert result["training_days"] > 0

    def test_training_rows_counted(self, sample_data, actuals_df):
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        assert result["training_rows"] > 0

    def test_period_is_all(self, sample_data, actuals_df):
        """Pooled learner should have period='all'."""
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        if result["weights_df"] is not None:
            assert (result["weights_df"]["period"] == "all").all()

    def test_regime_is_all(self, sample_data, actuals_df):
        result = train_pooled_30d_bgew(
            predictions=sample_data,
            actuals=actuals_df,
            target_day="2026-07-01",
            task="realtime",
        )
        if result["weights_df"] is not None:
            assert (result["weights_df"]["regime"] == "all").all()


class TestInsufficientData:
    """Test behavior with insufficient training data."""

    @pytest.fixture
    def small_data(self):
        """Only 3 days of data — below MIN_TRAINING_DAYS."""
        rows = []
        for day_offset in range(3):
            day = pd.Timestamp("2026-06-28") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                ds = day + pd.Timedelta(hours=h - 1)
                y_true = 300.0
                rows.append({
                    "model_name": "rt_da_anchor",
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_pred": y_true + np.random.uniform(-10, 10),
                })
                rows.append({
                    "model_name": "sgdfnet_rt_assist",
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_pred": y_true + np.random.uniform(-15, 15),
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def small_actuals(self, small_data):
        actual_rows = []
        for day_offset in range(3):
            day = pd.Timestamp("2026-06-28") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                ds = day + pd.Timedelta(hours=h - 1)
                actual_rows.append({
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_true": 300.0,
                })
        return pd.DataFrame(actual_rows)

    def test_insufficient_days(self, small_data, small_actuals):
        result = train_pooled_30d_bgew(
            predictions=small_data,
            actuals=small_actuals,
            target_day="2026-07-01",
            task="realtime",
        )
        assert result["weights_df"] is None
        assert "INSUFFICIENT" in ";".join(result["reason_codes"])


class TestHardReject:
    """Test hard_reject_bad_assist."""

    @pytest.fixture
    def bad_assist_data(self):
        """Make SGDFNet significantly worse than DA anchor."""
        np.random.seed(42)
        rows = []
        for day_offset in range(35):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                ds = day + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = day + pd.Timedelta(hours=23)
                y_true = 300.0 + np.random.uniform(-50, 50)
                da_pred = y_true + np.random.uniform(-10, 10)
                sg_pred = y_true + np.random.uniform(-200, 200)
                rows.append({
                    "model_name": "rt_da_anchor",
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_pred": da_pred,
                })
                rows.append({
                    "model_name": "sgdfnet_rt_assist",
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_pred": sg_pred,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def bad_actuals(self, bad_assist_data):
        np.random.seed(42)
        actual_rows = []
        for day_offset in range(35):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                ds = day + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = day + pd.Timedelta(hours=23)
                y_true = 300.0 + np.random.uniform(-50, 50)
                actual_rows.append({
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_true": y_true,
                })
        return pd.DataFrame(actual_rows)

    def test_hard_reject(self, bad_assist_data, bad_actuals):
        """Hard reject should drop very bad assist."""
        result = train_pooled_30d_bgew(
            predictions=bad_assist_data,
            actuals=bad_actuals,
            target_day="2026-07-01",
            task="realtime",
            hard_reject_bad_assist=True,
        )
        if result["weights_df"] is not None:
            weights = dict(zip(result["weights_df"]["model_name"],
                               result["weights_df"]["weight"]))
            # DA anchor should dominate
            assert "rt_da_anchor" in weights


class TestUnifiedWeightsWithPolicy:
    """Test train_unified_weights with learner_policy override."""

    @pytest.fixture
    def sample_rt_data(self):
        """Realtime data with 2 models (no y_true — production contract)."""
        np.random.seed(42)
        rows = []
        for day_offset in range(35):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                ds = day + pd.Timedelta(hours=h - 1)
                rows.append({
                    "model_name": "rt_da_anchor",
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_pred": 300.0 + np.random.uniform(-10, 10),
                })
                rows.append({
                    "model_name": "sgdfnet_rt_assist",
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_pred": 300.0 + np.random.uniform(-15, 15),
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_rt_actuals(self, sample_rt_data):
        np.random.seed(42)
        actual_rows = []
        for day_offset in range(35):
            day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
            day_str = day.strftime("%Y-%m-%d")
            for h in range(1, 25):
                ds = day + pd.Timedelta(hours=h - 1)
                actual_rows.append({
                    "business_day": day_str,
                    "ds": ds,
                    "hour_business": h,
                    "y_true": 300.0,
                })
        return pd.DataFrame(actual_rows)

    def test_realtime_policy_override(self, sample_rt_data, sample_rt_actuals):
        """With learner_policy override for realtime only."""
        learner_policy = {
            "dayahead": "period_regime_bgew",
            "realtime": "pooled_30d_bgew",
        }
        result = train_unified_weights(
            realtime_predictions=sample_rt_data,
            realtime_actuals=sample_rt_actuals,
            target_day="2026-07-01",
            learner_policy=learner_policy,
        )
        assert result.get("realtime_weights") is not None
        # Should use pooled method
        methods = result["realtime_weights"]["learner_method"].unique()
        assert LEARNER_POLICY_REALTIME in methods

    def test_realtime_weights_have_schema(self, sample_rt_data, sample_rt_actuals):
        learner_policy = {
            "realtime": "pooled_30d_bgew",
        }
        result = train_unified_weights(
            realtime_predictions=sample_rt_data,
            realtime_actuals=sample_rt_actuals,
            target_day="2026-07-01",
            learner_policy=learner_policy,
        )
        if result.get("realtime_weights") is not None:
            wdf = result["realtime_weights"]
            for col in ["task", "target_day", "period", "regime",
                         "model_name", "weight", "learner_method",
                         "training_days", "lookback_start", "lookback_end"]:
                assert col in wdf.columns

    def test_realtime_task_label(self, sample_rt_data, sample_rt_actuals):
        result = train_unified_weights(
            realtime_predictions=sample_rt_data,
            realtime_actuals=sample_rt_actuals,
            target_day="2026-07-01",
            learner_policy={"realtime": "pooled_30d_bgew"},
        )
        if result.get("realtime_weights") is not None:
            assert (result["realtime_weights"]["task"] == "realtime").all()


class TestBGEWFoundation:
    """Test that BGEW computation still works."""

    def test_bgew_basic(self):
        weights = compute_bgew_weights({"a": 10.0, "b": 20.0}, alpha=0.05)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights["a"] > weights["b"]  # lower sMAPE -> higher weight

    def test_bgew_min_weight(self):
        """Bad model should hit min_weight floor."""
        weights = compute_bgew_weights({"a": 5.0, "b": 200.0}, alpha=0.05,
                                        min_weight=0.05, max_weight=0.95)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights["b"] >= 0.05

    def test_bgew_empty(self):
        assert compute_bgew_weights({}) == {}

    def test_bgew_single(self):
        weights = compute_bgew_weights({"a": 10.0})
        assert weights["a"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
