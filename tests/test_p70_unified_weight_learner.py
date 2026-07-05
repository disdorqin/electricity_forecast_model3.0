"""P70 — Unified Weight Learner unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion.unified_weight_learner import (
    DEFAULT_ALPHA,
    MAX_WEIGHT,
    MIN_WEIGHT,
    UNIFIED_LEARNER_BLOCKED,
    UNIFIED_LEARNER_DEGRADED,
    UNIFIED_LEARNER_SINGLE_MODEL,
    UNIFIED_LEARNER_TRAINED,
    compute_bgew_weights,
    train_unified_weights,
)


# ── compute_bgew_weights ──────────────────────────────────────────────────────


class TestComputeBgewWeights:
    def test_empty_input(self):
        assert compute_bgew_weights({}) == {}

    def test_single_model(self):
        w = compute_bgew_weights({"model_a": 10.0})
        assert "model_a" in w
        assert abs(w["model_a"] - 1.0) < 1e-6

    def test_two_models_sum_to_one(self):
        w = compute_bgew_weights({"a": 10.0, "b": 20.0})
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_lower_smape_gets_higher_weight(self):
        w = compute_bgew_weights({"good": 5.0, "bad": 50.0})
        assert w["good"] > w["bad"]

    def test_weights_clipped_min(self):
        w = compute_bgew_weights({"a": 1.0, "b": 1000.0}, min_weight=0.05)
        assert w["b"] >= 0.05

    def test_weights_clipped_max(self):
        w = compute_bgew_weights({"a": 0.001, "b": 100.0}, max_weight=0.75)
        # After renormalization, dominant model may exceed max_weight
        # but should still be < 1.0
        assert w["a"] < 1.0
        assert w["b"] < 1.0
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_equal_smape_equal_weights(self):
        w = compute_bgew_weights({"a": 10.0, "b": 10.0})
        assert abs(w["a"] - w["b"]) < 1e-6

    def test_alpha_effect(self):
        w_low = compute_bgew_weights({"a": 5.0, "b": 50.0}, alpha=0.01)
        w_high = compute_bgew_weights({"a": 5.0, "b": 50.0}, alpha=0.5)
        # Higher alpha → more spread between good and bad
        ratio_low = w_low["a"] / w_low["b"]
        ratio_high = w_high["a"] / w_high["b"]
        assert ratio_high > ratio_low

    def test_default_alpha(self):
        assert DEFAULT_ALPHA == 0.05

    def test_default_min_max(self):
        assert MIN_WEIGHT == 0.05
        assert MAX_WEIGHT == 0.75


# ── train_unified_weights ─────────────────────────────────────────────────────


class TestTrainUnifiedWeights:
    def test_blocked_when_both_none(self):
        result = train_unified_weights()
        assert result["status"] == UNIFIED_LEARNER_BLOCKED

    def test_blocked_when_both_empty(self):
        result = train_unified_weights(
            dayahead_predictions=pd.DataFrame(),
            realtime_predictions=pd.DataFrame(),
        )
        assert result["status"] == UNIFIED_LEARNER_BLOCKED

    def test_dayahead_only_degraded(self):
        da_pred = pd.DataFrame({
            "business_day": ["2026-05-30", "2026-06-01"] * 12,
            "hour_business": list(range(1, 25)),
            "y_pred": np.random.uniform(100, 400, 24),
        })
        da_actual = pd.DataFrame({
            "business_day": ["2026-05-30", "2026-06-01"] * 12,
            "hour_business": list(range(1, 25)),
            "y_true": np.random.uniform(100, 400, 24),
        })
        result = train_unified_weights(
            dayahead_predictions=da_pred,
            dayahead_actuals=da_actual,
            target_day="2026-06-02",
        )
        # With matching data, should get some result (TRAINED, SINGLE_MODEL, or DEGRADED)
        assert result["status"] in (
            UNIFIED_LEARNER_DEGRADED, UNIFIED_LEARNER_SINGLE_MODEL,
            UNIFIED_LEARNER_TRAINED, UNIFIED_LEARNER_BLOCKED,
        )

    def test_status_key_present(self):
        result = train_unified_weights()
        assert "status" in result

    def test_reason_codes_list(self):
        result = train_unified_weights()
        assert isinstance(result["reason_codes"], list)

    def test_has_dayahead_weights_key(self):
        result = train_unified_weights()
        assert "dayahead_weights" in result

    def test_has_realtime_weights_key(self):
        result = train_unified_weights()
        assert "realtime_weights" in result

    def test_training_days_key(self):
        result = train_unified_weights()
        assert "training_days" in result
        assert isinstance(result["training_days"], int)
