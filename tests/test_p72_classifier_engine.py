"""P72 — Final Classifier Engine unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from classifiers import (
    CLASSIFIER_BLOCKED,
    CLASSIFIER_ML_READY,
    CLASSIFIER_RULE_FALLBACK,
    HIGH_SPIKE_THRESHOLD,
    NEGATIVE_PRICE_THRESHOLD,
    run_final_classifier,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def normal_prices():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "dayahead_price": [200.0] * 24,
    })


@pytest.fixture
def mixed_prices():
    prices = [200.0] * 20 + [-10.0, 600.0, 350.0, 50.0]
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "dayahead_price": prices,
    })


@pytest.fixture
def rt_normal():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "realtime_price": [250.0] * 24,
    })


# ── run_final_classifier ──────────────────────────────────────────────────────


class TestRunFinalClassifier:
    def test_both_none(self):
        result = run_final_classifier()
        assert result["dayahead"]["status"] == "NOT_RUN"
        assert result["realtime"]["status"] == "NOT_RUN"

    def test_dayahead_only(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        assert result["dayahead"]["status"] == "CLASSIFIED"
        assert result["realtime"]["status"] == "NOT_RUN"

    def test_realtime_only(self, rt_normal):
        result = run_final_classifier(realtime_fused=rt_normal)
        assert result["dayahead"]["status"] == "NOT_RUN"
        assert result["realtime"]["status"] == "CLASSIFIED"

    def test_both_classified(self, normal_prices, rt_normal):
        result = run_final_classifier(
            dayahead_fused=normal_prices, realtime_fused=rt_normal
        )
        assert result["dayahead"]["status"] == "CLASSIFIED"
        assert result["realtime"]["status"] == "CLASSIFIED"

    def test_rule_fallback_status(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        assert result["classifier_status"] == CLASSIFIER_RULE_FALLBACK

    def test_reason_codes_list(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        assert isinstance(result["reason_codes"], list)

    def test_normal_prices_classified_as_normal(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        output = result["dayahead"]["output"]
        assert (output["classifier_action"] == "NORMAL").all()

    def test_negative_detected(self, mixed_prices):
        result = run_final_classifier(dayahead_fused=mixed_prices)
        output = result["dayahead"]["output"]
        neg_rows = output[output["classifier_action"] == "NEGATIVE_DETECTED"]
        assert len(neg_rows) > 0

    def test_spike_detected(self, mixed_prices):
        result = run_final_classifier(dayahead_fused=mixed_prices)
        output = result["dayahead"]["output"]
        spike_rows = output[output["classifier_action"] == "SPIKE_DETECTED"]
        assert len(spike_rows) > 0

    def test_negative_risk_column(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        output = result["dayahead"]["output"]
        assert "negative_risk" in output.columns

    def test_spike_risk_column(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        output = result["dayahead"]["output"]
        assert "spike_risk" in output.columns

    def test_uncertainty_score_column(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        output = result["dayahead"]["output"]
        assert "uncertainty_score" in output.columns

    def test_delivery_warning_level(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        output = result["dayahead"]["output"]
        assert "delivery_warning_level" in output.columns

    def test_normal_trend_flag(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        output = result["dayahead"]["output"]
        assert "normal_trend_flag" in output.columns
        # 200 is in [0, 500]
        assert (output["normal_trend_flag"] == 1).all()

    def test_rows_count(self, normal_prices):
        result = run_final_classifier(dayahead_fused=normal_prices)
        assert result["dayahead"]["rows"] == 24


# ── Threshold Constants ──────────────────────────────────────────────────────


class TestThresholds:
    def test_negative_threshold(self):
        assert NEGATIVE_PRICE_THRESHOLD == 0.0

    def test_spike_threshold(self):
        assert HIGH_SPIKE_THRESHOLD == 500.0


# ── Edge Cases ────────────────────────────────────────────────────────────────


class TestClassifierEdgeCases:
    def test_no_price_column(self):
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        result = run_final_classifier(dayahead_fused=df)
        output = result["dayahead"]["output"]
        # Should handle gracefully — UNKNOWN or BLOCKED
        assert "classifier_action" in output.columns

    def test_empty_dataframe(self):
        result = run_final_classifier(dayahead_fused=pd.DataFrame())
        assert result["dayahead"]["status"] == "NOT_RUN"

    def test_fallback_column_resolution(self):
        """If dayahead_price missing, should try y_pred_corrected, y_pred, trend_pred."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "trend_pred": [200.0] * 24,
        })
        result = run_final_classifier(dayahead_fused=df)
        output = result["dayahead"]["output"]
        assert "classifier_action" in output.columns
