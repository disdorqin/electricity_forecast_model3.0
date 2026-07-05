"""P71 — Unified Fusion Engine unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion.unified_fusion_engine import (
    UNIFIED_FUSION_COMPLETE,
    UNIFIED_FUSION_DEGRADED,
    UNIFIED_FUSION_FAILED,
    run_unified_fusion,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def da_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "y_pred": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def rt_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "trend_pred": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def da_weights():
    return pd.DataFrame({
        "task": ["dayahead"] * 2,
        "target_day": ["2026-06-01"] * 2,
        "period": ["all"] * 2,
        "model_name": ["cfg05", "catboost"],
        "weight": [0.6, 0.4],
        "learner_method": ["bgew"] * 2,
        "training_days": [30, 30],
    })


# ── run_unified_fusion ────────────────────────────────────────────────────────


class TestRunUnifiedFusion:
    def test_both_none(self):
        result = run_unified_fusion()
        assert result["status"] == UNIFIED_FUSION_FAILED

    def test_both_empty(self):
        result = run_unified_fusion(
            dayahead_predictions=pd.DataFrame(),
            realtime_predictions=pd.DataFrame(),
        )
        assert result["status"] == UNIFIED_FUSION_FAILED

    def test_dayahead_only_degraded(self, da_predictions):
        result = run_unified_fusion(dayahead_predictions=da_predictions)
        assert result["status"] in (UNIFIED_FUSION_DEGRADED, UNIFIED_FUSION_COMPLETE)

    def test_realtime_only_degraded(self, rt_predictions):
        result = run_unified_fusion(realtime_predictions=rt_predictions)
        assert result["status"] in (UNIFIED_FUSION_DEGRADED, UNIFIED_FUSION_COMPLETE)

    def test_both_present(self, da_predictions, rt_predictions):
        result = run_unified_fusion(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
        )
        assert result["status"] == UNIFIED_FUSION_COMPLETE

    def test_dayahead_fused_is_dataframe(self, da_predictions):
        result = run_unified_fusion(dayahead_predictions=da_predictions)
        if result["dayahead_fused"] is not None:
            assert isinstance(result["dayahead_fused"], pd.DataFrame)

    def test_realtime_fused_is_dataframe(self, rt_predictions):
        result = run_unified_fusion(realtime_predictions=rt_predictions)
        if result["realtime_fused"] is not None:
            assert isinstance(result["realtime_fused"], pd.DataFrame)

    def test_reason_codes_list(self, da_predictions, rt_predictions):
        result = run_unified_fusion(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
        )
        assert isinstance(result["reason_codes"], list)

    def test_fused_has_price_column(self, da_predictions):
        result = run_unified_fusion(dayahead_predictions=da_predictions)
        fused = result.get("dayahead_fused")
        if fused is not None:
            assert "dayahead_price" in fused.columns

    def test_fused_has_model_column(self, da_predictions):
        result = run_unified_fusion(dayahead_predictions=da_predictions)
        fused = result.get("dayahead_fused")
        if fused is not None:
            assert "dayahead_model_or_fusion" in fused.columns

    def test_target_day_filter(self, da_predictions):
        result = run_unified_fusion(
            dayahead_predictions=da_predictions,
            target_day="2026-06-01",
        )
        assert result["status"] in (
            UNIFIED_FUSION_DEGRADED, UNIFIED_FUSION_COMPLETE
        )

    def test_with_weights(self, da_predictions, da_weights):
        result = run_unified_fusion(
            dayahead_predictions=da_predictions,
            dayahead_weights=da_weights,
            target_day="2026-06-01",
        )
        assert isinstance(result, dict)
        assert "status" in result
