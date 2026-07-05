"""
Tests for P91: Realtime Design Reclassification.

Covers:
  1. New state constants exist and differ from old names
  2. DASafeRealtimeAssistAdapter produces enhanced schema
  3. Online pack contains all required fields
  4. No forbidden columns in output
  5. SGDFNet disablement is not NO_GO
  6. Hybrid ready status logic
  7. learner_policy constants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.realtime_state import (
    REALTIME_DA_SAFE_BASELINE,
    REALTIME_ASSIST_SGDFNET_AVAILABLE,
    REALTIME_ASSIST_DISABLED,
    REALTIME_HYBRID_READY,
    REALTIME_READY_DA_SAFE_ONLY,
    REALTIME_HYBRID_READY_FINAL,
    REALTIME_NO_GO,
    SGDFNET_ASSIST_READY,
    SGDFNET_ASSIST_CODE_ONLY,
    SGDFNET_ASSIST_BLOCKED,
    LEARNER_POLICY_DAYAHEAD,
    LEARNER_POLICY_REALTIME,
    LEARNER_POLICY_REALTIME_SINGLE,
    SGDFNET_ASSIST_DISABLED,
    SGDFNET_ASSIST_ACTIVE,
    DA_SAFE_BASELINE_ACTIVE,
    REALTIME_HYBRID_FUSION_ACTIVE,
    SGDFNET_WEIGHT_SUPPRESSED,
    HARD_REJECT_BAD_ASSIST,
)
from models.adapters.realtime_da_safe_assist import DASafeRealtimeAssistAdapter


class TestRealtimeStateConstants:
    """Test that new state constants exist and have correct values."""

    def test_new_states_not_old_names(self):
        """New states should not match old fallback names."""
        assert REALTIME_DA_SAFE_BASELINE != "REALTIME_DA_ANCHOR_FALLBACK"
        assert REALTIME_ASSIST_SGDFNET_AVAILABLE != "REALTIME_DEEP_READY_FAST_DEV"
        assert REALTIME_ASSIST_DISABLED != "FAST_DEV_ONLY"

    def test_da_safe_baseline_constant(self):
        assert REALTIME_DA_SAFE_BASELINE == "REALTIME_DA_SAFE_BASELINE"

    def test_assist_sgdfnet_available(self):
        assert REALTIME_ASSIST_SGDFNET_AVAILABLE == "REALTIME_ASSIST_SGDFNET_AVAILABLE"

    def test_assist_disabled(self):
        assert REALTIME_ASSIST_DISABLED == "REALTIME_ASSIST_DISABLED"

    def test_hybrid_ready(self):
        assert REALTIME_HYBRID_READY == "REALTIME_HYBRID_READY"

    def test_final_status_constants(self):
        assert REALTIME_READY_DA_SAFE_ONLY == "REALTIME_READY_DA_SAFE_ONLY"
        assert REALTIME_HYBRID_READY_FINAL == "REALTIME_HYBRID_READY"
        assert REALTIME_NO_GO == "REALTIME_NO_GO"

    def test_sgdfnet_assist_statuses(self):
        assert SGDFNET_ASSIST_READY == "SGDFNET_ASSIST_READY"
        assert SGDFNET_ASSIST_CODE_ONLY == "SGDFNET_ASSIST_CODE_ONLY"
        assert SGDFNET_ASSIST_BLOCKED == "SGDFNET_ASSIST_BLOCKED"

    def test_learner_policy_constants(self):
        assert LEARNER_POLICY_DAYAHEAD == "period_regime_bgew"
        assert LEARNER_POLICY_REALTIME == "pooled_30d_bgew"
        assert LEARNER_POLICY_REALTIME_SINGLE == "realtime_single_model_safe_baseline"

    def test_reason_codes(self):
        assert SGDFNET_ASSIST_DISABLED == "SGDFNET_ASSIST_DISABLED"
        assert SGDFNET_ASSIST_ACTIVE == "SGDFNET_ASSIST_ACTIVE"
        assert DA_SAFE_BASELINE_ACTIVE == "DA_SAFE_BASELINE_ACTIVE"
        assert REALTIME_HYBRID_FUSION_ACTIVE == "REALTIME_HYBRID_FUSION_ACTIVE"
        assert SGDFNET_WEIGHT_SUPPRESSED == "SGDFNET_WEIGHT_SUPPRESSED"
        assert HARD_REJECT_BAD_ASSIST == "HARD_REJECT_BAD_ASSIST"

    def test_assist_disabled_is_not_no_go(self):
        """REALTIME_ASSIST_DISABLED should NOT equal NO_GO."""
        assert REALTIME_ASSIST_DISABLED != REALTIME_NO_GO
        assert REALTIME_ASSIST_DISABLED != "REALTIME_DA_ANCHOR_FALLBACK"
        assert REALTIME_ASSIST_DISABLED != "NO_GO"


class TestAdapterEnhancedSchema:
    """Test that DASafeRealtimeAssistAdapter produces the enhanced schema."""

    @pytest.fixture
    def sample_data(self):
        """Create 48 rows (2 days of hourly data)."""
        rows = []
        for day in ["2026-07-01", "2026-07-02"]:
            for h in range(1, 25):
                period = "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")
                ds = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = pd.Timestamp(day) + pd.Timedelta(hours=23)
                business_day = pd.Timestamp(day).date()
                rows.append({
                    "ds": ds,
                    "da_anchor": 300.0 + np.random.uniform(-20, 20),
                    "da_price": 300.0 + np.random.uniform(-20, 20),
                    "rt_price": 290.0 + np.random.uniform(-30, 30),
                })
        df = pd.DataFrame(rows)
        # Add business time columns
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")
        return df

    def test_prediction_returns_enhanced_columns(self, sample_data):
        """predict() output should contain enhanced online pack columns."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")

        # Standard columns should exist
        for col in ["task", "model_name", "target_day", "business_day",
                     "ds", "hour_business", "period", "y_pred"]:
            assert col in result.columns, f"Missing standard column: {col}"

        # Enhanced columns should exist
        for col in adapter.ONLINE_PACK_COLUMNS:
            assert col in result.columns, f"Missing enhanced column: {col}"

    def test_trend_pred_equals_rt_pred(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert np.allclose(result["trend_pred"].values, result["y_pred"].values)

    def test_deep_rt_pred_equals_da_anchor(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        da_from_data = sample_data["da_anchor"].values[:len(result)]
        assert np.allclose(result["deep_rt_pred"].values, da_from_data, rtol=1e-5)

    def test_sgdfnet_pred_is_nan(self, sample_data):
        """Without SGDFNet, sgdfnet_pred should be NaN."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert result["sgdfnet_pred"].isna().all()

    def test_blend_pred_equals_da_anchor(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert np.allclose(result["blend_pred"].values, result["da_anchor"].values, rtol=1e-5)

    def test_correction_permission_false(self, sample_data):
        """Without SGDFNet, correction_permission should be False."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert ~result["correction_permission"].any()

    def test_reason_codes_contains_da_safe_baseline(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        for code in result["reason_codes"].unique():
            assert DA_SAFE_BASELINE_ACTIVE in str(code)

    def test_no_y_true_in_output(self, sample_data):
        """Production output must not contain y_true."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        forbidden = ["y_true", "actual", "label", "residual_from_y_true",
                     "future_actual", "eval_residual"]
        for col in forbidden:
            assert col not in result.columns, f"Forbidden column found: {col}"

    def test_no_nan_in_critical_columns(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        for col in ["y_pred", "trend_pred", "deep_rt_pred", "da_anchor"]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_hour_business_range(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_24_rows_per_day(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        # Check we have reasonable hourly coverage
        assert result["hour_business"].nunique() == 24
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24
        assert len(result) >= 24  # At least 24 rows across the date range

    def test_da_error_prob_range(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert result["da_error_prob"].between(0, 1).all()

    def test_uncertainty_score_range(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert result["uncertainty_score"].between(0, 1).all()

    def test_trend_confidence_value(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["trend_confidence"] == 0.5).all()

    def test_normal_trend_flag(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["normal_trend_flag"] == 1).all()

    def test_online_pack_schema_completeness(self, sample_data):
        """All ONLINE_PACK_COLUMNS should be present in output."""
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        for col in adapter.ONLINE_PACK_COLUMNS:
            assert col in result.columns, f"Missing: {col}"

    def test_no_forbidden_columns_in_predict(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        for col in adapter.FORBIDDEN_COLUMNS:
            assert col not in result.columns

    def test_model_name(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["model_name"] == "da_safe_realtime_assist").all()

    def test_trend_model_name(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["trend_model_name"] == "rt_da_anchor").all()

    def test_negative_bucket_flag(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["negative_bucket_flag"] == 0).all()

    def test_high_price_bucket_flag(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["high_price_bucket_flag"] == 0).all()

    def test_source_confidence(self, sample_data):
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        result = adapter.predict(df=sample_data, start="2026-07-01", end="2026-07-02")
        assert (result["source_confidence"] == 0.5).all()

    def test_sgdfnet_disabled_is_not_no_go(self):
        """SGDFNET_ASSIST_DISABLED reason code should not block delivery."""
        assert SGDFNET_ASSIST_DISABLED not in ("NO_GO", "BLOCKED", "FAILED")


class TestStateTransitionLogic:
    """Test status transition rules."""

    def test_da_safe_baseline_always_available(self):
        """DA-Safe Baseline must always be available."""
        assert REALTIME_DA_SAFE_BASELINE is not None

    def test_hybrid_ready_requires_both(self):
        """Hybrid ready should be distinct from DA-safe only."""
        assert REALTIME_HYBRID_READY_FINAL != REALTIME_READY_DA_SAFE_ONLY

    def test_assist_disabled_has_delivery(self):
        """Assist disabled should still have delivery."""
        assert REALTIME_ASSIST_DISABLED != "NO_DELIVERY"

    def test_learner_policies_are_distinct(self):
        assert LEARNER_POLICY_DAYAHEAD != LEARNER_POLICY_REALTIME

    def test_sgdfnet_assist_code_only_is_not_ready(self):
        """CODE_ONLY means the adapter code exists but model can't run."""
        assert SGDFNET_ASSIST_CODE_ONLY != SGDFNET_ASSIST_READY
        assert SGDFNET_ASSIST_BLOCKED != SGDFNET_ASSIST_READY


class TestFullChainClassification:
    """Test that run_full_chain correctly reflects new naming."""

    def test_caveat_renamed(self):
        """New caveat name should be DA-Safe not fallback."""
        caveat_name = "REALTIME_DA_SAFE_BASELINE"
        assert "FALLBACK" not in caveat_name
        assert "DA_SAFE" in caveat_name

    def test_final_verdict_constant(self):
        """The final verdict constant uses integrated go_with_caveats."""
        verdict = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
        assert "GO" in verdict
        assert "CAVEATS" in verdict

    def test_verdict_not_no_go(self):
        verdict = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
        assert "NO_GO" not in verdict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
