"""P79 — Final Full Chain Audit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from classifiers import run_final_classifier
from delivery.final_output_builder import (
    FINAL_OUTPUT_COLUMNS,
    FORBIDDEN_COLUMNS,
    build_final_output,
)
from fusion.unified_fusion_engine import run_unified_fusion
from fusion.unified_weight_learner import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    compute_bgew_weights,
    train_unified_weights,
)
from models.adapters.realtime_deep_adapter import (
    FORBIDDEN_ONLINE_COLUMNS,
    ONLINE_PACK_COLUMNS,
    RealtimeDeepAdapter,
)
from residuals import run_residual_correction
from residuals.residual_correction_engine import run_full_chain_residual_correction
from safety.full_chain_safety_supervisor import (
    FORBIDDEN_IN_PRODUCTION,
    QUARANTINED_MODELS,
    run_full_chain_safety,
)


# ── Cross-module contract tests ───────────────────────────────────────────────


class TestCrossModuleContracts:
    """Verify invariants that span multiple modules."""

    def test_forbidden_columns_consistent(self):
        """FORBIDDEN_COLUMNS and FORBIDDEN_IN_PRODUCTION should overlap on y_true."""
        assert "y_true" in FORBIDDEN_COLUMNS
        assert "y_true" in FORBIDDEN_IN_PRODUCTION

    def test_online_pack_forbidden_subset(self):
        """y_true should be forbidden in both online pack and production."""
        assert "y_true" in FORBIDDEN_ONLINE_COLUMNS
        assert "y_true" in FORBIDDEN_IN_PRODUCTION
        assert "y_true" in FORBIDDEN_COLUMNS

    def test_weight_bounds(self):
        """BGEW weights should respect min_weight and sum to 1."""
        weights = compute_bgew_weights(
            {"a": 0.001, "b": 1000.0},
            min_weight=MIN_WEIGHT,
            max_weight=MAX_WEIGHT,
        )
        for w in weights.values():
            assert w >= MIN_WEIGHT
            assert w < 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_residual_output_has_delta(self):
        """Residual correction should always add residual_delta column."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_pred": [200.0] * 24,
        })
        result = run_residual_correction(df, task="dayahead")
        assert "residual_delta" in result["output"].columns

    def test_classifier_adds_all_risk_columns(self):
        """Classifier should add negative_risk, spike_risk, uncertainty_score."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
        })
        result = run_final_classifier(dayahead_fused=df)
        output = result["dayahead"]["output"]
        for col in ("negative_risk", "spike_risk", "uncertainty_score",
                     "normal_trend_flag", "classifier_action"):
            assert col in output.columns, f"Missing column: {col}"

    def test_final_output_17_columns(self):
        """Final output should have exactly 17 defined columns."""
        assert len(FINAL_OUTPUT_COLUMNS) == 17

    def test_quarantined_models_not_empty(self):
        """There should be at least one quarantined model."""
        assert len(QUARANTINED_MODELS) >= 1

    def test_online_pack_14_columns(self):
        """Online pack should have 14 columns."""
        assert len(ONLINE_PACK_COLUMNS) == 14


# ── Leakage invariants ────────────────────────────────────────────────────────


class TestLeakageInvariants:
    """Verify leakage prevention across all modules."""

    def test_no_y_true_in_residual_noop(self):
        """No-op residual should not introduce y_true."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_pred": [200.0] * 24,
        })
        result = run_residual_correction(df, task="dayahead")
        assert "y_true" not in result["output"].columns

    def test_no_y_true_in_classifier_output(self):
        """Classifier should not add y_true."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
        })
        result = run_final_classifier(dayahead_fused=df)
        assert "y_true" not in result["dayahead"]["output"].columns

    def test_no_y_true_in_final_output(self):
        """Final output builder should never include y_true."""
        result = build_final_output()
        assert "y_true" not in result["output"].columns

    def test_safety_detects_y_true_everywhere(self):
        """Safety supervisor should flag y_true in all inputs."""
        bad_df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_true": [200.0] * 24,
        })
        result = run_full_chain_safety(dayahead_predictions=bad_df)
        assert result["status"] == "FULL_CHAIN_SAFETY_FAILED"

    def test_adapter_strips_y_true_in_online_pack(self):
        """Adapter should strip y_true from the exported online pack."""
        adapter = RealtimeDeepAdapter()
        bad_df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_pred": [200.0] * 24,
            "da_anchor": [200.0] * 24,
            "y_true": [200.0] * 24,
        })
        result = adapter.export_online_pack(da_predictions=bad_df)
        # _build_online_pack only selects specific columns, so y_true is stripped
        assert result["status"] == "EXPORTED"
        if result.get("output_path"):
            pack = pd.read_csv(result["output_path"])
            assert "y_true" not in pack.columns


# ── Boundary / edge-case tests ────────────────────────────────────────────────


class TestBoundaryConditions:
    """Edge cases and boundary conditions."""

    def test_single_row_predictions(self):
        """Should handle single-row DataFrames."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"],
            "hour_business": [1],
            "y_pred": [200.0],
        })
        result = run_residual_correction(df, task="dayahead")
        assert result["rows"] == 1

    def test_large_price_values(self):
        """Should handle very large prices."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [10000.0] * 24,
        })
        result = run_final_classifier(dayahead_fused=df)
        output = result["dayahead"]["output"]
        # 10000 > 500 → SPIKE_DETECTED
        assert (output["classifier_action"] == "SPIKE_DETECTED").all()

    def test_zero_prices(self):
        """Should handle zero prices."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [0.0] * 24,
        })
        result = run_final_classifier(dayahead_fused=df)
        output = result["dayahead"]["output"]
        # 0 is in [0, 500] → NORMAL
        assert (output["classifier_action"] == "NORMAL").all()

    def test_negative_prices(self):
        """Should handle negative prices."""
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [-50.0] * 24,
        })
        result = run_final_classifier(dayahead_fused=df)
        output = result["dayahead"]["output"]
        assert (output["classifier_action"] == "NEGATIVE_DETECTED").all()

    def test_weight_sum_exactly_one(self):
        """BGEW weights should sum to 1.0."""
        weights = compute_bgew_weights({"a": 5.0, "b": 10.0, "c": 15.0})
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_same_smape(self):
        """Equal sMAPE values should give equal weights."""
        weights = compute_bgew_weights({"a": 10.0, "b": 10.0, "c": 10.0})
        assert abs(weights["a"] - weights["b"]) < 1e-6
        assert abs(weights["b"] - weights["c"]) < 1e-6

    def test_very_small_smape(self):
        """Very small sMAPE should still produce valid weights."""
        weights = compute_bgew_weights({"a": 0.001, "b": 0.002})
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_very_large_smape(self):
        """Very large sMAPE should still produce valid weights."""
        weights = compute_bgew_weights({"a": 10000.0, "b": 20000.0})
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for w in weights.values():
            assert w >= MIN_WEIGHT
            assert w <= MAX_WEIGHT
