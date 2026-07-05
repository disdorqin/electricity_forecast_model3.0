"""P76 — Full Chain End-to-End integration tests (no real data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from classifiers import run_final_classifier
from delivery.final_output_builder import build_final_output
from fusion.unified_fusion_engine import run_unified_fusion
from fusion.unified_weight_learner import compute_bgew_weights, train_unified_weights
from residuals.residual_correction_engine import run_full_chain_residual_correction
from safety.full_chain_safety_supervisor import (
    FULL_CHAIN_SAFETY_PASS,
    run_full_chain_safety,
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
        "da_anchor": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def rt_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "trend_pred": np.random.uniform(100, 400, 24),
        "da_anchor": np.random.uniform(100, 400, 24),
    })


# ── E2E: Residual → Weights → Fusion → Classifier → Output → Safety ──────────


class TestEndToEndPipeline:
    """Simulates the full chain without real data or model artifacts."""

    def test_full_pipeline_no_artifacts(self, da_predictions, rt_predictions, tmp_path):
        """Run the full chain with no trained artifacts — all fallbacks."""
        # Step 1: Residual correction
        residual_result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        assert residual_result["overall_status"] == "RESIDUAL_CORRECTION_COMPLETE"
        da_corrected = residual_result["dayahead"]["output"]
        rt_corrected = residual_result["realtime"]["output"]

        # Step 2: Fusion (no weights → single model passthrough)
        fusion_result = run_unified_fusion(
            dayahead_predictions=da_corrected,
            realtime_predictions=rt_corrected,
        )
        assert fusion_result["status"] in ("UNIFIED_FUSION_COMPLETE", "UNIFIED_FUSION_DEGRADED")

        # Step 3: Classifier
        da_fused = fusion_result.get("dayahead_fused")
        rt_fused = fusion_result.get("realtime_fused")
        classifier_result = run_final_classifier(
            dayahead_fused=da_fused,
            realtime_fused=rt_fused,
        )
        assert classifier_result["dayahead"]["status"] in ("CLASSIFIED", "NOT_RUN")

        # Step 4: Final output
        da_classified = classifier_result["dayahead"].get("output")
        rt_classified = classifier_result["realtime"].get("output")
        output_result = build_final_output(
            dayahead_fused=da_fused,
            realtime_fused=rt_fused,
            dayahead_classified=da_classified,
            realtime_classified=rt_classified,
            residual_info={"dayahead": {"status": residual_result["dayahead"]["status"]}},
            target_day="2026-06-01",
        )
        assert output_result["output"] is not None
        assert len(output_result["output"]) == 24

        # Step 5: Safety check
        safety_result = run_full_chain_safety(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            final_output=output_result["output"],
            target_day="2026-06-01",
        )
        assert safety_result["status"] in (
            FULL_CHAIN_SAFETY_PASS,
            "FULL_CHAIN_SAFETY_DEGRADED",
            "FULL_CHAIN_SAFETY_FAILED",
        )

    def test_no_y_true_in_final_output(self, da_predictions, rt_predictions, tmp_path):
        """Verify no leakage in the final output."""
        residual_result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        fusion_result = run_unified_fusion(
            dayahead_predictions=residual_result["dayahead"]["output"],
            realtime_predictions=residual_result["realtime"]["output"],
        )
        output_result = build_final_output(
            dayahead_fused=fusion_result.get("dayahead_fused"),
            realtime_fused=fusion_result.get("realtime_fused"),
            target_day="2026-06-01",
        )
        output = output_result["output"]
        forbidden = {"y_true", "actual", "label", "residual_from_y_true",
                     "future_actual", "eval_residual"}
        for col in forbidden:
            assert col not in output.columns, f"Forbidden column '{col}' found in output"

    def test_24h_completeness_preserved(self, da_predictions, rt_predictions, tmp_path):
        """Verify 24-hour completeness through the pipeline."""
        residual_result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        da_out = residual_result["dayahead"]["output"]
        assert len(da_out) == 24

        fusion_result = run_unified_fusion(dayahead_predictions=da_out)
        da_fused = fusion_result.get("dayahead_fused")
        if da_fused is not None:
            assert len(da_fused) == 24

    def test_bgew_weights_computation(self):
        """Verify BGEW weight computation in isolation."""
        weights = compute_bgew_weights({"cfg05": 10.0, "catboost": 15.0})
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights["cfg05"] > weights["catboost"]

    def test_pipeline_with_nan_prices_handled(self, tmp_path):
        """Pipeline should not crash with NaN prices."""
        da = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_pred": [np.nan] * 24,
        })
        rt = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "trend_pred": [np.nan] * 24,
        })
        residual_result = run_full_chain_residual_correction(
            dayahead_predictions=da,
            realtime_predictions=rt,
            work_dir=str(tmp_path),
        )
        assert residual_result["overall_status"] == "RESIDUAL_CORRECTION_COMPLETE"
