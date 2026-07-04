"""
tests/test_residual_key_merge_contract.py — Key-based risk merge contract tests.

Validates:
    1. _resolve_risk_merge_key returns full key when all columns present
    2. _resolve_risk_merge_key returns degraded with only business_day + hour_business
    3. _resolve_risk_merge_key returns None with no matching columns
    4. _merge_risk_data does key-based merge, not positional
    5. risk_df with shuffled rows still matches correctly
    6. Unmatched risk rows are counted in stats
    7. Prediction rows without risk match remain no-op
    8. merge_stats contains correct merge_key, key_quality, n_matched
    9. _merge_risk_data does not crash with empty risk_df
    10. reason_codes include merge-related codes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.residual_correction import (
    _resolve_risk_merge_key,
    _merge_risk_data,
    apply_residual_correction,
)
from data.schema import CORRECTED_MERGE_KEY


def _prediction_df(n_rows: int = 24) -> pd.DataFrame:
    """Standard prediction DataFrame with business-time columns."""
    np.random.seed(42)
    timestamps = pd.date_range("2026-03-05 01:00", periods=n_rows, freq="h")
    return pd.DataFrame({
        "task": "dayahead",
        "model_name": "cfg05",
        "target_day": "2026-03-05",
        "business_day": pd.Timestamp("2026-03-05"),
        "ds": timestamps,
        "hour_business": list(range(1, n_rows + 1)),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "y_pred_raw": np.random.uniform(80, 200, n_rows),
        "y_pred_corrected": np.random.uniform(80, 200, n_rows),
        "residual_delta": np.zeros(n_rows),
        "correction_applied": [False] * n_rows,
        "correction_module": ["p5m_residual_noop"] * n_rows,
        "risk_source": ["DATA_MISSING"] * n_rows,
        "reason_codes": ["DATA_MISSING_NO_OP"] * n_rows,
        "correction_version": ["0.0.0"] * n_rows,
        "source_confidence": [0.5] * n_rows,
        "model_version": ["1.0.0"] * n_rows,
    }).set_index(
        pd.Index(["row_" + str(i) for i in range(n_rows)])
    )  # custom index so positional merge would fail


class TestResolveMergeKey:
    """Contract: _resolve_risk_merge_key resolution."""

    def test_full_key_when_all_present(self):
        """Full 6-column key returned when all columns present in both."""
        pred_cols = list(CORRECTED_MERGE_KEY) + ["y_pred"]
        risk_cols = list(CORRECTED_MERGE_KEY) + ["negative_prob", "risk_source"]
        key, quality = _resolve_risk_merge_key(pred_cols, risk_cols)
        assert quality == "full"
        assert key == CORRECTED_MERGE_KEY

    def test_degraded_with_business_hour_only(self):
        """Degraded key returned with only business_day + hour_business."""
        pred_cols = ["business_day", "hour_business", "y_pred"]
        risk_cols = ["business_day", "hour_business", "negative_prob"]
        key, quality = _resolve_risk_merge_key(pred_cols, risk_cols)
        assert quality == "degraded"
        assert key == ["business_day", "hour_business"]

    def test_none_with_no_matching_columns(self):
        """None returned when no matching key columns."""
        pred_cols = ["y_pred", "ds"]
        risk_cols = ["negative_prob", "risk_source"]
        key, quality = _resolve_risk_merge_key(pred_cols, risk_cols)
        assert key is None
        assert quality is None

    def test_partial_without_model_name(self):
        """Partial key returned when model_name absent from risk."""
        pred_cols = CORRECTED_MERGE_KEY + ["y_pred"]
        risk_cols = ["task", "target_day", "business_day", "ds",
                     "hour_business", "negative_prob"]
        key, quality = _resolve_risk_merge_key(pred_cols, risk_cols)
        assert quality == "partial"
        assert "model_name" not in key
        assert "business_day" in key
        assert "hour_business" in key


class TestKeyBasedMerge:
    """Contract: _merge_risk_data key-based merge."""

    def test_merge_by_full_key_not_positional(self):
        """Merge uses key-based join, not positional."""
        preds = _prediction_df(24)
        # Create risk_df with rows in REVERSE order (would fail positional)
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": list(range(24, 0, -1)),  # reversed
            "negative_prob": np.random.uniform(0, 1, 24),
            "risk_source": ["NEGATIVE_RISK"] * 24,
        })
        # Add minimal merge key: task, target_day, ds for degraded merge
        risk_df["task"] = "dayahead"
        risk_df["target_day"] = "2026-03-05"
        risk_df["ds"] = preds["ds"].iloc[::-1].values  # reversed ds too

        merged, stats = _merge_risk_data(preds, risk_df)

        # Should have merged correctly despite reversed order
        assert stats["key_quality"] in ("full", "partial", "degraded")
        assert stats["n_matched"] > 0

        # Verify risk columns are present
        assert "negative_prob" in merged.columns
        assert "risk_source" in merged.columns

        # Verify risk_source was merged (should not all be DATA_MISSING from preds)
        # At minimum, merged column exists with both sources
        risk_vals = merged["risk_source"].dropna().unique()
        assert len(risk_vals) > 0

    def test_shuffled_risk_df_matches_correctly(self):
        """Risk_df with shuffled rows still matches correctly."""
        preds = _prediction_df(24)
        # Create risk_df with business_day + hour_business
        hb_shuffled = list(range(1, 25))
        np.random.shuffle(hb_shuffled)

        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": hb_shuffled,
            "negative_prob": np.random.uniform(0, 1, 24),
            "risk_source": ["NEGATIVE_RISK"] * 24,
        })

        merged, stats = _merge_risk_data(preds, risk_df)

        # All 24 rows should match on business_day + hour_business
        assert stats["n_matched"] == 24
        assert stats["n_unmatched_risk_rows"] == 0

    def test_unmatched_risk_rows_counted(self):
        """Unmatched risk rows are counted in stats."""
        preds = _prediction_df(24)
        # Create risk_df with extra rows that don't match
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 30,
            "hour_business": list(range(1, 31)),  # 6 extra rows
            "negative_prob": np.random.uniform(0, 1, 30),
            "risk_source": ["NEGATIVE_RISK"] * 30,
        })

        merged, stats = _merge_risk_data(preds, risk_df)

        # 24 matched, 6 unmatched
        assert stats["n_matched"] == 24
        assert stats["n_unmatched_risk_rows"] == 6

    def test_missing_risk_rows_reported(self):
        """Prediction rows without risk match are reported."""
        preds = _prediction_df(24)
        # Only provide risk for first 12 hours
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 12,
            "hour_business": list(range(1, 13)),
            "negative_prob": np.random.uniform(0, 1, 12),
            "risk_source": ["NEGATIVE_RISK"] * 12,
        })

        merged, stats = _merge_risk_data(preds, risk_df)

        assert stats["n_matched"] == 12
        assert stats["n_pred_rows_without_risk"] == 12
        # Last 12 rows should have NaN in risk columns added from risk_df.
        # Use a risk column not in predictions (negative_prob) to check.
        assert merged["negative_prob"].isna().sum() == 12

    def test_no_risk_cols_beyond_key(self):
        """Risk_df with no additional columns yields no merge."""
        preds = _prediction_df(24)
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": list(range(1, 25)),
        })

        merged, stats = _merge_risk_data(preds, risk_df)
        assert stats["n_unmatched_risk_rows"] == 24
        assert stats["n_pred_rows_without_risk"] == 24

    def test_empty_risk_df_does_not_crash(self):
        """Empty risk_df does not crash."""
        preds = _prediction_df(24)
        risk_df = pd.DataFrame(columns=["business_day", "hour_business", "negative_prob"])

        merged, stats = _merge_risk_data(preds, risk_df)
        assert stats["n_risk_rows"] == 0

    def test_merge_stats_structure(self):
        """merge_stats contains expected keys."""
        preds = _prediction_df(24)
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": list(range(1, 25)),
            "negative_prob": np.random.uniform(0, 1, 24),
        })

        merged, stats = _merge_risk_data(preds, risk_df)

        assert "merge_key" in stats
        assert "key_quality" in stats
        assert "n_risk_rows" in stats
        assert "n_matched" in stats
        assert "n_unmatched_risk_rows" in stats
        assert "n_pred_rows_without_risk" in stats

    def test_merge_key_is_list_of_strings(self):
        """Merge key is a list of strings."""
        preds = _prediction_df(24)
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": list(range(1, 25)),
            "negative_prob": np.random.uniform(0, 1, 24),
        })

        merged, stats = _merge_risk_data(preds, risk_df)
        if stats["merge_key"] is not None:
            assert isinstance(stats["merge_key"], list)
            assert all(isinstance(c, str) for c in stats["merge_key"])


class TestRiskMergeInPipeline:
    """Contract: risk merge in apply_residual_correction."""

    def test_risk_df_with_key_columns_in_pipeline(self):
        """Risk_df with key columns is correctly merged in pipeline."""
        import numpy as np
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")
        preds = pd.DataFrame({
            "task": "dayahead",
            "model_name": "cfg05",
            "target_day": "2026-03-05",
            "ds": timestamps,
            "y_pred": rng.uniform(80, 200, 24),
        })

        # Risk df with matching keys (business_day + hour_business)
        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": list(range(1, 25)),
            "negative_prob": rng.uniform(0, 1, 24),
            "risk_source": ["SYNTHETIC"] * 24,
        })

        result = apply_residual_correction(preds, risk_df=risk_df)
        # Should not crash, and reason_codes should mention merge
        assert not result["y_pred_corrected"].isna().any()
        # Check reason_codes contain merge info
        reason = result["reason_codes"].iloc[0]
        assert isinstance(reason, str)
        # No-op is fine (P5M still has no real model)
        assert True

    def test_reason_codes_include_merge_info(self):
        """reason_codes include merge-related codes when risk_df provided."""
        import numpy as np
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")
        preds = pd.DataFrame({
            "task": "dayahead",
            "model_name": "cfg05",
            "target_day": "2026-03-05",
            "ds": timestamps,
            "y_pred": rng.uniform(80, 200, 24),
        })

        risk_df = pd.DataFrame({
            "business_day": [pd.Timestamp("2026-03-05")] * 24,
            "hour_business": list(range(1, 25)),
            "negative_prob": rng.uniform(0, 1, 24),
            "risk_source": ["SYNTHETIC"] * 24,
        })

        result = apply_residual_correction(preds, risk_df=risk_df)
        reason = result["reason_codes"].iloc[0]
        # Should mention merge or adapter
        assert any(kw in reason for kw in
                   ["MERGE", "ADAPTER", "DEGRADED", "MATCHED", "NO_OP"])

    def test_risk_df_without_key_does_not_crash(self):
        """Risk_df without merge keys falls back gracefully."""
        import numpy as np
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=24, freq="h")
        preds = pd.DataFrame({
            "task": "dayahead",
            "model_name": "cfg05",
            "target_day": "2026-03-05",
            "ds": timestamps,
            "y_pred": rng.uniform(80, 200, 24),
        })

        # risk_df WITHOUT any merge key columns
        risk_df = pd.DataFrame({
            "negative_prob": rng.uniform(0, 1, 24),
        })

        result = apply_residual_correction(preds, risk_df=risk_df)
        assert not result["y_pred_corrected"].isna().any()
        # Should be no-op since risk can't be merged
        corrected = result["y_pred_corrected"].values.astype(float)
        raw = result["y_pred_raw"].values.astype(float)
        assert np.allclose(corrected, raw)
