"""
tests/test_residual_correction_schema.py — Residual correction schema contract tests.

Validates:
    1. CORRECTED_PREDICTION_COLUMNS has expected columns
    2. CORRECTED_UNIQUE_KEY columns exist in CORRECTED_PREDICTION_COLUMNS
    3. CORRECTED_REQUIRED_KEYS columns exist in CORRECTED_PREDICTION_COLUMNS
    4. Schema column order matches specification
    5. Key columns (task, model_name, business_day, hour_business) preserved
    6. New fields (y_pred_raw, y_pred_corrected, residual_delta, etc.) present
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    CORRECTED_UNIQUE_KEY,
    CORRECTED_MERGE_KEY,
    CORRECTED_REQUIRED_KEYS,
    PREDICTION_OUTPUT_COLUMNS,
)


class TestCorrectedSchemaCompleteness:
    """Contract: corrected prediction schema definition."""

    def test_has_17_columns(self):
        """CORRECTED_PREDICTION_COLUMNS has exactly 17 columns."""
        assert len(CORRECTED_PREDICTION_COLUMNS) == 17

    def test_includes_all_prediction_key_columns(self):
        """Corrected schema preserves key columns from prediction output."""
        key_from_pred = ["task", "model_name", "target_day", "business_day", "ds",
                         "hour_business", "period"]
        for col in key_from_pred:
            assert col in CORRECTED_PREDICTION_COLUMNS, f"Missing key: {col}"

    def test_has_new_correction_fields(self):
        """Corrected schema has all P3-specific fields."""
        correction_fields = [
            "y_pred_raw", "y_pred_corrected", "residual_delta",
            "correction_applied", "correction_module", "risk_source",
            "reason_codes", "correction_version",
        ]
        for col in correction_fields:
            assert col in CORRECTED_PREDICTION_COLUMNS, f"Missing correction field: {col}"

    def test_still_has_confidence_and_version(self):
        """Corrected schema keeps source_confidence and model_version."""
        assert "source_confidence" in CORRECTED_PREDICTION_COLUMNS
        assert "model_version" in CORRECTED_PREDICTION_COLUMNS

    def test_unique_key_is_subset(self):
        """CORRECTED_UNIQUE_KEY columns all exist in CORRECTED_PREDICTION_COLUMNS."""
        for col in CORRECTED_UNIQUE_KEY:
            assert col in CORRECTED_PREDICTION_COLUMNS, f"{col} missing"

    def test_required_keys_are_subset(self):
        """CORRECTED_REQUIRED_KEYS columns all exist in CORRECTED_PREDICTION_COLUMNS."""
        for col in CORRECTED_REQUIRED_KEYS:
            assert col in CORRECTED_PREDICTION_COLUMNS, f"{col} missing"

    def test_unique_key_has_5_columns(self):
        """CORRECTED_UNIQUE_KEY has exactly 5 columns."""
        assert len(CORRECTED_UNIQUE_KEY) == 5
        assert CORRECTED_UNIQUE_KEY == [
            "task", "model_name", "target_day", "business_day", "hour_business",
        ]

    def test_merge_key_has_6_columns(self):
        """CORRECTED_MERGE_KEY has exactly 6 columns."""
        assert len(CORRECTED_MERGE_KEY) == 6
        assert CORRECTED_MERGE_KEY == [
            "task", "model_name", "target_day", "business_day", "ds", "hour_business",
        ]

    def test_merge_key_all_in_corrected_columns(self):
        """All CORRECTED_MERGE_KEY columns exist in CORRECTED_PREDICTION_COLUMNS."""
        for col in CORRECTED_MERGE_KEY:
            assert col in CORRECTED_PREDICTION_COLUMNS, f"{col} missing"


class TestCorrectedDataFrameConstruct:
    """Contract: constructing a corrected DataFrame with the schema."""

    def test_can_construct_corrected_df(self):
        """Can construct a minimal corrected prediction DataFrame."""
        df = pd.DataFrame({
            "task": ["dayahead"],
            "model_name": ["cfg05"],
            "target_day": ["2026-03-05"],
            "business_day": [pd.Timestamp("2026-03-05")],
            "ds": [pd.Timestamp("2026-03-05 01:00")],
            "hour_business": [1],
            "period": ["1_8"],
            "y_pred_raw": [100.0],
            "y_pred_corrected": [100.0],
            "residual_delta": [0.0],
            "correction_applied": [False],
            "correction_module": ["p5m_residual_noop"],
            "risk_source": ["DATA_MISSING"],
            "reason_codes": ["DATA_MISSING_NO_OP"],
            "correction_version": ["0.0.0"],
            "source_confidence": [0.5],
            "model_version": ["1.0.0"],
        })
        for col in CORRECTED_PREDICTION_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_noop_row_values(self):
        """No-op row has correct default values."""
        df = pd.DataFrame({
            "task": ["dayahead"],
            "model_name": ["cfg05"],
            "target_day": ["2026-03-05"],
            "business_day": [pd.Timestamp("2026-03-05")],
            "ds": [pd.Timestamp("2026-03-05 01:00")],
            "hour_business": [1],
            "period": ["1_8"],
            "y_pred_raw": [100.0],
            "y_pred_corrected": [100.0],
            "residual_delta": [0.0],
            "correction_applied": [False],
            "correction_module": ["p5m_residual_noop"],
            "risk_source": ["DATA_MISSING"],
            "reason_codes": ["DATA_MISSING_NO_OP"],
            "correction_version": ["0.0.0"],
            "source_confidence": [0.5],
            "model_version": ["1.0.0"],
        })
        assert df.loc[0, "y_pred_corrected"] == df.loc[0, "y_pred_raw"]
        assert df.loc[0, "residual_delta"] == 0.0
        assert df.loc[0, "correction_applied"] == False
        assert df.loc[0, "correction_module"] == "p5m_residual_noop"
        assert df.loc[0, "risk_source"] == "DATA_MISSING"

    def test_corrected_row_values(self):
        """Corrected row has delta != 0 and correction_applied == True."""
        df = pd.DataFrame({
            "task": ["dayahead"],
            "model_name": ["cfg05"],
            "target_day": ["2026-03-05"],
            "business_day": [pd.Timestamp("2026-03-05")],
            "ds": [pd.Timestamp("2026-03-05 01:00")],
            "hour_business": [1],
            "period": ["1_8"],
            "y_pred_raw": [100.0],
            "y_pred_corrected": [95.0],
            "residual_delta": [-5.0],
            "correction_applied": [True],
            "correction_module": ["p5m_residual_plugin"],
            "risk_source": ["NEGATIVE_RISK"],
            "reason_codes": ["P5M_ADAPTER_CORRECTION;RISK_DATA_AVAILABLE"],
            "correction_version": ["1.0.0"],
            "source_confidence": [0.5],
            "model_version": ["1.0.0"],
        })
        assert df.loc[0, "y_pred_corrected"] == 95.0
        assert df.loc[0, "residual_delta"] == -5.0
        assert df.loc[0, "correction_applied"] == True

    def test_dataframe_can_be_sorted(self):
        """Corrected DataFrame can be sorted by business_day and hour_business."""
        df = pd.DataFrame({
            "task": ["dayahead", "dayahead"],
            "model_name": ["cfg05", "cfg05"],
            "target_day": ["2026-03-05", "2026-03-05"],
            "business_day": [pd.Timestamp("2026-03-05"), pd.Timestamp("2026-03-05")],
            "ds": [pd.Timestamp("2026-03-05 02:00"), pd.Timestamp("2026-03-05 01:00")],
            "hour_business": [2, 1],
            "period": ["1_8", "1_8"],
            "y_pred_raw": [110.0, 100.0],
            "y_pred_corrected": [110.0, 100.0],
            "residual_delta": [0.0, 0.0],
            "correction_applied": [False, False],
            "correction_module": ["p5m_residual_noop", "p5m_residual_noop"],
            "risk_source": ["DATA_MISSING", "DATA_MISSING"],
            "reason_codes": ["DATA_MISSING_NO_OP", "DATA_MISSING_NO_OP"],
            "correction_version": ["0.0.0", "0.0.0"],
            "source_confidence": [0.5, 0.5],
            "model_version": ["1.0.0", "1.0.0"],
        })
        sorted_df = df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        assert sorted_df.loc[0, "hour_business"] == 1
        assert sorted_df.loc[1, "hour_business"] == 2
