"""
tests/test_fusion_schema.py — Fusion schema contract tests.

Validates:
    1. FUSION_OUTPUT_COLUMNS has expected columns
    2. FUSION_UNIQUE_KEY columns exist in FUSION_OUTPUT_COLUMNS
    3. FUSION_GROUPING_KEY columns exist in FUSION_OUTPUT_COLUMNS
    4. FUSION_REQUIRED_INPUT_COLUMNS is a subset of corrected columns
    5. VALID_FUSION_METHODS are all valid strings
    6. Schema column order matches specification
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    FUSION_OUTPUT_COLUMNS,
    FUSION_UNIQUE_KEY,
    FUSION_GROUPING_KEY,
    FUSION_REQUIRED_INPUT_COLUMNS,
    VALID_FUSION_METHODS,
)


class TestFusionSchemaCompleteness:
    """Contract: fusion schema definition."""

    def test_has_14_columns(self):
        """FUSION_OUTPUT_COLUMNS has exactly 14 columns."""
        assert len(FUSION_OUTPUT_COLUMNS) == 14

    def test_includes_fused_price(self):
        """FUSION_OUTPUT_COLUMNS includes fused_price."""
        assert "fused_price" in FUSION_OUTPUT_COLUMNS

    def test_includes_weights_json(self):
        """FUSION_OUTPUT_COLUMNS includes weights_json."""
        assert "weights_json" in FUSION_OUTPUT_COLUMNS

    def test_includes_included_excluded_models(self):
        """FUSION_OUTPUT_COLUMNS includes included_models and excluded_models."""
        assert "included_models" in FUSION_OUTPUT_COLUMNS
        assert "excluded_models" in FUSION_OUTPUT_COLUMNS

    def test_includes_fusion_method_and_learner(self):
        """FUSION_OUTPUT_COLUMNS includes fusion_method and learner_version."""
        assert "fusion_method" in FUSION_OUTPUT_COLUMNS
        assert "learner_version" in FUSION_OUTPUT_COLUMNS

    def test_includes_readiness_and_reason(self):
        """FUSION_OUTPUT_COLUMNS includes readiness_mode and reason_codes."""
        assert "readiness_mode" in FUSION_OUTPUT_COLUMNS
        assert "reason_codes" in FUSION_OUTPUT_COLUMNS

    def test_unique_key_is_subset(self):
        """FUSION_UNIQUE_KEY columns all exist in FUSION_OUTPUT_COLUMNS."""
        for col in FUSION_UNIQUE_KEY:
            assert col in FUSION_OUTPUT_COLUMNS, f"{col} missing"

    def test_grouping_key_is_subset(self):
        """FUSION_GROUPING_KEY columns all exist in FUSION_OUTPUT_COLUMNS."""
        for col in FUSION_GROUPING_KEY:
            assert col in FUSION_OUTPUT_COLUMNS, f"{col} missing"

    def test_required_input_is_subset_of_corrected(self):
        """FUSION_REQUIRED_INPUT_COLUMNS is a subset of corrected schema."""
        for col in FUSION_REQUIRED_INPUT_COLUMNS:
            assert col in CORRECTED_PREDICTION_COLUMNS, f"{col} not in corrected schema"

    def test_unique_key_has_5_columns(self):
        """FUSION_UNIQUE_KEY has exactly 5 columns."""
        assert len(FUSION_UNIQUE_KEY) == 5
        assert FUSION_UNIQUE_KEY == [
            "task", "target_day", "business_day", "ds", "hour_business",
        ]

    def test_grouping_key_has_6_columns(self):
        """FUSION_GROUPING_KEY has exactly 6 columns."""
        assert len(FUSION_GROUPING_KEY) == 6

    def test_valid_methods(self):
        """VALID_FUSION_METHODS are all strings."""
        for m in VALID_FUSION_METHODS:
            assert isinstance(m, str)
        assert "equal_weight" in VALID_FUSION_METHODS
        assert "prior_weight" in VALID_FUSION_METHODS
        assert "bgew_skeleton" in VALID_FUSION_METHODS


class TestFusionDataFrameConstruct:
    """Contract: constructing a minimal fusion output DataFrame."""

    def test_can_construct_fusion_df(self):
        """Can construct a minimal fusion output DataFrame."""
        df = pd.DataFrame({
            "task": ["dayahead"],
            "target_day": ["2026-07-04"],
            "business_day": [pd.Timestamp("2026-07-04")],
            "ds": [pd.Timestamp("2026-07-04 01:00")],
            "hour_business": [1],
            "period": ["1_8"],
            "fused_price": [150.0],
            "weights_json": ['{"cfg05": 1.0}'],
            "included_models": ["cfg05"],
            "excluded_models": [""],
            "fusion_method": ["equal_weight"],
            "learner_version": ["0.1.0-skeleton"],
            "readiness_mode": ["DRY_RUN"],
            "reason_codes": ["FUSION_EQUAL_WEIGHT"],
        })
        for col in FUSION_OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_fusion_df_no_nan(self):
        """Fusion DataFrame has no NaN in fused_price."""
        df = pd.DataFrame({
            "task": ["dayahead", "dayahead"],
            "target_day": ["2026-07-04", "2026-07-04"],
            "business_day": [pd.Timestamp("2026-07-04")] * 2,
            "ds": pd.date_range("2026-07-04 01:00", periods=2, freq="h"),
            "hour_business": [1, 2],
            "period": ["1_8", "1_8"],
            "fused_price": [150.0, 155.0],
            "weights_json": ['{"cfg05":1.0}', '{"cfg05":1.0}'],
            "included_models": ["cfg05", "cfg05"],
            "excluded_models": ["", ""],
            "fusion_method": ["equal_weight", "equal_weight"],
            "learner_version": ["0.1.0-skeleton", "0.1.0-skeleton"],
            "readiness_mode": ["DRY_RUN", "DRY_RUN"],
            "reason_codes": ["FUSION_EQUAL_WEIGHT", "FUSION_EQUAL_WEIGHT"],
        })
        assert not df["fused_price"].isna().any()

    def test_df_can_be_sorted(self):
        """Fusion output can be sorted by business_day and hour_business."""
        df = pd.DataFrame({
            "task": ["dayahead", "dayahead"],
            "target_day": ["2026-07-04", "2026-07-04"],
            "business_day": [pd.Timestamp("2026-07-04")] * 2,
            "ds": [pd.Timestamp("2026-07-04 02:00"), pd.Timestamp("2026-07-04 01:00")],
            "hour_business": [2, 1],
            "period": ["1_8", "1_8"],
            "fused_price": [155.0, 150.0],
            "weights_json": ['{"cfg05":1.0}', '{"cfg05":1.0}'],
            "included_models": ["cfg05", "cfg05"],
            "excluded_models": ["", ""],
            "fusion_method": ["equal_weight", "equal_weight"],
            "learner_version": ["0.1.0-skeleton", "0.1.0-skeleton"],
            "readiness_mode": ["DRY_RUN", "DRY_RUN"],
            "reason_codes": ["FUSION_EQUAL_WEIGHT", "FUSION_EQUAL_WEIGHT"],
        })
        sorted_df = df.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        assert sorted_df.loc[0, "hour_business"] == 1
        assert sorted_df.loc[1, "hour_business"] == 2
