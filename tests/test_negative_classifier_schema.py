"""
tests/test_negative_classifier_schema.py — Final output schema contract tests.

Validates:
    1. FINAL_OUTPUT_COLUMNS has expected columns
    2. FINAL_UNIQUE_KEY columns exist in FINAL_OUTPUT_COLUMNS
    3. VALID_NEGATIVE_SEVERITY values are correct
    4. NEGATIVE_CLASSIFIER_NOOP / RULE / EXTREMPRICE constants defined
    5. Schema column count
    6. Model lineage column is included
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FINAL_UNIQUE_KEY,
    VALID_NEGATIVE_SEVERITY,
    NEGATIVE_CLASSIFIER_NOOP,
    NEGATIVE_CLASSIFIER_RULE,
    NEGATIVE_CLASSIFIER_EXTREMPRICE,
    FUSION_OUTPUT_COLUMNS,
)


class TestFinalOutputSchemaCompleteness:
    """Contract: final output schema definition."""

    def test_has_17_columns(self):
        """FINAL_OUTPUT_COLUMNS has exactly 17 columns."""
        assert len(FINAL_OUTPUT_COLUMNS) == 17

    def test_includes_fused_price_and_final_price(self):
        """FINAL_OUTPUT_COLUMNS includes fused_price and final_price."""
        assert "fused_price" in FINAL_OUTPUT_COLUMNS
        assert "final_price" in FINAL_OUTPUT_COLUMNS

    def test_includes_negative_fields(self):
        """FINAL_OUTPUT_COLUMNS includes negative_prob, negative_flag, negative_severity."""
        assert "negative_prob" in FINAL_OUTPUT_COLUMNS
        assert "negative_flag" in FINAL_OUTPUT_COLUMNS
        assert "negative_severity" in FINAL_OUTPUT_COLUMNS

    def test_includes_classifier_fields(self):
        """FINAL_OUTPUT_COLUMNS includes classifier_applied, classifier_module, classifier_version."""
        assert "classifier_applied" in FINAL_OUTPUT_COLUMNS
        assert "classifier_module" in FINAL_OUTPUT_COLUMNS
        assert "classifier_version" in FINAL_OUTPUT_COLUMNS

    def test_includes_risk_and_reason(self):
        """FINAL_OUTPUT_COLUMNS includes risk_source and reason_codes."""
        assert "risk_source" in FINAL_OUTPUT_COLUMNS
        assert "reason_codes" in FINAL_OUTPUT_COLUMNS

    def test_includes_model_lineage(self):
        """FINAL_OUTPUT_COLUMNS includes model_lineage_json."""
        assert "model_lineage_json" in FINAL_OUTPUT_COLUMNS

    def test_final_unique_key_is_subset(self):
        """FINAL_UNIQUE_KEY columns all exist in FINAL_OUTPUT_COLUMNS."""
        for col in FINAL_UNIQUE_KEY:
            assert col in FINAL_OUTPUT_COLUMNS, f"{col} missing from final output columns"

    def test_final_unique_key_4_columns(self):
        """FINAL_UNIQUE_KEY has exactly 4 columns."""
        assert len(FINAL_UNIQUE_KEY) == 4
        assert FINAL_UNIQUE_KEY == [
            "task", "target_day", "business_day", "hour_business",
        ]

    def test_valid_negative_severity_values(self):
        """VALID_NEGATIVE_SEVERITY contains correct values."""
        assert "none" in VALID_NEGATIVE_SEVERITY
        assert "low" in VALID_NEGATIVE_SEVERITY
        assert "medium" in VALID_NEGATIVE_SEVERITY
        assert "high" in VALID_NEGATIVE_SEVERITY
        assert len(VALID_NEGATIVE_SEVERITY) == 4

    def test_classifier_constants_defined(self):
        """Classifier constants are non-empty strings."""
        assert isinstance(NEGATIVE_CLASSIFIER_NOOP, str)
        assert isinstance(NEGATIVE_CLASSIFIER_RULE, str)
        assert isinstance(NEGATIVE_CLASSIFIER_EXTREMPRICE, str)
        assert len(NEGATIVE_CLASSIFIER_NOOP) > 0
        assert len(NEGATIVE_CLASSIFIER_RULE) > 0
        assert len(NEGATIVE_CLASSIFIER_EXTREMPRICE) > 0

    def test_core_keys_from_fusion_present(self):
        """Core key fields (task, target_day, business_day, ds, hour_business, period) in final."""
        for col in ["task", "target_day", "business_day", "ds",
                     "hour_business", "period"]:
            assert col in FINAL_OUTPUT_COLUMNS, f"{col} missing from final output"

    def test_final_output_can_construct_df(self):
        """Can construct a minimal final output DataFrame."""
        df = pd.DataFrame({
            "task": ["dayahead"],
            "target_day": ["2026-07-04"],
            "business_day": [pd.Timestamp("2026-07-04")],
            "ds": [pd.Timestamp("2026-07-04 01:00")],
            "hour_business": [1],
            "period": ["1_8"],
            "fused_price": [150.0],
            "final_price": [150.0],
            "negative_prob": [0.0],
            "negative_flag": [False],
            "negative_severity": ["none"],
            "classifier_applied": [False],
            "classifier_module": ["negative_classifier_noop"],
            "classifier_version": ["0.0.0-noop"],
            "risk_source": ["CLASSIFIER_ARTIFACT_MISSING"],
            "reason_codes": ["NEGATIVE_CLASSIFIER_NO_OP"],
            "model_lineage_json": ['{"fusion_method":"equal_weight"}'],
        })
        for col in FINAL_OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) == 1
