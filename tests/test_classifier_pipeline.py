"""
tests/test_classifier_pipeline.py — Classifier pipeline contract tests.

Validates:
    1. run_negative_classifier with valid fusion input produces final output
    2. Pipeline produces no duplicate final keys
    3. final_price has no NaN
    4. Production mode does not require y_true
    5. Empty fusion input raises ValueError
    6. Missing fused_price raises ValueError
    7. Negative prices are flagged
    8. Pipeline handles no-op correctly
    9. Pipeline handles rule fallback correctly
    10. model_lineage_json is valid JSON
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FINAL_UNIQUE_KEY,
    FUSION_OUTPUT_COLUMNS,
)
from pipelines.classifier_pipeline import (
    run_negative_classifier,
    validate_fusion_input,
    _build_synthetic_fusion,
)


class TestValidateFusionInput:
    """Contract: validate_fusion_input."""

    def test_valid_passes(self):
        """Valid fusion input passes validation."""
        fusion = _build_synthetic_fusion(24)
        issues = validate_fusion_input(fusion)
        assert issues == []

    def test_empty_returns_issue(self):
        """Empty DataFrame returns issue."""
        issues = validate_fusion_input(pd.DataFrame())
        assert len(issues) > 0

    def test_missing_column_detected(self):
        """Missing fused_price is detected."""
        fusion = _build_synthetic_fusion(24)
        bad = fusion.drop(columns=["fused_price"])
        issues = validate_fusion_input(bad)
        assert any("fused_price" in i for i in issues)

    def test_nan_fused_detected(self):
        """NaN in fused_price is detected."""
        fusion = _build_synthetic_fusion(24)
        fusion["fused_price"] = np.nan
        issues = validate_fusion_input(fusion)
        assert any("NaN" in i for i in issues)

    def test_invalid_hour_detected(self):
        """hour_business outside 1..24 is detected."""
        fusion = _build_synthetic_fusion(24)
        fusion.loc[0, "hour_business"] = 99
        issues = validate_fusion_input(fusion)
        assert any("hour_business" in i for i in issues)


class TestClassifierPipeline:
    """Contract: run_negative_classifier."""

    def test_produces_final_output_schema(self):
        """Pipeline produces correct schema."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion)
        assert list(result.columns) == FINAL_OUTPUT_COLUMNS
        assert len(result) == 24

    def test_no_duplicate_final_keys(self):
        """No duplicate final keys in output."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion)
        dups = result.duplicated(subset=FINAL_UNIQUE_KEY, keep=False)
        assert dups.sum() == 0

    def test_no_nan_final_price(self):
        """final_price has no NaN."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion)
        assert not result["final_price"].isna().any()

    def test_no_y_true_in_production(self):
        """Production mode output has no y_true."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion, production=True)
        assert "y_true" not in result.columns

    def test_empty_input_raises(self):
        """Empty fusion input raises ValueError."""
        empty = pd.DataFrame(columns=FUSION_OUTPUT_COLUMNS)
        with pytest.raises(ValueError, match="validation failed"):
            run_negative_classifier(empty)

    def test_missing_fused_raises(self):
        """Missing fused_price raises ValueError."""
        fusion = _build_synthetic_fusion(24)
        bad = fusion.drop(columns=["fused_price"])
        with pytest.raises(ValueError, match="validation failed"):
            run_negative_classifier(bad)

    def test_negative_prices_flagged(self):
        """Negative fused_price rows are flagged."""
        fusion = _build_synthetic_fusion(24, include_negative=True)
        result = run_negative_classifier(fusion, rule_fallback=True)

        neg_in_fusion = (fusion["fused_price"] < 0).sum()
        neg_in_result = result["negative_flag"].sum()
        assert neg_in_result == neg_in_fusion

    def test_rule_fallback_disabled(self):
        """No flagging when rule_fallback=False."""
        fusion = _build_synthetic_fusion(24, include_negative=True)
        result = run_negative_classifier(fusion, rule_fallback=False)

        assert not result["negative_flag"].any()

    def test_model_lineage_json_valid(self):
        """model_lineage_json is valid JSON."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion)
        for raw in result["model_lineage_json"]:
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)

    def test_output_sorted(self):
        """Output is sorted by business_day, hour_business."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion)
        day_diffs = result["business_day"].diff().dropna()
        # All diffs should be >= 0 (sorted ascending)
        assert (day_diffs >= pd.Timedelta(0)).all()

    def test_classifier_not_applied(self):
        """No-artifact pipeline: classifier_applied is False."""
        fusion = _build_synthetic_fusion(24)
        result = run_negative_classifier(fusion)
        assert not result["classifier_applied"].any()
