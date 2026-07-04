"""
tests/test_negative_classifier_adapter.py — NegativeClassifierAdapter contract tests.

Validates:
    1. No-artifact fallback does not crash
    2. No-artifact: final_price == fused_price
    3. No-artifact: classifier_applied == False
    4. No-artifact: reason_codes contains NEGATIVE_CLASSIFIER_NO_OP
    5. Rule fallback: fused_price < 0 → negative_flag == True
    6. Rule fallback: positive price → no negative_flag
    7. Rule fallback: reason_codes contains RULE_NEGATIVE_PRICE
    8. Load with non-existent model_dir does not crash
    9. Load with existent model_dir patterns (tmp_path)
    10. Empty fusion input returns empty output
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FUSION_OUTPUT_COLUMNS,
    NEGATIVE_CLASSIFIER_NOOP,
    NEGATIVE_CLASSIFIER_RULE,
)
from extreme.negative_classifier import NegativeClassifierAdapter


def _fusion_df(n_hours: int = 24, include_negative: bool = False) -> pd.DataFrame:
    """Synthetic fusion output for testing."""
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    if include_negative:
        base = -20.0
        prices = base + rng.uniform(-10, 30, n_hours)
    else:
        base = 120.0
        prices = base + rng.uniform(-5, 25, n_hours)

    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    rows: list[dict] = []
    for i in range(n_hours):
        w1 = round(rng.uniform(0.3, 0.7), 4)
        w2 = round(1.0 - w1, 4)
        rows.append({
            "task": "dayahead",
            "target_day": "2026-07-04",
            "ds": timestamps[i],
            "fused_price": float(prices[i]),
            "weights_json": json.dumps({"cfg05": w1, "best_two_average": w2}),
            "included_models": "cfg05;best_two_average",
            "excluded_models": "",
            "fusion_method": "equal_weight",
            "learner_version": "0.1.0-skeleton",
            "readiness_mode": "DRY_RUN",
            "reason_codes": "FUSION_EQUAL_WEIGHT",
        })

    df = pd.DataFrame(rows)
    df = add_business_time_columns(df, timestamp_col="ds")

    for c in FUSION_OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = None

    return df[FUSION_OUTPUT_COLUMNS]


class TestNoArtifactFallback:
    """Contract: no-artifact fallback behavior."""

    def test_no_artifact_does_not_crash(self):
        """No-artifact fallback runs without error."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        assert list(result.columns) == FINAL_OUTPUT_COLUMNS
        assert len(result) == 24

    def test_no_artifact_final_price_equals_fused(self):
        """No-artifact: final_price == fused_price for all rows."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        pd.testing.assert_series_equal(
            result["final_price"].reset_index(drop=True),
            result["fused_price"].reset_index(drop=True),
            check_names=False,
        )

    def test_no_artifact_classifier_applied_false(self):
        """No-artifact: classifier_applied == False for all rows."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        assert not result["classifier_applied"].any()

    def test_no_artifact_reason_contains_noop(self):
        """No-artifact: reason_codes contains NEGATIVE_CLASSIFIER_NO_OP."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        for codes in result["reason_codes"]:
            assert "NEGATIVE_CLASSIFIER_NO_OP" in str(codes)

    def test_no_artifact_risk_source(self):
        """No-artifact: risk_source is CLASSIFIER_ARTIFACT_MISSING."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        assert (result["risk_source"] == "CLASSIFIER_ARTIFACT_MISSING").all()

    def test_no_artifact_module(self):
        """No-artifact: classifier_module is negative_classifier_noop."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        assert (result["classifier_module"] == NEGATIVE_CLASSIFIER_NOOP).all()

    def test_empty_input_returns_empty(self):
        """Empty fusion input returns empty final output."""
        adapter = NegativeClassifierAdapter()
        empty = pd.DataFrame(columns=FUSION_OUTPUT_COLUMNS)
        result = adapter.predict(empty)
        assert list(result.columns) == FINAL_OUTPUT_COLUMNS
        assert len(result) == 0


class TestRuleFallback:
    """Contract: rule-based fallback behavior."""

    def test_negative_prices_flagged(self):
        """fused_price < 0 → negative_flag == True."""
        adapter = NegativeClassifierAdapter(rule_fallback=True)
        fusion = _fusion_df(24, include_negative=True)
        result = adapter.predict(fusion)

        neg_rows = result[result["fused_price"] < 0]
        if len(neg_rows) > 0:
            assert neg_rows["negative_flag"].all()

    def test_positive_prices_not_flagged(self):
        """fused_price >= 0 → negative_flag == False."""
        adapter = NegativeClassifierAdapter(rule_fallback=True)
        fusion = _fusion_df(24, include_negative=False)
        result = adapter.predict(fusion)

        pos_rows = result[result["fused_price"] >= 0]
        if len(pos_rows) > 0:
            assert not pos_rows["negative_flag"].any()

    def test_negative_price_reason_code(self):
        """Negative price rows include RULE_NEGATIVE_PRICE in reason_codes."""
        adapter = NegativeClassifierAdapter(rule_fallback=True)
        fusion = _fusion_df(24, include_negative=True)
        result = adapter.predict(fusion)

        neg_rows = result[result["fused_price"] < 0]
        if len(neg_rows) > 0:
            for codes in neg_rows["reason_codes"]:
                assert "RULE_NEGATIVE_PRICE" in str(codes)

    def test_rule_fallback_module(self):
        """Rule fallback rows use negative_classifier_rule module."""
        adapter = NegativeClassifierAdapter(rule_fallback=True)
        fusion = _fusion_df(1, include_negative=True)
        result = adapter.predict(fusion)

        neg_rows = result[result["fused_price"] < 0]
        if len(neg_rows) > 0:
            assert (neg_rows["classifier_module"] == NEGATIVE_CLASSIFIER_RULE).all()
            assert (neg_rows["risk_source"] == "RULE_FALLBACK").all()
            assert (neg_rows["negative_severity"] == "high").all()
            assert (neg_rows["negative_prob"] == 1.0).all()

    def test_rule_fallback_disabled(self):
        """rule_fallback=False does not flag negative prices."""
        adapter = NegativeClassifierAdapter(rule_fallback=False)
        fusion = _fusion_df(24, include_negative=True)
        result = adapter.predict(fusion)

        # Without rule fallback, no-op mode means no negative flag
        assert not result["negative_flag"].any()


class TestLoadBehavior:
    """Contract: adapter load behavior."""

    def test_load_none_model_dir(self):
        """load(None) does not crash and stays no-op."""
        adapter = NegativeClassifierAdapter()
        adapter.load(model_dir=None)
        assert not adapter._artifact_found

    def test_load_nonexistent_dir(self):
        """load with non-existent directory does not crash."""
        adapter = NegativeClassifierAdapter()
        adapter.load(model_dir="/nonexistent/path/for/testing")
        assert not adapter._artifact_found

    def test_load_with_tmp_path(self, tmp_path):
        """load with a directory containing artifact files."""
        # Create a dummy artifact
        artifact = tmp_path / "ExtremPriceClf.pkl"
        artifact.write_text("dummy_artifact")

        adapter = NegativeClassifierAdapter()
        adapter.load(model_dir=str(tmp_path))
        assert adapter._artifact_found

    def test_load_with_named_pattern(self, tmp_path):
        """load with extreme_price_radar artifact in dir."""
        artifact = tmp_path / "extreme_price_radar"
        artifact.write_text("dummy")

        adapter = NegativeClassifierAdapter()
        adapter.load(model_dir=str(tmp_path))
        assert adapter._artifact_found

    def test_model_lineage_json_valid(self):
        """model_lineage_json is valid JSON."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(1)
        result = adapter.predict(fusion)
        for raw in result["model_lineage_json"]:
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)

    def test_output_sorted_columns(self):
        """Output columns match FINAL_OUTPUT_COLUMNS order."""
        adapter = NegativeClassifierAdapter()
        fusion = _fusion_df(24)
        result = adapter.predict(fusion)
        assert list(result.columns) == FINAL_OUTPUT_COLUMNS
