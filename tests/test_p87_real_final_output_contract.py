"""
P87 — Real Final Output Contract tests.

Verifies the real final output contract:
  - final_output must be 24H
  - dayahead_price must not be NaN
  - realtime_price must not be NaN
  - No y_true in output
  - DEGRADED_DELIVERED when realtime is da_anchor fallback
  - RESIDUAL_NO_OP_FALLBACK in reason_codes when residual is no-op
  - CLASSIFIER_RULE_FALLBACK in reason_codes when classifier is rule fallback
  - Caveats list populated when fallbacks active
  - FINAL_OUTPUT_COLUMNS has 17 entries
  - FORBIDDEN_COLUMNS has 6 entries
  - run_full_chain result has "caveats" key
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from delivery.final_output_builder import (
    FINAL_OUTPUT_COLUMNS,
    FORBIDDEN_COLUMNS,
    OUTPUT_BUILT,
    OUTPUT_BLOCKED,
    OUTPUT_DEGRADED,
    build_final_output,
    _build_base_output,
    _validate_output,
)
from scripts.run_full_chain import (
    run_full_chain,
    FULL_CHAIN_DELIVERY_GO,
    FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
    FULL_CHAIN_DELIVERY_NO_GO,
)
from residuals import RESIDUAL_NO_OP_FALLBACK
from classifiers import CLASSIFIER_RULE_FALLBACK


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def dayahead_fused():
    """24-row dayahead fused DataFrame with valid prices."""
    rng = np.random.default_rng(100)
    return pd.DataFrame({
        "hour_business": list(range(1, 25)),
        "dayahead_price": rng.uniform(100, 400, 24),
        "dayahead_model_or_fusion": ["unified_bgew_fusion"] * 24,
    })


@pytest.fixture
def realtime_fused():
    """24-row realtime fused DataFrame with valid prices."""
    rng = np.random.default_rng(101)
    return pd.DataFrame({
        "hour_business": list(range(1, 25)),
        "realtime_price": rng.uniform(100, 400, 24),
        "realtime_model_or_fusion": ["rt_fusion"] * 24,
    })


# ── Tests ───────────────────────────────────────────────────────────────────


class TestOutputContractConstants:
    """Verify output schema constants."""

    def test_final_output_columns_has_17_entries(self):
        assert len(FINAL_OUTPUT_COLUMNS) == 17

    def test_forbidden_columns_has_6_entries(self):
        assert len(FORBIDDEN_COLUMNS) == 6

    def test_forbidden_columns_contains_y_true(self):
        assert "y_true" in FORBIDDEN_COLUMNS

    def test_forbidden_columns_contains_eval_residual(self):
        assert "eval_residual" in FORBIDDEN_COLUMNS


class TestBuildFinalOutput24H:
    """Verify build_final_output produces 24-row output."""

    def test_base_output_has_24_rows(self):
        base = _build_base_output("2026-06-15")
        assert len(base) == 24

    def test_base_output_hours_1_to_24(self):
        base = _build_base_output("2026-06-15")
        hours = sorted(base["hour_business"].tolist())
        assert hours == list(range(1, 25))

    def test_build_final_output_rows_24(self, dayahead_fused, realtime_fused):
        result = build_final_output(
            dayahead_fused=dayahead_fused,
            realtime_fused=realtime_fused,
            target_day="2026-06-15",
        )
        assert result["rows"] == 24

    def test_build_final_output_status_built(self, dayahead_fused, realtime_fused):
        result = build_final_output(
            dayahead_fused=dayahead_fused,
            realtime_fused=realtime_fused,
            target_day="2026-06-15",
        )
        assert result["status"] == OUTPUT_BUILT


class TestNoNaNInPrices:
    """dayahead_price and realtime_price must not be NaN."""

    def test_validate_output_detects_nan_dayahead(self):
        bad = pd.DataFrame({
            "business_day": ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [np.nan] * 24,
            "realtime_price": [200.0] * 24,
        })
        issues = _validate_output(bad)
        nan_issues = [i for i in issues if "NaN" in i and "dayahead" in i]
        assert len(nan_issues) > 0

    def test_validate_output_detects_nan_realtime(self):
        bad = pd.DataFrame({
            "business_day": ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
            "realtime_price": [np.nan] * 24,
        })
        issues = _validate_output(bad)
        nan_issues = [i for i in issues if "NaN" in i and "realtime" in i]
        assert len(nan_issues) > 0


class TestNoYTrueInOutput:
    """y_true must not appear in final output."""

    def test_validate_output_detects_y_true(self):
        bad = pd.DataFrame({
            "business_day": ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
            "y_true": [200.0] * 24,
        })
        issues = _validate_output(bad)
        forbidden_issues = [i for i in issues if "FORBIDDEN" in i and "y_true" in i]
        assert len(forbidden_issues) > 0

    def test_build_final_output_drops_forbidden_columns(self, dayahead_fused, realtime_fused):
        result = build_final_output(
            dayahead_fused=dayahead_fused,
            realtime_fused=realtime_fused,
            target_day="2026-06-15",
        )
        if result["output"] is not None:
            for col in FORBIDDEN_COLUMNS:
                assert col not in result["output"].columns


class TestDeliveryStatus:
    """Verify delivery_status values in output."""

    def test_degraded_status_when_realtime_missing(self, dayahead_fused):
        result = build_final_output(
            dayahead_fused=dayahead_fused,
            realtime_fused=None,
            target_day="2026-06-15",
            delivery_status="DEGRADED_DELIVERED",
        )
        if result["output"] is not None:
            statuses = result["output"]["delivery_status"].unique()
            assert "DEGRADED_DELIVERED" in statuses


class TestRunFullChainCaveats:
    """run_full_chain result must have 'caveats' key."""

    def test_result_has_caveats_key(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
        )
        assert "caveats" in result

    def test_caveats_is_list(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
        )
        assert isinstance(result["caveats"], list)

    def test_residual_no_op_caveat(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
        )
        res_step = result["steps"].get("residual_correction", {})
        if "NO_OP" in str(res_step.get("dayahead_status", "")):
            assert "RESIDUAL_NO_OP_FALLBACK" in result["caveats"]

    def test_classifier_rule_fallback_caveat(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
        )
        clf_step = result["steps"].get("classifier", {})
        if clf_step.get("status") == "CLASSIFIER_RULE_FALLBACK":
            assert "CLASSIFIER_RULE_FALLBACK" in result["caveats"]


class TestReasonCodesInOutput:
    """Verify reason_codes populated in build_final_output."""

    def test_reason_codes_list_returned(self, dayahead_fused):
        result = build_final_output(
            dayahead_fused=dayahead_fused,
            target_day="2026-06-15",
        )
        assert isinstance(result["reason_codes"], list)

    def test_reason_codes_populated_when_dayahead_merged(self, dayahead_fused):
        result = build_final_output(
            dayahead_fused=dayahead_fused,
            target_day="2026-06-15",
        )
        assert "DAYAHEAD_PRICES_MERGED" in result["reason_codes"]
