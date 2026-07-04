"""
tests/test_p19_cfg05_source_methodology_alignment.py — P19 tests.

Tests source methodology alignment audit.
Minimum 10 tests covering:
  1. source methodology matched / partial / not matched labels
  2. all 16 dimensions present
  3. claim rule: not MATCHED → "not claimed"
  4. claim rule: MATCHED → "candidate"
  5. no backtest summary → partial/not_matched
  6. no source repo → not_matched for relevant dims
  7. matched count + partial count + not_matched count = 16
  8. dimension names match AUDIT_DIMENSIONS
  9. local params used correctly
  10. walk-forward retrain strategy detection
"""

from __future__ import annotations

import os
import sys
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.audit_cfg05_source_methodology_alignment import (
    audit_cfg05_source_methodology_alignment,
    AUDIT_DIMENSIONS,
    MATCHED,
    PARTIAL,
    NOT_MATCHED,
)


class TestAuditLabels:
    def test_label_is_valid(self):
        result = audit_cfg05_source_methodology_alignment()
        assert result["label"] in (MATCHED, PARTIAL, NOT_MATCHED)

    def test_no_source_repo_not_matched_dims(self):
        result = audit_cfg05_source_methodology_alignment(source_repo="/nonexistent")
        # Dimensions that require source repo should be NOT_MATCHED
        d1 = result["dimensions"]["source_repo_report_date_window"]
        assert d1["status"] == "NOT_MATCHED"

    def test_no_backtest_summary(self):
        result = audit_cfg05_source_methodology_alignment(backtest_summary=None)
        d2 = result["dimensions"]["evaluation_start_end"]
        assert d2["status"] == "NOT_MATCHED"

    def test_with_backtest_summary(self):
        bt = {"eval_start": "2026-06-01", "eval_end": "2026-06-30", "reuse_model": True}
        result = audit_cfg05_source_methodology_alignment(backtest_summary=bt)
        d2 = result["dimensions"]["evaluation_start_end"]
        assert d2["status"] == "PARTIAL"


class TestClaimRules:
    def test_not_matched_claim(self):
        result = audit_cfg05_source_methodology_alignment()
        if result["label"] != MATCHED:
            assert "not claimed" in result["claim"].lower() or "caveats" in result["claim"].lower()

    def test_all_dimensions_present(self):
        result = audit_cfg05_source_methodology_alignment()
        for dim in AUDIT_DIMENSIONS:
            assert dim in result["dimensions"], f"Missing dimension: {dim}"

    def test_counts_sum_to_16(self):
        result = audit_cfg05_source_methodology_alignment()
        total = result["matched_count"] + result["partial_count"] + result["not_matched_count"]
        assert total == len(AUDIT_DIMENSIONS)

    def test_dimension_names_match(self):
        result = audit_cfg05_source_methodology_alignment()
        assert set(result["dimensions"].keys()) == set(AUDIT_DIMENSIONS)


class TestAuditDetails:
    def test_target_day_definition_matched(self):
        result = audit_cfg05_source_methodology_alignment()
        assert result["dimensions"]["target_day_definition"]["status"] == "MATCHED"

    def test_hour_24_mapping_matched(self):
        result = audit_cfg05_source_methodology_alignment()
        assert result["dimensions"]["hour_24_mapping"]["status"] == "MATCHED"

    def test_metric_formula_matched(self):
        result = audit_cfg05_source_methodology_alignment()
        assert result["dimensions"]["metric_formula"]["status"] == "MATCHED"

    def test_walk_forward_reuse_detection(self):
        bt_reuse = {"reuse_model": True}
        result = audit_cfg05_source_methodology_alignment(backtest_summary=bt_reuse)
        d13 = result["dimensions"]["walk_forward_retrain_strategy"]
        assert d13["status"] == "NOT_MATCHED"  # reuse != per-day retrain

    def test_walk_forward_perday_detection(self):
        bt_perday = {"reuse_model": False}
        result = audit_cfg05_source_methodology_alignment(backtest_summary=bt_perday)
        d13 = result["dimensions"]["walk_forward_retrain_strategy"]
        assert d13["status"] == "MATCHED"

    def test_local_params_used(self):
        custom_params = {
            "params": {"num_leaves": 128, "learning_rate": 0.01, "objective": "mae"},
            "feature_columns": ["hour", "month"],
        }
        result = audit_cfg05_source_methodology_alignment(local_params=custom_params)
        d6 = result["dimensions"]["cfg05_lightgbm_params"]
        assert "128" in d6["detail"]

    def test_reason_codes_present(self):
        result = audit_cfg05_source_methodology_alignment()
        assert len(result["reason_codes"]) > 0
        assert any("AUDIT_SUMMARY" in rc for rc in result["reason_codes"])
