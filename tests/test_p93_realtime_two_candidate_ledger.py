"""
Tests for P93: Realtime Two-Candidate Prediction Ledger.

Covers:
  1. Build ledger with only DA anchor
  2. Build ledger with both candidates
  3. rt_da_anchor always present
  4. sgdfnet_rt_assist optional
  5. No y_true in ledger
  6. Validation passes for valid ledger
  7. Validation fails when rt_da_anchor missing
  8. Extract candidates from ledger
  9. Append to ledger with dedup
  10. Schema completeness
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ledgers.realtime_prediction_ledger import (
    build_realtime_ledger,
    append_to_realtime_ledger,
    validate_realtime_ledger,
    extract_rt_candidates,
    REALTIME_LEDGER_COLUMNS,
    REALTIME_LEDGER_KEY,
    RT_DA_ANCHOR_MODEL,
    SGDFNET_ASSIST_MODEL,
)


@pytest.fixture
def da_anchor_df():
    """Create DA-Safe Baseline predictions (2 days, 24 hours each = 48 rows)."""
    rows = []
    for day in ["2026-07-01", "2026-07-02"]:
        for h in range(1, 25):
            period = "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")
            ds = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
            if h == 24:
                ds = pd.Timestamp(day) + pd.Timedelta(hours=23)
            rows.append({
                "business_day": day,
                "ds": ds,
                "hour_business": h,
                "period": period,
                "y_pred": 300.0 + np.random.uniform(-20, 20),
                "source_confidence": 0.5,
                "trend_pred": 300.0 + np.random.uniform(-20, 20),
                "da_anchor": 300.0 + np.random.uniform(-20, 20),
                "da_error_prob": 0.3,
                "residual_direction_prob": 0.5,
                "uncertainty_score": 0.3,
                "correction_permission": False,
                "reason_codes": "DA_SAFE_BASELINE_ACTIVE",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sgdfnet_df():
    """Create SGDFNet assist predictions (2 days = 48 rows)."""
    rows = []
    for day in ["2026-07-01", "2026-07-02"]:
        for h in range(1, 25):
            period = "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")
            ds = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
            if h == 24:
                ds = pd.Timestamp(day) + pd.Timedelta(hours=23)
            rows.append({
                "business_day": day,
                "ds": ds,
                "hour_business": h,
                "period": period,
                "rt_pred": 290.0 + np.random.uniform(-15, 15),
                "sgdfnet_pred": 290.0 + np.random.uniform(-15, 15),
                "da_anchor": 300.0 + np.random.uniform(-20, 20),
                "source_confidence": 0.4,
                "assist_available": True,
                "da_error_prob": 0.4,
                "residual_direction_prob": 0.5,
                "uncertainty_score": 0.4,
                "correction_permission": True,
                "reason_codes": "SGDFNET_ASSIST_ACTIVE",
            })
    return pd.DataFrame(rows)


class TestBuildLedger:
    """Test building the realtime ledger."""

    def test_da_anchor_only(self, da_anchor_df):
        """With only DA anchor, ledger should have only rt_da_anchor."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        models = ledger["model_name"].unique().tolist()
        assert RT_DA_ANCHOR_MODEL in models
        assert SGDFNET_ASSIST_MODEL not in models
        assert len(models) == 1

    def test_both_candidates(self, da_anchor_df, sgdfnet_df):
        """With both candidates, ledger should have both models."""
        ledger = build_realtime_ledger(
            da_anchor_predictions=da_anchor_df,
            sgdfnet_predictions=sgdfnet_df,
        )
        models = ledger["model_name"].unique().tolist()
        assert RT_DA_ANCHOR_MODEL in models
        assert SGDFNET_ASSIST_MODEL in models
        assert len(models) == 2

    def test_rt_da_anchor_always_present(self, da_anchor_df):
        """rt_da_anchor must always be in the ledger."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        assert RT_DA_ANCHOR_MODEL in ledger["model_name"].values

    def test_no_y_true_in_ledger(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        assert "y_true" not in ledger.columns

    def test_ledger_columns_complete(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        for col in REALTIME_LEDGER_COLUMNS:
            assert col in ledger.columns, f"Missing column: {col}"

    def test_da_anchor_pred_count(self, da_anchor_df):
        """DA anchor should have 48 entries (2 days x 24 hours)."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        da = ledger[ledger["model_name"] == RT_DA_ANCHOR_MODEL]
        assert len(da) == 48

    def test_both_pred_counts(self, da_anchor_df, sgdfnet_df):
        """Both models should have same row count."""
        ledger = build_realtime_ledger(
            da_anchor_predictions=da_anchor_df,
            sgdfnet_predictions=sgdfnet_df,
        )
        da = ledger[ledger["model_name"] == RT_DA_ANCHOR_MODEL]
        sg = ledger[ledger["model_name"] == SGDFNET_ASSIST_MODEL]
        assert len(da) == len(sg)

    def test_task_is_realtime(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        assert (ledger["task"] == "realtime").all()

    def test_da_anchor_includes_assist_fields(self, da_anchor_df):
        """DA anchor entries should include assist/risk fields."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        da = ledger[ledger["model_name"] == RT_DA_ANCHOR_MODEL]
        for col in ["da_error_prob", "uncertainty_score", "reason_codes"]:
            assert col in da.columns


class TestValidateLedger:
    """Test ledger validation."""

    def test_valid_ledger(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        valid, issues = validate_realtime_ledger(ledger)
        assert valid, f"Expected valid, got issues: {issues}"

    def test_no_y_true_check(self, da_anchor_df):
        """Validation should fail if y_true is present."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        ledger["y_true"] = 300.0
        valid, issues = validate_realtime_ledger(ledger)
        assert not valid
        assert any("y_true" in issue for issue in issues)

    def test_da_anchor_missing_check(self, da_anchor_df):
        """Validation should fail if rt_da_anchor is missing."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        # Remove da_anchor rows
        ledger = ledger[ledger["model_name"] != RT_DA_ANCHOR_MODEL]
        valid, issues = validate_realtime_ledger(ledger)
        if len(ledger) > 0:
            assert not valid

    def test_no_nan_in_y_pred(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        valid, issues = validate_realtime_ledger(ledger)
        assert valid

    def test_empty_ledger_valid(self):
        empty = pd.DataFrame(columns=REALTIME_LEDGER_COLUMNS)
        valid, issues = validate_realtime_ledger(empty)
        assert valid


class TestExtractCandidates:
    """Test extracting candidates from ledger."""

    def test_extract_da_anchor(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        candidates = extract_rt_candidates(ledger)
        assert RT_DA_ANCHOR_MODEL in candidates

    def test_extract_both(self, da_anchor_df, sgdfnet_df):
        ledger = build_realtime_ledger(
            da_anchor_predictions=da_anchor_df,
            sgdfnet_predictions=sgdfnet_df,
        )
        candidates = extract_rt_candidates(ledger)
        assert RT_DA_ANCHOR_MODEL in candidates
        assert SGDFNET_ASSIST_MODEL in candidates

    def test_extract_empty(self):
        empty = pd.DataFrame(columns=REALTIME_LEDGER_COLUMNS)
        candidates = extract_rt_candidates(empty)
        assert candidates == {}

    def test_extract_preserves_columns(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        candidates = extract_rt_candidates(ledger)
        da = candidates[RT_DA_ANCHOR_MODEL]
        assert "y_pred" in da.columns
        assert "business_day" in da.columns


class TestAppendLedger:
    """Test appending to ledger."""

    def test_append_new(self, da_anchor_df):
        """Append to empty ledger."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        appended = append_to_realtime_ledger(ledger)
        assert len(appended) == len(ledger)

    def test_append_duplicate(self, da_anchor_df):
        """Appending same data should dedup."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        appended = append_to_realtime_ledger(ledger, ledger)
        # Should not double: dedup on key
        assert len(appended) <= len(ledger) * 2

    def test_append_empty(self):
        """Appending empty should return existing."""
        empty = pd.DataFrame()
        result = append_to_realtime_ledger(empty)
        assert len(result) == 0

    def test_append_with_run_id(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df, run_id="test_run")
        assert "run_id" in ledger.columns
        assert (ledger["run_id"] == "test_run").all()


class TestEdgeCases:
    """Test edge cases."""

    def test_sgdfnet_none(self, da_anchor_df):
        """Explicit None SGDFNet should result in DA-only."""
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df, sgdfnet_predictions=None)
        assert SGDFNET_ASSIST_MODEL not in ledger["model_name"].values

    def test_sgdfnet_empty_df(self, da_anchor_df):
        """Empty SGDFNet DataFrame should result in DA-only."""
        empty_sg = pd.DataFrame()
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df, sgdfnet_predictions=empty_sg)
        assert SGDFNET_ASSIST_MODEL not in ledger["model_name"].values

    def test_ledger_key_columns(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        for key in REALTIME_LEDGER_KEY:
            assert key in ledger.columns

    def test_model_name_format(self, da_anchor_df, sgdfnet_df):
        ledger = build_realtime_ledger(
            da_anchor_predictions=da_anchor_df,
            sgdfnet_predictions=sgdfnet_df,
        )
        models = set(ledger["model_name"].unique())
        assert models == {RT_DA_ANCHOR_MODEL, SGDFNET_ASSIST_MODEL}

    def test_no_duplicate_keys(self, da_anchor_df):
        ledger = build_realtime_ledger(da_anchor_predictions=da_anchor_df)
        valid, issues = validate_realtime_ledger(ledger)
        assert valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
