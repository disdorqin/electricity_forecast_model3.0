#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for P141 - Negative / Spike Specialized Performance Audit
================================================================
At least 10 tests covering:
  - negative hour detection
  - spike hour detection
  - per-category metrics computation
  - top-50 error hours output format
  - hourly heatmap has 24 entries
  - period breakdown
  - month breakdown
  - audit summary answers key questions
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the scripts directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_p141_negative_spike_performance_audit import (
    classify_hour,
    compute_metrics,
    compute_smape_floor50,
    get_period,
    run_p141_negative_spike_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_predictions_df(
    n_hours: int = 48,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a small synthetic predictions DataFrame for testing."""
    rng = np.random.RandomState(seed)
    ds = pd.date_range("2025-01-01 01:00:00", periods=n_hours, freq="h")
    y_true_vals = rng.uniform(-100, 800, size=n_hours)
    y_pred_vals = y_true_vals + rng.normal(0, 30, size=n_hours)
    hours = [(d.hour if d.hour != 0 else 24) for d in ds]
    periods = [get_period(h) for h in hours]
    return pd.DataFrame({
        "task": "dayahead",
        "model_name": "test_model",
        "target_day": ds.normalize().strftime("%Y-%m-%d"),
        "business_day": ds.normalize().strftime("%Y-%m-%d"),
        "ds": ds.astype(str),
        "hour_business": hours,
        "period": periods,
        "y_pred": y_pred_vals,
        "source_confidence": 1.0,
        "model_version": "1.0.0",
        "y_true": y_true_vals,
    })


@pytest.fixture
def synthetic_predictions_csv(tmp_path: Path) -> str:
    """Write a synthetic predictions CSV and return its path."""
    df = _make_predictions_df(n_hours=720, seed=7)
    path = str(tmp_path / "all_predictions.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_raw_csv(tmp_path: Path) -> str:
    """Write a minimal raw-data CSV (GBK) and return its path."""
    # We only need it to exist; the audit script does not actually read it
    # for the core logic, but we create one so the signature is satisfied.
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    path = str(tmp_path / "raw.csv")
    df.to_csv(path, index=False, encoding="gbk")
    return path


@pytest.fixture
def audit_result(tmp_path: Path, synthetic_predictions_csv: str, synthetic_raw_csv: str) -> dict:
    """Run the audit and return the summary dict."""
    out_dir = str(tmp_path / "p141_out")
    return run_p141_negative_spike_audit(synthetic_predictions_csv, synthetic_raw_csv, out_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifyHour:
    """Test hour-level category classification."""

    def test_negative_detection(self):
        """y_true < 0 -> 'negative'."""
        assert classify_hour(-10.0) == "negative"
        assert classify_hour(-0.01) == "negative"
        assert classify_hour(-100.0) == "negative"

    def test_spike_detection(self):
        """y_true > 500 -> 'spike'."""
        assert classify_hour(500.01) == "spike"
        assert classify_hour(1500.0) == "spike"

    def test_low_price_detection(self):
        """0 <= y_true < 50 -> 'low_price'."""
        assert classify_hour(0.0) == "low_price"
        assert classify_hour(49.99) == "low_price"

    def test_high_price_detection(self):
        """300 <= y_true <= 500 -> 'high_price'."""
        assert classify_hour(300.0) == "high_price"
        assert classify_hour(500.0) == "high_price"

    def test_normal_detection(self):
        """50 <= y_true < 300 -> 'normal'."""
        assert classify_hour(50.0) == "normal"
        assert classify_hour(200.0) == "normal"
        assert classify_hour(299.99) == "normal"


class TestGetPeriod:
    """Test period assignment."""

    def test_period_1_8(self):
        for h in range(1, 9):
            assert get_period(h) == "1_8"

    def test_period_9_16(self):
        for h in range(9, 17):
            assert get_period(h) == "9_16"

    def test_period_17_24(self):
        for h in range(17, 25):
            assert get_period(h) == "17_24"


class TestComputeMetrics:
    """Test metric computation."""

    def test_per_category_metrics_keys(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        m = compute_metrics(y_true, y_pred)
        assert "smape_floor50" in m
        assert "mae" in m
        assert "rmse" in m
        assert "count" in m
        assert m["count"] == 3

    def test_empty_metrics(self):
        m = compute_metrics(np.array([]), np.array([]))
        assert m["count"] == 0
        assert m["smape_floor50"] is None

    def test_smape_floor50_perfect(self):
        y = np.array([100.0, 200.0])
        assert compute_smape_floor50(y, y) == pytest.approx(0.0, abs=1e-6)


class TestAuditOutputs:
    """Integration tests on the full audit run."""

    def test_top50_error_hours_format(self, tmp_path: Path, audit_result: dict):
        """top_50_error_hours.csv has the right columns and <= 50 rows."""
        csv_path = tmp_path / "p141_out" / "top_50_error_hours.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        expected_cols = {"ds", "y_true", "y_pred", "abs_error", "category"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) <= 50
        # abs_error should be descending
        assert df["abs_error"].is_monotonic_decreasing

    def test_hourly_heatmap_has_24_entries(self, tmp_path: Path, audit_result: dict):
        """hourly_heatmap.json has exactly 24 entries."""
        json_path = tmp_path / "p141_out" / "hourly_heatmap.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 24
        hours = [entry["hour"] for entry in data]
        assert hours == list(range(1, 25))
        for entry in data:
            assert "avg_error" in entry
            assert "count" in entry

    def test_period_breakdown(self, audit_result: dict):
        """Audit summary has all three periods."""
        pb = audit_result["period_breakdown"]
        for period_label in ["1_8", "9_16", "17_24"]:
            assert period_label in pb
            assert "smape_floor50" in pb[period_label]
            assert "count" in pb[period_label]

    def test_month_breakdown(self, audit_result: dict):
        """Audit summary has month breakdown with numeric keys."""
        mb = audit_result["month_breakdown"]
        assert len(mb) >= 1
        for key, val in mb.items():
            assert key.isdigit()
            assert "smape_floor50" in val

    def test_audit_summary_answers_key_questions(self, audit_result: dict):
        """Audit summary addresses 'where does error come from'."""
        assert "worst_error_category" in audit_result
        assert "category_error_share_pct" in audit_result
        assert "key_findings" in audit_result
        assert isinstance(audit_result["key_findings"], list)
        assert len(audit_result["key_findings"]) >= 2
        # Error shares should sum close to 100
        total_share = sum(audit_result["category_error_share_pct"].values())
        assert total_share == pytest.approx(100.0, abs=1.0)

    def test_category_json_files_exist(self, tmp_path: Path, audit_result: dict):
        """Per-category JSON files are written."""
        out = tmp_path / "p141_out"
        for cat in ["normal", "negative", "spike", "low_price", "high_price"]:
            fname = f"{cat}_hours_metrics.json"
            assert (out / fname).exists(), f"Missing {fname}"
            with open(out / fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["category"] == cat
            assert "smape_floor50" in data

    def test_negative_and_spike_counts(self, audit_result: dict):
        """Negative and spike counts are non-negative integers."""
        assert audit_result["negative_hours_count"] >= 0
        assert audit_result["spike_hours_count"] >= 0

    def test_overall_metrics_present(self, audit_result: dict):
        """Overall metrics are in the summary."""
        ov = audit_result["overall"]
        assert ov["count"] > 0
        assert ov["smape_floor50"] is not None
        assert ov["mae"] is not None
        assert ov["rmse"] is not None
