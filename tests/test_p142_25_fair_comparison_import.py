#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for P142 - 2.5 Fair Comparison (If Available)
=====================================================
At least 8 tests covering:
  - search logic checks all expected paths
  - missing artefacts -> 2.5_COMPARISON_UNAVAILABLE
  - no fabricated data when unavailable
  - comparison format when available
  - same-window evaluation requirement
  - search log records all paths checked
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the scripts directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_p142_25_fair_comparison_import import (
    STATUS_AVAILABLE,
    STATUS_UNAVAILABLE,
    _detect_columns,
    _evaluate_25_predictions,
    _find_predictions_csv,
    _looks_like_25,
    _search_paths,
    compute_metrics,
    compute_smape_floor50,
    run_p142_fair_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_repo(tmp_path: Path) -> Path:
    """Create a minimal repo structure with no 2.5 artefacts."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / ".local_artifacts").mkdir()
    (repo / ".local_artifacts" / "p2025_full" / "dayahead").mkdir(parents=True)
    return repo


@pytest.fixture
def repo_with_25(tmp_path: Path) -> Path:
    """Create a repo structure with synthetic 2.5 predictions."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / ".local_artifacts").mkdir()

    # Create a fake 2.5 directory with predictions
    v25_dir = repo / ".local_artifacts" / "model_v2.5_predictions"
    v25_dir.mkdir()

    rng = np.random.RandomState(99)
    n = 200
    ds = pd.date_range("2025-01-01 01:00:00", periods=n, freq="h")
    y_true = rng.uniform(50, 600, size=n)
    y_pred = y_true + rng.normal(0, 25, size=n)

    pred_df = pd.DataFrame({
        "ds": ds.astype(str),
        "y_pred": y_pred,
        "y_true": y_true,
    })
    pred_df.to_csv(v25_dir / "all_predictions.csv", index=False)

    # Also create a 3.0 cfg05 predictions file for comparison
    cfg05_dir = repo / ".local_artifacts" / "p2025_full" / "dayahead"
    cfg05_dir.mkdir(parents=True)
    y_pred_30 = y_true + rng.normal(0, 35, size=n)
    cfg05_df = pd.DataFrame({
        "task": "dayahead",
        "model_name": "lightgbm_cfg05_dayahead",
        "target_day": ds.normalize().strftime("%Y-%m-%d"),
        "business_day": ds.normalize().strftime("%Y-%m-%d"),
        "ds": ds.astype(str),
        "hour_business": [d.hour if d.hour != 0 else 24 for d in ds],
        "period": ["1_8"] * n,
        "y_pred": y_pred_30,
        "source_confidence": 1.0,
        "model_version": "1.0.0",
        "y_true": y_true,
    })
    cfg05_df.to_csv(cfg05_dir / "all_predictions.csv", index=False)

    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSearchLogic:
    """Test that search logic checks all expected paths."""

    def test_checks_all_candidate_paths(self, empty_repo: Path):
        """Search should check at least the 3 well-known candidate paths."""
        log = _search_paths(empty_repo)
        # At minimum the 3 candidate relative paths should be in the log
        assert len(log) >= 3
        paths_checked = [e["path"] for e in log]
        # The well-known paths should appear (resolved to absolute)
        for rel in ["../electricity_forecast_model2.5",
                     "../electricity_forecast_model2.0_exp",
                     ".local_artifacts/source_repos/electricity_forecast_model2.5"]:
            expected = (empty_repo / rel).resolve()
            assert any(str(expected) in p for p in paths_checked), \
                f"Missing expected search path: {expected}"

    def test_search_log_records_existence(self, empty_repo: Path):
        """Each log entry records whether the directory exists."""
        log = _search_paths(empty_repo)
        for entry in log:
            assert "exists" in entry
            assert isinstance(entry["exists"], bool)

    def test_finds_predictions_when_present(self, repo_with_25: Path):
        """Search should find the predictions CSV in the 2.5 directory."""
        log = _search_paths(repo_with_25)
        found = [e for e in log if e["predictions"] is not None]
        assert len(found) >= 1, "Should find at least one predictions CSV"
        assert "all_predictions.csv" in found[0]["predictions"]


class TestUnavailablePath:
    """Test behaviour when 2.5 artefacts are NOT found."""

    def test_status_unavailable(self, empty_repo: Path, tmp_path: Path):
        """When no 2.5 data exists, status must be 2.5_COMPARISON_UNAVAILABLE."""
        out_dir = str(tmp_path / "p142_out")
        # Patch repo_root to point to our empty repo
        script_path = empty_repo / "scripts" / "run_p142_25_fair_comparison_import.py"
        script_path.write_text("# placeholder")
        with patch("run_p142_25_fair_comparison_import.Path") as mock_path_cls:
            # We need a more surgical approach: just call with the empty repo
            pass
        # Simpler: directly call with the empty repo by temporarily changing __file__
        import run_p142_25_fair_comparison_import as mod
        original_file = mod.__file__
        mod.__file__ = str(empty_repo / "scripts" / "run_p142_25_fair_comparison_import.py")
        try:
            result = run_p142_fair_comparison(out_dir)
        finally:
            mod.__file__ = original_file

        assert result["status"] == STATUS_UNAVAILABLE

    def test_no_fabricated_data(self, empty_repo: Path, tmp_path: Path):
        """When unavailable, comparison must be None -- no fabricated data."""
        out_dir = str(tmp_path / "p142_out")
        import run_p142_25_fair_comparison_import as mod
        original_file = mod.__file__
        mod.__file__ = str(empty_repo / "scripts" / "run_p142_25_fair_comparison_import.py")
        try:
            result = run_p142_fair_comparison(out_dir)
        finally:
            mod.__file__ = original_file

        assert result["comparison"] is None
        assert result["paths_with_predictions"] == 0
        assert "reason" in result
        assert "NOT fabricated" in result["reason"]

    def test_search_log_written(self, empty_repo: Path, tmp_path: Path):
        """search_log.json is written even when unavailable."""
        out_dir = str(tmp_path / "p142_out")
        import run_p142_25_fair_comparison_import as mod
        original_file = mod.__file__
        mod.__file__ = str(empty_repo / "scripts" / "run_p142_25_fair_comparison_import.py")
        try:
            run_p142_fair_comparison(out_dir)
        finally:
            mod.__file__ = original_file

        log_path = Path(out_dir) / "search_log.json"
        assert log_path.exists()
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        assert isinstance(log_data, list)
        assert len(log_data) >= 3


class TestAvailablePath:
    """Test behaviour when 2.5 artefacts ARE found."""

    def test_status_available(self, repo_with_25: Path, tmp_path: Path):
        """When 2.5 data exists, status must be 2.5_COMPARISON_AVAILABLE."""
        out_dir = str(tmp_path / "p142_out")
        import run_p142_25_fair_comparison_import as mod
        original_file = mod.__file__
        mod.__file__ = str(repo_with_25 / "scripts" / "run_p142_25_fair_comparison_import.py")
        try:
            result = run_p142_fair_comparison(out_dir)
        finally:
            mod.__file__ = original_file

        assert result["status"] == STATUS_AVAILABLE

    def test_comparison_format(self, repo_with_25: Path, tmp_path: Path):
        """Comparison dict has expected structure when available."""
        out_dir = str(tmp_path / "p142_out")
        import run_p142_25_fair_comparison_import as mod
        original_file = mod.__file__
        mod.__file__ = str(repo_with_25 / "scripts" / "run_p142_25_fair_comparison_import.py")
        try:
            result = run_p142_fair_comparison(out_dir)
        finally:
            mod.__file__ = original_file

        comp = result["comparison"]
        assert comp is not None
        assert "model_25" in comp
        assert "model_30_cfg05" in comp
        assert "window" in comp
        assert comp["window"]["start"] == "2025-01-01"
        assert comp["window"]["end"] == "2025-12-31"

    def test_same_window_evaluation(self, repo_with_25: Path, tmp_path: Path):
        """2.5 and 3.0 are evaluated on the same date window."""
        out_dir = str(tmp_path / "p142_out")
        import run_p142_25_fair_comparison_import as mod
        original_file = mod.__file__
        mod.__file__ = str(repo_with_25 / "scripts" / "run_p142_25_fair_comparison_import.py")
        try:
            result = run_p142_fair_comparison(out_dir, day_start="2025-01-01", day_end="2025-01-05")
        finally:
            mod.__file__ = original_file

        comp = result["comparison"]
        assert comp is not None
        assert comp["window"]["start"] == "2025-01-01"
        assert comp["window"]["end"] == "2025-01-05"
        # Both models should have been evaluated
        assert comp["model_25"]["count"] > 0
        assert comp["model_30_cfg05"] is not None
        assert comp["model_30_cfg05"]["count"] > 0


class TestHelpers:
    """Test helper functions."""

    def test_looks_like_25(self):
        assert _looks_like_25("model_v2.5_predictions") is True
        assert _looks_like_25("electricity_forecast_model2.5") is True
        assert _looks_like_25("some_2_5_data") is True
        assert _looks_like_25("p2025_full") is False
        assert _looks_like_25("random_dir") is False

    def test_detect_columns(self):
        df = pd.DataFrame({"ds": ["2025-01-01"], "y_pred": [100], "y_true": [110]})
        result = _detect_columns(df)
        assert result is not None
        assert result == ("ds", "y_pred", "y_true")

    def test_detect_columns_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert _detect_columns(df) is None

    def test_evaluate_25_predictions_window_filter(self, tmp_path: Path):
        """Predictions outside the window are excluded."""
        rng = np.random.RandomState(0)
        ds = pd.date_range("2025-06-01", periods=48, freq="h")
        df = pd.DataFrame({
            "ds": ds.astype(str),
            "y_pred": rng.uniform(100, 400, 48),
            "y_true": rng.uniform(100, 400, 48),
        })
        csv_path = tmp_path / "pred.csv"
        df.to_csv(csv_path, index=False)

        # Window that doesn't overlap
        result = _evaluate_25_predictions(csv_path, "2025-01-01", "2025-01-02")
        assert result is None

        # Window that overlaps
        result = _evaluate_25_predictions(csv_path, "2025-06-01", "2025-06-02")
        assert result is not None
        assert result["count"] > 0
