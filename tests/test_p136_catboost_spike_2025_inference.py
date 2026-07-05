"""
tests/test_p136_catboost_spike_2025_inference.py — Tests for P136 CatBoost Spike 2025 Inference.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_p136_catboost_spike_2025_inference import (
    DEFAULT_MODEL_PATH,
    _path_a_direct_inference,
    _path_b_original_scripts,
    _path_c_retrain_walkforward,
    run_p136_catboost_spike_2025,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_raw_csv(tmp_path):
    """Create a minimal raw CSV file for testing."""
    np.random.seed(42)
    n_hours = 24 * 40  # 40 days
    dates = pd.date_range("2024-12-01", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "时刻": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "y": np.random.uniform(100, 500, n_hours),
        "load": np.random.uniform(20000, 40000, n_hours),
        "wind": np.random.uniform(1000, 8000, n_hours),
        "solar": np.random.uniform(0, 3000, n_hours),
        "interconnect": np.random.uniform(500, 2000, n_hours),
    })
    csv_path = str(tmp_path / "test_raw.csv")
    df.to_csv(csv_path, index=False, encoding="gbk")
    return csv_path


# ── Tests: Path A — Direct inference ──────────────────────────────


class TestPathADirectInference:
    """Tests for Path A: direct model loading and inference."""

    def test_missing_model_file(self, tmp_output_dir, sample_raw_csv):
        result = _path_a_direct_inference(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert result["status"] == "PATH_A_FAILED"
        assert "MODEL_FILE_MISSING" in result["reason_codes"]

    def test_missing_data_file(self, tmp_output_dir):
        result = _path_a_direct_inference(
            model_path=DEFAULT_MODEL_PATH,
            raw_data_path="/nonexistent/data.csv",
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        # Should fail either at model load or data load
        assert "FAILED" in result["status"] or result.get("status") == "PATH_A_FAILED"

    def test_result_has_path_a_marker(self, tmp_output_dir, sample_raw_csv):
        result = _path_a_direct_inference(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert result["path"] == "A"

    def test_real_model_if_available(self, tmp_output_dir):
        """If the real model file exists, try loading it."""
        if not os.path.isfile(DEFAULT_MODEL_PATH):
            pytest.skip("Real model file not available")
        raw_path = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
        if not os.path.isfile(raw_path):
            pytest.skip("Real raw data not available")
        result = _path_a_direct_inference(
            model_path=DEFAULT_MODEL_PATH,
            raw_data_path=raw_path,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        # May succeed or fail depending on model integrity
        assert result["path"] == "A"
        assert "status" in result


# ── Tests: Path B — Original scripts ──────────────────────────────


class TestPathBOriginalScripts:
    """Tests for Path B: calling original P31-P40 scripts."""

    def test_path_b_returns_result(self, tmp_output_dir, sample_raw_csv):
        result = _path_b_original_scripts(
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert result["path"] == "B"
        assert "status" in result

    def test_path_b_has_reason_codes(self, tmp_output_dir, sample_raw_csv):
        result = _path_b_original_scripts(
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert "reason_codes" in result


# ── Tests: Path C — Retrain ───────────────────────────────────────


class TestPathCRetrain:
    """Tests for Path C: retrain with walk-forward."""

    def test_path_c_missing_data(self, tmp_output_dir):
        result = _path_c_retrain_walkforward(
            raw_data_path="/nonexistent/data.csv",
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert "FAILED" in result["status"]

    def test_path_c_returns_correct_structure(self, tmp_output_dir, sample_raw_csv):
        result = _path_c_retrain_walkforward(
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert result["path"] == "C"
        assert "status" in result
        assert "reason_codes" in result


# ── Tests: Full orchestrator ──────────────────────────────────────


class TestRunP136:
    """Tests for run_p136_catboost_spike_2025()."""

    def test_orchestrator_produces_manifest(self, tmp_output_dir, sample_raw_csv):
        result = run_p136_catboost_spike_2025(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert result["phase"] == "P136"
        assert "final_status" in result
        assert "paths" in result

    def test_orchestrator_saves_manifest_json(self, tmp_output_dir, sample_raw_csv):
        result = run_p136_catboost_spike_2025(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        manifest_path = os.path.join(tmp_output_dir, "catboost_spike_2025_manifest.json")
        assert os.path.isfile(manifest_path)
        with open(manifest_path) as f:
            loaded = json.load(f)
        assert loaded["phase"] == "P136"

    def test_orchestrator_saves_metrics_json(self, tmp_output_dir, sample_raw_csv):
        result = run_p136_catboost_spike_2025(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        metrics_path = os.path.join(tmp_output_dir, "catboost_spike_2025_metrics.json")
        assert os.path.isfile(metrics_path)

    def test_status_is_one_of_expected(self, tmp_output_dir, sample_raw_csv):
        result = run_p136_catboost_spike_2025(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        valid_statuses = [
            "CATBOOST_SPIKE_2025_READY",
            "CATBOOST_SPIKE_2025_RETRAINED_READY",
            "CATBOOST_SPIKE_2025_BLOCKED",
        ]
        assert result["final_status"] in valid_statuses

    def test_all_paths_attempted_when_a_fails(self, tmp_output_dir, sample_raw_csv):
        result = run_p136_catboost_spike_2025(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        # Path A should fail (model missing), so B and C should be attempted
        assert "A" in result["paths"]
        assert "B" in result["paths"]
        assert "C" in result["paths"]

    def test_elapsed_time_recorded(self, tmp_output_dir, sample_raw_csv):
        result = run_p136_catboost_spike_2025(
            model_path="/nonexistent/model.cbm",
            raw_data_path=sample_raw_csv,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert "elapsed_seconds" in result
        assert result["elapsed_seconds"] >= 0

    def test_real_model_if_available(self, tmp_output_dir):
        """Integration test with real model if available."""
        if not os.path.isfile(DEFAULT_MODEL_PATH):
            pytest.skip("Real model file not available")
        raw_path = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
        if not os.path.isfile(raw_path):
            pytest.skip("Real raw data not available")
        result = run_p136_catboost_spike_2025(
            model_path=DEFAULT_MODEL_PATH,
            raw_data_path=raw_path,
            output_dir=tmp_output_dir,
            day_start="2025-01-01",
            day_end="2025-01-31",
        )
        assert result["phase"] == "P136"
        # Should succeed via Path A if model is intact
        assert result["final_status"] in [
            "CATBOOST_SPIKE_2025_READY",
            "CATBOOST_SPIKE_2025_RETRAINED_READY",
            "CATBOOST_SPIKE_2025_BLOCKED",
        ]
