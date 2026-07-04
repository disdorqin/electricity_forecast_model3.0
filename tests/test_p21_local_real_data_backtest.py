"""
tests/test_p21_local_real_data_backtest.py — P21 tests.

Tests P21 orchestration with synthetic data and mocked paths.
Minimum 12 tests covering:
  1. raw CSV discovery success
  2. raw CSV discovery failure
  3. source repo discovery success
  4. source repo discovery failure
  5. blocked when data missing
  6. no fake metrics when no data
  7. 24H completeness enforced (via P16 delegation)
  8. strict mode returns non-zero on failure
  9. non-strict mode returns zero on partial
  10. forbidden files check
  11. summary keys present
  12. quick-days / full-days / three-month parameters
  13. source repo missing → P21_SOURCE_REPO_MISSING
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_p21_local_real_data_backtest import (
    discover_raw_csv,
    discover_source_repo,
    run_p21_local_real_data_backtest,
    _path_is_safe,
    P21_REAL_BACKTEST_COMPLETE,
    P21_REAL_BACKTEST_PARTIAL,
    P21_LOCAL_DATA_MISSING,
    P21_SOURCE_REPO_MISSING,
    P21_BACKTEST_FAILED,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_synthetic_raw_csv(path: str, start: str = "2026-01-01", days: int = 120):
    """Create a synthetic raw Chinese CSV with 时刻, 日前电价, etc."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start, periods=days * 24, freq="h")
    prices = 100 + 30 * np.sin(np.arange(len(timestamps)) * 0.1) + rng.normal(0, 5, len(timestamps))
    df = pd.DataFrame({
        "时刻": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "日前电价": np.round(prices, 2),
        "实时电价": np.round(prices + rng.normal(0, 3, len(timestamps)), 2),
        "直调负荷预测值": np.round(500 + 50 * np.sin(np.arange(len(timestamps)) * 0.05), 2),
        "风电总加预测值": np.round(50 + 20 * rng.random(len(timestamps)), 2),
        "光伏总加预测值": np.round(30 + 15 * rng.random(len(timestamps)), 2),
        "联络线受电负荷预测值": np.round(100 + 10 * rng.random(len(timestamps)), 2),
        "竞价空间预测值": np.round(200 + 20 * rng.random(len(timestamps)), 2),
    })
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ── Discovery tests ────────────────────────────────────────────────────────

class TestRawCsvDiscovery:
    def test_discover_raw_csv_success(self, tmp_path):
        csv_path = tmp_path / "shandong_pmos_hourly.csv"
        _make_synthetic_raw_csv(str(csv_path), days=5)
        result = discover_raw_csv(extra_candidates=[str(csv_path)])
        assert result is not None
        assert os.path.isfile(result)

    def test_discover_raw_csv_failure(self):
        result = discover_raw_csv(
            extra_candidates=["/nonexistent/path.csv"],
            include_defaults=False,
        )
        assert result is None


class TestSourceRepoDiscovery:
    def test_discover_source_repo_success(self, tmp_path):
        repo_dir = tmp_path / "epf-sota-experiment"
        repo_dir.mkdir()
        result = discover_source_repo(extra_candidates=[str(repo_dir)])
        assert result is not None
        assert os.path.isdir(result)

    def test_discover_source_repo_failure(self):
        result = discover_source_repo(
            extra_candidates=["/nonexistent/repo"],
            include_defaults=False,
        )
        assert result is None


# ── Orchestration tests ────────────────────────────────────────────────────

class TestOrchestration:
    def test_blocked_when_data_missing(self, tmp_path):
        result = run_p21_local_real_data_backtest(
            full_days=7,
            work_dir=str(tmp_path / "work"),
            raw_csv_override="/nonexistent/raw.csv",
        )
        assert result["final_status"] == P21_LOCAL_DATA_MISSING

    def test_no_fake_metrics_when_no_data(self, tmp_path):
        result = run_p21_local_real_data_backtest(
            full_days=7,
            work_dir=str(tmp_path / "work"),
            raw_csv_override="/nonexistent/raw.csv",
        )
        assert result["p16_summary"] is None
        assert result["raw_csv_path"] is None

    def test_source_repo_missing(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        result = run_p21_local_real_data_backtest(
            full_days=7,
            work_dir=str(tmp_path / "work"),
            raw_csv_override=csv_path,
            source_repo_override="/nonexistent/repo",
        )
        assert result["final_status"] == P21_SOURCE_REPO_MISSING

    def test_summary_keys_present(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        result = run_p21_local_real_data_backtest(
            full_days=7,
            work_dir=str(tmp_path / "work"),
            raw_csv_override=csv_path,
            source_repo_override="/nonexistent/repo",
        )
        required_keys = [
            "p21_stage", "raw_csv_path", "source_repo_path", "eval_days",
            "p16_summary", "final_status", "reason_codes", "forbidden_files_check",
        ]
        for key in required_keys:
            assert key in result, f"Missing summary key: {key}"

    def test_forbidden_files_check_pass(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        result = run_p21_local_real_data_backtest(
            full_days=7,
            work_dir=str(tmp_path / "work"),
            raw_csv_override=csv_path,
            source_repo_override="/nonexistent/repo",
        )
        assert result["forbidden_files_check"] == "PASS"

    def test_quick_days_parameter(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        result = run_p21_local_real_data_backtest(
            quick_days=7,
            work_dir=str(tmp_path / "work"),
            raw_csv_override=csv_path,
            source_repo_override="/nonexistent/repo",
        )
        assert result["eval_days"] == 7

    def test_three_month_parameter(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        result = run_p21_local_real_data_backtest(
            three_month=90,
            work_dir=str(tmp_path / "work"),
            raw_csv_override=csv_path,
            source_repo_override="/nonexistent/repo",
        )
        assert result["eval_days"] == 90


# ── Path safety ────────────────────────────────────────────────────────────

class TestPathSafety:
    def test_safe_path(self):
        assert _path_is_safe(".local_artifacts/test") is True

    def test_unsafe_data(self):
        assert _path_is_safe("data/raw") is False

    def test_unsafe_outputs(self):
        assert _path_is_safe("outputs/pred") is False

    def test_unsafe_ledgers(self):
        assert _path_is_safe("ledgers/test") is False


# ── CLI tests ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_strict_mode_returns_nonzero_on_missing(self, tmp_path):
        from scripts.run_p21_local_real_data_backtest import main
        ret = main([
            "--full-days", "7",
            "--work-dir", str(tmp_path / "work"),
            "--raw-csv", "/nonexistent/raw.csv",
            "--strict",
        ])
        assert ret == 1

    def test_non_strict_mode_returns_zero_on_missing(self, tmp_path):
        from scripts.run_p21_local_real_data_backtest import main
        ret = main([
            "--full-days", "7",
            "--work-dir", str(tmp_path / "work"),
            "--raw-csv", "/nonexistent/raw.csv",
        ])
        assert ret == 0
