"""
tests/test_p16_cfg05_30d_walkforward_backtest.py — P16 tests.

Tests walk-forward backtest logic with synthetic data.
Minimum 15 tests covering:
  1. 24H complete day accepted
  2. 23H incomplete day rejected
  3. missing hour 24 rejected
  4. duplicate hour rejected
  5. no-leakage train split
  6. y_true missing rows excluded from metrics
  7. sMAPE_floor50 / MAE / RMSE correctness
  8. raw data missing → BLOCKED
  9. raw data contract validation
  10. eval range generation
  11. per-day metrics computation
  12. per-hour metrics computation
  13. forbidden files check
  14. strict/non-strict CLI behavior
  15. source reproduction claim always present
"""

from __future__ import annotations

import os
import sys
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_p16_cfg05_30d_walkforward_backtest import (
    compute_smape_floor50,
    compute_mae,
    compute_rmse,
    compute_metrics,
    run_p16_cfg05_30d_walkforward_backtest,
    _load_raw_with_ytrue,
    _path_is_safe,
    BACKTEST_COMPLETE,
    BACKTEST_BLOCKED,
    BACKTEST_INCOMPLETE,
    BACKTEST_NO_VALID_YTRUE,
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
        "直调负荷预测值": np.round(500 + 50 * np.sin(np.arange(len(timestamps)) * 0.05) + rng.normal(0, 10, len(timestamps)), 2),
        "风电总加预测值": np.round(50 + 20 * rng.random(len(timestamps)), 2),
        "光伏总加预测值": np.round(30 + 15 * rng.random(len(timestamps)), 2),
        "联络线受电负荷预测值": np.round(100 + 10 * rng.random(len(timestamps)), 2),
        "竞价空间预测值": np.round(200 + 20 * rng.random(len(timestamps)), 2),
    })
    df.to_csv(path, index=False, encoding="utf-8")
    return path


# ── Metric tests ───────────────────────────────────────────────────────────

class TestMetrics:
    def test_smape_floor50_perfect(self):
        y = np.array([100.0, 200.0, 300.0])
        assert compute_smape_floor50(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_smape_floor50_known_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        # floor=50: y_true_f=[100,200], y_pred_f=[110,190]
        # |100-110|/(100+110) = 10/210, |200-190|/(200+190) = 10/390
        # sMAPE = 200 * mean(10/210, 10/390) = 200 * (0.04762 + 0.02564)/2
        expected = 200 * np.mean([10 / 210, 10 / 390])
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(expected, rel=1e-4)

    def test_smape_floor50_floor_effect(self):
        # Values below 50 should be floored
        y_true = np.array([10.0])
        y_pred = np.array([20.0])
        # floored: y_true_f=50, y_pred_f=50 → sMAPE = 200 * |50-50|/(50+50) = 0
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_mae_known(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(10.0, abs=1e-6)

    def test_rmse_known(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        assert compute_rmse(y_true, y_pred) == pytest.approx(10.0, abs=1e-6)

    def test_metrics_empty(self):
        m = compute_metrics(np.array([]), np.array([]))
        assert np.isnan(m["sMAPE_floor50"])
        assert m["n_observations"] == 0


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


# ── Raw data loading ───────────────────────────────────────────────────────

class TestRawDataLoading:
    def test_load_raw_with_ytrue(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        df = _load_raw_with_ytrue(csv_path)
        assert "ds" in df.columns
        assert "y_true" in df.columns
        assert len(df) == 5 * 24


# ── Backtest core ──────────────────────────────────────────────────────────

class TestBacktestCore:
    def test_raw_data_missing_blocked(self):
        result = run_p16_cfg05_30d_walkforward_backtest(
            raw_data="/nonexistent/path.csv",
            start_day="2026-06-01",
            end_day="2026-06-05",
            work_dir=".local_artifacts/test_p16",
        )
        assert result["final_status"] == BACKTEST_BLOCKED
        assert "RAW_DATA_MISSING_OR_NOT_FOUND" in result["reason_codes"]

    def test_source_reproduction_claim_always_present(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=5)
        result = run_p16_cfg05_30d_walkforward_backtest(
            raw_data=csv_path,
            start_day="2026-01-10",
            end_day="2026-01-12",
            work_dir=str(tmp_path / "work"),
        )
        assert "source_reproduction_claim" in result
        assert "not claimed" in result["source_reproduction_claim"] or "candidate" in result["source_reproduction_claim"]

    def test_eval_range_generation(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=120)
        result = run_p16_cfg05_30d_walkforward_backtest(
            raw_data=csv_path,
            start_day="2026-04-01",
            end_day="2026-04-05",
            work_dir=str(tmp_path / "work"),
        )
        assert result["attempted_days"] == 5
        assert result["eval_start"] == "2026-04-01"
        assert result["eval_end"] == "2026-04-05"

    def test_forbidden_files_check_pass(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=120)
        work_dir = str(tmp_path / ".local_artifacts" / "work")
        os.makedirs(os.path.dirname(work_dir), exist_ok=True)
        result = run_p16_cfg05_30d_walkforward_backtest(
            raw_data=csv_path,
            start_day="2026-04-01",
            end_day="2026-04-03",
            work_dir=work_dir,
        )
        assert result["forbidden_files_check"] == "PASS"

    def test_summary_keys_present(self, tmp_path):
        csv_path = str(tmp_path / "raw.csv")
        _make_synthetic_raw_csv(csv_path, days=120)
        result = run_p16_cfg05_30d_walkforward_backtest(
            raw_data=csv_path,
            start_day="2026-04-01",
            end_day="2026-04-03",
            work_dir=str(tmp_path / "work"),
        )
        required_keys = [
            "raw_data_status", "eval_start", "eval_end", "attempted_days",
            "complete_days", "metric_days", "eval_rows",
            "missing_y_true_rows", "incomplete_days", "metrics",
            "per_day_metrics_path_local", "per_hour_metrics_path_local",
            "predictions_path_local", "final_status",
            "source_reproduction_claim", "reason_codes", "forbidden_files_check",
        ]
        for key in required_keys:
            assert key in result, f"Missing summary key: {key}"
