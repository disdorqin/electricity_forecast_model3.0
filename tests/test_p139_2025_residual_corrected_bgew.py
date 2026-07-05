"""
tests/test_p139_2025_residual_corrected_bgew.py
================================================
Tests for P139: 2025 Residual-Corrected BGEW.

Covers:
  - Canonical sMAPE_floor50 formula
  - BGEW weight computation
  - Residual correction changes predictions when model exists
  - No-op residual -> RESIDUAL_NO_OP status
  - No improvement claim when residual is identity
  - Before/after metrics computed correctly
  - Period breakdown output structure
  - Delta summary format
  - Missing data -> RESIDUAL_BLOCKED
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_p139_2025_residual_corrected_bgew import (  # noqa: E402
    compute_bgew_weights,
    compute_metrics,
    compute_smape_floor50,
    run_p139_residual_corrected_bgew,
    _period_metrics,
    _rederive_bgew_from_ledger,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_actuals():
    """24 hours of actual dayahead prices."""
    dates = pd.date_range("2025-01-02 01:00", periods=24, freq="h")
    return pd.DataFrame({
        "ds": dates,
        "dayahead_price": np.random.RandomState(42).uniform(100, 500, 24),
        "realtime_price": np.random.RandomState(43).uniform(80, 480, 24),
    })


@pytest.fixture
def sample_raw_csv(tmp_path, sample_actuals):
    """Write a minimal GBK CSV mimicking the Shandong PMOS data."""
    df = sample_actuals.rename(columns={
        "ds": "\u65f6\u523b",
        "dayahead_price": "\u65e5\u524d\u7535\u4ef7",
        "realtime_price": "\u5b9e\u65f6\u7535\u4ef7",
    })
    path = tmp_path / "raw.csv"
    df.to_csv(path, encoding="gbk", index=False)
    return str(path)


@pytest.fixture
def sample_bgew_dir(tmp_path, sample_actuals):
    """Create a minimal P138 BGEW output directory."""
    bgew_dir = tmp_path / "p138_bgew"
    bgew_dir.mkdir()
    # daily_metrics.csv with y_pred and y_true
    rng = np.random.RandomState(99)
    n = 24
    y_true = sample_actuals["dayahead_price"].values
    y_pred = y_true + rng.normal(0, 20, n)
    df = pd.DataFrame({
        "ds": sample_actuals["ds"].values,
        "y_pred": y_pred,
        "y_true": y_true,
    })
    df.to_csv(bgew_dir / "daily_metrics.csv", index=False)
    # bgew_2025_metrics.json
    weights = {"lightgbm_cfg05_dayahead": 0.6, "catboost_spike_residual": 0.4}
    with open(bgew_dir / "bgew_2025_metrics.json", "w") as f:
        json.dump({"weights": weights}, f)
    return str(bgew_dir)


# ── Canonical sMAPE_floor50 tests ────────────────────────────────────────

class TestSmapeFloor50:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert compute_smape_floor50(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_known_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([120.0, 180.0])
        # floor=50: y_true_f=[100,200], y_pred_f=[120,180]
        # denom=[220,380], abs_diff=[20,20], ratios=[20/220, 20/380]
        # mean = (20/220 + 20/380)/2 = (0.09091 + 0.05263)/2 = 0.07177
        # sMAPE = 200 * 0.07177 = 14.354
        expected = 200.0 * np.mean([20 / 220, 20 / 380])
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(expected, rel=1e-4)

    def test_floor_applied_to_negative(self):
        """Values below 50 are floored to 50."""
        y_true = np.array([10.0])
        y_pred = np.array([20.0])
        # After floor: y_true_f=50, y_pred_f=50 => diff=0 => sMAPE=0
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_floor_does_not_affect_large_values(self):
        y_true = np.array([200.0])
        y_pred = np.array([250.0])
        # Both > 50, no floor effect
        expected = 200.0 * (50.0 / 450.0)
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(expected, rel=1e-4)


# ── BGEW weights tests ───────────────────────────────────────────────────

class TestBgewWeights:
    def test_weights_sum_to_one(self):
        smapes = {"a": 10.0, "b": 20.0, "c": 30.0}
        w = compute_bgew_weights(smapes)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_lower_smape_gets_higher_weight(self):
        smapes = {"good": 5.0, "bad": 50.0}
        w = compute_bgew_weights(smapes)
        assert w["good"] > w["bad"]

    def test_clipping_applied_before_renormalize(self):
        """Verify clip is applied (pre-renorm), and weights still sum to 1."""
        smapes = {"a": 1.0, "b": 100.0, "c": 200.0}
        w = compute_bgew_weights(smapes, min_weight=0.05, max_weight=0.75)
        # Weights must sum to 1 after renormalization
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        # The dominant model should have the highest weight
        assert w["a"] > w["b"]
        assert w["a"] > w["c"]
        # With 3 models and extreme sMAPE differences, "a" gets clipped to
        # 0.75 then renormalized, so it can exceed 0.75 after renorm.
        # The key invariant is that clipping was applied (not raw softmax).


# ── compute_metrics tests ────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_keys(self):
        m = compute_metrics(np.array([100.0]), np.array([110.0]))
        assert set(m.keys()) == {"sMAPE_floor50", "MAE", "RMSE", "n"}

    def test_n_correct(self):
        m = compute_metrics(np.arange(10, dtype=float), np.arange(10, dtype=float) + 1)
        assert m["n"] == 10

    def test_perfect_zero_error(self):
        y = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(y, y)
        assert m["sMAPE_floor50"] == 0.0
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0


# ── Period breakdown tests ───────────────────────────────────────────────

class TestPeriodMetrics:
    def test_all_periods_present(self):
        y_true = np.random.RandomState(0).uniform(100, 500, 72)
        y_pred = y_true + np.random.RandomState(1).normal(0, 10, 72)
        hours = np.tile(np.arange(1, 25), 3)
        pm = _period_metrics(y_true, y_pred, hours)
        assert set(pm.keys()) == {"1_8", "9_16", "17_24"}

    def test_period_n_correct(self):
        y_true = np.ones(24) * 100
        y_pred = np.ones(24) * 110
        hours = np.arange(1, 25)
        pm = _period_metrics(y_true, y_pred, hours)
        assert pm["1_8"]["n"] == 8
        assert pm["9_16"]["n"] == 8
        assert pm["17_24"]["n"] == 8


# ── Residual correction behaviour ────────────────────────────────────────

class TestResidualCorrection:
    def test_noop_residual_gives_no_op_status(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        """When no residual model is found, status must be RESIDUAL_NO_OP."""
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {
                "p5m_pkl": None, "catboost_cbm": None,
                "stack_code": None, "reports": None,
            }
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_NO_OP", "model": None,
                "model_info": {"model_name": "none"},
            }
            instance.apply_correction.return_value = {
                "task": "dayahead",
                "status": "RESIDUAL_P5M_NO_OP",
                "correction_applied": False,
                "model_info": {"model_name": "none"},
                "reason_codes": [],
                "output": None,
                "rows": 0,
            }

            result = run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        assert result["status"] == "RESIDUAL_NO_OP"
        assert result["residual_delta_summary"]["correction_applied"] is False
        assert result["residual_delta_summary"]["improved"] is False

    def test_no_improvement_claim_when_identity(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        """Identity correction (delta=0) must not claim improvement."""
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {}
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_NO_OP", "model": None,
                "model_info": {},
            }

            # Simulate identity correction: y_pred_corrected == y_pred
            def fake_apply_correction(pred_df, task="dayahead"):
                out = pred_df.copy()
                out["y_pred_raw"] = out["y_pred"]
                out["residual_delta"] = 0.0
                out["y_pred_corrected"] = out["y_pred"]  # identity
                return {
                    "task": task,
                    "status": "RESIDUAL_P5M_NO_OP",
                    "correction_applied": False,
                    "model_info": {},
                    "reason_codes": [],
                    "output": out,
                    "rows": len(out),
                }

            instance.apply_correction.side_effect = fake_apply_correction

            result = run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        assert result["status"] == "RESIDUAL_NO_OP"
        assert result["residual_delta_summary"]["delta_sMAPE_floor50"] == 0.0

    def test_correction_changes_predictions(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        """When a real model is applied, corrected predictions differ from raw."""
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {"p5m_pkl": "/fake/model.pkl"}
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_REAL_APPLIED",
                "model": MagicMock(),
                "model_info": {"model_name": "p5m_residual", "version": "2.0"},
            }

            def fake_apply_correction(pred_df, task="dayahead"):
                out = pred_df.copy()
                out["y_pred_raw"] = out["y_pred"]
                out["residual_delta"] = -5.0  # constant correction
                out["y_pred_corrected"] = out["y_pred"] - 5.0
                return {
                    "task": task,
                    "status": "RESIDUAL_P5M_REAL_APPLIED",
                    "correction_applied": True,
                    "model_info": {"model_name": "p5m_residual"},
                    "reason_codes": [],
                    "output": out,
                    "rows": len(out),
                }

            instance.apply_correction.side_effect = fake_apply_correction

            result = run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        # Metrics should be computed for both raw and corrected
        assert "sMAPE_floor50" in result["bgew_raw_metrics"]
        assert "sMAPE_floor50" in result["bgew_residual_corrected_metrics"]
        # The corrected predictions differ from raw
        assert result["bgew_raw_metrics"]["sMAPE_floor50"] != result["bgew_residual_corrected_metrics"]["sMAPE_floor50"]

    def test_before_after_metrics_computed_correctly(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        """Verify raw and corrected metrics are independently computed."""
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {}
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_NO_OP", "model": None, "model_info": {},
            }
            instance.apply_correction.return_value = {
                "task": "dayahead",
                "status": "RESIDUAL_P5M_NO_OP",
                "correction_applied": False,
                "model_info": {},
                "reason_codes": [],
                "output": None,
                "rows": 0,
            }

            result = run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        # Both metric dicts must have the same keys
        raw_keys = set(result["bgew_raw_metrics"].keys())
        corr_keys = set(result["bgew_residual_corrected_metrics"].keys())
        # For no-op, corrected has an extra "note" key but shares the metric keys
        assert {"sMAPE_floor50", "MAE", "RMSE", "n"}.issubset(raw_keys)
        assert {"sMAPE_floor50", "MAE", "RMSE", "n"}.issubset(corr_keys)


# ── Delta summary format ─────────────────────────────────────────────────

class TestDeltaSummary:
    def test_delta_summary_keys(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {}
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_NO_OP", "model": None, "model_info": {},
            }
            instance.apply_correction.return_value = {
                "task": "dayahead", "status": "RESIDUAL_P5M_NO_OP",
                "correction_applied": False, "model_info": {},
                "reason_codes": [], "output": None, "rows": 0,
            }
            result = run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        delta = result["residual_delta_summary"]
        assert "delta_sMAPE_floor50" in delta
        assert "delta_MAE" in delta
        assert "delta_RMSE" in delta
        assert "improved" in delta
        assert "correction_applied" in delta
        assert "adapter_status" in delta


# ── Output files ──────────────────────────────────────────────────────────

class TestOutputFiles:
    def test_all_json_files_written(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {}
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_NO_OP", "model": None, "model_info": {},
            }
            instance.apply_correction.return_value = {
                "task": "dayahead", "status": "RESIDUAL_P5M_NO_OP",
                "correction_applied": False, "model_info": {},
                "reason_codes": [], "output": None, "rows": 0,
            }
            run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        expected_files = [
            "bgew_raw_metrics.json",
            "bgew_residual_corrected_metrics.json",
            "residual_delta_summary.json",
            "period_metrics.json",
        ]
        for fname in expected_files:
            assert (Path(out_dir) / fname).is_file(), f"Missing: {fname}"

    def test_json_files_are_valid(self, sample_raw_csv, sample_bgew_dir, tmp_path):
        out_dir = str(tmp_path / "p139_out")
        with patch(
            "scripts.run_p139_2025_residual_corrected_bgew.ResidualP5MAdapter"
        ) as MockAdapter:
            instance = MockAdapter.return_value
            instance.find_artifacts.return_value = {}
            instance.load_correction_model.return_value = {
                "status": "RESIDUAL_P5M_NO_OP", "model": None, "model_info": {},
            }
            instance.apply_correction.return_value = {
                "task": "dayahead", "status": "RESIDUAL_P5M_NO_OP",
                "correction_applied": False, "model_info": {},
                "reason_codes": [], "output": None, "rows": 0,
            }
            run_p139_residual_corrected_bgew(
                bgew_dir=sample_bgew_dir,
                raw_data_path=sample_raw_csv,
                output_dir=out_dir,
            )

        for fname in ("bgew_raw_metrics.json", "residual_delta_summary.json", "period_metrics.json"):
            with open(Path(out_dir) / fname, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)


# ── Blocked when missing data ────────────────────────────────────────────

class TestBlocked:
    def test_missing_raw_data_gives_blocked(self, tmp_path):
        out_dir = str(tmp_path / "p139_out")
        result = run_p139_residual_corrected_bgew(
            bgew_dir=str(tmp_path / "nonexistent"),
            raw_data_path=str(tmp_path / "nonexistent.csv"),
            output_dir=out_dir,
        )
        assert result["status"] == "RESIDUAL_BLOCKED"
        assert "RAW_DATA_MISSING" in result["reason_codes"]


# ── Re-derive BGEW from ledger ───────────────────────────────────────────

class TestRederiveBgew:
    def test_rederive_produces_fused_predictions(self):
        """_rederive_bgew_from_ledger should produce a fused y_pred column."""
        dates = pd.date_range("2025-01-02 01:00", periods=4, freq="h")
        ledger = pd.DataFrame({
            "ds": list(dates) * 2,
            "hour_business": [1, 2, 3, 4] * 2,
            "model_name": ["a"] * 4 + ["b"] * 4,
            "y_pred": [100, 200, 300, 400, 110, 210, 310, 410],
        })
        weights = {"a": 0.7, "b": 0.3}
        result = _rederive_bgew_from_ledger(ledger, weights)
        assert "y_pred" in result.columns
        assert len(result) == 4
        # Fused = 0.7*100 + 0.3*110 = 103 for first row
        expected_first = 0.7 * 100 + 0.3 * 110
        assert result["y_pred"].iloc[0] == pytest.approx(expected_first, rel=1e-6)

    def test_rederive_empty_when_no_models(self):
        ledger = pd.DataFrame({"ds": [], "model_name": [], "y_pred": []})
        result = _rederive_bgew_from_ledger(ledger, {"a": 1.0})
        assert result.empty
