"""tests/test_p26_cfg05_per_day_retrain_backtest.py — P26 per-day retrain backtest tests."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Status constants ───────────────────────────────────────────────────────

def test_p26_status_constants():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        P26_PER_DAY_RETRAIN_BLOCKED,
        P26_PER_DAY_RETRAIN_COMPLETE,
        P26_PER_DAY_RETRAIN_PARTIAL,
        P26_LOCAL_DATA_MISSING,
    )
    assert P26_PER_DAY_RETRAIN_COMPLETE == "P26_PER_DAY_RETRAIN_COMPLETE"
    assert P26_PER_DAY_RETRAIN_PARTIAL == "P26_PER_DAY_RETRAIN_PARTIAL"
    assert P26_PER_DAY_RETRAIN_BLOCKED == "P26_PER_DAY_RETRAIN_BLOCKED"
    assert P26_LOCAL_DATA_MISSING == "P26_LOCAL_DATA_MISSING"


def test_p26_missing_raw_data():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        P26_LOCAL_DATA_MISSING,
        run_p26_cfg05_per_day_retrain_backtest,
    )
    result = run_p26_cfg05_per_day_retrain_backtest(raw_data="/nonexistent/path.csv")
    assert result["final_status"] == P26_LOCAL_DATA_MISSING
    assert "RAW_DATA_MISSING" in result["reason_codes"]


def test_p26_mode_is_per_day_retrain():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    result = run_p26_cfg05_per_day_retrain_backtest(raw_data="/nonexistent.csv")
    assert result["p26_mode"] == "per_day_retrain"


def test_p26_output_keys():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    result = run_p26_cfg05_per_day_retrain_backtest(raw_data="/nonexistent.csv")
    required_keys = [
        "attempted_days", "complete_days", "metric_days", "eval_rows",
        "sMAPE_floor50", "MAE", "RMSE", "training_time_seconds",
        "failed_days", "reason_codes", "final_status", "improvement_vs_p21",
        "forbidden_files_check",
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_p26_training_time_recorded():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    result = run_p26_cfg05_per_day_retrain_backtest(raw_data="/nonexistent.csv")
    assert result["training_time_seconds"] >= 0


def test_p26_improvement_vs_p21_structure():
    """When no metrics, improvement should be None."""
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    result = run_p26_cfg05_per_day_retrain_backtest(raw_data="/nonexistent.csv")
    assert result["improvement_vs_p21"] is None


def test_p26_forbidden_files_check_default_pass():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    result = run_p26_cfg05_per_day_retrain_backtest(raw_data="/nonexistent.csv")
    assert result["forbidden_files_check"] == "PASS"


def test_p26_delegates_to_p16_with_no_reuse():
    """Verify P26 calls P16 with reuse_model=False."""
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30,
            "complete_days": 30,
            "metric_days": 30,
            "eval_rows": 720,
            "metrics": {"sMAPE_floor50": 18.0, "MAE": 60.0, "RMSE": 80.0, "n_observations": 720},
            "per_day_metrics_path_local": None,
            "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_COMPLETE",
            "reason_codes": [],
        }
        # Need a real file for raw_data check
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            mock_p16.assert_called_once()
            call_kwargs = mock_p16.call_args[1]
            assert call_kwargs.get("reuse_model") is False, "P26 must call P16 with reuse_model=False"
        finally:
            os.unlink(tmp_path)


def test_p26_improvement_computed_when_metrics_exist():
    """When P16 returns metrics, improvement vs P21 should be computed."""
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        P26_PER_DAY_RETRAIN_COMPLETE,
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30,
            "complete_days": 30,
            "metric_days": 30,
            "eval_rows": 720,
            "metrics": {"sMAPE_floor50": 18.0, "MAE": 60.0, "RMSE": 80.0, "n_observations": 720},
            "per_day_metrics_path_local": None,
            "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_COMPLETE",
            "reason_codes": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            assert result["improvement_vs_p21"] is not None
            assert result["improvement_vs_p21"]["p21_smape_floor50"] == 20.71
            assert result["improvement_vs_p21"]["p26_smape_floor50"] == 18.0
            assert result["improvement_vs_p21"]["direction"] == "IMPROVED"
        finally:
            os.unlink(tmp_path)


def test_p26_no_improvement_when_worse():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30,
            "complete_days": 30,
            "metric_days": 30,
            "eval_rows": 720,
            "metrics": {"sMAPE_floor50": 25.0, "MAE": 80.0, "RMSE": 100.0, "n_observations": 720},
            "per_day_metrics_path_local": None,
            "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_COMPLETE",
            "reason_codes": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            assert result["improvement_vs_p21"]["direction"] == "NO_IMPROVEMENT"
        finally:
            os.unlink(tmp_path)


def test_p26_complete_status_mapping():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        P26_PER_DAY_RETRAIN_COMPLETE,
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30, "complete_days": 30, "metric_days": 30,
            "eval_rows": 720,
            "metrics": {"sMAPE_floor50": 18.0, "MAE": 60.0, "RMSE": 80.0, "n_observations": 720},
            "per_day_metrics_path_local": None, "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_COMPLETE", "reason_codes": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            assert result["final_status"] == P26_PER_DAY_RETRAIN_COMPLETE
        finally:
            os.unlink(tmp_path)


def test_p26_partial_status_mapping():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        P26_PER_DAY_RETRAIN_PARTIAL,
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30, "complete_days": 25, "metric_days": 25,
            "eval_rows": 600,
            "metrics": {"sMAPE_floor50": 19.0, "MAE": 65.0, "RMSE": 85.0, "n_observations": 600},
            "per_day_metrics_path_local": None, "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_COMPLETE", "reason_codes": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            assert result["final_status"] == P26_PER_DAY_RETRAIN_PARTIAL
        finally:
            os.unlink(tmp_path)


def test_p26_blocked_status_mapping():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        P26_PER_DAY_RETRAIN_BLOCKED,
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30, "complete_days": 0, "metric_days": 0,
            "eval_rows": 0, "metrics": None,
            "per_day_metrics_path_local": None, "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_NO_VALID_YTRUE", "reason_codes": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            assert result["final_status"] == P26_PER_DAY_RETRAIN_BLOCKED
        finally:
            os.unlink(tmp_path)


def test_p26_failed_days_counted():
    from scripts.run_p26_cfg05_per_day_retrain_backtest import (
        run_p26_cfg05_per_day_retrain_backtest,
    )
    with patch("scripts.run_p16_cfg05_30d_walkforward_backtest.run_p16_cfg05_30d_walkforward_backtest") as mock_p16:
        mock_p16.return_value = {
            "attempted_days": 30, "complete_days": 28, "metric_days": 28,
            "eval_rows": 672,
            "metrics": {"sMAPE_floor50": 19.0, "MAE": 65.0, "RMSE": 85.0, "n_observations": 672},
            "per_day_metrics_path_local": None, "per_hour_metrics_path_local": None,
            "predictions_path_local": None,
            "final_status": "CFG05_BACKTEST_COMPLETE", "reason_codes": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            result = run_p26_cfg05_per_day_retrain_backtest(
                raw_data=tmp_path, work_dir=tempfile.mkdtemp()
            )
            assert result["failed_days"] == 2
        finally:
            os.unlink(tmp_path)
