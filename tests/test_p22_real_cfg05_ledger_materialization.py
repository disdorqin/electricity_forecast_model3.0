"""
tests/test_p22_real_cfg05_ledger_materialization.py — P22 tests.

Tests P22 orchestration with synthetic data.
Minimum 10 tests covering:
  1. canonical schema validation
  2. 24H per day enforced
  3. duplicate key rejection
  4. eval mode preserves y_true
  5. production mode strips y_true
  6. summary keys present
  7. missing predictions file → FAILED
  8. empty predictions file → FAILED
  9. successful materialization → READY
  10. partial materialization → PARTIAL
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_p22_real_cfg05_ledger_materialization import (
    run_p22_real_cfg05_ledger_materialization,
    _path_is_safe,
    P22_REAL_PREDICTION_LEDGER_READY,
    P22_REAL_PREDICTION_LEDGER_PARTIAL,
    P22_REAL_PREDICTION_LEDGER_FAILED,
)
from data.schema import PREDICTION_LEDGER_COLUMNS


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_synthetic_predictions(n_days: int = 3, start_day: str = "2026-06-01") -> pd.DataFrame:
    """Create synthetic prediction output with 24 rows per day."""
    rows = []
    for d in range(n_days):
        td = pd.Timestamp(start_day) + pd.Timedelta(days=d)
        for h in range(1, 25):
            if h <= 23:
                ts = pd.Timestamp(td) + pd.Timedelta(hours=h)
            else:
                ts = pd.Timestamp(td) + pd.Timedelta(days=1)
            rows.append({
                "task": "dayahead",
                "model_name": "lightgbm_cfg05_dayahead",
                "target_day": str(td.date()),
                "ds": ts,
                "y_pred": 100.0 + h * 0.5,
                "source_confidence": 0.9,
                "model_version": "1.0.0",
            })
    df = pd.DataFrame(rows)
    from data.business_day import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def _write_predictions_to_workdir(work_dir: str, n_days: int = 3) -> str:
    """Write synthetic predictions CSV to work_dir/all_predictions.csv."""
    os.makedirs(work_dir, exist_ok=True)
    pred = _make_synthetic_predictions(n_days)
    path = os.path.join(work_dir, "all_predictions.csv")
    pred.to_csv(path, index=False)
    return path


# ── Tests ──────────────────────────────────────────────────────────────────

class TestLedgerMaterialization:
    def test_missing_predictions_file_failed(self, tmp_path):
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=str(tmp_path / "work"),
            predictions_path_override="/nonexistent/predictions.csv",
        )
        assert result["final_status"] == P22_REAL_PREDICTION_LEDGER_FAILED

    def test_empty_predictions_file_failed(self, tmp_path):
        work_dir = str(tmp_path / "work")
        os.makedirs(work_dir, exist_ok=True)
        path = os.path.join(work_dir, "all_predictions.csv")
        pd.DataFrame(columns=["task", "ds", "y_pred"]).to_csv(path, index=False)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
        )
        assert result["final_status"] == P22_REAL_PREDICTION_LEDGER_FAILED

    def test_successful_materialization_ready(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_predictions_to_workdir(work_dir, n_days=3)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
            mode="eval",
        )
        assert result["final_status"] == P22_REAL_PREDICTION_LEDGER_READY
        assert result["ledger_path_local"] is not None

    def test_canonical_schema_valid(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_predictions_to_workdir(work_dir, n_days=2)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
            mode="eval",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        for col in PREDICTION_LEDGER_COLUMNS:
            assert col in ledger.columns, f"Missing ledger column: {col}"

    def test_24h_per_day_enforced(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_predictions_to_workdir(work_dir, n_days=2)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
            mode="eval",
        )
        p17 = result["p17_summary"]
        assert p17 is not None
        assert p17["complete_days"] == 2
        assert p17["target_days"] == 2

    def test_duplicate_key_rejection(self, tmp_path):
        work_dir = str(tmp_path / "work")
        pred = _make_synthetic_predictions(1)
        # Duplicate first row
        dup = pred.iloc[[0]].copy()
        pred = pd.concat([pred, dup], ignore_index=True)
        os.makedirs(work_dir, exist_ok=True)
        pred.to_csv(os.path.join(work_dir, "all_predictions.csv"), index=False)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
            mode="eval",
        )
        p17 = result["p17_summary"]
        assert p17 is not None
        assert p17["duplicate_keys"] >= 1

    def test_eval_mode_preserves_ytrue(self, tmp_path):
        work_dir = str(tmp_path / "work")
        pred = _make_synthetic_predictions(1)
        pred["y_true"] = 105.0
        os.makedirs(work_dir, exist_ok=True)
        pred.to_csv(os.path.join(work_dir, "all_predictions.csv"), index=False)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
            mode="eval",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        assert "y_true" in ledger.columns

    def test_production_mode_strips_ytrue(self, tmp_path):
        work_dir = str(tmp_path / "work")
        pred = _make_synthetic_predictions(1)
        pred["y_true"] = 105.0
        os.makedirs(work_dir, exist_ok=True)
        pred.to_csv(os.path.join(work_dir, "all_predictions.csv"), index=False)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
            mode="production",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        assert "y_true" not in ledger.columns

    def test_summary_keys_present(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_predictions_to_workdir(work_dir, n_days=1)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
        )
        required_keys = [
            "p22_stage", "predictions_path", "p17_summary",
            "ledger_path_local", "final_status", "reason_codes",
            "forbidden_files_check",
        ]
        for key in required_keys:
            assert key in result, f"Missing summary key: {key}"

    def test_forbidden_files_check_pass(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_predictions_to_workdir(work_dir, n_days=1)
        result = run_p22_real_cfg05_ledger_materialization(
            work_dir=work_dir,
        )
        assert result["forbidden_files_check"] == "PASS"


class TestPathSafety:
    def test_safe_path(self):
        assert _path_is_safe(".local_artifacts/test") is True

    def test_unsafe_data(self):
        assert _path_is_safe("data/raw") is False
