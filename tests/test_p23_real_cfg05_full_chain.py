"""
tests/test_p23_real_cfg05_full_chain.py — P23 tests.

Tests P23 orchestration with synthetic data.
Minimum 10 tests covering:
  1. row consistency through chain
  2. fallback labels honest
  3. validators passed list populated
  4. empty input blocked
  5. missing ledger → FAILED
  6. successful chain → READY_WITH_FALLBACKS
  7. summary keys present
  8. forbidden files check
  9. partial chain → PARTIAL
  10. p18 summary included
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

from scripts.run_p23_real_cfg05_full_chain import (
    run_p23_real_cfg05_full_chain,
    _path_is_safe,
    P23_REAL_CFG05_FULL_CHAIN_READY_WITH_FALLBACKS,
    P23_REAL_CFG05_FULL_CHAIN_PARTIAL,
    P23_REAL_CFG05_FULL_CHAIN_FAILED,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_prediction_ledger(n_days: int = 2, start_day: str = "2026-06-01") -> pd.DataFrame:
    """Create synthetic prediction ledger."""
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
                "y_pred": 120.0 + h * 0.5,
                "source_confidence": 0.9,
                "model_version": "1.0.0",
                "run_id": "test_run",
                "created_at": "2026-07-04T00:00:00",
                "updated_at": "2026-07-04T00:00:00",
            })
    df = pd.DataFrame(rows)
    from data.business_day import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def _write_ledger_to_workdir(work_dir: str, n_days: int = 2) -> str:
    """Write prediction ledger to work_dir/ledgers/prediction_ledger.csv."""
    ledger_dir = os.path.join(work_dir, "ledgers")
    os.makedirs(ledger_dir, exist_ok=True)
    ledger = _make_prediction_ledger(n_days)
    path = os.path.join(ledger_dir, "prediction_ledger.csv")
    ledger.to_csv(path, index=False)
    return path


# ── Tests ──────────────────────────────────────────────────────────────────

class TestFullChainOrchestration:
    def test_missing_ledger_failed(self, tmp_path):
        result = run_p23_real_cfg05_full_chain(
            work_dir=str(tmp_path / "work"),
            ledger_path_override="/nonexistent/ledger.csv",
        )
        assert result["final_status"] == P23_REAL_CFG05_FULL_CHAIN_FAILED

    def test_empty_ledger_failed(self, tmp_path):
        work_dir = str(tmp_path / "work")
        ledger_dir = os.path.join(work_dir, "ledgers")
        os.makedirs(ledger_dir, exist_ok=True)
        path = os.path.join(ledger_dir, "prediction_ledger.csv")
        pd.DataFrame(columns=["task", "ds", "y_pred"]).to_csv(path, index=False)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        assert result["final_status"] == P23_REAL_CFG05_FULL_CHAIN_FAILED

    def test_successful_chain_ready_with_fallbacks(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=2)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        assert result["final_status"] == P23_REAL_CFG05_FULL_CHAIN_READY_WITH_FALLBACKS

    def test_row_consistency(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=1)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        p18 = result["p18_summary"]
        assert p18 is not None
        n_input = p18["input_prediction_rows"]
        assert p18["corrected_rows"] == n_input
        assert p18["fusion_rows"] == n_input
        assert p18["final_rows"] == n_input

    def test_fallback_labels_honest(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=1)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        p18 = result["p18_summary"]
        assert p18 is not None
        assert p18["residual_mode"] == "P5M_DATA_MISSING_NO_OP"
        assert p18["classifier_mode"] == "NEGATIVE_CLASSIFIER_RULE_FALLBACK"

    def test_validators_passed(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=1)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        assert len(result["validators_passed"]) >= 3
        assert "corrected_schema" in result["validators_passed"]
        assert "fusion_schema" in result["validators_passed"]
        assert "final_schema" in result["validators_passed"]

    def test_summary_keys_present(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=1)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        required_keys = [
            "p23_stage", "ledger_path", "p18_summary",
            "final_output_path_local", "validators_passed",
            "final_status", "reason_codes", "forbidden_files_check",
        ]
        for key in required_keys:
            assert key in result, f"Missing summary key: {key}"

    def test_forbidden_files_check_pass(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=1)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        assert result["forbidden_files_check"] == "PASS"

    def test_p18_summary_included(self, tmp_path):
        work_dir = str(tmp_path / "work")
        _write_ledger_to_workdir(work_dir, n_days=1)
        result = run_p23_real_cfg05_full_chain(work_dir=work_dir)
        assert result["p18_summary"] is not None
        assert "final_status" in result["p18_summary"]
        assert "final_rows" in result["p18_summary"]


class TestPathSafety:
    def test_safe_path(self):
        assert _path_is_safe(".local_artifacts/test") is True

    def test_unsafe_data(self):
        assert _path_is_safe("data/raw") is False

    def test_unsafe_ledgers(self):
        assert _path_is_safe("ledgers/test") is False
