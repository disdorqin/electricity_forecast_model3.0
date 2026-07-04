"""
tests/test_p18_cfg05_real_full_chain_local.py — P18 tests.

Tests full chain: prediction ledger → residual → fusion → classifier → final.
Minimum 12 tests covering:
  1. residual fallback label honest (P5M_DATA_MISSING_NO_OP)
  2. fusion with cfg05 single real model works
  3. negative classifier fallback honest
  4. row counts consistent through chain
  5. corrected schema valid
  6. fusion schema valid
  7. final schema valid
  8. empty input → BLOCKED
  9. no input path → BLOCKED
  10. validators passed list populated
  11. readiness label assigned
  12. forbidden files check
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

from scripts.run_p18_cfg05_real_full_chain_local import (
    run_p18_cfg05_real_full_chain_local,
    _prepare_prediction_input,
    CHAIN_READY,
    CHAIN_READY_FALLBACKS,
    CHAIN_BLOCKED,
    CHAIN_INVALID,
    RESIDUAL_MODE,
    FUSION_MODE,
    CLASSIFIER_MODE,
)
from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    FUSION_OUTPUT_COLUMNS,
    FINAL_OUTPUT_COLUMNS,
)


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


class TestFullChain:
    def test_residual_fallback_honest(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["residual_mode"] == RESIDUAL_MODE

    def test_fusion_mode_label(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["fusion_mode"] == FUSION_MODE

    def test_classifier_fallback_honest(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["classifier_mode"] == CLASSIFIER_MODE

    def test_row_counts_consistent(self, tmp_path):
        pred = _make_prediction_ledger(2)
        n = len(pred)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["corrected_rows"] == n
        assert result["fusion_rows"] == n
        assert result["final_rows"] == n

    def test_corrected_schema_valid(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert "corrected_schema" in result["validators_passed"]

    def test_fusion_schema_valid(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert "fusion_schema" in result["validators_passed"]

    def test_final_schema_valid(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert "final_schema" in result["validators_passed"]

    def test_empty_input_blocked(self, tmp_path):
        pred = pd.DataFrame(columns=["task", "ds", "y_pred", "hour_business"])
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["final_status"] == CHAIN_BLOCKED

    def test_no_input_blocked(self, tmp_path):
        result = run_p18_cfg05_real_full_chain_local(
            prediction_ledger_path="/nonexistent.csv",
            work_dir=str(tmp_path),
        )
        assert result["final_status"] == CHAIN_BLOCKED

    def test_validators_populated(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert len(result["validators_passed"]) >= 3

    def test_readiness_label(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["readiness_label"] in ("LOCAL_CHAIN_READY", "LOCAL_CHAIN_PARTIAL", "NOT_ASSESSED")

    def test_forbidden_files_check(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["forbidden_files_check"] == "PASS"

    def test_summary_keys(self, tmp_path):
        pred = _make_prediction_ledger(1)
        result = run_p18_cfg05_real_full_chain_local(
            prediction_df=pred,
            work_dir=str(tmp_path),
        )
        required_keys = [
            "input_prediction_rows", "corrected_rows", "fusion_rows", "final_rows",
            "validators_passed", "residual_mode", "fusion_mode", "classifier_mode",
            "prediction_ledger_path_local", "corrected_ledger_path_local",
            "fusion_ledger_path_local", "final_output_path_local",
            "readiness_label", "final_status", "reason_codes", "forbidden_files_check",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestPreparePredictionInput:
    def test_strips_y_true(self):
        df = _make_prediction_ledger(1)
        df["y_true"] = 105.0
        result = _prepare_prediction_input(df)
        assert "y_true" not in result.columns

    def test_ensures_task_column(self):
        df = _make_prediction_ledger(1)
        df = df.drop(columns=["task"])
        result = _prepare_prediction_input(df)
        assert "task" in result.columns
        assert (result["task"] == "dayahead").all()
