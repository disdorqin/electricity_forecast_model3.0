"""P69 — Residual Correction (full chain) unit tests."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from residuals import (
    RESIDUAL_BLOCKED_NO_DATA,
    RESIDUAL_CORRECTION_APPLIED,
    RESIDUAL_NO_OP_FALLBACK,
    run_residual_correction,
)
from residuals.residual_correction_engine import (
    run_full_chain_residual_correction,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def da_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "y_pred": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def rt_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "trend_pred": np.random.uniform(100, 400, 24),
    })


# ── run_residual_correction (single task) ─────────────────────────────────────


class TestRunResidualCorrection:
    def test_blocked_when_none(self):
        result = run_residual_correction(None, task="dayahead")
        assert result["status"] == RESIDUAL_BLOCKED_NO_DATA

    def test_blocked_when_empty(self):
        result = run_residual_correction(pd.DataFrame(), task="dayahead")
        assert result["status"] == RESIDUAL_BLOCKED_NO_DATA

    def test_noop_fallback_when_no_artifact(self, da_predictions, tmp_path):
        result = run_residual_correction(
            da_predictions, task="dayahead", work_dir=str(tmp_path)
        )
        assert result["status"] == RESIDUAL_NO_OP_FALLBACK
        assert result["correction_applied"] is False

    def test_output_has_delta_column(self, da_predictions, tmp_path):
        result = run_residual_correction(
            da_predictions, task="dayahead", work_dir=str(tmp_path)
        )
        output = result["output"]
        assert "residual_delta" in output.columns
        assert (output["residual_delta"] == 0.0).all()

    def test_output_has_corrected_column(self, da_predictions, tmp_path):
        result = run_residual_correction(
            da_predictions, task="dayahead", work_dir=str(tmp_path)
        )
        output = result["output"]
        assert "y_pred_corrected" in output.columns

    def test_rows_count(self, da_predictions, tmp_path):
        result = run_residual_correction(
            da_predictions, task="dayahead", work_dir=str(tmp_path)
        )
        assert result["rows"] == 24

    def test_task_echoed(self, da_predictions, tmp_path):
        result = run_residual_correction(
            da_predictions, task="realtime", work_dir=str(tmp_path)
        )
        assert result["task"] == "realtime"

    def test_reason_codes_is_list(self, da_predictions, tmp_path):
        result = run_residual_correction(
            da_predictions, task="dayahead", work_dir=str(tmp_path)
        )
        assert isinstance(result["reason_codes"], list)

    def test_noop_with_trend_pred_column(self, tmp_path):
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "trend_pred": np.random.uniform(100, 400, 24),
        })
        result = run_residual_correction(df, task="dayahead", work_dir=str(tmp_path))
        assert result["status"] == RESIDUAL_NO_OP_FALLBACK
        assert "y_pred_corrected" in result["output"].columns

    def test_noop_no_price_col(self, tmp_path):
        df = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        result = run_residual_correction(df, task="dayahead", work_dir=str(tmp_path))
        output = result["output"]
        assert output["y_pred_corrected"].isna().all()


# ── run_full_chain_residual_correction ────────────────────────────────────────


class TestFullChainResidualCorrection:
    def test_both_none(self):
        result = run_full_chain_residual_correction()
        assert result["overall_status"] == "RESIDUAL_CORRECTION_FAILED"

    def test_both_empty(self):
        result = run_full_chain_residual_correction(
            dayahead_predictions=pd.DataFrame(),
            realtime_predictions=pd.DataFrame(),
        )
        assert result["overall_status"] == "RESIDUAL_CORRECTION_FAILED"

    def test_dayahead_only(self, da_predictions, tmp_path):
        result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            work_dir=str(tmp_path),
        )
        assert result["overall_status"] == "RESIDUAL_PARTIAL_DAYAHEAD_ONLY"
        assert result["dayahead"]["status"] == RESIDUAL_NO_OP_FALLBACK

    def test_realtime_only(self, rt_predictions, tmp_path):
        result = run_full_chain_residual_correction(
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        # da is None -> FAILED (asymmetric)
        assert result["overall_status"] == "RESIDUAL_CORRECTION_FAILED"

    def test_both_present(self, da_predictions, rt_predictions, tmp_path):
        result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        assert result["overall_status"] == "RESIDUAL_CORRECTION_COMPLETE"

    def test_reason_codes_list(self, da_predictions, rt_predictions, tmp_path):
        result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        assert isinstance(result["reason_codes"], list)

    def test_dayahead_output_is_dataframe(self, da_predictions, rt_predictions, tmp_path):
        result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        assert isinstance(result["dayahead"]["output"], pd.DataFrame)

    def test_realtime_output_is_dataframe(self, da_predictions, rt_predictions, tmp_path):
        result = run_full_chain_residual_correction(
            dayahead_predictions=da_predictions,
            realtime_predictions=rt_predictions,
            work_dir=str(tmp_path),
        )
        assert isinstance(result["realtime"]["output"], pd.DataFrame)
