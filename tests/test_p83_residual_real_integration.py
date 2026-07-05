"""P83 -- ResidualP5MAdapter real-integration tests."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapters.residual_p5m_adapter import (
    RESIDUAL_P5M_CATBOOST_APPLIED,
    RESIDUAL_P5M_CODE_ONLY,
    RESIDUAL_P5M_NO_OP,
    RESIDUAL_P5M_REAL_APPLIED,
    ResidualP5MAdapter,
)


# -- Fixtures -------------------------------------------------------------------


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """Bare adapter with no source repo, isolated from real .local_artifacts."""
    monkeypatch.chdir(tmp_path)
    return ResidualP5MAdapter()


@pytest.fixture
def predictions_24():
    """Minimal 24-row prediction DataFrame."""
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "y_pred": np.linspace(100, 400, 24),
    })


@pytest.fixture
def source_repo_with_code(tmp_path):
    """Source repo with residual_stack/orchestrator.py but no model weights."""
    repo = tmp_path / "model2.0_exp"
    repo.mkdir()
    stack_dir = repo / "residual_stack"
    stack_dir.mkdir()
    (stack_dir / "orchestrator.py").write_text("# P5M residual stack code\n")
    return str(repo)


@pytest.fixture
def source_repo_empty(tmp_path):
    """Source repo that exists but has no relevant artifacts."""
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    return str(repo)


# -- find_artifacts() -----------------------------------------------------------


class TestFindArtifacts:
    def test_no_repo_returns_empty_results(self, adapter):
        result = adapter.find_artifacts()
        assert result["p5m_pkl"] is None
        assert result["catboost_cbm"] is None
        assert result["stack_code"] is None
        assert result["reports"] is None

    def test_real_repo_finds_code_at_least(self, adapter, source_repo_with_code):
        adapter.residual_source_repo = source_repo_with_code
        result = adapter.find_artifacts()
        assert result["stack_code"] is not None
        assert "orchestrator.py" in result["stack_code"]

    def test_empty_repo_no_artifacts(self, adapter, source_repo_empty):
        adapter.residual_source_repo = source_repo_empty
        result = adapter.find_artifacts()
        assert result["p5m_pkl"] is None
        assert result["catboost_cbm"] is None
        assert result["stack_code"] is None

    def test_find_artifacts_returns_dict(self, adapter):
        result = adapter.find_artifacts()
        assert isinstance(result, dict)
        for key in ("p5m_pkl", "catboost_cbm", "stack_code", "reports"):
            assert key in result


# -- load_correction_model() ----------------------------------------------------


class TestLoadCorrectionModel:
    def test_no_artifacts_no_op_status(self, adapter):
        result = adapter.load_correction_model()
        assert result["status"] == RESIDUAL_P5M_NO_OP

    def test_no_artifacts_model_is_none(self, adapter):
        result = adapter.load_correction_model()
        assert result["model"] is None

    def test_code_only_status(self, adapter, source_repo_with_code):
        adapter.residual_source_repo = source_repo_with_code
        result = adapter.load_correction_model()
        assert result["status"] == RESIDUAL_P5M_CODE_ONLY

    def test_load_result_has_model_info(self, adapter):
        result = adapter.load_correction_model()
        assert "model_info" in result
        assert "model_name" in result["model_info"]

    def test_status_property_matches(self, adapter):
        adapter.load_correction_model()
        assert adapter.status == RESIDUAL_P5M_NO_OP


# -- apply_correction() ---------------------------------------------------------


class TestApplyCorrection:
    def test_no_model_no_op_fallback(self, adapter, predictions_24):
        result = adapter.apply_correction(predictions_24)
        assert result["status"] == RESIDUAL_P5M_NO_OP
        assert result["correction_applied"] is False

    def test_preserves_24_rows(self, adapter, predictions_24):
        result = adapter.apply_correction(predictions_24)
        assert result["rows"] == 24
        assert len(result["output"]) == 24

    def test_no_nan_in_y_pred_corrected(self, adapter, predictions_24):
        result = adapter.apply_correction(predictions_24)
        output = result["output"]
        assert "y_pred_corrected" in output.columns
        assert not output["y_pred_corrected"].isna().any()

    def test_never_uses_y_true(self, adapter, predictions_24):
        """Correction output must not reference y_true."""
        predictions_24["y_true"] = np.linspace(110, 390, 24)
        result = adapter.apply_correction(predictions_24)
        output = result["output"]
        # y_pred_corrected should equal y_pred (no-op), not y_true
        np.testing.assert_array_almost_equal(
            output["y_pred_corrected"].values,
            predictions_24["y_pred"].values,
        )

    def test_empty_input_returns_no_op(self, adapter):
        result = adapter.apply_correction(pd.DataFrame())
        assert result["status"] == RESIDUAL_P5M_NO_OP

    def test_result_has_task_key(self, adapter, predictions_24):
        result = adapter.apply_correction(predictions_24, task="dayahead")
        assert result["task"] == "dayahead"

    def test_result_has_reason_codes(self, adapter, predictions_24):
        result = adapter.apply_correction(predictions_24)
        assert "reason_codes" in result
        assert isinstance(result["reason_codes"], list)

    def test_noop_corrected_equals_raw(self, adapter, predictions_24):
        result = adapter.apply_correction(predictions_24)
        output = result["output"]
        np.testing.assert_array_almost_equal(
            output["y_pred_corrected"].values,
            output["y_pred_raw"].values,
        )


# -- Status constants -----------------------------------------------------------


class TestStatusConstants:
    def test_residual_p5m_real_applied(self):
        assert RESIDUAL_P5M_REAL_APPLIED == "RESIDUAL_P5M_REAL_APPLIED"

    def test_residual_p5m_catboost_applied(self):
        assert RESIDUAL_P5M_CATBOOST_APPLIED == "RESIDUAL_P5M_CATBOOST_APPLIED"

    def test_residual_p5m_code_only(self):
        assert RESIDUAL_P5M_CODE_ONLY == "RESIDUAL_P5M_CODE_ONLY"

    def test_residual_p5m_no_op(self):
        assert RESIDUAL_P5M_NO_OP == "RESIDUAL_P5M_NO_OP"


# -- Integration with run_full_chain_residual_correction ------------------------


class TestFullChainIntegration:
    def test_full_chain_importable(self):
        from residuals.residual_correction_engine import (
            run_full_chain_residual_correction,
        )
        assert callable(run_full_chain_residual_correction)

    def test_full_chain_with_p5m_source_repo(self, source_repo_with_code):
        from residuals.residual_correction_engine import (
            run_full_chain_residual_correction,
        )
        preds = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
            "y_pred": np.linspace(100, 400, 24),
        })
        result = run_full_chain_residual_correction(
            dayahead_predictions=preds,
            residual_source_repo=source_repo_with_code,
        )
        assert isinstance(result, dict)
        assert "dayahead" in result
        assert "overall_status" in result
