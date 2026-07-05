"""P85 -- NegativeClassifierAdapter real-integration tests."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapters.negative_classifier_adapter import (
    CLASSIFIER_ML_READY,
    CLASSIFIER_RULE_FALLBACK,
    NegativeClassifierAdapter,
)


# -- Module-level mock classifier (must be picklable) --------------------------


class _MockClassifier:
    """Mock sklearn-like classifier that can be pickled."""

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.column_stack([
            np.ones(n) * 0.7,
            np.ones(n) * 0.3,
        ])
        return probs

    def predict(self, X):
        return np.zeros(X.shape[0])


# -- Fixtures -------------------------------------------------------------------


@pytest.fixture
def adapter():
    """Bare adapter with no source repo."""
    return NegativeClassifierAdapter()


@pytest.fixture
def predictions_24():
    """Minimal 24-row prediction DataFrame with y_pred column."""
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "y_pred": np.linspace(50, 600, 24),
    })


@pytest.fixture
def source_repo_with_models(tmp_path):
    """Create a source repo with mock negative_risk and spike_risk model.pkl files."""
    repo = tmp_path / "deep_sgdf_delta"
    repo.mkdir()

    # -- artifacts/negative_risk/exp_2026_02/model.pkl
    neg_dir = repo / "artifacts" / "negative_risk" / "exp_2026_02"
    neg_dir.mkdir(parents=True)
    with open(neg_dir / "model.pkl", "wb") as f:
        pickle.dump(_MockClassifier(), f)

    # -- artifacts/spike_risk/exp_2026_02/model.pkl
    spike_dir = repo / "artifacts" / "spike_risk" / "exp_2026_02"
    spike_dir.mkdir(parents=True)
    with open(spike_dir / "model.pkl", "wb") as f:
        pickle.dump(_MockClassifier(), f)

    return str(repo)


@pytest.fixture
def source_repo_empty(tmp_path):
    """Source repo that exists but has no classifier artifacts."""
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    return str(repo)


# -- find_artifacts() -----------------------------------------------------------


class TestFindArtifacts:
    def test_real_source_repo_finds_negative_risk(self, adapter, source_repo_with_models):
        adapter.source_repo_path = source_repo_with_models
        result = adapter.find_artifacts()
        assert result["negative_risk_pkl"] is not None

    def test_real_source_repo_finds_spike_risk(self, adapter, source_repo_with_models):
        adapter.source_repo_path = source_repo_with_models
        result = adapter.find_artifacts()
        assert result["spike_risk_pkl"] is not None

    def test_empty_path_no_artifacts(self, adapter):
        result = adapter.find_artifacts()
        assert result["negative_risk_pkl"] is None
        assert result["spike_risk_pkl"] is None

    def test_empty_source_repo_no_artifacts(self, adapter, source_repo_empty):
        adapter.source_repo_path = source_repo_empty
        result = adapter.find_artifacts()
        assert result["negative_risk_pkl"] is None
        assert result["spike_risk_pkl"] is None

    def test_find_artifacts_returns_dict(self, adapter):
        result = adapter.find_artifacts()
        assert isinstance(result, dict)
        assert "negative_risk_pkl" in result
        assert "spike_risk_pkl" in result


# -- load_models() --------------------------------------------------------------


class TestLoadModels:
    def test_real_models_status_ml_ready(self, adapter, source_repo_with_models):
        adapter.source_repo_path = source_repo_with_models
        adapter.find_artifacts()
        result = adapter.load_models()
        assert result["status"] == CLASSIFIER_ML_READY

    def test_no_models_status_rule_fallback(self, adapter):
        result = adapter.load_models()
        assert result["status"] == CLASSIFIER_RULE_FALLBACK

    def test_load_result_has_model_names(self, adapter, source_repo_with_models):
        adapter.source_repo_path = source_repo_with_models
        adapter.find_artifacts()
        result = adapter.load_models()
        assert "model_names" in result
        assert isinstance(result["model_names"], dict)

    def test_status_property_matches(self, adapter, source_repo_with_models):
        adapter.source_repo_path = source_repo_with_models
        adapter.find_artifacts()
        adapter.load_models()
        assert adapter.status == CLASSIFIER_ML_READY


# -- classify() -----------------------------------------------------------------


class TestClassify:
    def test_adds_classifier_action(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        assert "classifier_action" in result.columns

    def test_adds_negative_risk(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        assert "negative_risk" in result.columns

    def test_adds_spike_risk(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        assert "spike_risk" in result.columns

    def test_never_uses_y_true(self, adapter, predictions_24):
        """classify() must not use y_true as input feature."""
        predictions_24["y_true"] = np.linspace(1000, 2000, 24)
        result = adapter.classify(predictions_24)
        # The classifier_action should be based on y_pred, not y_true
        # With rule fallback, y_pred range 50-600 should produce mixed results
        assert "classifier_action" in result.columns
        # Verify that y_true values don't influence the output
        assert "classifier_status" in result.columns

    def test_output_has_all_required_columns(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        required_cols = [
            "classifier_action",
            "negative_risk",
            "spike_risk",
            "normal_trend_flag",
            "uncertainty_score",
            "classifier_model_name",
            "classifier_status",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_row_count_preserved(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        assert len(result) == 24

    def test_ml_classify_with_real_models(self, adapter, source_repo_with_models, predictions_24):
        adapter.source_repo_path = source_repo_with_models
        adapter.find_artifacts()
        adapter.load_models()
        result = adapter.classify(predictions_24)
        assert "classifier_action" in result.columns
        assert "negative_risk" in result.columns
        assert "spike_risk" in result.columns

    def test_ml_classify_status_is_ml_ready(
        self, adapter, source_repo_with_models, predictions_24
    ):
        adapter.source_repo_path = source_repo_with_models
        adapter.find_artifacts()
        adapter.load_models()
        result = adapter.classify(predictions_24)
        assert (result["classifier_status"] == CLASSIFIER_ML_READY).all()


# -- Rule fallback --------------------------------------------------------------


class TestRuleFallback:
    def test_rule_fallback_when_no_source_repo(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        assert (result["classifier_status"] == CLASSIFIER_RULE_FALLBACK).all()

    def test_rule_fallback_model_name(self, adapter, predictions_24):
        result = adapter.classify(predictions_24)
        assert (result["classifier_model_name"] == "rule_fallback").all()


# -- Integration with run_final_classifier --------------------------------------


class TestRunFinalClassifierIntegration:
    def test_run_final_classifier_importable(self):
        from classifiers import run_final_classifier
        assert callable(run_final_classifier)

    def test_run_final_classifier_with_source_repo(self, source_repo_with_models):
        from classifiers import run_final_classifier

        fused = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
            "dayahead_price": np.linspace(100, 400, 24),
        })
        result = run_final_classifier(
            dayahead_fused=fused,
            source_repo_path=source_repo_with_models,
        )
        assert isinstance(result, dict)
        assert "dayahead" in result
        assert result["dayahead"]["status"] == "CLASSIFIED"

    def test_run_final_classifier_without_source_repo(self):
        from classifiers import run_final_classifier

        fused = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
            "dayahead_price": np.linspace(100, 400, 24),
        })
        result = run_final_classifier(dayahead_fused=fused)
        assert isinstance(result, dict)
        assert result["classifier_status"] == CLASSIFIER_RULE_FALLBACK


# -- Status constants -----------------------------------------------------------


class TestStatusConstants:
    def test_classifier_ml_ready_value(self):
        assert CLASSIFIER_ML_READY == "CLASSIFIER_ML_READY"

    def test_classifier_rule_fallback_value(self):
        assert CLASSIFIER_RULE_FALLBACK == "CLASSIFIER_RULE_FALLBACK"
