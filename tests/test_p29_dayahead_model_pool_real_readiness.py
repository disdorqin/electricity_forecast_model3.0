"""tests/test_p29_dayahead_model_pool_real_readiness.py — P29 model pool audit tests."""

import os
import tempfile
from unittest.mock import patch

import pytest


def test_p29_readiness_labels():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        ARTIFACT_MISSING,
        BACKTESTED,
        DATA_MISSING,
        INVALID_BANNED,
        PREDICTION_ONLY,
        REAL_24H_READY,
        REAL_READY,
        TRAINABLE_LOCAL,
    )
    labels = [REAL_READY, REAL_24H_READY, BACKTESTED, TRAINABLE_LOCAL,
              PREDICTION_ONLY, DATA_MISSING, ARTIFACT_MISSING, INVALID_BANNED]
    assert len(set(labels)) == 8  # all unique


def test_p29_candidate_models_list():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import CANDIDATE_MODELS
    assert "cfg05" in CANDIDATE_MODELS
    assert "best_two_average" in CANDIDATE_MODELS
    assert "stage3_business_fixed" in CANDIDATE_MODELS
    assert "catboost_spike_residual" in CANDIDATE_MODELS
    assert "catboost_sota" in CANDIDATE_MODELS


def test_p29_banned_models_list():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import BANNED_MODELS
    assert "lgbm_spike_residual_1127" in BANNED_MODELS
    assert "stage3_old_1164" in BANNED_MODELS
    assert "lightgbm_90d_orig_1197" in BANNED_MODELS


def test_p29_banned_models_in_registry():
    """Banned models should be detected by the registry."""
    from src.registry.dayahead_models import is_invalid_model
    assert is_invalid_model("lgbm_spike_residual_1127")
    assert is_invalid_model("stage3_old_1164")
    assert is_invalid_model("lightgbm_90d_orig_1197")


def test_p29_valid_models_not_banned():
    from src.registry.dayahead_models import is_invalid_model, is_valid_model
    for m in ["cfg05", "best_two_average", "catboost_sota"]:
        assert is_valid_model(m), f"{m} should be valid"
        assert not is_invalid_model(m), f"{m} should not be banned"


def test_p29_audit_returns_dict():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        audit_p29_dayahead_model_pool_real_readiness,
    )
    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo="/nonexistent",
        work_dir="/nonexistent",
    )
    assert isinstance(result, dict)
    assert "candidates_evaluated" in result
    assert "real_ready_count" in result
    assert "final_status" in result


def test_p29_all_candidates_evaluated():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        CANDIDATE_MODELS,
        audit_p29_dayahead_model_pool_real_readiness,
    )
    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo="/nonexistent",
        work_dir="/nonexistent",
    )
    assert len(result["candidates_evaluated"]) == len(CANDIDATE_MODELS)


def test_p29_each_candidate_has_required_fields():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        audit_p29_dayahead_model_pool_real_readiness,
    )
    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo="/nonexistent",
        work_dir="/nonexistent",
    )
    required_fields = [
        "model_id", "artifact_exists", "training_script_exists",
        "can_predict_24h", "schema_ready", "backtest_ready",
        "readiness_label", "blocker",
    ]
    for entry in result["candidates_evaluated"]:
        for f in required_fields:
            assert f in entry, f"Missing field {f} for model {entry.get('model_id')}"


def test_p29_banned_models_not_in_fusion_pool():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        BANNED_MODELS,
        audit_p29_dayahead_model_pool_real_readiness,
    )
    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo="/nonexistent",
        work_dir="/nonexistent",
    )
    for banned_id in BANNED_MODELS:
        assert banned_id not in result["fusion_pool_models"]


def test_p29_can_form_pool_requires_two():
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        audit_p29_dayahead_model_pool_real_readiness,
    )
    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo="/nonexistent",
        work_dir="/nonexistent",
    )
    # can_form_fusion_pool should be True only if >= 2 real-ready models
    if result["real_ready_count"] >= 2:
        assert result["can_form_fusion_pool"] is True
    else:
        assert result["can_form_fusion_pool"] is False


def test_p29_cfg05_should_be_backtested_or_ready():
    """cfg05 should have BACKTESTED or better label."""
    from scripts.audit_p29_dayahead_model_pool_real_readiness import (
        audit_p29_dayahead_model_pool_real_readiness,
    )
    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo="/nonexistent",
        work_dir="/nonexistent",
    )
    cfg05_entry = next(e for e in result["candidates_evaluated"] if e["model_id"] == "cfg05")
    assert cfg05_entry["readiness_label"] in ("BACKTESTED", "REAL_24H_READY", "REAL_READY")
