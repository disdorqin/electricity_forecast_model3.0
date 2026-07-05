"""tests/test_p27_cfg05_feature_builder_alignment.py — P27 feature builder alignment tests."""

import os
import sys
from unittest.mock import patch

import pytest


def test_p27_alignment_labels():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        FEATURE_ALIGNMENT_MATCHED,
        FEATURE_ALIGNMENT_PARTIAL,
        FEATURE_ALIGNMENT_NOT_MATCHED,
    )
    assert FEATURE_ALIGNMENT_MATCHED == "FEATURE_ALIGNMENT_MATCHED"
    assert FEATURE_ALIGNMENT_PARTIAL == "FEATURE_ALIGNMENT_PARTIAL"
    assert FEATURE_ALIGNMENT_NOT_MATCHED == "FEATURE_ALIGNMENT_NOT_MATCHED"


def test_p27_cfg05_feature_count():
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
    assert len(CFG05_FEATURE_COLUMNS) == 56


def test_p27_get_cfg05_features():
    from scripts.audit_p27_cfg05_feature_builder_alignment import _get_cfg05_features
    feats = _get_cfg05_features()
    assert len(feats) == 56
    assert "hour" in feats
    assert "lag_price_target" in feats


def test_p27_get_cfg05_params():
    from scripts.audit_p27_cfg05_feature_builder_alignment import _get_cfg05_params
    params = _get_cfg05_params()
    assert params["objective"] == "mae"
    assert params["num_leaves"] == 191
    assert params["learning_rate"] == 0.015


def test_p27_compare_feature_lists_identical():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        FEATURE_ALIGNMENT_MATCHED,
        _compare_feature_lists,
    )
    feats = ["a", "b", "c"]
    result = _compare_feature_lists(feats, feats, None)
    assert result["alignment_label"] == FEATURE_ALIGNMENT_MATCHED


def test_p27_compare_feature_lists_subset():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        FEATURE_ALIGNMENT_PARTIAL,
        _compare_feature_lists,
    )
    cfg05 = ["a", "b", "c", "d", "e"]
    v3 = ["a", "b", "c", "d", "e", "f", "g", "h"]
    result = _compare_feature_lists(cfg05, None, v3)
    # cfg05 is subset of v3, 5/8 = 62.5% > 50% overlap
    assert result["alignment_label"] == FEATURE_ALIGNMENT_PARTIAL


def test_p27_compare_feature_lists_mismatch():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        FEATURE_ALIGNMENT_NOT_MATCHED,
        _compare_feature_lists,
    )
    cfg05 = ["x", "y", "z"]
    v3 = ["a", "b", "c", "d", "e", "f", "g"]
    result = _compare_feature_lists(cfg05, None, v3)
    assert result["alignment_label"] == FEATURE_ALIGNMENT_NOT_MATCHED


def test_p27_audit_returns_dict():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        audit_p27_cfg05_feature_builder_alignment,
    )
    result = audit_p27_cfg05_feature_builder_alignment(source_repo="/nonexistent")
    assert isinstance(result, dict)
    assert "alignment_label" in result
    assert "cfg05_feature_count" in result


def test_p27_audit_dimensions_checked():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        audit_p27_cfg05_feature_builder_alignment,
    )
    result = audit_p27_cfg05_feature_builder_alignment(source_repo="/nonexistent")
    assert result["dimensions_checked"] >= 3


def test_p27_audit_has_recommendation():
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        audit_p27_cfg05_feature_builder_alignment,
    )
    result = audit_p27_cfg05_feature_builder_alignment(source_repo="/nonexistent")
    assert "migration_recommendation" in result
    assert len(result["migration_recommendation"]) > 0


def test_p27_v2_vs_v3_difference():
    """v3 should have more features than v2."""
    from scripts.audit_p27_cfg05_feature_builder_alignment import (
        _get_source_v2_features,
        _get_source_v3_features,
    )
    source_repo = os.path.join(".local_artifacts", "source_repos", "epf-sota-experiment")
    if not os.path.isdir(source_repo):
        pytest.skip("Source repo not available")
    v2 = _get_source_v2_features(source_repo)
    v3 = _get_source_v3_features(source_repo)
    if v2 and v3:
        assert len(v3) > len(v2), f"v3 ({len(v3)}) should have more features than v2 ({len(v2)})"
