"""P81 -- RealtimeDeepAdapter real-pack and real-artifact integration tests."""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.adapters.realtime_deep_adapter import (
    FORBIDDEN_ONLINE_COLUMNS,
    ONLINE_PACK_COLUMNS,
    REALTIME_DEEP_BLOCKED_NO_ARTIFACT,
    REALTIME_DEEP_REAL_PACK_LOADED,
    REALTIME_DEEP_REAL_READY,
    REALTIME_DEEP_READY_FAST_DEV,
    RealtimeDeepAdapter,
)


# -- Fixtures -------------------------------------------------------------------


@pytest.fixture
def adapter():
    """Bare adapter with no source repo."""
    return RealtimeDeepAdapter()


@pytest.fixture
def da_predictions():
    """Minimal day-ahead predictions DataFrame (24 rows)."""
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "y_pred": np.linspace(100, 400, 24),
        "da_anchor": np.linspace(105, 395, 24),
    })


@pytest.fixture
def source_repo(tmp_path):
    """Create a minimal fake deep_sgdf_delta source repo with artifacts."""
    repo = tmp_path / "deep_sgdf_delta"
    repo.mkdir()

    # -- artifacts/delta_supply/exp_2026_02/model.pkl
    delta_dir = repo / "artifacts" / "delta_supply" / "exp_2026_02"
    delta_dir.mkdir(parents=True)
    with open(delta_dir / "model.pkl", "wb") as f:
        pickle.dump({"type": "delta_supply_mock"}, f)

    # -- artifacts/negative_risk/exp_2026_02/model.pkl
    neg_dir = repo / "artifacts" / "negative_risk" / "exp_2026_02"
    neg_dir.mkdir(parents=True)
    with open(neg_dir / "model.pkl", "wb") as f:
        pickle.dump({"type": "negative_risk_mock"}, f)

    # -- artifacts/spike_risk/exp_2026_02/model.pkl
    spike_dir = repo / "artifacts" / "spike_risk" / "exp_2026_02"
    spike_dir.mkdir(parents=True)
    with open(spike_dir / "model.pkl", "wb") as f:
        pickle.dump({"type": "spike_risk_mock"}, f)

    return str(repo)


@pytest.fixture
def pack_repo(tmp_path):
    """Create a source repo with a valid trend_prediction_pack.csv."""
    repo = tmp_path / "deep_sgdf_delta_pack"
    repo.mkdir()
    pack_dir = repo / "reports" / "local" / "phase3" / "export"
    pack_dir.mkdir(parents=True)

    pack_df = pd.DataFrame({
        "trend_pred": np.linspace(100, 400, 24),
        "trend_model_name": ["SGDFNet_v2"] * 24,
        "trend_confidence": [0.9] * 24,
        "deep_rt_pred": np.linspace(102, 398, 24),
        "blend_pred": np.linspace(101, 399, 24),
    })
    pack_df.to_csv(pack_dir / "trend_prediction_pack.csv", index=False)
    return str(repo)


@pytest.fixture
def insufficient_pack_repo(tmp_path):
    """Source repo with a pack CSV that has < 24 rows."""
    repo = tmp_path / "deep_sgdf_delta_short"
    repo.mkdir()
    pack_dir = repo / "reports" / "local" / "phase3" / "export"
    pack_dir.mkdir(parents=True)

    pack_df = pd.DataFrame({
        "trend_pred": [100.0] * 10,
        "trend_model_name": ["mock"] * 10,
        "trend_confidence": [0.5] * 10,
        "deep_rt_pred": [100.0] * 10,
    })
    pack_df.to_csv(pack_dir / "trend_prediction_pack.csv", index=False)
    return str(repo)


# -- find_real_packs() ----------------------------------------------------------


class TestFindRealPacks:
    def test_no_packs_found_returns_pack_found_false(self, adapter):
        result = adapter.find_real_packs(source_repo_path="/nonexistent/path")
        assert result["pack_found"] is False

    def test_no_packs_found_has_issue(self, adapter):
        result = adapter.find_real_packs(source_repo_path="/nonexistent/path")
        assert len(result["issues"]) > 0
        assert "SOURCE_REPO_MISSING" in result["issues"]

    def test_valid_pack_found(self, adapter, pack_repo):
        result = adapter.find_real_packs(source_repo_path=pack_repo)
        assert result["pack_found"] is True
        assert result["pack_valid"] is True

    def test_valid_pack_validates_columns(self, adapter, pack_repo):
        result = adapter.find_real_packs(source_repo_path=pack_repo)
        assert result["pack_rows"] == 24
        assert "trend_pred" not in str(result.get("issues", []))

    def test_insufficient_rows_marked(self, adapter, insufficient_pack_repo):
        result = adapter.find_real_packs(source_repo_path=insufficient_pack_repo)
        assert result["pack_found"] is True
        assert result["pack_valid"] is False
        assert "PACK_INSUFFICIENT_ROWS" in result["issues"]

    def test_empty_repo_path(self, adapter):
        result = adapter.find_real_packs(source_repo_path="")
        assert result["pack_found"] is False


# -- load_real_artifacts() ------------------------------------------------------


class TestLoadRealArtifacts:
    def test_finds_delta_supply(self, adapter, source_repo):
        result = adapter.load_real_artifacts(source_repo_path=source_repo)
        assert result["delta_supply"] is not None

    def test_finds_negative_risk(self, adapter, source_repo):
        result = adapter.load_real_artifacts(source_repo_path=source_repo)
        assert result["negative_risk"] is not None

    def test_finds_spike_risk(self, adapter, source_repo):
        result = adapter.load_real_artifacts(source_repo_path=source_repo)
        assert result["spike_risk"] is not None

    def test_any_loaded_true(self, adapter, source_repo):
        result = adapter.load_real_artifacts(source_repo_path=source_repo)
        assert result["any_loaded"] is True

    def test_empty_path_no_artifacts(self, adapter):
        result = adapter.load_real_artifacts(source_repo_path="")
        assert result["any_loaded"] is False
        assert "SOURCE_REPO_MISSING" in result["issues"]

    def test_nonexistent_path_no_artifacts(self, adapter):
        result = adapter.load_real_artifacts(source_repo_path="/no/such/path")
        assert result["any_loaded"] is False


# -- select_champion() ----------------------------------------------------------


class TestSelectChampion:
    def test_strict_mode_no_real_pack_blocked(self, adapter):
        result = adapter.select_champion(strict=True)
        assert result["verdict"] == "BLOCKED"

    def test_strict_mode_status_blocked(self, adapter):
        adapter.select_champion(strict=True)
        assert adapter.status == REALTIME_DEEP_BLOCKED_NO_ARTIFACT

    def test_non_strict_mode_no_real_pack_fallback(self, adapter):
        result = adapter.select_champion(strict=False)
        assert result["verdict"] == "FAST_DEV_ONLY"

    def test_non_strict_status_fast_dev(self, adapter):
        adapter.select_champion(strict=False)
        assert adapter.status == REALTIME_DEEP_READY_FAST_DEV

    def test_real_artifacts_gives_real_verdict(self, adapter, source_repo):
        adapter.source_repo_path = source_repo
        result = adapter.select_champion(strict=False)
        assert result["verdict"] == "REAL_ARTIFACTS"

    def test_real_artifacts_status_real_ready(self, adapter, source_repo):
        adapter.source_repo_path = source_repo
        adapter.select_champion(strict=False)
        assert adapter.status == REALTIME_DEEP_REAL_READY

    def test_real_pack_gives_real_pack_verdict(self, adapter, pack_repo):
        adapter.source_repo_path = pack_repo
        result = adapter.select_champion(strict=False)
        assert result["verdict"] == "REAL_PACK"

    def test_real_pack_status_pack_loaded(self, adapter, pack_repo):
        adapter.source_repo_path = pack_repo
        adapter.select_champion(strict=False)
        assert adapter.status == REALTIME_DEEP_REAL_PACK_LOADED


# -- export_online_pack() -------------------------------------------------------


class TestExportOnlinePack:
    def test_strict_mode_no_real_pack_blocked(self, adapter, da_predictions, tmp_path):
        result = adapter.export_online_pack(
            da_predictions=da_predictions,
            output_dir=str(tmp_path),
            strict=True,
        )
        assert result["status"] == "BLOCKED"

    def test_strict_blocked_reason_codes(self, adapter, da_predictions, tmp_path):
        result = adapter.export_online_pack(
            da_predictions=da_predictions,
            output_dir=str(tmp_path),
            strict=True,
        )
        assert "STRICT_MODE" in result["reason_codes"]

    def test_real_pack_exported(self, adapter, pack_repo, da_predictions, tmp_path):
        adapter.source_repo_path = pack_repo
        adapter.find_real_packs()
        result = adapter.export_online_pack(
            da_predictions=da_predictions,
            output_dir=str(tmp_path),
        )
        assert result["status"] == "EXPORTED"
        assert "REAL_PACK_USED" in result["reason_codes"]


# -- _build_online_pack() -------------------------------------------------------


class TestBuildOnlinePack:
    def test_real_artifacts_different_trend_model_name(self, adapter, source_repo, da_predictions):
        adapter.source_repo_path = source_repo
        adapter.load_real_artifacts()
        pack = adapter._build_online_pack(da_predictions)
        # With delta_supply loaded, trend_model_name should NOT be the fallback
        model_names = pack["trend_model_name"].unique()
        assert all("FALLBACK" not in name for name in model_names)

    def test_fallback_trend_model_name(self, adapter, da_predictions):
        pack = adapter._build_online_pack(da_predictions)
        assert (pack["trend_model_name"] == "rt_da_anchor_FALLBACK").all()

    def test_online_pack_never_contains_y_true(self, adapter, da_predictions):
        pack = adapter._build_online_pack(da_predictions)
        assert "y_true" not in pack.columns

    def test_online_pack_columns_count(self):
        assert len(ONLINE_PACK_COLUMNS) == 14


# -- Status constants -----------------------------------------------------------


class TestStatusConstants:
    def test_realtime_deep_real_ready_exists(self):
        assert REALTIME_DEEP_REAL_READY == "REALTIME_DEEP_REAL_READY"

    def test_realtime_deep_real_pack_loaded_exists(self):
        assert REALTIME_DEEP_REAL_PACK_LOADED == "REALTIME_DEEP_REAL_PACK_LOADED"
