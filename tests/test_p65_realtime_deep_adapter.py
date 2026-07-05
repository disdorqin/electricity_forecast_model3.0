"""P65 — RealtimeDeepAdapter unit tests."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from models.adapters.realtime_deep_adapter import (
    FORBIDDEN_ONLINE_COLUMNS,
    ONLINE_PACK_COLUMNS,
    REALTIME_DEEP_BLOCKED_LEAKAGE,
    REALTIME_DEEP_BLOCKED_NO_ARTIFACT,
    REALTIME_DEEP_READY_FAST_DEV,
    RealtimeDeepAdapter,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def adapter():
    return RealtimeDeepAdapter()


@pytest.fixture
def da_predictions():
    """Minimal day-ahead predictions DataFrame."""
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "y_pred": np.random.uniform(100, 400, 24),
        "da_anchor": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def da_predictions_with_y_true():
    """Day-ahead predictions that contain a forbidden column."""
    df = pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "y_pred": np.random.uniform(100, 400, 24),
        "da_anchor": np.random.uniform(100, 400, 24),
        "y_true": np.random.uniform(100, 400, 24),
    })
    return df


# ── Constructor / Status ──────────────────────────────────────────────────────


class TestAdapterInit:
    def test_default_status_is_blocked(self, adapter):
        assert adapter.status == REALTIME_DEEP_BLOCKED_NO_ARTIFACT

    def test_default_work_dir(self, adapter):
        assert adapter.work_dir == os.path.join(".local_artifacts", "realtime")

    def test_custom_work_dir(self):
        a = RealtimeDeepAdapter(work_dir="/tmp/custom")
        assert a.work_dir == "/tmp/custom"

    def test_champion_info_initially_empty(self, adapter):
        assert adapter._champion_info == {}


# ── check_environment ─────────────────────────────────────────────────────────


class TestCheckEnvironment:
    def test_returns_dict(self, adapter):
        result = adapter.check_environment()
        assert isinstance(result, dict)

    def test_has_required_keys(self, adapter):
        result = adapter.check_environment()
        for key in ("source_repo_exists", "sgdfnet_root_exists",
                     "raw_data_exists", "torch_available", "status"):
            assert key in result

    def test_status_is_valid(self, adapter):
        result = adapter.check_environment()
        assert result["status"] in (
            "ENVIRONMENT_READY", "ENVIRONMENT_PARTIAL", "ENVIRONMENT_NOT_READY"
        )

    def test_empty_paths_give_not_ready(self, adapter):
        result = adapter.check_environment()
        assert result["source_repo_exists"] is False
        assert result["raw_data_exists"] is False


# ── train_if_needed ───────────────────────────────────────────────────────────


class TestTrainIfNeeded:
    def test_returns_dict(self, adapter):
        result = adapter.train_if_needed()
        assert isinstance(result, dict)

    def test_has_status_key(self, adapter):
        result = adapter.train_if_needed()
        assert "status" in result

    def test_has_reason_codes(self, adapter):
        result = adapter.train_if_needed()
        assert "reason_codes" in result
        assert isinstance(result["reason_codes"], list)

    def test_fallback_model_type(self, adapter):
        result = adapter.train_if_needed()
        assert result.get("model_type") == "da_anchor_fallback"


# ── select_champion ───────────────────────────────────────────────────────────


class TestSelectChampion:
    def test_champion_selected(self, adapter):
        result = adapter.select_champion()
        assert result["model_type"] == "da_anchor_fallback"
        assert result["model_name"] == "rt_da_anchor"

    def test_verdict_is_fast_dev(self, adapter):
        result = adapter.select_champion()
        assert result["verdict"] == "FAST_DEV_ONLY"

    def test_status_updated(self, adapter):
        adapter.select_champion()
        assert adapter.status == REALTIME_DEEP_READY_FAST_DEV

    def test_smape_is_none(self, adapter):
        result = adapter.select_champion()
        assert result["sMAPE_floor50"] is None


# ── run_backtest ──────────────────────────────────────────────────────────────


class TestRunBacktest:
    def test_returns_dict(self, adapter):
        result = adapter.run_backtest()
        assert isinstance(result, dict)

    def test_status_key(self, adapter):
        result = adapter.run_backtest()
        assert "status" in result


# ── export_online_pack ────────────────────────────────────────────────────────


class TestExportOnlinePack:
    def test_export_with_predictions(self, adapter, da_predictions, tmp_path):
        result = adapter.export_online_pack(
            da_predictions=da_predictions,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, dict)
        assert "status" in result

    def test_export_strips_y_true_from_input(self, adapter, da_predictions_with_y_true, tmp_path):
        """y_true in input should not leak into the exported pack."""
        result = adapter.export_online_pack(
            da_predictions=da_predictions_with_y_true,
            output_dir=str(tmp_path),
        )
        # The adapter builds pack from da_anchor, so y_true is not in output
        assert result["status"] == "EXPORTED"
        if result.get("output_path"):
            pack = pd.read_csv(result["output_path"])
            assert "y_true" not in pack.columns

    def test_export_without_predictions(self, adapter, tmp_path):
        result = adapter.export_online_pack(output_dir=str(tmp_path))
        assert isinstance(result, dict)

    def test_output_columns_defined(self):
        assert len(ONLINE_PACK_COLUMNS) == 14
        assert "business_day" in ONLINE_PACK_COLUMNS
        assert "da_anchor" in ONLINE_PACK_COLUMNS

    def test_forbidden_columns_defined(self):
        assert "y_true" in FORBIDDEN_ONLINE_COLUMNS
        assert len(FORBIDDEN_ONLINE_COLUMNS) == 4


# ── validate_online_pack ──────────────────────────────────────────────────────


class TestValidateOnlinePack:
    def test_validate_missing_file(self, adapter):
        result = adapter.validate_online_pack("/nonexistent/path.csv")
        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_good_pack(self, adapter, tmp_path):
        pack = pd.DataFrame({
            col: [1] * 24 for col in ONLINE_PACK_COLUMNS
        })
        pack_path = str(tmp_path / "pack.csv")
        pack.to_csv(pack_path, index=False)
        result = adapter.validate_online_pack(pack_path)
        assert result["valid"] is True

    def test_validate_forbidden_column_detected(self, adapter, tmp_path):
        pack = pd.DataFrame({
            col: [1] * 24 for col in ONLINE_PACK_COLUMNS
        })
        pack["y_true"] = [100.0] * 24
        pack_path = str(tmp_path / "pack_bad.csv")
        pack.to_csv(pack_path, index=False)
        result = adapter.validate_online_pack(pack_path)
        assert result["valid"] is False


# ── _build_online_pack ────────────────────────────────────────────────────────


class TestBuildOnlinePack:
    def test_build_from_y_pred(self, adapter, da_predictions):
        pack = adapter._build_online_pack(da_predictions)
        assert isinstance(pack, pd.DataFrame)
        assert "da_anchor" in pack.columns
        assert "trend_pred" in pack.columns

    def test_rt_pred_equals_da_anchor(self, adapter, da_predictions):
        pack = adapter._build_online_pack(da_predictions)
        pd.testing.assert_series_equal(
            pack["deep_rt_pred"], pack["da_anchor"], check_names=False
        )

    def test_trend_confidence_is_05(self, adapter, da_predictions):
        pack = adapter._build_online_pack(da_predictions)
        assert (pack["trend_confidence"] == 0.5).all()

    def test_trend_model_name(self, adapter, da_predictions):
        pack = adapter._build_online_pack(da_predictions)
        assert (pack["trend_model_name"] == "rt_da_anchor_FALLBACK").all()
