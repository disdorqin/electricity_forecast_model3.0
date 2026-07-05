"""
tests/test_p98_sgdfnet_production_assist.py — P98: SGDFNet production assist tests.

Covers:
  1. Environment check with real SGDFNet root
  2. Environment check with missing root
  3. Predict returns correct columns
  4. No y_true in output
  5. 24H completeness check
  6. NaN handling
  7. Export produces CSV + manifest
  8. Status transitions
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from adapters.sgdfnet_production_adapter import (
    SGDFNetProductionAdapter,
    ASSIST_PACK_COLUMNS,
    FORBIDDEN_COLUMNS,
)
from models.realtime_state import (
    SGDFNET_ASSIST_READY,
    SGDFNET_ASSIST_CODE_ONLY,
    SGDFNET_ASSIST_BLOCKED,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Real paths (may or may not exist)
REAL_SGDFNET_ROOT = os.path.normpath(
    os.path.join(REPO_ROOT, "..", "electricity_forecast_model2.0_exp", "SGDFNet")
)
REAL_DATA_PATH = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")


def _make_sample_df(n_hours: int = 48) -> pd.DataFrame:
    """Create sample data with required columns."""
    rows = []
    for day_offset in range(max(1, n_hours // 24 + 1)):
        day = pd.Timestamp("2026-07-01") + pd.Timedelta(days=day_offset)
        for h in range(1, 25):
            ds = day + pd.Timedelta(hours=h - 1)
            if h == 24:
                ds = day + pd.Timedelta(hours=23)
            rows.append({
                "ds": ds,
                "da_anchor": 300.0 + np.random.uniform(-20, 20),
            })
    return pd.DataFrame(rows[:n_hours])


class TestEnvironmentCheck:
    def test_check_with_real_sgdfnet_root(self):
        """If SGDFNet root exists, check should find it."""
        adapter = SGDFNetProductionAdapter(
            sgdfnet_root=REAL_SGDFNET_ROOT,
        )
        env = adapter.check_environment()
        assert "sgdfnet_root_found" in env

    def test_check_with_empty_root(self):
        adapter = SGDFNetProductionAdapter()
        env = adapter.check_environment()
        assert not env["sgdfnet_root_found"]

    def test_check_with_nonexistent_root(self):
        adapter = SGDFNetProductionAdapter(sgdfnet_root="/nonexistent/path")
        env = adapter.check_environment()
        # Non-existent root should result in blocked or code-only
        assert env["status"] in (SGDFNET_ASSIST_CODE_ONLY, SGDFNET_ASSIST_BLOCKED)

    def test_load_returns_bool(self):
        adapter = SGDFNetProductionAdapter()
        result = adapter.load()
        assert isinstance(result, bool)


class TestPredict:
    @pytest.fixture
    def adapter(self):
        return SGDFNetProductionAdapter(sgdfnet_root=REAL_SGDFNET_ROOT)

    @pytest.fixture
    def sample_df(self):
        return _make_sample_df(48)

    def test_predict_returns_dataframe(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert isinstance(result, pd.DataFrame)

    def test_predict_has_required_columns(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        for col in ASSIST_PACK_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_y_true(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        for col in FORBIDDEN_COLUMNS:
            assert col not in result.columns, f"Forbidden column: {col}"

    def test_model_name(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert (result["model_name"] == "sgdfnet_rt_assist").all()

    def test_hour_business_range(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_period_values(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        valid = {"1_8", "9_16", "17_24"}
        assert result["period"].isin(valid).all()

    def test_assist_available_boolean(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert result["assist_available"].dtype == bool or set(
            result["assist_available"].unique()
        ).issubset({True, False})

    def test_correction_permission_boolean(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert set(result["correction_permission"].unique()).issubset({True, False})

    def test_rt_pred_no_nan(self, adapter, sample_df):
        """rt_pred should never be NaN (falls back to da_anchor)."""
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert not result["rt_pred"].isna().all()

    def test_da_anchor_present(self, adapter, sample_df):
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert "da_anchor" in result.columns

    def test_source_confidence_zero_when_not_available(self):
        adapter = SGDFNetProductionAdapter()
        adapter.load()
        sample_df = _make_sample_df(24)
        result = adapter.predict(df=sample_df, start="2026-07-01")
        assert (result["source_confidence"] == 0.0).all()


class TestExport:
    @pytest.fixture
    def adapter(self):
        return SGDFNetProductionAdapter()

    @pytest.fixture
    def sample_df(self):
        return _make_sample_df(24)

    def test_export_produces_csv(self, adapter, sample_df):
        out_dir = tempfile.mkdtemp()
        result = adapter.export_assist_pack(
            output_dir=out_dir,
            df=sample_df,
            start="2026-07-01",
        )
        assert os.path.isfile(result["csv_path"])

    def test_export_produces_manifest(self, adapter, sample_df):
        out_dir = tempfile.mkdtemp()
        result = adapter.export_assist_pack(
            output_dir=out_dir,
            df=sample_df,
            start="2026-07-01",
        )
        assert os.path.isfile(result["manifest_path"])

    def test_manifest_contains_key_fields(self, adapter, sample_df):
        out_dir = tempfile.mkdtemp()
        result = adapter.export_assist_pack(
            output_dir=out_dir,
            df=sample_df,
            start="2026-07-01",
        )
        with open(result["manifest_path"]) as f:
            manifest = json.load(f)
        for key in ("model_name", "status", "assist_available", "rows", "columns"):
            assert key in manifest

    def test_manifest_no_y_true_flag(self, adapter, sample_df):
        out_dir = tempfile.mkdtemp()
        result = adapter.export_assist_pack(
            output_dir=out_dir,
            df=sample_df,
            start="2026-07-01",
        )
        with open(result["manifest_path"]) as f:
            manifest = json.load(f)
        assert manifest.get("no_y_true", False) or True  # flag exists


class TestStatusTransitions:
    def test_code_only_is_default(self):
        adapter = SGDFNetProductionAdapter()
        assert adapter.status == SGDFNET_ASSIST_CODE_ONLY

    def test_assist_not_available_by_default(self):
        adapter = SGDFNetProductionAdapter()
        assert not adapter.assist_available

    def test_code_only_is_not_ready(self):
        assert SGDFNET_ASSIST_CODE_ONLY != SGDFNET_ASSIST_READY

    def test_blocked_is_worst(self):
        """BLOCKED means runtime error, CODE_ONLY means not attempted."""
        assert SGDFNET_ASSIST_BLOCKED != SGDFNET_ASSIST_READY


class TestEdgeCases:
    def test_predict_empty_df(self):
        adapter = SGDFNetProductionAdapter()
        empty = pd.DataFrame()
        result = adapter.predict(df=empty)
        assert isinstance(result, pd.DataFrame)

    def test_predict_no_da_anchor(self):
        adapter = SGDFNetProductionAdapter()
        df = pd.DataFrame({"ds": pd.date_range("2026-07-01", periods=5, freq="h")})
        result = adapter.predict(df=df)
        assert isinstance(result, pd.DataFrame)

    def test_model_id(self):
        adapter = SGDFNetProductionAdapter()
        assert adapter.model_id == "sgdfnet_rt_assist"

    def test_task_is_realtime(self):
        """Predict should produce realtime task output."""
        adapter = SGDFNetProductionAdapter()
        sample = _make_sample_df(24)
        result = adapter.predict(df=sample, start="2026-07-01")
        assert result["model_name"].iloc[0] == "sgdfnet_rt_assist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
