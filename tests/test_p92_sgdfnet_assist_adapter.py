"""
Tests for P92: SGDFNet Assist Adapter.

Covers:
  1. Adapter loads without error (code-only or ready)
  2. Predict output has correct schema
  3. Without SGDFNet root, status = CODE_ONLY
  4. SGDFNet unavailable → sgdfnet_pred = NaN, assist_available = False
  5. No y_true in output
  6. Pack columns all present
  7. Correction_permission = False when SGDFNet unavailable
  8. Reason codes contain SGDFNET_ASSIST_DISABLED when unavailable
  9. Export produces both CSV and manifest
  10. Manifest contains correct fields
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from models.adapters.sgdfnet_assist_adapter import SGDFNetAssistAdapter
from models.realtime_state import (
    SGDFNET_ASSIST_READY,
    SGDFNET_ASSIST_CODE_ONLY,
    SGDFNET_ASSIST_BLOCKED,
    SGDFNET_ASSIST_ACTIVE,
)


class TestAdapterLoad:
    """Test adapter loading behavior."""

    def test_load_without_root(self):
        """Without sgdfnet_root, status should be CODE_ONLY."""
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        assert adapter.assist_status == SGDFNET_ASSIST_CODE_ONLY
        assert adapter._loaded

    def test_load_with_nonexistent_root(self):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="/nonexistent/path")
        adapter.load()
        assert adapter.assist_status == SGDFNET_ASSIST_CODE_ONLY

    @pytest.fixture
    def sample_df(self):
        """Create sample data with da_anchor."""
        rows = []
        for day in ["2026-07-01", "2026-07-02"]:
            for h in range(1, 25):
                ds = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = pd.Timestamp(day) + pd.Timedelta(hours=23)
                rows.append({"ds": ds, "da_anchor": 300.0 + np.random.uniform(-20, 20)})
        df = pd.DataFrame(rows)
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")
        return df

    def test_predict_without_sgdfnet(self, sample_df):
        """Without SGDFNet, predict should return NaN sgdfnet_pred."""
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert len(result) > 0
        assert result["sgdfnet_pred"].isna().all()
        assert ~result["assist_available"].any()

    def test_predict_columns(self, sample_df):
        """Predict output should contain all PACK_COLUMNS."""
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        for col in adapter.PACK_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_y_true(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        for col in adapter.FORBIDDEN_COLUMNS:
            assert col not in result.columns

    def test_rt_pred_equals_da_anchor_when_no_sgdfnet(self, sample_df):
        """Without SGDFNet, rt_pred should fall back to da_anchor."""
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert np.allclose(result["rt_pred"].values, result["da_anchor"].values, rtol=1e-5)

    def test_correction_permission_false_when_no_sgdfnet(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert ~result["correction_permission"].any()

    def test_reason_codes_disabled(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        for code in result["reason_codes"].unique():
            assert "DISABLED" in str(code)

    def test_source_confidence_zero_when_no_sgdfnet(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert (result["source_confidence"] == 0.0).all()

    def test_model_name_constant(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert (result["model_name"] == "sgdfnet_rt_assist").all()

    def test_da_anchor_present(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert "da_anchor" in result.columns
        assert not result["da_anchor"].isna().all()

    def test_hour_business_range(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        assert result["hour_business"].min() >= 1
        assert result["hour_business"].max() <= 24

    def test_period_values(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        result = adapter.predict(df=sample_df, start="2026-07-01", end="2026-07-02")
        valid_periods = {"1_8", "9_16", "17_24"}
        for p in result["period"].unique():
            assert p in valid_periods


class TestExportAssistPack:
    """Test export_assist_pack method."""

    @pytest.fixture
    def sample_df(self):
        rows = []
        for day in ["2026-07-01", "2026-07-02"]:
            for h in range(1, 25):
                ds = pd.Timestamp(day) + pd.Timedelta(hours=h - 1)
                if h == 24:
                    ds = pd.Timestamp(day) + pd.Timedelta(hours=23)
                rows.append({"ds": ds, "da_anchor": 300.0})
        df = pd.DataFrame(rows)
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")
        return df

    def test_export_produces_csv(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = adapter.export_assist_pack(
                output_dir=tmpdir,
                df=sample_df,
                start="2026-07-01",
                end="2026-07-02",
            )
            csv_path = result.get("csv_path", "")
            assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"

    def test_export_produces_manifest(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = adapter.export_assist_pack(
                output_dir=tmpdir,
                df=sample_df,
                start="2026-07-01",
                end="2026-07-02",
            )
            manifest_path = result.get("manifest_path", "")
            assert os.path.isfile(manifest_path)

    def test_manifest_contains_key_fields(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = adapter.export_assist_pack(
                output_dir=tmpdir,
                df=sample_df,
                start="2026-07-01",
                end="2026-07-02",
            )
            with open(result["manifest_path"]) as f:
                manifest = json.load(f)
            assert manifest["model_name"] == "sgdfnet_rt_assist"
            assert "assist_status" in manifest
            assert "rows" in manifest
            assert "columns" in manifest

    def test_manifest_rows_match_csv(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = adapter.export_assist_pack(
                output_dir=tmpdir,
                df=sample_df,
                start="2026-07-01",
                end="2026-07-02",
            )
            csv_df = pd.read_csv(result["csv_path"])
            assert len(csv_df) == result["rows"]

    def test_csv_no_y_true(self, sample_df):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = adapter.export_assist_pack(
                output_dir=tmpdir,
                df=sample_df,
                start="2026-07-01",
                end="2026-07-02",
            )
            csv_df = pd.read_csv(result["csv_path"])
            for col in adapter.FORBIDDEN_COLUMNS:
                assert col not in csv_df.columns


class TestAssistStatusTransitions:
    """Test status transitions."""

    def test_code_only_is_not_blocked(self):
        assert SGDFNET_ASSIST_CODE_ONLY != SGDFNET_ASSIST_BLOCKED
        assert SGDFNET_ASSIST_CODE_ONLY != SGDFNET_ASSIST_READY

    def test_ready_is_best(self):
        """READY is the only positive status."""
        assert SGDFNET_ASSIST_READY == "SGDFNET_ASSIST_READY"
        assert SGDFNET_ASSIST_READY != SGDFNET_ASSIST_CODE_ONLY

    def test_blocked_is_worst(self):
        assert SGDFNET_ASSIST_BLOCKED != SGDFNET_ASSIST_READY
        assert SGDFNET_ASSIST_BLOCKED == "SGDFNET_ASSIST_BLOCKED"

    def test_assist_active_reason_code(self):
        assert SGDFNET_ASSIST_ACTIVE == "SGDFNET_ASSIST_ACTIVE"


class TestEdgeCases:
    """Test edge cases."""

    def test_predict_empty_df(self):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        empty_df = pd.DataFrame(columns=["ds", "da_anchor"])
        result = adapter.predict(df=empty_df)
        assert len(result) == 0 or result.empty

    def test_predict_no_da_anchor_raises(self):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        df = pd.DataFrame({"ds": pd.date_range("2026-07-01", periods=5, freq="h"), "other": [1]*5})
        with pytest.raises(ValueError):
            adapter.predict(df=df)

    def test_predict_single_row(self):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        df = pd.DataFrame({"ds": [pd.Timestamp("2026-07-01 01:00")], "da_anchor": [300.0]})
        result = adapter.predict(df=df)
        assert len(result) > 0

    def test_model_id(self):
        adapter = SGDFNetAssistAdapter()
        assert adapter.model_id == "sgdfnet_rt_assist"

    def test_task_is_realtime(self):
        adapter = SGDFNetAssistAdapter()
        assert adapter.task == "realtime"

    def test_load_idempotent(self):
        adapter = SGDFNetAssistAdapter(sgdfnet_root="")
        adapter.load()
        adapter.load()  # second call should not error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
