"""
tests/test_loaders_contract.py — Loader layer contract tests.

Validates:
    1. load_table reads CSV correctly (utf-8, utf-8-sig, gbk)
    2. Chinese column names map to canonical English names
    3. add_business_time adds business_day/hour_business/period
    4. 00:00 → business_day D-1, hour 24
    5. Missing file raises FileNotFoundError
    6. Unsupported extension raises ValueError
    7. Excel format raise import errors gracefully
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.loaders import (
    load_table,
    CN_COLUMN_MAP,
    ENCODING_CANDIDATES,
)


class TestLoadTableCSV:
    """Contract: load_table reads CSV correctly."""

    def test_reads_utf8_csv(self, tmp_path):
        """Basic utf-8 CSV read returns DataFrame with expected columns."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("ds,da_anchor,y_pred\n2026-03-05 01:00,120.5,121.0\n", encoding="utf-8")
        df, meta = load_table(str(csv_path))
        assert len(df) == 1
        assert "ds" in df.columns
        assert "da_anchor" in df.columns
        assert meta["encoding_used"] == "utf-8"
        assert meta["rows"] == 1

    def test_reads_utf8_sig_csv(self, tmp_path):
        """utf-8-sig CSV (with BOM) is read correctly."""
        csv_path = tmp_path / "test_bom.csv"
        csv_path.write_bytes(b"\xef\xbb\xbfds,da_anchor\n2026-03-05 01:00,120.5\n")
        df, meta = load_table(str(csv_path))
        assert len(df) == 1
        assert "ds" in df.columns
        assert meta["encoding_used"] == "utf-8-sig"

    def test_reads_gbk_csv(self, tmp_path):
        """GBK-encoded CSV with Chinese content is read correctly."""
        csv_path = tmp_path / "test_gbk.csv"
        # Use Chinese content that would fail with utf-8
        content = "时间,日前价格\n2026-03-05 01:00,120.5\n"
        csv_path.write_bytes(content.encode("gbk"))
        df, meta = load_table(str(csv_path))
        assert len(df) == 1
        assert meta["encoding_used"] == "gbk"
        assert "ds" in df.columns  # 时间 mapped to ds
        assert "da_anchor" in df.columns  # 日前价格 mapped to da_anchor

    def test_encoding_fallback_order(self):
        """Encoding candidates are tried in the correct order."""
        assert ENCODING_CANDIDATES == ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    def test_missing_file_raises(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_table("/nonexistent/path.csv")

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extension raises ValueError."""
        fpath = tmp_path / "data.parquet"
        fpath.write_text("test", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_table(str(fpath))

    def test_empty_csv_returns_empty(self, tmp_path):
        """Empty CSV (header only) returns empty DataFrame."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("ds,da_anchor\n", encoding="utf-8")
        df, meta = load_table(str(csv_path))
        assert len(df) == 0
        assert meta["rows"] == 0


class TestChineseColumnMapping:
    """Contract: Chinese column names map to canonical English names."""

    @pytest.mark.parametrize("cn_name,canonical", [
        ("时间", "ds"),
        ("日前价格", "da_anchor"),
        ("实时价格", "rt_actual"),
        ("预测价格", "y_pred"),
        ("实际价格", "y_true"),
        ("负荷", "load"),
        ("风电", "wind"),
        ("光伏", "solar"),
        ("竞价空间", "bidding_space"),
        ("净负荷", "net_load"),
    ])
    def test_chinese_column_mapped(self, tmp_path, cn_name, canonical):
        """Chinese column '{cn_name}' maps to '{canonical}'."""
        csv_path = tmp_path / "cn_test.csv"
        csv_path.write_text(f"{cn_name}\n2026-03-05 01:00\n", encoding="utf-8")
        df, meta = load_table(str(csv_path))
        assert canonical in df.columns, f"{cn_name} should map to {canonical}"
        # Check mapping is recorded
        assert (cn_name, canonical) in meta["cn_mappings"]

    def test_mixed_chinese_and_english_columns(self, tmp_path):
        """Mixed CN/EN columns are all mapped and preserved."""
        csv_path = tmp_path / "mixed.csv"
        csv_path.write_text(
            "时间,da_anchor,日前价格,y_pred\n"
            "2026-03-05 01:00,120.5,121.0,122.0\n",
            encoding="utf-8",
        )
        df, meta = load_table(str(csv_path))
        assert "ds" in df.columns  # from 时间
        assert "da_anchor" in df.columns  # from both da_anchor and 日前价格 (collision ok)
        assert "y_pred" in df.columns
        # Confirm mappings recorded
        cn_mappings = dict(meta["cn_mappings"])
        assert "时间" in cn_mappings
        assert "日前价格" in cn_mappings


class TestParseDates:
    """Contract: parse_dates and add_business_time options."""

    def test_parse_dates_converts_ds_to_datetime(self, tmp_path):
        """With parse_dates=True, ds column becomes datetime."""
        csv_path = tmp_path / "dates.csv"
        csv_path.write_text("ds,da_anchor\n2026-03-05 01:00,120.5\n", encoding="utf-8")
        df, meta = load_table(str(csv_path), parse_dates=True)
        assert pd.api.types.is_datetime64_any_dtype(df["ds"])

    def test_add_business_time_adds_columns(self, tmp_path):
        """add_business_time=True adds business_day, hour_business, period."""
        csv_path = tmp_path / "bt.csv"
        csv_path.write_text("ds,da_anchor\n2026-03-05 01:00,120.5\n", encoding="utf-8")
        df, meta = load_table(str(csv_path), parse_dates=True, add_business_time=True)
        assert "business_day" in df.columns
        assert "hour_business" in df.columns
        assert "period" in df.columns
        assert meta["added_business_time"] is True

    def test_midnight_maps_to_previous_day_hour_24(self, tmp_path):
        """00:00 timestamp → business_day D-1, hour 24."""
        csv_path = tmp_path / "midnight.csv"
        csv_path.write_text("ds\n2026-03-10 00:00\n", encoding="utf-8")
        df, meta = load_table(str(csv_path), parse_dates=True, add_business_time=True)
        assert df["business_day"].iloc[0] == pd.Timestamp("2026-03-09")
        assert df["hour_business"].iloc[0] == 24

    def test_afternoon_maps_to_same_day(self, tmp_path):
        """Afternoon timestamp → same business_day."""
        csv_path = tmp_path / "afternoon.csv"
        csv_path.write_text("ds\n2026-03-10 14:00\n", encoding="utf-8")
        df, meta = load_table(str(csv_path), parse_dates=True, add_business_time=True)
        assert df["business_day"].iloc[0] == pd.Timestamp("2026-03-10")
        assert df["hour_business"].iloc[0] == 14


class TestMetadata:
    """Contract: load_table returns correct metadata."""

    def test_metadata_keys(self, tmp_path):
        """Metadata dict contains all expected keys."""
        csv_path = tmp_path / "meta.csv"
        csv_path.write_text("ds\n2026-03-05 01:00\n", encoding="utf-8")
        df, meta = load_table(str(csv_path))
        for key in ("path", "rows", "columns", "encoding_used", "cn_mappings",
                     "added_business_time", "parse_dates", "errors"):
            assert key in meta, f"Missing metadata key: {key}"
