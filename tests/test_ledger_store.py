"""
tests/test_ledger_store.py — Generic ledger store contract tests.

Validates:
    1. load_ledger returns empty DataFrame for missing path
    2. load_ledger returns DataFrame with specified columns
    3. save_ledger writes and load_ledger reads back
    4. append_ledger deduplicates on key with keep="latest"
    5. append_ledger deduplicates with keep="first"
    6. append_ledger handles empty inputs
    7. validate_ledger_keys detects duplicates
    8. validate_ledger_keys passes clean DataFrame
    9. add_run_metadata adds timestamps
"""

from __future__ import annotations

import pandas as pd
import pytest

from ledgers.store import (
    load_ledger,
    save_ledger,
    append_ledger,
    validate_ledger_keys,
    add_run_metadata,
)


class TestLoadLedger:
    """Contract: load_ledger."""

    def test_missing_path_returns_empty(self):
        """load_ledger with missing path returns empty DataFrame."""
        df = load_ledger("/nonexistent/path.csv", columns=["a", "b"])
        assert len(df) == 0
        assert list(df.columns) == ["a", "b"]

    def test_none_path_returns_empty(self):
        """load_ledger with None path returns empty DataFrame."""
        df = load_ledger(None, columns=["a", "b"])
        assert len(df) == 0

    def test_no_columns_returns_empty(self):
        """load_ledger with no columns returns empty DataFrame."""
        df = load_ledger(None)
        assert len(df) == 0
        assert list(df.columns) == []


class TestSaveLoadRoundTrip:
    """Contract: save + load round-trip via tmp_path."""

    def test_save_and_load_csv(self, tmp_path):
        """Save and load a CSV file."""
        path = str(tmp_path / "test.csv")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        save_ledger(df, path)
        loaded = load_ledger(path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["a", "b"]

    def test_load_adds_missing_columns(self, tmp_path):
        """load_ledger adds missing columns if specified."""
        path = str(tmp_path / "test.csv")
        save_ledger(pd.DataFrame({"a": [1]}), path)
        loaded = load_ledger(path, columns=["a", "b"])
        assert "b" in loaded.columns


class TestAppendLedger:
    """Contract: append_ledger dedup logic."""

    def test_dedup_same_key_keeps_latest(self):
        """Same key appended: keep="latest" keeps the last entry."""
        existing = pd.DataFrame({
            "id": [1, 2],
            "value": ["a", "b"],
            "updated_at": ["2026-01-01", "2026-01-02"],
        })
        new = pd.DataFrame({
            "id": [1],
            "value": ["c"],
            "updated_at": ["2026-01-03"],
        })
        result = append_ledger(existing, new, key_cols=["id"], keep="latest")
        assert len(result) == 2  # id=1 and id=2
        assert result[result["id"] == 1]["value"].iloc[0] == "c"  # latest

    def test_dedup_same_key_keeps_first(self):
        """Same key appended: keep="first" keeps the original."""
        existing = pd.DataFrame({
            "id": [1, 2],
            "value": ["a", "b"],
        })
        new = pd.DataFrame({
            "id": [1],
            "value": ["c"],
        })
        result = append_ledger(existing, new, key_cols=["id"], keep="first")
        assert len(result) == 2
        assert result[result["id"] == 1]["value"].iloc[0] == "a"  # first

    def test_no_duplicates_no_change(self):
        """No duplicate keys: all rows preserved."""
        existing = pd.DataFrame({"id": [1], "value": ["a"]})
        new = pd.DataFrame({"id": [2], "value": ["b"]})
        result = append_ledger(existing, new, key_cols=["id"])
        assert len(result) == 2

    def test_empty_existing(self):
        """Empty existing: just returns new."""
        existing = pd.DataFrame(columns=["id", "value"])
        new = pd.DataFrame({"id": [1], "value": ["a"]})
        result = append_ledger(existing, new, key_cols=["id"])
        assert len(result) == 1

    def test_empty_new(self):
        """Empty new: returns copy of existing."""
        existing = pd.DataFrame({"id": [1], "value": ["a"]})
        new = pd.DataFrame(columns=["id", "value"])
        result = append_ledger(existing, new, key_cols=["id"])
        assert len(result) == 1

    def test_unknown_keep_raises(self):
        """Unknown keep strategy raises ValueError."""
        existing = pd.DataFrame({"id": [1]})
        new = pd.DataFrame({"id": [2]})
        with pytest.raises(ValueError, match="Unknown keep strategy"):
            append_ledger(existing, new, key_cols=["id"], keep="invalid")


class TestValidateLedgerKeys:
    """Contract: validate_ledger_keys."""

    def test_passes_clean(self):
        """No duplicates returns True."""
        df = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        valid, errors = validate_ledger_keys(df, ["id"])
        assert valid
        assert len(errors) == 0

    def test_detects_duplicates(self):
        """Duplicates are detected."""
        df = pd.DataFrame({"id": [1, 1], "value": ["a", "b"]})
        valid, errors = validate_ledger_keys(df, ["id"])
        assert not valid
        assert any("duplicate" in e for e in errors)

    def test_missing_key_columns(self):
        """Missing key columns returns error."""
        df = pd.DataFrame({"id": [1]})
        valid, errors = validate_ledger_keys(df, ["nonexistent"])
        assert not valid

    def test_empty_df(self):
        """Empty DataFrame passes."""
        df = pd.DataFrame(columns=["id"])
        valid, errors = validate_ledger_keys(df, ["id"])
        assert valid


class TestAddRunMetadata:
    """Contract: add_run_metadata."""

    def test_adds_columns(self):
        """add_run_metadata adds run_id, created_at, updated_at."""
        df = pd.DataFrame({"value": [1]})
        result = add_run_metadata(df, run_id="test_run")
        assert "run_id" in result.columns
        assert "created_at" in result.columns
        assert "updated_at" in result.columns
        assert result["run_id"].iloc[0] == "test_run"

    def test_does_not_overwrite_existing(self):
        """Does not overwrite existing run_id if provided."""
        df = pd.DataFrame({"value": [1], "run_id": ["existing"]})
        result = add_run_metadata(df, run_id="new_run")
        assert result["run_id"].iloc[0] == "new_run"  # overwritten
