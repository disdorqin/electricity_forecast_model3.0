"""
tests/test_p137_trusted_2025_prediction_ledger_v2.py — Tests for P137.

At least 12 tests covering:
  - Ledger combines both models
  - y_true is stripped
  - No NaN y_pred
  - model_count >= 2 check
  - Single model -> BLOCKED
  - Output file exists after run
  - Manifest JSON structure
  - Column schema compliance
  - business_day / hour_business correctness
  - Forbidden column detection
  - Date range correctness
  - Rows per model accounting
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_p137_trusted_2025_prediction_ledger_v2 import (
    DEFAULT_CFG05_PATH,
    DEFAULT_CATBOOST_SPIKE_PATH,
    DEFAULT_OUTPUT_DIR,
    FORBIDDEN_COLUMNS,
    LEDGER_COLUMNS,
    _derive_target_day,
    _read_cfg05,
    _read_catboost_spike,
    _safety_checks,
    _standardize,
    run_p137_trusted_ledger,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_cfg05_csv(n_days: int = 5, with_y_true: bool = True) -> str:
    """Create a temporary cfg05 predictions CSV."""
    rows = []
    for d in range(n_days):
        day_str = f"2025-01-{d+1:02d}"
        for h in range(1, 25):
            # hour_business 24 → wall-clock next day 00:00
            if h == 24:
                ds_val = (pd.Timestamp(day_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                ds_val = f"2025-01-{d+1:02d} {h:02d}:00:00"
            rows.append({
                "task": "dayahead",
                "model_name": "lightgbm_cfg05_dayahead",
                "target_day": day_str,
                "business_day": day_str,
                "ds": ds_val,
                "hour_business": h,
                "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                "y_pred": 200.0 + h + d,
                "source_confidence": 1.0,
                "model_version": "1.0.0",
            })
            if with_y_true:
                rows[-1]["y_true"] = 210.0 + h + d
    df = pd.DataFrame(rows)
    path = os.path.join(tempfile.gettempdir(), "test_cfg05_predictions.csv")
    df.to_csv(path, index=False)
    return path


def _make_catboost_csv(n_days: int = 5) -> str:
    """Create a temporary catboost_spike predictions CSV."""
    rows = []
    for d in range(n_days):
        day_str = f"2025-01-{d+1:02d}"
        for h in range(1, 25):
            # hour_business 24 → wall-clock next day 00:00
            if h == 24:
                ds_val = (pd.Timestamp(day_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            else:
                ds_val = f"2025-01-{d+1:02d} {h:02d}:00:00"
            rows.append({
                "business_day": day_str,
                "ds": ds_val,
                "hour_business": h,
                "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                "y_pred": 195.0 + h + d,
                "model_name": "catboost_spike_residual",
                "task": "dayahead",
                "model_version": "p136_path_a",
                "source_confidence": 1.0,
            })
    df = pd.DataFrame(rows)
    path = os.path.join(tempfile.gettempdir(), "test_catboost_predictions.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def temp_ledger_dir():
    """Provide a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ── Tests ─────────────────────────────────────────────────────────────


class TestLedgerCombinesModels:
    """Test that the ledger combines both models."""

    def test_ledger_has_two_models(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        assert result["model_count"] == 2
        assert "lightgbm_cfg05_dayahead" in result["models"]
        assert "catboost_spike_residual" in result["models"]

    def test_ledger_total_rows_sum(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv(n_days=3)
        catboost_path = _make_catboost_csv(n_days=3)
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        expected = 3 * 24 * 2  # 3 days x 24 hours x 2 models
        assert result["total_rows"] == expected


class TestYTrueStripped:
    """Test that y_true is stripped from the ledger."""

    def test_y_true_not_in_ledger_columns(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv(with_y_true=True)
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        if ledger_path and os.path.isfile(ledger_path):
            ledger = pd.read_csv(ledger_path)
            assert "y_true" not in ledger.columns

    def test_forbidden_columns_absent(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv(with_y_true=True)
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        if ledger_path and os.path.isfile(ledger_path):
            ledger = pd.read_csv(ledger_path)
            for col in FORBIDDEN_COLUMNS:
                assert col not in ledger.columns, f"Forbidden column {col} found in ledger"


class TestNoNaNYpred:
    """Test no NaN in y_pred."""

    def test_no_nan_in_y_pred(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        if ledger_path and os.path.isfile(ledger_path):
            ledger = pd.read_csv(ledger_path)
            assert ledger["y_pred"].isna().sum() == 0


class TestModelCountCheck:
    """Test model_count >= 2 check."""

    def test_single_model_blocked(self, temp_ledger_dir):
        """Single model should produce BLOCKED status."""
        # Create two CSVs with the same model name
        cfg05_path = _make_cfg05_csv()
        # Overwrite catboost to have same model name
        df = pd.read_csv(cfg05_path)
        catboost_path = os.path.join(tempfile.gettempdir(), "test_single_model.csv")
        df.to_csv(catboost_path, index=False)

        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)
        assert result["status"] == "TRUSTED_LEDGER_BLOCKED_SINGLE_MODEL"

    def test_two_models_ready(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)
        assert result["status"] == "TRUSTED_LEDGER_V2_READY"


class TestOutputFileExists:
    """Test output file exists after successful run."""

    def test_ledger_csv_exists(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        assert os.path.isfile(ledger_path), f"Ledger file not found: {ledger_path}"

    def test_manifest_json_exists(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        manifest_path = os.path.join(temp_ledger_dir, "manifest.json")
        assert os.path.isfile(manifest_path)


class TestManifestStructure:
    """Test manifest JSON structure."""

    def test_manifest_has_required_keys(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        for key in ("model_count", "total_rows", "rows_per_model", "date_range", "status"):
            assert key in result, f"Missing manifest key: {key}"

    def test_manifest_rows_per_model(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv(n_days=2)
        catboost_path = _make_catboost_csv(n_days=2)
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        rpm = result["rows_per_model"]
        assert isinstance(rpm, dict)
        assert len(rpm) == 2
        for model_name, count in rpm.items():
            assert count == 2 * 24  # 2 days x 24 hours


class TestColumnSchema:
    """Test column schema compliance."""

    def test_ledger_columns_match_schema(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        if ledger_path and os.path.isfile(ledger_path):
            ledger = pd.read_csv(ledger_path)
            for col in LEDGER_COLUMNS:
                assert col in ledger.columns, f"Missing column: {col}"


class TestBusinessDayCorrectness:
    """Test business_day / hour_business correctness."""

    def test_hour_business_range(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        if ledger_path and os.path.isfile(ledger_path):
            ledger = pd.read_csv(ledger_path)
            assert ledger["hour_business"].min() >= 1
            assert ledger["hour_business"].max() <= 24

    def test_business_day_format(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv()
        catboost_path = _make_catboost_csv()
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        ledger_path = result.get("ledger_path", "")
        if ledger_path and os.path.isfile(ledger_path):
            ledger = pd.read_csv(ledger_path)
            # business_day should be parseable dates
            bd = pd.to_datetime(ledger["business_day"])
            assert bd.notna().all()


class TestDeriveTargetDay:
    """Test _derive_target_day helper."""

    def test_derive_target_day_basic(self):
        ds = pd.Series(["2025-01-01 01:00:00", "2025-01-01 12:00:00"])
        result = _derive_target_day(ds)
        # ds=Jan 1 01:00 → normalize=Jan 1 → +1 = Jan 2
        assert result.iloc[0] == "2025-01-02"
        assert result.iloc[1] == "2025-01-02"

    def test_derive_target_day_midnight(self):
        ds = pd.Series(["2025-01-02 00:00:00"])
        result = _derive_target_day(ds)
        # ds=Jan 2 00:00 → normalize=Jan 2 → +1 = Jan 3
        assert result.iloc[0] == "2025-01-03"

    def test_derive_target_day_hour24(self):
        """Hour=24 in ds string should be handled gracefully."""
        ds = pd.Series(["2025-01-01 24:00:00"])
        result = _derive_target_day(ds)
        # 24:00 → replaced with 00:00, then +1 day for hour24 → Jan 2 → +1 = Jan 3
        assert result.iloc[0] == "2025-01-03"


class TestSafetyChecksFunction:
    """Test _safety_checks directly."""

    def test_blocked_single_model(self):
        df = pd.DataFrame({
            "model_name": ["model_a"] * 24,
            "y_pred": np.random.randn(24),
            "business_day": ["2025-01-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        status, codes = _safety_checks(df)
        assert status == "TRUSTED_LEDGER_BLOCKED_SINGLE_MODEL"

    def test_blocked_nan_ypred(self):
        df = pd.DataFrame({
            "model_name": ["model_a"] * 12 + ["model_b"] * 12,
            "y_pred": [1.0] * 11 + [np.nan] + [2.0] * 12,
            "business_day": ["2025-01-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        status, codes = _safety_checks(df)
        assert status == "TRUSTED_LEDGER_BLOCKED_NAN_YPRED"

    def test_blocked_forbidden_column(self):
        df = pd.DataFrame({
            "model_name": ["model_a"] * 12 + ["model_b"] * 12,
            "y_pred": [1.0] * 24,
            "business_day": ["2025-01-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_true": [2.0] * 24,
        })
        status, codes = _safety_checks(df)
        assert status == "TRUSTED_LEDGER_BLOCKED_YTRUE_LEAK"

    def test_ready_status(self):
        df = pd.DataFrame({
            "model_name": ["model_a"] * 12 + ["model_b"] * 12,
            "y_pred": [1.0] * 24,
            "business_day": ["2025-01-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        status, codes = _safety_checks(df)
        assert status == "TRUSTED_LEDGER_V2_READY"


class TestDateRange:
    """Test date range in manifest."""

    def test_date_range_present(self, temp_ledger_dir):
        cfg05_path = _make_cfg05_csv(n_days=3)
        catboost_path = _make_catboost_csv(n_days=3)
        result = run_p137_trusted_ledger(cfg05_path, catboost_path, temp_ledger_dir)

        dr = result.get("date_range", {})
        assert "start" in dr
        assert "end" in dr
        assert dr["start"] == "2025-01-01"
        assert dr["end"] == "2025-01-03"


# ── Integration test with real artifacts (skip if not available) ─────


class TestRealArtifacts:
    """Integration tests using real artifact files."""

    @pytest.fixture(autouse=True)
    def _check_artifacts(self):
        if not os.path.isfile(DEFAULT_CFG05_PATH):
            pytest.skip("cfg05 predictions not available")
        if not os.path.isfile(DEFAULT_CATBOOST_SPIKE_PATH):
            pytest.skip("catboost_spike predictions not available")

    def test_real_ledger_builds(self, temp_ledger_dir):
        result = run_p137_trusted_ledger(
            DEFAULT_CFG05_PATH, DEFAULT_CATBOOST_SPIKE_PATH, temp_ledger_dir
        )
        assert result["status"] == "TRUSTED_LEDGER_V2_READY"
        assert result["model_count"] == 2
        assert result["total_rows"] > 0

    def test_real_ledger_no_y_true(self, temp_ledger_dir):
        result = run_p137_trusted_ledger(
            DEFAULT_CFG05_PATH, DEFAULT_CATBOOST_SPIKE_PATH, temp_ledger_dir
        )
        ledger = pd.read_csv(result["ledger_path"])
        assert "y_true" not in ledger.columns
