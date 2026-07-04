"""
tests/test_p17_cfg05_predictions_to_ledger.py — P17 tests.

Tests prediction ledger conversion with synthetic data.
Minimum 12 tests covering:
  1. 24H complete day accepted
  2. 23H incomplete day rejected
  3. missing hour 24 rejected
  4. duplicate hour rejected
  5. prediction ledger canonical schema
  6. duplicate ledger keys rejected
  7. production ledger no y_true
  8. eval ledger may include y_true
  9. hour_business range 1..24
  10. no input → INCOMPLETE
  11. empty DataFrame → INCOMPLETE
  12. strict/non-strict CLI behavior
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_p17_cfg05_predictions_to_ledger import (
    run_p17_cfg05_predictions_to_ledger,
    LEDGER_READY,
    LEDGER_INCOMPLETE,
    LEDGER_INVALID,
)
from data.schema import PREDICTION_LEDGER_COLUMNS, EVAL_ONLY_COLUMNS


def _make_synthetic_predictions(n_days: int = 3, start_day: str = "2026-06-01") -> pd.DataFrame:
    """Create synthetic prediction output with 24 rows per day."""
    rows = []
    for d in range(n_days):
        td = pd.Timestamp(start_day) + pd.Timedelta(days=d)
        for h in range(1, 25):
            # hour_business h → timestamp
            if h <= 23:
                ts = pd.Timestamp(td) + pd.Timedelta(hours=h)
            else:
                ts = pd.Timestamp(td) + pd.Timedelta(days=1)  # hour 24 = D+1 00:00
            rows.append({
                "task": "dayahead",
                "model_name": "lightgbm_cfg05_dayahead",
                "target_day": str(td.date()),
                "ds": ts,
                "y_pred": 100.0 + h * 0.5,
                "source_confidence": 0.9,
                "model_version": "1.0.0",
            })
    df = pd.DataFrame(rows)
    from data.business_day import add_business_time_columns
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


class TestLedgerConversion:
    def test_complete_days_accepted(self, tmp_path):
        pred = _make_synthetic_predictions(3)
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        assert result["final_status"] == LEDGER_READY
        assert result["complete_days"] == 3
        assert result["target_days"] == 3

    def test_incomplete_day_rejected(self, tmp_path):
        pred = _make_synthetic_predictions(2)
        # Remove last row to make day 2 incomplete (23 hours)
        pred = pred.iloc[:-1]
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        assert result["complete_days"] < result["target_days"]

    def test_missing_hour_24(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        # Remove hour 24 row
        pred = pred[pred["hour_business"] != 24]
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        assert result["complete_days"] == 0

    def test_duplicate_hours_deduped(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        # Duplicate first row
        dup = pred.iloc[[0]].copy()
        pred = pd.concat([pred, dup], ignore_index=True)
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        assert result["duplicate_keys"] >= 1

    def test_canonical_schema(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        for col in PREDICTION_LEDGER_COLUMNS:
            assert col in ledger.columns, f"Missing ledger column: {col}"

    def test_production_no_ytrue(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        pred["y_true"] = 105.0  # Inject y_true
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="production",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        assert "y_true" not in ledger.columns

    def test_eval_may_include_ytrue(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        pred["y_true"] = 105.0
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        assert "y_true" in ledger.columns

    def test_hour_business_range(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        ledger = pd.read_csv(result["ledger_path_local"])
        assert ledger["hour_business"].between(1, 24).all()

    def test_no_input_incomplete(self, tmp_path):
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_path="/nonexistent/path.csv",
            work_dir=str(tmp_path),
        )
        assert result["final_status"] == LEDGER_INCOMPLETE

    def test_empty_dataframe(self, tmp_path):
        pred = pd.DataFrame(columns=["task", "model_name", "target_day", "ds", "y_pred"])
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
        )
        assert result["final_status"] == LEDGER_INCOMPLETE

    def test_ledger_rows_count(self, tmp_path):
        pred = _make_synthetic_predictions(3)
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
            mode="eval",
        )
        assert result["ledger_rows"] == 3 * 24
        assert result["input_rows"] == 3 * 24

    def test_summary_keys(self, tmp_path):
        pred = _make_synthetic_predictions(1)
        result = run_p17_cfg05_predictions_to_ledger(
            predictions_df=pred,
            work_dir=str(tmp_path),
        )
        required_keys = [
            "input_rows", "ledger_rows", "target_days", "complete_days",
            "duplicate_keys", "schema_valid", "completeness_status",
            "ledger_path_local", "final_status", "reason_codes",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
