"""
tests/test_prediction_ledger.py — Prediction ledger contract tests.

Validates:
    1. append_predictions_to_ledger from P2 output produces correct schema
    2. append_predictions_to_ledger deduplicates on key
    3. append_corrected_predictions_to_ledger from P3 output produces correct schema
    4. append_corrected_predictions_to_ledger deduplicates on key
    5. validate_prediction_ledger passes valid ledger
    6. validate_prediction_ledger detects missing columns
    7. validate_corrected_ledger passes valid ledger
    8. Empty input handling
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    PREDICTION_LEDGER_KEY,
    CORRECTED_LEDGER_COLUMNS,
    CORRECTED_LEDGER_KEY,
)
from ledgers.prediction_ledger import (
    append_predictions_to_ledger,
    append_corrected_predictions_to_ledger,
    validate_prediction_ledger,
    validate_corrected_ledger,
)


def _p2_prediction_df(n_hours: int = 24) -> pd.DataFrame:
    """Synthetic P2 prediction output."""
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    df = pd.DataFrame({
        "task": ["dayahead"] * n_hours,
        "model_name": ["cfg05"] * n_hours,
        "target_day": ["2026-07-04"] * n_hours,
        "ds": timestamps,
        "y_pred": rng.uniform(80, 200, n_hours),
        "source_confidence": [0.9] * n_hours,
        "model_version": ["1.0.0"] * n_hours,
    })
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def _p3_corrected_df(n_hours: int = 24) -> pd.DataFrame:
    """Synthetic P3 corrected prediction output."""
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    df = pd.DataFrame({
        "task": ["dayahead"] * n_hours,
        "model_name": ["cfg05"] * n_hours,
        "target_day": ["2026-07-04"] * n_hours,
        "ds": timestamps,
        "y_pred_raw": rng.uniform(80, 200, n_hours),
        "y_pred_corrected": rng.uniform(80, 200, n_hours),
        "residual_delta": rng.uniform(-5, 5, n_hours),
        "correction_applied": [True] * n_hours,
        "correction_module": ["p5m_residual_noop"] * n_hours,
        "risk_source": ["NONE"] * n_hours,
        "reason_codes": ["NO_CORRECTION_NEEDED"] * n_hours,
        "correction_version": ["1.0.0"] * n_hours,
        "source_confidence": [0.9] * n_hours,
        "model_version": ["1.0.0"] * n_hours,
    })
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


class TestPredictionLedger:
    """Contract: prediction ledger."""

    def test_appends_p2_output(self):
        """append_predictions_to_ledger produces correct schema."""
        preds = _p2_prediction_df(24)
        result = append_predictions_to_ledger(preds, run_id="test_run")
        assert list(result.columns) == PREDICTION_LEDGER_COLUMNS
        assert len(result) == 24
        assert result["run_id"].iloc[0] == "test_run"

    def test_dedup_on_key(self):
        """append_predictions_to_ledger deduplicates on key."""
        preds = _p2_prediction_df(24)
        ledger = append_predictions_to_ledger(preds, run_id="run1")

        # Ensure distinct updated_at for reliable dedup
        time.sleep(0.02)
        preds2 = preds.copy()
        preds2["y_pred"] = preds["y_pred"] * 0.9
        result = append_predictions_to_ledger(preds2, ledger_df=ledger, run_id="run2")

        # Should still be 24 rows (deduped), with run2 values
        assert len(result) == 24
        # All rows should have run_id = last appended
        assert (result["run_id"] == "run2").all()

    def test_empty_input(self):
        """Empty prediction input returns empty ledger."""
        result = append_predictions_to_ledger(pd.DataFrame(), run_id="test")
        assert list(result.columns) == PREDICTION_LEDGER_COLUMNS
        assert len(result) == 0

    def test_validate_passes(self):
        """validate_prediction_ledger passes valid ledger."""
        preds = _p2_prediction_df(24)
        ledger = append_predictions_to_ledger(preds, run_id="test")
        valid, issues = validate_prediction_ledger(ledger)
        assert valid

    def test_validate_missing_columns(self):
        """validate_prediction_ledger detects missing columns."""
        bad = pd.DataFrame({"wrong_col": [1]})
        valid, issues = validate_prediction_ledger(bad)
        assert not valid


class TestCorrectedLedger:
    """Contract: corrected prediction ledger."""

    def test_appends_p3_output(self):
        """append_corrected_predictions_to_ledger produces correct schema."""
        corrected = _p3_corrected_df(24)
        result = append_corrected_predictions_to_ledger(corrected, run_id="test_run")
        assert list(result.columns) == CORRECTED_LEDGER_COLUMNS
        assert len(result) == 24

    def test_dedup_on_key(self):
        """append_corrected_predictions_to_ledger deduplicates on key."""
        corrected = _p3_corrected_df(24)
        ledger = append_corrected_predictions_to_ledger(corrected, run_id="run1")

        time.sleep(0.02)
        corrected2 = corrected.copy()
        corrected2["y_pred_corrected"] = corrected["y_pred_corrected"] * 0.9
        result = append_corrected_predictions_to_ledger(
            corrected2, ledger_df=ledger, run_id="run2",
        )

        assert len(result) == 24
        assert (result["run_id"] == "run2").all()

    def test_empty_input(self):
        """Empty corrected input returns empty ledger."""
        result = append_corrected_predictions_to_ledger(pd.DataFrame(), run_id="test")
        assert list(result.columns) == CORRECTED_LEDGER_COLUMNS
        assert len(result) == 0

    def test_validate_passes(self):
        """validate_corrected_ledger passes valid ledger."""
        corrected = _p3_corrected_df(24)
        ledger = append_corrected_predictions_to_ledger(corrected, run_id="test")
        valid, issues = validate_corrected_ledger(ledger)
        assert valid
