"""
tests/test_actual_ledger.py — Actual ledger contract tests.

Validates:
    1. append_actuals_to_ledger produces correct schema
    2. append_actuals_to_ledger deduplicates on key
    3. validate_actual_ledger passes valid ledger
    4. filter_actuals_for_training only returns business_day < target_day
    5. filter_actuals_for_training does NOT leak target_day actuals
    6. filter_actuals_for_training returns empty when no history
    7. Empty input handling
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from data.schema import ACTUAL_LEDGER_COLUMNS, ACTUAL_LEDGER_KEY
from ledgers.actual_ledger import (
    append_actuals_to_ledger,
    validate_actual_ledger,
    filter_actuals_for_training,
)


def _actuals_df(days: list[str] = None) -> pd.DataFrame:
    """Synthetic actuals DataFrame."""
    from data.business_day import add_business_time_columns

    if days is None:
        days = ["2026-07-01", "2026-07-02", "2026-07-03"]

    rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(42)
    for day in days:
        n_hours = 24
        ts = pd.date_range(f"{day} 01:00", periods=n_hours, freq="h")
        df = pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "target_day": [day] * n_hours,
            "ds": ts,
            "y_true": rng.uniform(80, 200, n_hours),
            "actual_source": ["market_feed"] * n_hours,
        })
        df = add_business_time_columns(df, timestamp_col="ds")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


class TestActualLedger:
    """Contract: actual ledger."""

    def test_appends_actuals(self):
        """append_actuals_to_ledger produces correct schema."""
        actuals = _actuals_df()
        result = append_actuals_to_ledger(actuals, run_id="test_run")
        assert list(result.columns) == ACTUAL_LEDGER_COLUMNS
        assert len(result) == 72  # 3 days * 24 hours

    def test_dedup_on_key(self):
        """append_actuals_to_ledger deduplicates on key."""
        actuals = _actuals_df()
        ledger = append_actuals_to_ledger(actuals, run_id="run1")

        time.sleep(0.02)
        actuals2 = actuals.copy()
        actuals2["y_true"] = actuals["y_true"] * 0.9
        result = append_actuals_to_ledger(actuals2, ledger_df=ledger, run_id="run2")

        assert len(result) == 72
        assert (result["run_id"] == "run2").all()

    def test_validate_passes(self):
        """validate_actual_ledger passes valid ledger."""
        actuals = _actuals_df()
        ledger = append_actuals_to_ledger(actuals, run_id="test")
        valid, issues = validate_actual_ledger(ledger)
        assert valid

    def test_empty_input(self):
        """Empty actuals input returns empty ledger."""
        result = append_actuals_to_ledger(pd.DataFrame(), run_id="test")
        assert list(result.columns) == ACTUAL_LEDGER_COLUMNS
        assert len(result) == 0


class TestFilterActualsForTraining:
    """Contract: filter_actuals_for_training (no-leakage)."""

    def test_returns_only_past_actuals(self):
        """Only business_day < target_day rows returned."""
        days = ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"]
        actuals = _actuals_df(days)
        ledger = append_actuals_to_ledger(actuals, run_id="test")

        trained = filter_actuals_for_training(ledger, target_day="2026-07-04")
        # Should only have July 1-3 (3 * 24 = 72 rows)
        assert len(trained) == 72

        # Verify no leakage
        assert (trained["business_day"] < pd.Timestamp("2026-07-04")).all()

    def test_no_leakage_of_target_day(self):
        """No actuals from target_day leak in."""
        days = ["2026-07-04"]
        actuals = _actuals_df(days)
        ledger = append_actuals_to_ledger(actuals, run_id="test")

        trained = filter_actuals_for_training(ledger, target_day="2026-07-04")
        assert len(trained) == 0  # no past data

    def test_empty_ledger(self):
        """Empty ledger returns empty."""
        ledger = pd.DataFrame(columns=ACTUAL_LEDGER_COLUMNS)
        trained = filter_actuals_for_training(ledger, target_day="2026-07-04")
        assert len(trained) == 0

    def test_window_limits_rows(self):
        """Window parameter limits returned rows."""
        days = ["2026-06-01", "2026-07-01", "2026-07-02", "2026-07-03"]
        actuals = _actuals_df(days)
        ledger = append_actuals_to_ledger(actuals, run_id="test")

        # Window of 5 days should exclude June 1
        trained = filter_actuals_for_training(ledger, target_day="2026-07-04", window=5)
        assert len(trained) == 72  # July 1-3 only
        assert (trained["business_day"] >= pd.Timestamp("2026-07-01")).all()

    def test_mixed_dates_no_future(self):
        """Even if ledger has future dates, filter prevents leakage."""
        days = ["2026-07-01", "2026-07-04", "2026-07-05"]
        actuals = _actuals_df(days)
        ledger = append_actuals_to_ledger(actuals, run_id="test")

        trained = filter_actuals_for_training(ledger, target_day="2026-07-04")
        # Only July 1 (24 rows)
        assert len(trained) == 24
