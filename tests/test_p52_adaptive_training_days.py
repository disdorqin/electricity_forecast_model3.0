"""
tests/test_p52_adaptive_training_days.py — P52 Adaptive Training Days contract tests.

Validates:
    1. COMPLETE_30D when all data is clean
    2. DEGRADED_MIN_DAYS with 7-29 days
    3. INSUFFICIENT_DAYS with 0-6 days
    4. NO_VALID_DAYS with missing files
    5. Missing hours detected
    6. NaN y_pred detected
    7. Duplicate keys detected
    8. Models missing from prediction ledger
    9. Skipped days logged correctly
    10. Ledger not found returns NO_VALID_DAYS
    11. DEGRADED with >=7 days returns correct status
    12. COMPLETE_30D counts training_rows correctly
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from fusion.adaptive_training_days import select_complete_training_days

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAKE_MODELS = ["model_a", "model_b"]
"""Two trusted models used in most tests."""

_EXPECTED_HOURS = list(range(1, 25))
"""Business hours 1..24."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _make_prediction_rows(
    days: list[str],
    models: list[str],
    *,
    seed: int = 42,
) -> list[dict]:
    """Generate complete 24-hour prediction rows for every (day, model)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for day in days:
        day_dt = pd.Timestamp(day)
        for model in models:
            for h in range(1, 25):
                ts = day_dt + pd.Timedelta(hours=h)
                rows.append({
                    "task": "dayahead",
                    "model_name": model,
                    "target_day": day_dt,
                    "business_day": day_dt,
                    "ds": ts,
                    "hour_business": h,
                    "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
                    "y_pred": float(rng.uniform(80, 200)),
                    "source_confidence": 0.9,
                    "model_version": "1.0",
                    "run_id": "test_run",
                })
    return rows


def _make_actual_rows(
    days: list[str],
    *,
    seed: int = 42,
) -> list[dict]:
    """Generate complete 24-hour actual rows for every day."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for day in days:
        day_dt = pd.Timestamp(day)
        for h in range(1, 25):
            ts = day_dt + pd.Timedelta(hours=h)
            rows.append({
                "task": "dayahead",
                "target_day": day_dt,
                "business_day": day_dt,
                "ds": ts,
                "hour_business": h,
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
                "y_true": float(rng.uniform(80, 200)),
                "actual_source": "market_feed",
                "run_id": "test_run",
            })
    return rows


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to parquet, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _make_complete_days(n_days: int, start: str = "2026-06-01") -> list[str]:
    """Return *n_days* consecutive date strings starting from *start*."""
    start_dt = pd.Timestamp(start)
    return [(start_dt + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def _target_date_after(n_days: int, start: str = "2026-06-01") -> str:
    """Return the day after the last generated day."""
    start_dt = pd.Timestamp(start)
    last_day = start_dt + pd.Timedelta(days=n_days - 1)
    return (last_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Yield a temporary directory path for ledger parquet files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def _setup_clean(tmp_dir: str, n_days: int = 35,
                  models: list[str] | None = None,
                  start_date: str = "2026-06-01",
                  ) -> tuple[str, str]:
    """Write clean prediction + actual ledgers. Returns (pred_path, act_path)."""
    if models is None:
        models = list(_FAKE_MODELS)
    days = _make_complete_days(n_days, start=start_date)
    pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")
    act_path = os.path.join(tmp_dir, "actual_ledger.parquet")

    pred_df = pd.DataFrame(_make_prediction_rows(days, models=models))
    act_df = pd.DataFrame(_make_actual_rows(days))

    _write_parquet(pred_df, pred_path)
    _write_parquet(act_df, act_path)
    return pred_path, act_path


# ---------------------------------------------------------------------------
# Tests: COMPLETE_30D
# ---------------------------------------------------------------------------


class TestComplete30D:
    """COMPLETE_30D status when >= 30 clean days available."""

    def test_complete_30d_returns_correct_status(self, tmp_dir):
        """COMPLETE_30D with 35 clean days."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35)
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "COMPLETE_30D"
        assert result["selected_count"] == 30
        assert len(result["selected_days"]) == 30
        assert result["latest_selected_day"] == result["selected_days"][0]
        assert result["oldest_selected_day"] == result["selected_days"][-1]
        assert result["latest_selected_day"] is not None
        assert result["oldest_selected_day"] is not None

    def test_complete_30d_counts_training_rows_correctly(self, tmp_dir):
        """COMPLETE_30D training_rows = selected_count * n_models * 24."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35)
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        n_models = len(_FAKE_MODELS)
        assert result["selected_count"] == 30
        assert result["training_rows"] == 30 * n_models * 24
        assert result["actual_rows"] == 30 * 24

    def test_complete_30d_no_errors(self, tmp_dir):
        """Clean 30-day selection produces no errors."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35)
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        # All 30 newest days are clean — no errors, no skipped days
        assert len(result["errors"]) == 0
        assert result["selected_count"] == 30


# ---------------------------------------------------------------------------
# Tests: DEGRADED_MIN_DAYS
# ---------------------------------------------------------------------------


class TestDegradedMinDays:
    """DEGRADED_MIN_DAYS status when 7-29 complete days found."""

    def test_degraded_with_exactly_7_days(self, tmp_dir):
        """7 clean days returns DEGRADED_MIN_DAYS."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=7)
        target = _target_date_after(7)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "DEGRADED_MIN_DAYS"
        assert result["selected_count"] == 7

    def test_degraded_with_15_days(self, tmp_dir):
        """15 clean days returns DEGRADED_MIN_DAYS."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=15)
        target = _target_date_after(15)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "DEGRADED_MIN_DAYS"
        assert result["selected_count"] == 15

    def test_degraded_returns_correct_training_rows(self, tmp_dir):
        """DEGRADED_MIN_DAYS computes training_rows correctly."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=7)
        target = _target_date_after(7)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["selected_count"] == 7
        assert result["training_rows"] == 7 * len(_FAKE_MODELS) * 24
        assert result["actual_rows"] == 7 * 24


# ---------------------------------------------------------------------------
# Tests: INSUFFICIENT_DAYS
# ---------------------------------------------------------------------------


class TestInsufficientDays:
    """INSUFFICIENT_DAYS status when 1-6 complete days found."""

    def test_insufficient_with_3_days(self, tmp_dir):
        """3 clean days returns INSUFFICIENT_DAYS."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=3)
        target = _target_date_after(3)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "INSUFFICIENT_DAYS"
        assert result["selected_count"] == 3

    def test_insufficient_with_1_day(self, tmp_dir):
        """1 clean day returns INSUFFICIENT_DAYS."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=1)
        target = _target_date_after(1)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "INSUFFICIENT_DAYS"
        assert result["selected_count"] == 1


# ---------------------------------------------------------------------------
# Tests: NO_VALID_DAYS
# ---------------------------------------------------------------------------


class TestNoValidDays:
    """NO_VALID_DAYS when files are missing or empty."""

    def test_missing_prediction_ledger(self, tmp_dir):
        """Missing prediction ledger returns NO_VALID_DAYS."""
        act_path = os.path.join(tmp_dir, "actual_ledger.parquet")
        _write_parquet(pd.DataFrame(_make_actual_rows(["2026-07-01"])), act_path)
        pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "NO_VALID_DAYS"
        assert any("prediction ledger not found" in e for e in result["errors"])

    def test_missing_actual_ledger(self, tmp_dir):
        """Missing actual ledger returns NO_VALID_DAYS."""
        pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")
        _write_parquet(pd.DataFrame(_make_prediction_rows(["2026-07-01"], _FAKE_MODELS)), pred_path)
        act_path = os.path.join(tmp_dir, "actual_ledger.parquet")

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "NO_VALID_DAYS"
        assert any("actual ledger not found" in e for e in result["errors"])

    def test_missing_both_ledgers(self, tmp_dir):
        """Both missing returns NO_VALID_DAYS."""
        pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")
        act_path = os.path.join(tmp_dir, "actual_ledger.parquet")

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "NO_VALID_DAYS"
        assert len(result["errors"]) >= 1

    def test_empty_prediction_ledger(self, tmp_dir):
        """Empty prediction ledger returns NO_VALID_DAYS."""
        pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")
        act_path = os.path.join(tmp_dir, "actual_ledger.parquet")
        _write_parquet(pd.DataFrame(columns=["task", "model_name"]), pred_path)
        _write_parquet(pd.DataFrame(_make_actual_rows(["2026-07-01"])), act_path)

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "NO_VALID_DAYS"
        assert any("empty" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# Tests: Data quality detection (day-specific issues)
# ---------------------------------------------------------------------------


def _setup_with_issue(
    clean_days: list[str],
    bad_day: str,
    *,
    modify_pred: callable = None,
    modify_act: callable = None,
    models: list[str] | None = None,
    tmp_dir: str = "",
) -> tuple[str, str]:
    """Write ledgers where *bad_day* has data issues.

    *modify_pred* is called with (rows, day, models) and returns modified rows.
    *modify_act* is called with (rows, day) and returns modified rows.
    """
    if models is None:
        models = list(_FAKE_MODELS)

    pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")
    act_path = os.path.join(tmp_dir, "actual_ledger.parquet")

    # Separate clean and bad days
    clean_list = [d for d in clean_days if d != bad_day]

    # Build clean rows
    pred_rows = _make_prediction_rows(clean_list, models=models, seed=42)
    act_rows = _make_actual_rows(clean_list, seed=42)

    # Build bad-day rows (may be modified)
    bad_pred = _make_prediction_rows([bad_day], models=models, seed=99)
    bad_act = _make_actual_rows([bad_day], seed=99)

    if modify_pred:
        bad_pred = modify_pred(bad_pred, bad_day, models)
    if modify_act:
        bad_act = modify_act(bad_act, bad_day)

    pred_df = pd.DataFrame(pred_rows + bad_pred)
    act_df = pd.DataFrame(act_rows + bad_act)

    _write_parquet(pred_df, pred_path)
    _write_parquet(act_df, act_path)
    return pred_path, act_path


def _drop_pred_hours(rows: list[dict], day: str, models: list[str]) -> list[dict]:
    """Remove hours 5-8 for model_a from the given rows."""
    return [r for r in rows
            if not (r["model_name"] == "model_a" and r["hour_business"] in (5, 6, 7, 8))]


def _set_nan_y_pred(rows: list[dict], day: str, models: list[str]) -> list[dict]:
    """Set y_pred to NaN at hour 12 for model_a."""
    for r in rows:
        if r["model_name"] == "model_a" and r["hour_business"] == 12:
            r["y_pred"] = float("nan")
    return rows


def _duplicate_pred_hour(rows: list[dict], day: str, models: list[str]) -> list[dict]:
    """Duplicate hour 8 for model_a."""
    dups = [dict(r) for r in rows
            if r["model_name"] == "model_a" and r["hour_business"] == 8]
    return rows + dups


def _drop_act_hours(rows: list[dict], day: str) -> list[dict]:
    """Remove hours 20-24 from actual rows."""
    return [r for r in rows if r["hour_business"] not in (20, 21, 22, 23, 24)]


def _set_nan_y_true(rows: list[dict], day: str) -> list[dict]:
    """Set y_true to NaN at hour 12."""
    for r in rows:
        if r["hour_business"] == 12:
            r["y_true"] = float("nan")
    return rows


class TestDataQualityDetection:
    """Detection of missing hours, NaN, duplicates, and missing models."""

    def test_missing_hours_detected(self, tmp_dir):
        """Day with missing hours in prediction is skipped."""
        days = _make_complete_days(31)
        bad_day = days[-1]  # newest day (scanned first — will be skipped)
        pred_path, act_path = _setup_with_issue(
            days, bad_day,
            modify_pred=_drop_pred_hours,
            tmp_dir=tmp_dir,
        )
        target = _target_date_after(31)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        # One day was skipped, but 30 remaining days are clean
        assert result["selected_count"] == 30
        skipped_days = dict(result["skipped_days"])
        assert any("model_a" in str(v) and "hours" in str(v).lower()
                   for v in skipped_days.values())

    def test_nan_y_pred_detected(self, tmp_dir):
        """Day with NaN y_pred is skipped."""
        days = _make_complete_days(31)
        bad_day = days[-1]
        pred_path, act_path = _setup_with_issue(
            days, bad_day,
            modify_pred=_set_nan_y_pred,
            tmp_dir=tmp_dir,
        )
        target = _target_date_after(31)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["selected_count"] == 30
        skipped_days = dict(result["skipped_days"])
        assert any("nan" in str(v).lower() for v in skipped_days.values())

    def test_duplicate_keys_detected_and_handled(self, tmp_dir):
        """Day with duplicate prediction keys is handled gracefully.

        The dedup step (drop_duplicates by hour_business) resolves the
        duplicate before the key-check, so the day is still selected."""
        days = _make_complete_days(31)
        bad_day = days[-1]
        pred_path, act_path = _setup_with_issue(
            days, bad_day,
            modify_pred=_duplicate_pred_hour,
            tmp_dir=tmp_dir,
        )
        target = _target_date_after(31)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        # Dedup by hour_business resolves the duplicate — 30 selected
        assert result["selected_count"] == 30
        assert result["status"] == "COMPLETE_30D"

    def test_models_missing_from_prediction_ledger(self, tmp_dir):
        """Day missing a trusted model is skipped."""
        days = _make_complete_days(31)
        bad_day = days[-1]

        def _missing_model_b(rows, day, models):
            return [r for r in rows if r["model_name"] != "model_b"]

        pred_path, act_path = _setup_with_issue(
            days, bad_day,
            modify_pred=_missing_model_b,
            tmp_dir=tmp_dir,
        )
        target = _target_date_after(31)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["selected_count"] == 30
        skipped_days = dict(result["skipped_days"])
        assert any("model_b" in str(v).lower() for v in skipped_days.values())

    def test_skipped_days_logged_correctly(self, tmp_dir):
        """Skipped days with reasons appear in result."""
        # Use 3 bad days + 5 clean days = 8 total; only 5 clean
        clean_days = _make_complete_days(5)
        # Build data in two batches where all days have issues
        bad_days = ["2026-07-01", "2026-07-02", "2026-07-03"]
        all_days = bad_days + clean_days
        target = "2026-07-09"  # after all days

        # Build clean prediction for all days
        pred_rows = _make_prediction_rows(all_days, _FAKE_MODELS, seed=42)
        # Build actuals where first 3 days have issues
        act_rows = _make_actual_rows(clean_days, seed=42)

        # Add actuals for bad days with NaN
        rng = np.random.default_rng(99)
        for day in bad_days:
            day_dt = pd.Timestamp(day)
            for h in range(1, 25):
                act_rows.append({
                    "task": "dayahead",
                    "target_day": day_dt,
                    "business_day": day_dt,
                    "ds": day_dt + pd.Timedelta(hours=h),
                    "hour_business": h,
                    "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
                    "y_true": float("nan"),  # NaN makes them incomplete
                    "actual_source": "market_feed",
                    "run_id": "test_run",
                })

        pred_path = os.path.join(tmp_dir, "prediction_ledger.parquet")
        act_path = os.path.join(tmp_dir, "actual_ledger.parquet")
        _write_parquet(pd.DataFrame(pred_rows), pred_path)
        _write_parquet(pd.DataFrame(act_rows), act_path)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert len(result["skipped_days"]) >= 1
        for day, reason in result["skipped_days"]:
            assert isinstance(day, str)
            assert isinstance(reason, str)
            assert len(day) == 10  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Tests: Actual ledger data quality (day-specific)
# ---------------------------------------------------------------------------


class TestActualLedgerQuality:
    """Actual ledger quality checks."""

    def test_nan_y_true_detected(self, tmp_dir):
        """Day with NaN y_true in actual ledger is skipped."""
        days = _make_complete_days(31)
        bad_day = days[-1]
        pred_path, act_path = _setup_with_issue(
            days, bad_day,
            modify_act=_set_nan_y_true,
            tmp_dir=tmp_dir,
        )
        target = _target_date_after(31)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["selected_count"] == 30
        skipped_days = dict(result["skipped_days"])
        assert any("nan" in str(v).lower() for v in skipped_days.values())

    def test_missing_actual_hours_detected(self, tmp_dir):
        """Day with missing hours in actual ledger is skipped."""
        days = _make_complete_days(31)
        bad_day = days[-1]
        pred_path, act_path = _setup_with_issue(
            days, bad_day,
            modify_act=_drop_act_hours,
            tmp_dir=tmp_dir,
        )
        target = _target_date_after(31)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["selected_count"] == 30
        skipped_days = dict(result["skipped_days"])
        assert any("actual" in str(v).lower() for v in skipped_days.values())


# ---------------------------------------------------------------------------
# Tests: Custom parameters
# ---------------------------------------------------------------------------


class TestCustomParameters:
    """Custom required_days, max_lookback_days, min_days_for_degraded."""

    def test_custom_required_days_10(self, tmp_dir):
        """Custom required_days=10 with 12 available -> COMPLETE_30D."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=12)
        target = _target_date_after(12)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            required_days=10,
        )

        assert result["status"] == "COMPLETE_30D"
        assert result["selected_count"] == 10
        assert result["training_rows"] == 10 * len(_FAKE_MODELS) * 24

    def test_required_50_with_35_clean(self, tmp_dir):
        """required_days=50 with 35 available collects all 35 -> DEGRADED."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35)
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            required_days=50,
        )

        assert result["status"] == "DEGRADED_MIN_DAYS"
        # All 35 clean days are available — should collect all of them
        assert result["selected_count"] == 35

    def test_max_lookback_limits_selection(self, tmp_dir):
        """max_lookback_days=5 returns at most 5 days."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35)
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            max_lookback_days=5,
        )

        assert result["selected_count"] <= 5
        assert result["status"] in ("INSUFFICIENT_DAYS", "DEGRADED_MIN_DAYS")

    def test_min_days_for_degraded_3_with_5_days(self, tmp_dir):
        """min_days_for_degraded=3 with 5 clean days -> DEGRADED."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=5)
        target = _target_date_after(5)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            required_days=30,
            min_days_for_degraded=3,
        )

        assert result["status"] == "DEGRADED_MIN_DAYS"
        assert result["selected_count"] == 5

    def test_min_days_for_degraded_20_with_5_days(self, tmp_dir):
        """min_days_for_degraded=20 with 5 clean days -> INSUFFICIENT."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=5)
        target = _target_date_after(5)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=list(_FAKE_MODELS),
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            required_days=30,
            min_days_for_degraded=20,
        )

        assert result["status"] == "INSUFFICIENT_DAYS"
        assert result["selected_count"] == 5


# ---------------------------------------------------------------------------
# Tests: Trusted models edge cases
# ---------------------------------------------------------------------------


class TestTrustedModelsEdgeCases:
    """Edge cases with trusted_models input."""

    def test_empty_trusted_models_returns_no_valid(self, tmp_dir):
        """Empty trusted_models returns NO_VALID_DAYS."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=5)
        target = _target_date_after(5)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=[],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "NO_VALID_DAYS"
        assert any("empty" in e.lower() for e in result["errors"])

    def test_single_trusted_model(self, tmp_dir):
        """Single trusted model works correctly."""
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35, models=["model_a"])
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=["model_a"],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "COMPLETE_30D"
        assert result["selected_count"] == 30
        assert result["training_rows"] == 30 * 1 * 24
        assert result["actual_rows"] == 30 * 24

    def test_three_trusted_models(self, tmp_dir):
        """Three trusted models work correctly."""
        models = ["model_a", "model_b", "model_c"]
        pred_path, act_path = _setup_clean(tmp_dir, n_days=35, models=models)
        target = _target_date_after(35)

        result = select_complete_training_days(
            target_date=target,
            trusted_models=models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == "COMPLETE_30D"
        assert result["selected_count"] == 30
        assert result["training_rows"] == 30 * 3 * 24
