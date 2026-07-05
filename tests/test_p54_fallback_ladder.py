"""
tests/test_p54_fallback_ladder.py — P54 tests (14+).

Tests the fallback ladder implementation:
    - NORMAL delivery via trusted_bgew_fusion (level 1)
    - DEGRADED_DELIVERED via equal_weight, best_single, cfg05, historical_median
    - FAILED_NO_DELIVERY when everything fails
    - _validate_fallback_output edge cases
    - Attempt tracking and warnings accumulation
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from delivery.fallback_ladder import (
    FALLBACK_LEVEL_NAMES,
    _build_output_from_predictions,
    _validate_fallback_output,
    _run_postflight,
    _try_trusted_bgew_fusion,
    _try_trusted_equal_weight,
    _try_best_trusted_single,
    _try_cfg05_baseline,
    _try_historical_median,
    run_fallback_ladder,
)


# ──────────────────────────────────────────────
# Helpers  (use tmp_path for file creation)
# ──────────────────────────────────────────────


def _period_from_hour(hb: int) -> str:
    if 1 <= hb <= 8:
        return "1_8"
    if 9 <= hb <= 16:
        return "9_16"
    return "17_24"


def _make_24h_predictions(
    target_date: str = "2026-07-01",
    model_name: str = "trusted_model_a",
    base_price: float = 300.0,
    task: str = "dayahead",
    n_hours: int = 24,
    add_noise: bool = False,
) -> pd.DataFrame:
    """Create a 24-row prediction DataFrame for a single model."""
    rows: list[dict] = []
    target = pd.Timestamp(target_date)
    rng = np.random.default_rng(42)

    for hb in range(1, n_hours + 1):
        if hb == 24:
            ds = target + timedelta(days=1)
            bd = target - timedelta(days=1)
        else:
            ds = target + timedelta(hours=hb)
            bd = target

        price = base_price + hb * 5.0
        if add_noise:
            price += rng.normal(0, 10)

        rows.append({
            "task": task,
            "model_name": model_name,
            "target_day": target_date,
            "business_day": bd,
            "ds": ds,
            "hour_business": hb,
            "period": _period_from_hour(hb),
            "y_pred": price,
            "y_pred_corrected": price,
            "source_confidence": 1.0,
            "model_version": "1.0.0",
        })

    return pd.DataFrame(rows)


def _make_24h_actuals(
    target_date: str = "2026-07-01",
    base_price: float = 300.0,
    n_days: int = 30,
    task: str = "dayahead",
) -> pd.DataFrame:
    """Create historical actuals for training BGEW weights."""
    rows: list[dict] = []
    target = pd.Timestamp(target_date)

    for day_offset in range(1, n_days + 1):
        day = target - timedelta(days=day_offset)
        for hb in range(1, 25):
            if hb == 24:
                ds = day + timedelta(days=1)
                bd = day - timedelta(days=1)
            else:
                ds = day + timedelta(hours=hb)
                bd = day

            price = base_price + hb * 5.0 + np.random.default_rng(day_offset * 100 + hb).normal(0, 15)
            rows.append({
                "task": task,
                "target_day": str(day.date()),
                "business_day": bd,
                "ds": ds,
                "hour_business": hb,
                "period": _period_from_hour(hb),
                "y_true": price,
                "actual_source": "test",
            })

    return pd.DataFrame(rows)


def _write_csv(tmp_path: str, df: pd.DataFrame, name: str = "data.csv") -> str:
    """Write a DataFrame to a CSV file under *tmp_path* and return the path."""
    path = os.path.join(tmp_path, name)
    df.to_csv(path, index=False)
    return path


def _make_raw_data_csv(
    target_date: str,
    n_days: int,
    tmp_path: str,
) -> str:
    """Create a CSV with historical raw dayahead prices under *tmp_path*."""
    rows: list[dict] = []
    target = pd.Timestamp(target_date)
    rng = np.random.default_rng(12345)

    for day_offset in range(1, n_days + 1):
        day = target - timedelta(days=day_offset)
        for hb in range(1, 25):
            if hb == 24:
                ds = day + timedelta(days=1)
                bd = day - timedelta(days=1)
            else:
                ds = day + timedelta(hours=hb)
                bd = day

            price = 280.0 + hb * 5.0 + rng.normal(0, 20)
            rows.append({
                "ds": ds,
                "da_anchor": price,
            })

    df = pd.DataFrame(rows)
    return _write_csv(tmp_path, df, "raw_data.csv")


# ──────────────────────────────────────────────
# _validate_fallback_output tests  (6)
# ──────────────────────────────────────────────


class TestValidateFallbackOutput:
    """Tests for ``_validate_fallback_output``."""

    def test_valid_24_row_output(self):
        """Valid 24-row output passes."""
        rows = []
        for hb in range(1, 25):
            rows.append({
                "business_day": "2026-07-01",
                "ds": f"2026-07-01 {hb:02d}:00:00",
                "hour_business": hb,
                "period": _period_from_hour(hb),
                "dayahead_price": 300.0 + hb * 5.0,
                "realtime_price": None,
            })
        df = pd.DataFrame(rows)
        valid, issues = _validate_fallback_output(df, "2026-07-01")
        assert valid is True, f"Expected valid, got issues: {issues}"
        assert len(issues) == 0

    def test_fails_for_23_rows(self):
        """23 rows should fail validation."""
        rows = []
        for hb in range(1, 24):  # only 23 rows
            rows.append({
                "business_day": "2026-07-01",
                "ds": f"2026-07-01 {hb:02d}:00:00",
                "hour_business": hb,
                "period": _period_from_hour(hb),
                "dayahead_price": 300.0 + hb * 5.0,
                "realtime_price": None,
            })
        df = pd.DataFrame(rows)
        valid, issues = _validate_fallback_output(df, "2026-07-01")
        assert valid is False
        assert any("24" in i for i in issues) or any("rows" in i for i in issues)

    def test_fails_for_nan(self):
        """NaN in dayahead_price should fail validation."""
        rows = []
        for hb in range(1, 25):
            price = 300.0 + hb * 5.0 if hb != 12 else float("nan")
            rows.append({
                "business_day": "2026-07-01",
                "ds": f"2026-07-01 {hb:02d}:00:00",
                "hour_business": hb,
                "period": _period_from_hour(hb),
                "dayahead_price": price,
                "realtime_price": None,
            })
        df = pd.DataFrame(rows)
        valid, issues = _validate_fallback_output(df, "2026-07-01")
        assert valid is False
        assert any("NaN" in i for i in issues)

    def test_fails_for_duplicates(self):
        """Duplicate hour_business should fail validation."""
        rows = []
        for hb in range(1, 25):
            actual_hb = hb if hb <= 23 else 23  # duplicate hour 23
            rows.append({
                "business_day": "2026-07-01",
                "ds": f"2026-07-01 {hb:02d}:00:00",
                "hour_business": actual_hb,
                "period": _period_from_hour(actual_hb),
                "dayahead_price": 300.0 + hb * 5.0,
                "realtime_price": None,
            })
        df = pd.DataFrame(rows)
        valid, issues = _validate_fallback_output(df, "2026-07-01")
        assert valid is False
        assert any("duplicate" in i.lower() or "Duplicate" in i for i in issues)

    def test_fails_for_wrong_schema(self):
        """Missing columns should fail validation."""
        df = pd.DataFrame({"hour_business": list(range(1, 25))})  # missing required cols
        valid, issues = _validate_fallback_output(df, "2026-07-01")
        assert valid is False
        assert any("Missing" in i for i in issues)

    def test_passes_for_none_realtime(self):
        """None in realtime_price is acceptable (dayahead-only delivery)."""
        rows = []
        for hb in range(1, 25):
            rows.append({
                "business_day": "2026-07-01",
                "ds": f"2026-07-01 {hb:02d}:00:00",
                "hour_business": hb,
                "period": _period_from_hour(hb),
                "dayahead_price": 300.0 + hb * 5.0,
                "realtime_price": None,
            })
        df = pd.DataFrame(rows)
        valid, issues = _validate_fallback_output(df, "2026-07-01")
        assert valid is True


# ──────────────────────────────────────────────
# _run_postflight tests  (2)
# ──────────────────────────────────────────────


class TestRunPostflight:
    """Tests for ``_run_postflight``."""

    def test_postflight_passes_for_valid_output(self):
        rows = []
        for hb in range(1, 25):
            rows.append({
                "business_day": "2026-07-01",
                "ds": f"2026-07-01 {hb:02d}:00:00",
                "hour_business": hb,
                "period": _period_from_hour(hb),
                "dayahead_price": 300.0 + hb * 5.0,
                "realtime_price": None,
            })
        df = pd.DataFrame(rows)
        status, warnings = _run_postflight(df)
        assert status == "PASS"
        assert len(warnings) == 0

    def test_postflight_fails_for_empty_output(self):
        status, warnings = _run_postflight(pd.DataFrame())
        assert status == "FAIL"
        assert len(warnings) > 0


# ──────────────────────────────────────────────
# _build_output_from_predictions tests  (2)
# ──────────────────────────────────────────────


class TestBuildOutputFromPredictions:
    """Tests for ``_build_output_from_predictions``."""

    def test_builds_24_rows_successfully(self):
        df = _make_24h_predictions("2026-07-01", "test_model", 300.0)
        output = _build_output_from_predictions(df, "2026-07-01", "y_pred")
        assert output is not None
        assert len(output) == 24
        assert list(output.columns) == [
            "business_day", "ds", "hour_business", "period",
            "dayahead_price", "realtime_price",
        ]

    def test_returns_none_for_missing_hours(self):
        df = _make_24h_predictions("2026-07-01", "test_model", 300.0, n_hours=20)
        output = _build_output_from_predictions(df, "2026-07-01", "y_pred")
        assert output is None


# ──────────────────────────────────────────────
# Level-specific try-function tests  (9)
# ──────────────────────────────────────────────


class TestTryTrustedBgewFusion:
    """Tests for ``_try_trusted_bgew_fusion``."""

    def test_success_with_actuals(self):
        """BGEW fusion succeeds when trusted models and actuals are available."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        preds_a = _make_24h_predictions(target, "model_a", 300.0)
        preds_b = _make_24h_predictions(target, "model_b", 310.0, add_noise=True)
        preds = pd.concat([preds_a, preds_b], ignore_index=True)

        actuals = _make_24h_actuals(target, 305.0, n_days=30)

        result = _try_trusted_bgew_fusion(target, models, preds, actuals)
        assert result is not None
        assert result["success"] is True
        assert "output" in result
        assert len(result["output"]) == 24

    def test_returns_none_with_empty_actuals(self):
        """Returns None when actuals ledger is empty."""
        target = "2026-07-05"
        models = ["model_a"]
        preds = _make_24h_predictions(target, "model_a", 300.0)
        actuals = pd.DataFrame()

        result = _try_trusted_bgew_fusion(target, models, preds, actuals)
        assert result is None

    def test_fails_with_no_model_data(self):
        """Returns failed result when no predictions exist for trusted models."""
        target = "2026-07-05"
        models = ["nonexistent_model"]
        preds = _make_24h_predictions(target, "model_a", 300.0)
        actuals = _make_24h_actuals(target, 305.0, n_days=10)

        result = _try_trusted_bgew_fusion(target, models, preds, actuals)
        assert result is not None
        assert result["success"] is False


class TestTryTrustedEqualWeight:
    """Tests for ``_try_trusted_equal_weight``."""

    def test_success_with_two_models(self):
        """Equal weight succeeds when multiple trusted models are available."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        preds_a = _make_24h_predictions(target, "model_a", 300.0)
        preds_b = _make_24h_predictions(target, "model_b", 310.0)
        preds = pd.concat([preds_a, preds_b], ignore_index=True)

        result = _try_trusted_equal_weight(target, models, preds)
        assert result is not None
        assert result["success"] is True
        assert "output" in result
        assert len(result["output"]) == 24

    def test_single_trusted_model(self):
        """Works with only 1 trusted model (same as passing it through)."""
        target = "2026-07-05"
        models = ["model_a"]
        preds = _make_24h_predictions(target, "model_a", 300.0)

        result = _try_trusted_equal_weight(target, models, preds)
        assert result is not None
        assert result["success"] is True
        assert len(result["output"]) == 24


class TestTryBestTrustedSingle:
    """Tests for ``_try_best_trusted_single``."""

    def test_selects_best_model_by_mae(self):
        """Selects the model with lowest MAE against actuals."""
        target = "2026-07-05"
        models = ["good_model", "bad_model"]

        preds_good = _make_24h_predictions(target, "good_model", 300.0)
        preds_bad = _make_24h_predictions(target, "bad_model", 500.0)  # far off
        preds = pd.concat([preds_good, preds_bad], ignore_index=True)

        actuals = _make_24h_actuals(target, 305.0, n_days=15)

        result = _try_best_trusted_single(target, models, preds, actuals)
        assert result is not None
        assert result["success"] is True
        assert "good_model" in result["reason"] or "good_model" in str(result)

    def test_defaults_to_first_model_without_actuals(self):
        """Defaults to first trusted model when actuals are unavailable."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        preds_a = _make_24h_predictions(target, "model_a", 300.0)
        preds_b = _make_24h_predictions(target, "model_b", 310.0)
        preds = pd.concat([preds_a, preds_b], ignore_index=True)

        result = _try_best_trusted_single(target, models, preds, pd.DataFrame())
        assert result is not None
        assert result["success"] is True
        assert len(result["output"]) == 24


class TestTryCfg05Baseline:
    """Tests for ``_try_cfg05_baseline``."""

    def test_finds_and_uses_cfg05_model(self):
        """Finds cfg05 model in ledger and produces valid output."""
        target = "2026-07-05"

        preds_cfg05 = _make_24h_predictions(target, "lightgbm_cfg05_dayahead", 300.0)
        preds_other = _make_24h_predictions(target, "other_model", 310.0)
        preds = pd.concat([preds_cfg05, preds_other], ignore_index=True)

        result = _try_cfg05_baseline(target, preds)
        assert result is not None
        assert result["success"] is True
        assert len(result["output"]) == 24

    def test_fails_when_no_cfg05_model(self):
        """Fails gracefully when no cfg05 model is in the ledger."""
        target = "2026-07-05"
        preds = _make_24h_predictions(target, "other_model", 300.0)

        result = _try_cfg05_baseline(target, preds)
        assert result is not None
        assert result["success"] is False
        assert "cfg05" in result.get("reason", "").lower()


class TestTryHistoricalMedian:
    """Tests for ``_try_historical_median``."""

    def test_computes_median_from_raw_data(self, tmp_path):
        """Computes per-hour median from raw CSV data."""
        target = "2026-07-05"
        csv_path = _make_raw_data_csv(target, n_days=60, tmp_path=tmp_path)

        result = _try_historical_median(target, csv_path)
        assert result is not None
        assert result["success"] is True
        assert len(result["output"]) == 24
        # Prices should be close to 280 + hb*5 across 60 days
        mean_price = result["output"]["dayahead_price"].mean()
        assert 280.0 < mean_price < 450.0

    def test_fails_without_raw_data_path(self):
        """Returns None when no raw_data_path is provided."""
        result = _try_historical_median("2026-07-05", None)
        assert result is None

    def test_fails_with_empty_raw_data(self, tmp_path):
        """Fails when raw data file is empty."""
        path = os.path.join(tmp_path, "empty.csv")
        pd.DataFrame().to_csv(path, index=False)
        result = _try_historical_median("2026-07-05", path)
        assert result is not None
        assert result["success"] is False

    def test_fails_with_no_price_column(self, tmp_path):
        """Fails when raw data has no recognizable price column."""
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        path = _write_csv(tmp_path, df, "no_price.csv")
        result = _try_historical_median("2026-07-05", path)
        assert result is not None
        assert result["success"] is False


# ──────────────────────────────────────────────
# Full run_fallback_ladder tests  (8)
# ──────────────────────────────────────────────


class TestRunFallbackLadder:
    """Tests for ``run_fallback_ladder`` — the main entry point."""

    def test_normal_when_trusted_bgew_succeeds(self, tmp_path):
        """NORMAL delivery when level 1 (BGEW fusion) succeeds."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        preds_a = _make_24h_predictions(target, "model_a", 300.0)
        preds_b = _make_24h_predictions(target, "model_b", 310.0, add_noise=True)
        preds = pd.concat([preds_a, preds_b], ignore_index=True)

        actuals = _make_24h_actuals(target, 305.0, n_days=30)

        pred_path = _write_csv(tmp_path, preds, "predictions.csv")
        act_path = _write_csv(tmp_path, actuals, "actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        assert result["success"] is True
        assert result["delivery_status"] == "NORMAL"
        assert result["fallback_level"] == 1
        assert result["fallback_method"] == "trusted_bgew_fusion"
        assert result["output"] is not None
        assert len(result["output"]) == 24

    def test_degraded_when_bgew_fails_equal_weight_succeeds(self, tmp_path):
        """DEGRADED_DELIVERED when level 1 fails and level 2 succeeds."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        preds_a = _make_24h_predictions(target, "model_a", 300.0)
        preds_b = _make_24h_predictions(target, "model_b", 310.0, add_noise=True)
        preds = pd.concat([preds_a, preds_b], ignore_index=True)

        # Empty actuals — BGEW returns None, but equal_weight still works
        empty_actuals = pd.DataFrame(columns=["task", "business_day", "hour_business", "y_true"])

        pred_path = _write_csv(tmp_path, preds, "predictions.csv")
        act_path = _write_csv(tmp_path, empty_actuals, "empty_actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        assert result["success"] is True
        assert result["delivery_status"] == "DEGRADED_DELIVERED"
        assert result["fallback_level"] in (2, 3)  # L2 equal_weight or L3 best_single

    def test_degraded_when_only_cfg05_works(self, tmp_path):
        """DEGRADED_DELIVERED when only cfg05 baseline works."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        # Only include cfg05 model in predictions (trusted models are model_a/b)
        preds = _make_24h_predictions(target, "lightgbm_cfg05_dayahead", 300.0)
        empty_actuals = pd.DataFrame(columns=["task", "business_day", "hour_business", "y_true"])

        pred_path = _write_csv(tmp_path, preds, "predictions.csv")
        act_path = _write_csv(tmp_path, empty_actuals, "empty_actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        assert result["success"] is True
        assert result["delivery_status"] == "DEGRADED_DELIVERED"
        assert result["fallback_level"] == 4
        assert result["fallback_method"] == "cfg05_baseline"

    def test_degraded_when_only_historical_median_works(self, tmp_path):
        """DEGRADED_DELIVERED when only historical median works."""
        target = "2026-07-05"

        raw_csv = _make_raw_data_csv(target, n_days=60, tmp_path=tmp_path)

        # Empty prediction ledger — no model data at all
        empty_schema = pd.DataFrame(columns=[
            "task", "model_name", "target_day", "business_day",
            "ds", "hour_business", "period", "y_pred",
        ])
        empty_actuals = pd.DataFrame(columns=["task", "business_day", "hour_business", "y_true"])

        pred_path = _write_csv(tmp_path, empty_schema, "empty_preds.csv")
        act_path = _write_csv(tmp_path, empty_actuals, "empty_actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=["model_a"],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            raw_data_path=raw_csv,
        )
        assert result["success"] is True
        assert result["delivery_status"] == "DEGRADED_DELIVERED"
        assert result["fallback_level"] == 5
        assert result["fallback_method"] == "historical_same_hour_median"

    def test_failed_no_delivery_when_everything_fails(self, tmp_path):
        """FAILED_NO_DELIVERY when all fallback levels fail."""
        target = "2026-07-05"

        header_preds = pd.DataFrame(columns=[
            "task", "model_name", "target_day", "hour_business",
        ])
        header_actuals = pd.DataFrame(columns=["task", "business_day", "hour_business", "y_true"])

        pred_path = _write_csv(tmp_path, header_preds, "empty_preds.csv")
        act_path = _write_csv(tmp_path, header_actuals, "empty_actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=["model_a"],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
            # no raw_data_path — level 5 not attempted
        )
        assert result["success"] is False
        assert result["delivery_status"] == "FAILED_NO_DELIVERY"
        assert result["fallback_level"] == 6
        assert result["fallback_method"] == "FAILED_NO_DELIVERY"
        assert result["output"] is None

    def test_fallback_level_recorded_correctly(self, tmp_path):
        """Each attempt has correct level, method, success, reason."""
        target = "2026-07-05"
        models = ["model_a", "model_b"]

        preds_a = _make_24h_predictions(target, "model_a", 300.0)
        preds_b = _make_24h_predictions(target, "model_b", 310.0, add_noise=True)
        preds = pd.concat([preds_a, preds_b], ignore_index=True)

        actuals = _make_24h_actuals(target, 305.0, n_days=30)

        pred_path = _write_csv(tmp_path, preds, "predictions.csv")
        act_path = _write_csv(tmp_path, actuals, "actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        assert len(result["attempts"]) >= 1
        first = result["attempts"][0]
        assert "level" in first
        assert "method" in first
        assert "success" in first
        assert "reason" in first
        # With good data, level 1 should succeed
        assert first["level"] == 1
        assert first["success"] is True

    def test_warnings_accumulated_from_each_attempt(self, tmp_path):
        """Warnings accumulate when multiple levels fail before one succeeds."""
        target = "2026-07-05"

        # Prediction ledger with only cfg05 model
        preds = _make_24h_predictions(target, "lightgbm_cfg05_dayahead", 300.0)
        empty_actuals = pd.DataFrame(columns=["task", "business_day", "hour_business", "y_true"])

        pred_path = _write_csv(tmp_path, preds, "predictions.csv")
        act_path = _write_csv(tmp_path, empty_actuals, "empty_actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=["model_a", "model_b"],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        # Should eventually succeed via cfg05 (level 4)
        assert result["success"] is True
        # Should have warnings from failed levels
        assert len(result["warnings"]) > 0
        # Should have multiple attempts
        assert len(result["attempts"]) >= 2

    def test_empty_input_handling(self, tmp_path):
        """Run with completely empty ledgers produces FAILED_NO_DELIVERY."""
        target = "2026-07-05"

        # Empty DataFrame (no columns at all)
        empty_preds = pd.DataFrame()
        empty_actuals = pd.DataFrame()

        pred_path = _write_csv(tmp_path, empty_preds, "empty.csv")
        act_path = _write_csv(tmp_path, empty_actuals, "empty.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=[],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        assert result["success"] is False
        assert result["delivery_status"] == "FAILED_NO_DELIVERY"

    def test_single_model_in_trusted_pool(self, tmp_path):
        """Works correctly when trusted pool has only 1 model."""
        target = "2026-07-05"
        models = ["model_a"]

        preds = _make_24h_predictions(target, "model_a", 300.0)
        empty_actuals = pd.DataFrame(columns=["task", "business_day", "hour_business", "y_true"])

        pred_path = _write_csv(tmp_path, preds, "predictions.csv")
        act_path = _write_csv(tmp_path, empty_actuals, "empty_actuals.csv")

        result = run_fallback_ladder(
            target_date=target,
            trusted_models=models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )
        assert result["success"] is True
        # With empty actuals, L1 returns None, L2 succeeds with single model
        assert result["delivery_status"] in ("DEGRADED_DELIVERED", "NORMAL")
        assert result["output"] is not None
        assert len(result["output"]) == 24


# ──────────────────────────────────────────────
# FALLBACK_LEVEL_NAMES  (1)
# ──────────────────────────────────────────────


class TestFallbackLevelNames:
    """Tests for ``FALLBACK_LEVEL_NAMES``."""

    def test_all_levels_defined(self):
        """All 6 fallback levels are named."""
        assert len(FALLBACK_LEVEL_NAMES) == 6
        assert FALLBACK_LEVEL_NAMES[1] == "trusted_bgew_fusion"
        assert FALLBACK_LEVEL_NAMES[2] == "trusted_equal_weight"
        assert FALLBACK_LEVEL_NAMES[3] == "best_trusted_single_model"
        assert FALLBACK_LEVEL_NAMES[4] == "cfg05_baseline"
        assert FALLBACK_LEVEL_NAMES[5] == "historical_same_hour_median"
        assert FALLBACK_LEVEL_NAMES[6] == "FAILED_NO_DELIVERY"
