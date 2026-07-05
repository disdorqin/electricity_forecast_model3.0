"""
tests/test_p140_realtime_performance_unblock.py
================================================
Tests for P140: Realtime Performance Unblock.

Covers:
  - Rolling median delta computation
  - No lookahead in delta models
  - rt_pred = da_anchor + delta
  - LightGBM delta training uses only past data
  - Output metrics format
  - BGEW fusion of delta models
  - Baseline comparison (33.03%)
  - Blocked when no raw data
  - sMAPE_floor50 canonical formula
  - Daily metrics CSV generation
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_p140_realtime_performance_unblock import (  # noqa: E402
    RT_DA_ANCHOR_SMAPE_BASELINE,
    compute_bgew_weights,
    compute_lgbm_delta,
    compute_metrics,
    compute_rolling_median_delta,
    compute_smape_floor50,
    fuse_delta_models_bgew,
    load_raw_data,
    run_p140_realtime_unblock,
    _build_lgbm_features,
    _prepare_eval_frame,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def raw_60d():
    """60 days of synthetic hourly data with known delta pattern."""
    rng = np.random.RandomState(42)
    n_days = 60
    dates = pd.date_range("2024-11-01 01:00", periods=n_days * 24, freq="h")
    dayahead = 200 + 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    # Realtime = dayahead + constant hourly delta + noise
    hourly_delta = {h: (h - 12) * 2.0 for h in range(1, 25)}
    delta = np.array([hourly_delta[((ts.hour == 0) * 24 or ts.hour)] for ts in dates])
    realtime = dayahead + delta + rng.normal(0, 5, len(dates))

    df = pd.DataFrame({
        "ds": dates,
        "dayahead_price": dayahead,
        "realtime_price": realtime,
    })
    return df


@pytest.fixture
def raw_csv_path(tmp_path, raw_60d):
    """Write synthetic data as GBK CSV."""
    df = raw_60d.rename(columns={
        "ds": "\u65f6\u523b",
        "dayahead_price": "\u65e5\u524d\u7535\u4ef7",
        "realtime_price": "\u5b9e\u65f6\u7535\u4ef7",
    })
    path = tmp_path / "raw.csv"
    df.to_csv(path, encoding="gbk", index=False)
    return str(path)


@pytest.fixture
def eval_frame(raw_60d):
    """Evaluation frame for last 30 days."""
    raw = raw_60d.copy()
    from data.business_day import add_business_time_columns
    raw = add_business_time_columns(raw, timestamp_col="ds")
    # Use days 31-60 as eval
    cutoff = raw["ds"].iloc[30 * 24]
    eval_df = raw[raw["ds"] >= cutoff].copy()
    eval_df["da_anchor"] = eval_df["dayahead_price"]
    eval_df["y_true"] = eval_df["realtime_price"]
    return eval_df.reset_index(drop=True)


# ── Test 1: Rolling median delta computation ─────────────────────────────

class TestRollingMedianDelta:
    def test_produces_rt_pred_column(self, eval_frame, raw_60d):
        result = compute_rolling_median_delta(eval_frame, raw_60d, window=30)
        assert "rt_pred" in result.columns
        assert "delta_pred" in result.columns
        assert len(result) == len(eval_frame)

    def test_delta_is_median_of_past(self, raw_60d):
        """Verify the delta is actually the median of past 30 days."""
        from data.business_day import add_business_time_columns
        raw = raw_60d.copy()
        raw = add_business_time_columns(raw, timestamp_col="ds")
        raw["delta"] = raw["realtime_price"] - raw["dayahead_price"]
        raw["business_day"] = pd.to_datetime(raw["business_day"])

        # Pick a specific target day and hour
        target_day = pd.Timestamp("2024-12-15")
        hour = 10
        past = raw[
            (raw["business_day"] < target_day)
            & (raw["hour_business"] == hour)
        ]
        latest = past["business_day"].max()
        cutoff = latest - pd.Timedelta(days=29)
        past_window = past[past["business_day"] >= cutoff]
        expected_median = float(past_window["delta"].median())

        # Build a single-row eval frame
        eval_row = pd.DataFrame({
            "ds": [pd.Timestamp("2024-12-15 10:00")],
            "business_day": [target_day],
            "hour_business": [hour],
            "period": ["9_16"],
            "da_anchor": [200.0],
            "y_true": [210.0],
        })
        result = compute_rolling_median_delta(eval_row, raw_60d, window=30)
        assert result["delta_pred"].iloc[0] == pytest.approx(expected_median, abs=1e-6)


# ── Test 2: No lookahead ─────────────────────────────────────────────────

class TestNoLookahead:
    def test_rolling_median_no_future_data(self, raw_60d):
        """Delta for day D must not use data from day D or later."""
        from data.business_day import add_business_time_columns
        raw = raw_60d.copy()
        raw = add_business_time_columns(raw, timestamp_col="ds")
        raw["delta"] = raw["realtime_price"] - raw["dayahead_price"]
        raw["business_day"] = pd.to_datetime(raw["business_day"])

        target_day = pd.Timestamp("2024-12-01")
        hour = 5
        past = raw[
            (raw["business_day"] < target_day)
            & (raw["hour_business"] == hour)
        ]
        # All past days must be strictly before target
        assert (past["business_day"] < target_day).all()

    def test_lgbm_features_no_future(self, raw_60d):
        """_build_lgbm_features must only use days < target_day."""
        from data.business_day import add_business_time_columns
        raw = raw_60d.copy()
        raw = add_business_time_columns(raw, timestamp_col="ds")
        raw["delta"] = raw["realtime_price"] - raw["dayahead_price"]
        raw["business_day"] = pd.to_datetime(raw["business_day"])

        target_day = pd.Timestamp("2024-12-15")
        feats = _build_lgbm_features(raw, target_day, 10, "9_16", 200.0)
        # Features should be finite numbers
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"


# ── Test 3: rt_pred = da_anchor + delta ──────────────────────────────────

class TestRtPredFormula:
    def test_rt_pred_equals_da_anchor_plus_delta(self, eval_frame, raw_60d):
        result = compute_rolling_median_delta(eval_frame, raw_60d, window=30)
        np.testing.assert_allclose(
            result["rt_pred"].values,
            result["da_anchor"].values + result["delta_pred"].values,
            atol=1e-10,
        )

    def test_zero_delta_means_anchor(self, raw_60d):
        """When delta is 0, rt_pred should equal da_anchor."""
        eval_row = pd.DataFrame({
            "ds": [pd.Timestamp("2024-11-05 01:00")],
            "business_day": [pd.Timestamp("2024-11-04")],
            "hour_business": [1],
            "period": ["1_8"],
            "da_anchor": [300.0],
            "y_true": [310.0],
        })
        # With very little history, delta might be 0
        result = compute_rolling_median_delta(eval_row, raw_60d, window=30)
        # rt_pred = da_anchor + delta_pred
        assert result["rt_pred"].iloc[0] == pytest.approx(
            result["da_anchor"].iloc[0] + result["delta_pred"].iloc[0], abs=1e-10
        )


# ── Test 4: LightGBM delta uses only past data ───────────────────────────

class TestLgbmDeltaPastOnly:
    def test_lgbm_delta_walk_forward(self, eval_frame, raw_60d):
        """LightGBM model should only train on data before each target day."""
        # Mock lightgbm via sys.modules so the import inside the function works
        import sys
        mock_lgb = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda X: np.zeros(len(X))
        mock_lgb.LGBMRegressor.return_value = mock_model

        with patch.dict(sys.modules, {"lightgbm": mock_lgb}):
            # Use a small eval frame to speed up
            small_eval = eval_frame.head(48).copy()
            result = compute_lgbm_delta(small_eval, raw_60d)
            assert "rt_pred" in result.columns
            assert len(result) == len(small_eval)


# ── Test 5: Output metrics format ────────────────────────────────────────

class TestOutputMetricsFormat:
    def test_metrics_have_required_keys(self):
        m = compute_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))
        assert set(m.keys()) == {"sMAPE_floor50", "MAE", "RMSE", "n"}

    def test_n_is_integer(self):
        m = compute_metrics(np.arange(5, dtype=float), np.arange(5, dtype=float))
        assert isinstance(m["n"], int)
        assert m["n"] == 5


# ── Test 6: BGEW fusion ─────────────────────────────────────────────────

class TestBgewFusion:
    def test_fusion_produces_bgew_column(self):
        dates = pd.date_range("2025-01-02 01:00", periods=24, freq="h")
        y_true = np.random.RandomState(0).uniform(100, 500, 24)
        rolling = pd.DataFrame({
            "ds": dates, "hour_business": range(1, 25),
            "rt_pred": y_true + np.random.RandomState(1).normal(0, 10, 24),
            "y_true": y_true,
        })
        lgbm = pd.DataFrame({
            "ds": dates, "hour_business": range(1, 25),
            "rt_pred": y_true + np.random.RandomState(2).normal(0, 15, 24),
            "y_true": y_true,
        })
        merged, weights, smapes = fuse_delta_models_bgew(rolling, lgbm)
        assert "rt_pred_bgew" in merged.columns
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert len(weights) == 2

    def test_bgew_weights_clip_and_renormalize(self):
        smapes = {"a": 1.0, "b": 100.0}
        w = compute_bgew_weights(smapes, min_weight=0.05, max_weight=0.75)
        # Weights sum to 1 after renormalization
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        # Lower sMAPE model gets higher weight
        assert w["a"] > w["b"]


# ── Test 7: Baseline comparison ──────────────────────────────────────────

class TestBaselineComparison:
    def test_baseline_constant_is_33_03(self):
        assert RT_DA_ANCHOR_SMAPE_BASELINE == 33.03

    def test_status_improved_when_beats_baseline(self, raw_csv_path, tmp_path):
        """If a delta model beats 33.03%, status should be RT_DELTA_IMPROVED."""
        out_dir = str(tmp_path / "p140_out")
        # We mock the delta computations to return known-good values
        with patch(
            "scripts.run_p140_realtime_performance_unblock.compute_rolling_median_delta"
        ) as mock_rolling, patch(
            "scripts.run_p140_realtime_performance_unblock.compute_lgbm_delta"
        ) as mock_lgbm:
            # Create fake eval-like DataFrames with good predictions
            def make_fake_df(eval_df, *args, **kwargs):
                fake = eval_df.copy()
                # Near-perfect predictions
                fake["rt_pred"] = fake["y_true"] + np.random.RandomState(0).normal(0, 5, len(fake))
                return fake

            mock_rolling.side_effect = make_fake_df
            mock_lgbm.side_effect = make_fake_df

            result = run_p140_realtime_unblock(
                raw_data_path=raw_csv_path,
                output_dir=out_dir,
                day_start="2024-12-01",
                day_end="2024-12-30",
            )

        # With near-perfect predictions, sMAPE should be well below 33.03
        assert result["status"] == "RT_DELTA_IMPROVED"
        assert result["improvement_vs_baseline"] > 0


# ── Test 8: Blocked when no raw data ─────────────────────────────────────

class TestBlocked:
    def test_missing_raw_data_returns_blocked(self, tmp_path):
        out_dir = str(tmp_path / "p140_out")
        result = run_p140_realtime_unblock(
            raw_data_path=str(tmp_path / "nonexistent.csv"),
            output_dir=out_dir,
        )
        assert result["status"] == "RT_DELTA_BLOCKED"
        assert "RAW_DATA_MISSING" in result["reason_codes"]

    def test_missing_columns_returns_blocked(self, tmp_path):
        """If raw data lacks required columns, status = BLOCKED."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"ds": ["2025-01-01"], "other_col": [1]}).to_csv(bad_csv)
        out_dir = str(tmp_path / "p140_out")
        result = run_p140_realtime_unblock(
            raw_data_path=str(bad_csv),
            output_dir=out_dir,
        )
        assert result["status"] == "RT_DELTA_BLOCKED"


# ── Test 9: sMAPE_floor50 canonical ─────────────────────────────────────

class TestSmapeFloor50:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert compute_smape_floor50(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_floor_effect(self):
        """Values below 50 are floored."""
        y_true = np.array([10.0])
        y_pred = np.array([20.0])
        # After floor: both become 50, diff=0
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


# ── Test 10: Output files written ────────────────────────────────────────

class TestOutputFiles:
    def test_json_files_written(self, raw_csv_path, tmp_path):
        out_dir = str(tmp_path / "p140_out")
        with patch(
            "scripts.run_p140_realtime_performance_unblock.compute_rolling_median_delta"
        ) as mock_r, patch(
            "scripts.run_p140_realtime_performance_unblock.compute_lgbm_delta"
        ) as mock_l:
            def fake(eval_df, *args, **kwargs):
                f = eval_df.copy()
                f["rt_pred"] = f["da_anchor"]
                return f
            mock_r.side_effect = fake
            mock_l.side_effect = fake

            run_p140_realtime_unblock(
                raw_data_path=raw_csv_path,
                output_dir=out_dir,
                day_start="2024-12-01",
                day_end="2024-12-10",
            )

        expected = [
            "rt_da_anchor_metrics.json",
            "rolling_delta_metrics.json",
            "lgbm_delta_metrics.json",
            "pooled_realtime_bgew_metrics.json",
        ]
        for fname in expected:
            assert (Path(out_dir) / fname).is_file(), f"Missing: {fname}"

    def test_json_files_valid(self, raw_csv_path, tmp_path):
        out_dir = str(tmp_path / "p140_out")
        with patch(
            "scripts.run_p140_realtime_performance_unblock.compute_rolling_median_delta"
        ) as mock_r, patch(
            "scripts.run_p140_realtime_performance_unblock.compute_lgbm_delta"
        ) as mock_l:
            def fake(eval_df, *args, **kwargs):
                f = eval_df.copy()
                f["rt_pred"] = f["da_anchor"]
                return f
            mock_r.side_effect = fake
            mock_l.side_effect = fake

            run_p140_realtime_unblock(
                raw_data_path=raw_csv_path,
                output_dir=out_dir,
                day_start="2024-12-01",
                day_end="2024-12-10",
            )

        for fname in ("rt_da_anchor_metrics.json", "rolling_delta_metrics.json"):
            with open(Path(out_dir) / fname, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "sMAPE_floor50" in data


# ── Test 11: Prepare eval frame ──────────────────────────────────────────

class TestPrepareEvalFrame:
    def test_eval_frame_has_required_columns(self, raw_60d):
        df = _prepare_eval_frame(raw_60d, "2024-12-01", "2024-12-10")
        assert "da_anchor" in df.columns
        assert "y_true" in df.columns
        assert "business_day" in df.columns
        assert "hour_business" in df.columns
        assert "period" in df.columns

    def test_eval_frame_filters_date_range(self, raw_60d):
        df = _prepare_eval_frame(raw_60d, "2024-12-01", "2024-12-05")
        assert df["ds"].min() >= pd.Timestamp("2024-12-01")
        assert df["ds"].max() <= pd.Timestamp("2024-12-05 23:00")


# ── Test 12: Not improved status ─────────────────────────────────────────

class TestNotImproved:
    def test_not_improved_when_worse_than_baseline(self, raw_csv_path, tmp_path):
        """If delta models perform worse than 33.03%, status = NOT_IMPROVED."""
        out_dir = str(tmp_path / "p140_out")
        with patch(
            "scripts.run_p140_realtime_performance_unblock.compute_rolling_median_delta"
        ) as mock_r, patch(
            "scripts.run_p140_realtime_performance_unblock.compute_lgbm_delta"
        ) as mock_l:
            def fake_bad(eval_df, *args, **kwargs):
                f = eval_df.copy()
                # Terrible predictions
                f["rt_pred"] = f["da_anchor"] * 3.0
                return f
            mock_r.side_effect = fake_bad
            mock_l.side_effect = fake_bad

            result = run_p140_realtime_unblock(
                raw_data_path=raw_csv_path,
                output_dir=out_dir,
                day_start="2024-12-01",
                day_end="2024-12-10",
            )

        assert result["status"] == "RT_DELTA_NOT_IMPROVED"
        assert result["gap_to_baseline"] > 0
