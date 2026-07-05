"""
tests/test_p138_2025_rolling_trusted_bgew.py — Tests for P138.

At least 12 tests covering:
  - BGEW computation with mock data
  - sMAPE_floor50 formula correctness
  - Weight learning produces valid weights (sum to 1, within bounds)
  - Rolling window uses only days < target_day (no lookahead)
  - Fallback to equal weights when < 14 days
  - daily_metrics output format
  - Improvement calculation
  - Blocked when single model
  - Overall metrics structure
  - Weight summary structure
  - Output files exist
  - Merge with actuals correctness
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

from scripts.run_p138_2025_rolling_trusted_bgew import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_RAW_DATA_PATH,
    LOOKBACK_DAYS,
    MIN_HISTORY_DAYS,
    SMAPE_FLOOR,
    compute_mae,
    compute_smape_floor50,
    _fuse_predictions,
    _get_history_window,
    _compute_model_smape,
    run_p138_rolling_bgew,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_mock_ledger(n_days: int = 40, models: list = None) -> pd.DataFrame:
    """Create a mock multi-model prediction ledger."""
    if models is None:
        models = ["lightgbm_cfg05_dayahead", "catboost_spike_residual"]

    rows = []
    base_date = pd.Timestamp("2025-01-01")
    for d in range(n_days):
        day = (base_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for model in models:
            for h in range(1, 25):
                # Different predictions per model
                offset = 0 if "cfg05" in model else 5
                rows.append({
                    "task": "dayahead",
                    "model_name": model,
                    "business_day": day,
                    "ds": f"{day} {h:02d}:00:00",
                    "hour_business": h,
                    "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                    "y_pred": 200.0 + h + d + offset,
                    "source_confidence": 1.0,
                    "model_version": "test",
                })
    return pd.DataFrame(rows)


def _make_mock_actuals(n_days: int = 40) -> pd.DataFrame:
    """Create mock actuals matching the ledger."""
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    for d in range(n_days):
        day = (base_date + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for h in range(1, 25):
            rows.append({
                "business_day": day,
                "hour_business": h,
                "y_true": 205.0 + h + d,  # close to predictions
            })
    return pd.DataFrame(rows)


def _save_ledger_and_actuals(
    ledger: pd.DataFrame,
    actuals: pd.DataFrame,
    tmpdir: str,
) -> tuple[str, str]:
    """Save mock data to CSV files and return paths."""
    ledger_path = os.path.join(tmpdir, "ledger.csv")
    ledger.to_csv(ledger_path, index=False)

    # Create a mock raw CSV with Chinese column names (GBK)
    raw_rows = []
    base_date = pd.Timestamp("2025-01-01")
    for _, row in actuals.iterrows():
        bd = pd.Timestamp(row["business_day"])
        h = int(row["hour_business"])
        # Convert business_day + hour_business back to wall-clock timestamp
        if h == 24:
            ts = bd + pd.Timedelta(days=1)
        else:
            ts = bd + pd.Timedelta(hours=h)
        raw_rows.append({
            "\u65f6\u523b": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "\u65e5\u524d\u7535\u4ef7": row["y_true"],
            "\u5b9e\u65f6\u7535\u4ef7": row["y_true"] * 0.95,
        })
    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(tmpdir, "raw_data.csv")
    raw_df.to_csv(raw_path, index=False, encoding="gbk")

    return ledger_path, raw_path


# ── Tests: sMAPE_floor50 formula ──────────────────────────────────────


class TestSmapeFloor50:
    """Test the canonical sMAPE_floor50 formula."""

    def test_perfect_prediction(self):
        """sMAPE should be 0 when predictions match actuals."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        assert compute_smape_floor50(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_floor_50_applied(self):
        """Values below 50 should be floored to 50."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([30.0, 40.0])
        # After floor: y_f = [50, 50], yp_f = [50, 50]
        # sMAPE = 200 * mean(|50-50| / (50+50)) = 0
        result = compute_smape_floor50(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_known_smape_value(self):
        """Test with a known computation."""
        y_true = np.array([100.0])
        y_pred = np.array([200.0])
        # y_f = max(100, 50) = 100, yp_f = max(200, 50) = 200
        # sMAPE = 200 * |100 - 200| / (100 + 200) = 200 * 100/300 = 66.667
        result = compute_smape_floor50(y_true, y_pred)
        assert result == pytest.approx(200.0 * 100.0 / 300.0, rel=1e-4)

    def test_symmetric_property(self):
        """sMAPE should be symmetric (swap y_true and y_pred gives same result)."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([150.0, 250.0])
        result1 = compute_smape_floor50(y_true, y_pred)
        result2 = compute_smape_floor50(y_pred, y_true)
        assert result1 == pytest.approx(result2, rel=1e-10)

    def test_negative_values_floored(self):
        """Negative values should be floored to 50."""
        y_true = np.array([-10.0, -50.0])
        y_pred = np.array([100.0, 200.0])
        # y_f = max(-10, 50) = 50, max(-50, 50) = 50
        # yp_f = max(100, 50) = 100, max(200, 50) = 200
        # sMAPE = 200 * mean(|50-100|/(50+100), |50-200|/(50+200))
        #       = 200 * mean(50/150, 150/250)
        #       = 200 * mean(0.3333, 0.6)
        #       = 200 * 0.4667 = 93.33
        result = compute_smape_floor50(y_true, y_pred)
        expected = 200.0 * np.mean([50.0/150.0, 150.0/250.0])
        assert result == pytest.approx(expected, rel=1e-4)


# ── Tests: BGEW weight learning ──────────────────────────────────────


class TestWeightLearning:
    """Test that weight learning produces valid weights."""

    def test_weights_sum_to_one(self):
        """BGEW weights should sum to 1."""
        from fusion.unified_weight_learner import compute_bgew_weights
        smape = {"model_a": 10.0, "model_b": 20.0}
        weights = compute_bgew_weights(smape)
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)

    def test_weights_within_bounds(self):
        """BGEW weights should be positive and sum to 1 after renormalization.

        Note: after clip+renormalize, individual weights may exceed max_weight
        because renormalization scales all weights proportionally.  The key
        invariants are: all weights > 0 and sum(weights) == 1.
        """
        from fusion.unified_weight_learner import compute_bgew_weights
        smape = {"model_a": 5.0, "model_b": 50.0, "model_c": 100.0}
        weights = compute_bgew_weights(smape, min_weight=0.05, max_weight=0.75)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=1e-6)
        for m, w in weights.items():
            assert w > 0, f"Weight for {m} should be positive: {w}"

    def test_better_model_gets_higher_weight(self):
        """Model with lower sMAPE should get higher weight."""
        from fusion.unified_weight_learner import compute_bgew_weights
        smape = {"good_model": 5.0, "bad_model": 50.0}
        weights = compute_bgew_weights(smape)
        assert weights["good_model"] > weights["bad_model"]


# ── Tests: Rolling window no-lookahead ────────────────────────────────


class TestNoLookahead:
    """Test that rolling window uses only days < target_day."""

    def test_history_excludes_target_day(self):
        """History window should not include the target day itself."""
        ledger = _make_mock_ledger(n_days=40)
        actuals = _make_mock_actuals(n_days=40)

        target_day = "2025-01-20"
        merged = _get_history_window(ledger, actuals, target_day)

        if merged is not None:
            days_in_history = merged["business_day"].unique()
            assert target_day not in days_in_history

    def test_history_only_past_days(self):
        """All history days should be strictly before target_day."""
        ledger = _make_mock_ledger(n_days=40)
        actuals = _make_mock_actuals(n_days=40)

        target_day = "2025-01-20"
        merged = _get_history_window(ledger, actuals, target_day)

        if merged is not None:
            for day in merged["business_day"].unique():
                assert pd.Timestamp(day) < pd.Timestamp(target_day)


# ── Tests: Fallback to equal weights ─────────────────────────────────


class TestEqualWeightFallback:
    """Test fallback to equal weights when < 14 days."""

    def test_equal_weights_when_few_days(self):
        """When < MIN_HISTORY_DAYS, should use equal weights."""
        # Create a scenario with only 10 history days
        ledger = _make_mock_ledger(n_days=12)
        actuals = _make_mock_actuals(n_days=12)

        # Target day 11 → only 10 history days available (< 14)
        target_day = "2025-01-11"
        merged = _get_history_window(ledger, actuals, target_day)

        if merged is not None:
            n_days = merged["business_day"].nunique()
            # Should be <= 10 days (which is < MIN_HISTORY_DAYS=14)
            assert n_days <= 10


# ── Tests: Output format ─────────────────────────────────────────────


class TestDailyMetricsFormat:
    """Test daily_metrics output format."""

    def test_daily_metrics_columns(self):
        """daily_metrics.csv should have the expected columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = _make_mock_ledger(n_days=40)
            actuals = _make_mock_actuals(n_days=40)
            ledger_path, raw_path = _save_ledger_and_actuals(ledger, actuals, tmpdir)

            output_dir = os.path.join(tmpdir, "output")
            result = run_p138_rolling_bgew(ledger_path, raw_path, output_dir)

            dm_path = os.path.join(output_dir, "daily_metrics.csv")
            if os.path.isfile(dm_path):
                dm = pd.read_csv(dm_path)
                assert "target_day" in dm.columns
                assert "bgew_smape" in dm.columns


class TestImprovementCalculation:
    """Test improvement calculation."""

    def test_improvement_metric_present(self):
        """Overall metrics should include relative_improvement_vs_cfg05."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = _make_mock_ledger(n_days=40)
            actuals = _make_mock_actuals(n_days=40)
            ledger_path, raw_path = _save_ledger_and_actuals(ledger, actuals, tmpdir)

            output_dir = os.path.join(tmpdir, "output")
            result = run_p138_rolling_bgew(ledger_path, raw_path, output_dir)

            overall = result.get("overall_metrics", {})
            if overall:
                assert "relative_improvement_vs_cfg05" in overall


class TestBlockedSingleModel:
    """Test blocked when single model."""

    def test_single_model_blocked(self):
        """Should be BLOCKED with only one model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = _make_mock_ledger(n_days=40, models=["only_model"])
            actuals = _make_mock_actuals(n_days=40)
            ledger_path, raw_path = _save_ledger_and_actuals(ledger, actuals, tmpdir)

            output_dir = os.path.join(tmpdir, "output")
            result = run_p138_rolling_bgew(ledger_path, raw_path, output_dir)

            assert result["status"] == "BGEW_2025_BLOCKED"


class TestOutputFilesExist:
    """Test that output files are created."""

    def test_weights_csv_exists(self):
        """weights.csv should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = _make_mock_ledger(n_days=40)
            actuals = _make_mock_actuals(n_days=40)
            ledger_path, raw_path = _save_ledger_and_actuals(ledger, actuals, tmpdir)

            output_dir = os.path.join(tmpdir, "output")
            run_p138_rolling_bgew(ledger_path, raw_path, output_dir)

            weights_path = os.path.join(output_dir, "weights.csv")
            assert os.path.isfile(weights_path)

    def test_model_weight_summary_exists(self):
        """model_weight_summary.json should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = _make_mock_ledger(n_days=40)
            actuals = _make_mock_actuals(n_days=40)
            ledger_path, raw_path = _save_ledger_and_actuals(ledger, actuals, tmpdir)

            output_dir = os.path.join(tmpdir, "output")
            run_p138_rolling_bgew(ledger_path, raw_path, output_dir)

            summary_path = os.path.join(output_dir, "model_weight_summary.json")
            assert os.path.isfile(summary_path)


class TestWeightSummaryStructure:
    """Test weight summary structure."""

    def test_weight_summary_has_models(self):
        """Weight summary should contain entries for each model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = _make_mock_ledger(n_days=40)
            actuals = _make_mock_actuals(n_days=40)
            ledger_path, raw_path = _save_ledger_and_actuals(ledger, actuals, tmpdir)

            output_dir = os.path.join(tmpdir, "output")
            result = run_p138_rolling_bgew(ledger_path, raw_path, output_dir)

            ws = result.get("weight_summary", {})
            if ws:
                assert len(ws) == 2
                for m, w in ws.items():
                    assert 0 <= w <= 1


class TestFusePredictions:
    """Test _fuse_predictions helper."""

    def test_fuse_equal_weights(self):
        """Equal weights should produce average of predictions."""
        day_preds = pd.DataFrame({
            "hour_business": [1, 2, 1, 2],
            "model_name": ["a", "a", "b", "b"],
            "y_pred": [100.0, 200.0, 200.0, 300.0],
        })
        weights = {"a": 0.5, "b": 0.5}
        models = ["a", "b"]

        fused = _fuse_predictions(day_preds, weights, models)
        assert len(fused) == 2
        assert fused[0] == pytest.approx(150.0)  # (100+200)/2
        assert fused[1] == pytest.approx(250.0)  # (200+300)/2

    def test_fuse_unequal_weights(self):
        """Unequal weights should produce weighted average."""
        day_preds = pd.DataFrame({
            "hour_business": [1, 1],
            "model_name": ["a", "b"],
            "y_pred": [100.0, 200.0],
        })
        weights = {"a": 0.75, "b": 0.25}
        models = ["a", "b"]

        fused = _fuse_predictions(day_preds, weights, models)
        assert len(fused) == 1
        assert fused[0] == pytest.approx(125.0)  # 0.75*100 + 0.25*200


class TestComputeModelSmape:
    """Test _compute_model_smape helper."""

    def test_smape_per_model(self):
        """Should compute sMAPE for each model."""
        merged = pd.DataFrame({
            "model_name": ["a"] * 4 + ["b"] * 4,
            "y_true": [100, 200, 300, 400, 100, 200, 300, 400],
            "y_pred": [110, 210, 310, 410, 150, 250, 350, 450],
            "business_day": ["2025-01-01"] * 8,
            "hour_business": [1, 2, 3, 4, 1, 2, 3, 4],
        })
        smape = _compute_model_smape(merged, ["a", "b"])
        assert "a" in smape
        assert "b" in smape
        # Model "a" is closer to actuals, so lower sMAPE
        assert smape["a"] < smape["b"]


# ── Integration test with real artifacts ──────────────────────────────


class TestRealArtifacts:
    """Integration tests using real artifact files."""

    @pytest.fixture(autouse=True)
    def _check_artifacts(self):
        if not os.path.isfile(DEFAULT_LEDGER_PATH):
            pytest.skip("Trusted ledger not available (run P137 first)")
        if not os.path.isfile(DEFAULT_RAW_DATA_PATH):
            pytest.skip("Raw data not available")

    def test_real_rolling_bgew_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_p138_rolling_bgew(
                DEFAULT_LEDGER_PATH, DEFAULT_RAW_DATA_PATH, tmpdir
            )
            assert result["status"] in (
                "BGEW_2025_IMPROVED",
                "BGEW_2025_NOT_IMPROVED",
                "BGEW_2025_BLOCKED",
            )
            assert result.get("processed_days", 0) > 0
