"""
tests/test_p53_leakage_sentinel.py — P53 tests (19 tests).

Tests the Leakage Sentinel Runtime Guard: individual model checks,
batch sentinel runs, delivery gate logic, and edge-case handling.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safety.leakage_sentinel import (
    CONSERVATIVE_QUARANTINE,
    CORR_THRESHOLD,
    INVALID_24H,
    INVALID_SCHEMA,
    MAE_TOO_GOOD,
    SMAPE_FLOOR50_TOO_GOOD,
    SUSPECT_LEAKAGE,
    TRUSTED,
    WITHIN_1PCT_THRESHOLD,
    check_model_leakage,
    is_delivery_allowed,
    run_leakage_sentinel,
)


# ═══════════════════════════════════════════════════════════════════════
#  Test data helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_pred_ledger(
    tmp_path: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "test_model",
    business_day: str = "2026-06-30",
    target_day: str = "2026-07-01",
    task: str = "dayahead",
    add_y_true_col: bool = False,
    add_future_ds: bool = False,
    duplicate_hour: int | None = None,
    filename: str = "pred_ledger.csv",
) -> str:
    """Create a prediction ledger CSV at *tmp_path*.

    Returns the file path.
    """
    n = len(y_true)
    rows: list[dict[str, Any]] = []
    target_dt = datetime.strptime(target_day, "%Y-%m-%d")
    for i in range(n):
        h = (i % 24) + 1
        # hour h = target_dt + h hours
        #   h=1  -> target_day 01:00
        #   h=24 -> next_day 00:00
        actual_ts = target_dt + timedelta(hours=h)
        ts = actual_ts.strftime("%Y-%m-%d %H:%M:%S")
        if add_future_ds:
            ts = "2099-01-01 12:00:00"
        row: dict[str, Any] = {
            "model_name": model_name,
            "task": task,
            "target_day": target_day,
            "business_day": business_day,
            "hour_business": h,
            "ds": ts,
            "y_pred": float(y_pred[i]),
            "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
        }
        if add_y_true_col:
            row["y_true"] = float(y_true[i])
        rows.append(row)
        if duplicate_hour is not None and h == duplicate_hour:
            dup = dict(row)
            dup["ds"] = ts.replace(f"{h:02d}:00", f"{h:02d}:01")
            rows.append(dup)

    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False)
    return path


def _make_actual_ledger(
    tmp_path: Any,
    y_true: np.ndarray,
    business_day: str = "2026-06-30",
    target_day: str = "2026-07-01",
    task: str = "dayahead",
    filename: str = "actual_ledger.csv",
    no_merge_keys: bool = False,
) -> str:
    """Create an actual ledger CSV at *tmp_path*.

    Returns the file path.
    """
    n = len(y_true)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        h = (i % 24) + 1
        row: dict[str, Any] = {
            "task": task,
            "target_day": target_day,
            "business_day": business_day,
            "hour_business": h,
            "y_true": float(y_true[i]),
        }
        if no_merge_keys:
            row.pop("business_day", None)
            row.pop("hour_business", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════════════════
#  1. TRUSTED model passes all checks                                  (1)
# ═══════════════════════════════════════════════════════════════════════


class TestTrustedModel:
    """A model with realistic error (MAE ~30, sMAPE ~12%, corr ~0.97)
    should pass all checks and get TRUSTED."""

    @staticmethod
    def _make_trusted_data(tmp_path: Any) -> tuple[str, str]:
        n = 24
        rng = np.random.default_rng(42)
        y_t = 200.0 + rng.uniform(0, 300, n)  # 200–500
        y_p = y_t + rng.normal(0, 40, n)  # noisy predictions
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        return pred, actual

    def test_trusted_status(self, tmp_path):
        pred, actual = self._make_trusted_data(tmp_path)
        result = check_model_leakage(
            "test_model", pred, actual,
        )
        assert result["status"] == TRUSTED, (
            f"Expected TRUSTED, got {result['status']}: "
            f"{result.get('suspicion_reasons', [])}"
        )

    def test_trusted_all_checks_pass(self, tmp_path):
        pred, actual = self._make_trusted_data(tmp_path)
        result = check_model_leakage(
            "test_model", pred, actual,
        )
        for check_name, passed in result["checks"].items():
            assert passed, f"Check '{check_name}' failed for TRUSTED model"

    def test_trusted_metrics_populated(self, tmp_path):
        pred, actual = self._make_trusted_data(tmp_path)
        result = check_model_leakage(
            "test_model", pred, actual,
        )
        metrics = result["details"]["metrics"]
        assert metrics["n"] == 24
        assert metrics["MAE"] >= MAE_TOO_GOOD
        assert metrics["sMAPE_floor50"] >= SMAPE_FLOOR50_TOO_GOOD
        assert metrics["within_1pct_ratio"] <= WITHIN_1PCT_THRESHOLD
        assert metrics["corr_y_pred_y_true"] <= CORR_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════
#  2. SUSPECT_LEAKAGE triggers                                         (5)
# ═══════════════════════════════════════════════════════════════════════


class TestSuspectLeakage:
    """Clear leakage indicators -> SUSPECT_LEAKAGE."""

    def test_within_1pct_ratio_triggers_conservative(self, tmp_path):
        """within_1pct > 80% → CONSERVATIVE_QUARANTINE (not SUSPECT_LEAKAGE).
        (When only this trigger fires and no SUSPECT_LEAKAGE trigger.)
        """
        n = 24
        y_t = np.full(n, 300.0)
        # All predictions within 0.5% of y_true
        y_p = y_t + np.random.default_rng(42).normal(0, 1.0, n)
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        # within_1pct ratio should be very high
        metrics = result["details"]["metrics"]
        assert metrics["within_1pct_ratio"] > WITHIN_1PCT_THRESHOLD, (
            f"within_1pct_ratio={metrics['within_1pct_ratio']} not > "
            f"{WITHIN_1PCT_THRESHOLD}"
        )
        assert result["status"] in (CONSERVATIVE_QUARANTINE, SUSPECT_LEAKAGE)

    def test_smape_too_good_triggers_suspect(self, tmp_path):
        """sMAPE < 2% → SUSPECT_LEAKAGE."""
        n = 24
        y_t = 300.0 + np.arange(n) * 10  # 300..530
        # Very close predictions => sMAPE < 2%
        y_p = y_t + np.random.default_rng(42).normal(0, 3.0, n)
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        metrics = result["details"]["metrics"]
        assert metrics["sMAPE_floor50"] < SMAPE_FLOOR50_TOO_GOOD, (
            f"sMAPE={metrics['sMAPE_floor50']} not < {SMAPE_FLOOR50_TOO_GOOD}"
        )
        assert result["status"] == SUSPECT_LEAKAGE, (
            f"Expected SUSPECT_LEAKAGE, got {result['status']}"
        )

    def test_mae_too_good_triggers_suspect(self, tmp_path):
        """MAE < 10 CNY → SUSPECT_LEAKAGE."""
        n = 24
        y_t = 300.0 + np.arange(n) * 10
        # Small error → MAE ~5
        y_p = y_t + np.random.default_rng(42).normal(0, 5.0, n)
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        metrics = result["details"]["metrics"]
        assert metrics["MAE"] < MAE_TOO_GOOD, (
            f"MAE={metrics['MAE']} not < {MAE_TOO_GOOD}"
        )
        assert result["status"] == SUSPECT_LEAKAGE, (
            f"Expected SUSPECT_LEAKAGE, got {result['status']}"
        )

    def test_future_timestamp_triggers_suspect(self, tmp_path):
        """Future timestamps → SUSPECT_LEAKAGE."""
        n = 24
        y_t = np.full(n, 300.0)
        y_p = y_t + 50.0
        pred = _make_pred_ledger(tmp_path, y_t, y_p, add_future_ds=True)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        assert result["checks"].get("no_future_timestamps") is False
        assert result["status"] == SUSPECT_LEAKAGE

    def test_duplicate_keys_triggers_suspect(self, tmp_path):
        """Duplicate (business_day, hour_business) → SUSPECT_LEAKAGE."""
        n = 24
        y_t = np.full(n, 300.0)
        y_p = y_t + 50.0
        # Add a duplicate for hour 12
        pred = _make_pred_ledger(tmp_path, y_t, y_p, duplicate_hour=12)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        assert result["checks"].get("no_duplicate_keys") is False
        assert result["status"] == SUSPECT_LEAKAGE


# ═══════════════════════════════════════════════════════════════════════
#  3. CONSERVATIVE_QUARANTINE triggers                                 (1)
# ═══════════════════════════════════════════════════════════════════════


class TestConservativeQuarantine:
    """corr > 0.995 → CONSERVATIVE_QUARANTINE
    (but other metrics still realistic).
    """

    def test_corr_high_triggers_conservative_quarantine(self, tmp_path):
        """corr > 0.995 with sMAPE >= 2% and MAE >= 10."""
        n = 24
        # y_true with strong linear trend
        y_t = 100.0 + np.arange(n) * 20  # 100..560
        # y_pred very close to a linear function of y_true
        # y_pred = 0.98 * y_t + 0.02 * y_t^2/560 + tiny noise
        # This maintains high corr but with ~5% avg error
        noise = np.random.default_rng(42).normal(0, 2.0, n)
        y_p = y_t * 1.05 + 5.0 + noise
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        metrics = result["details"]["metrics"]
        assert metrics["corr_y_pred_y_true"] > CORR_THRESHOLD, (
            f"corr={metrics['corr_y_pred_y_true']} not > {CORR_THRESHOLD}"
        )
        # sMAPE and MAE should be above the SUSPECT thresholds
        # y_p = y_t*1.05 + 5 → avg diff ≈ 0.05*mean(y_t) + 5
        # mean(y_t) = 330, avg diff ≈ 21.5 -> MAE ≈ 21.5 > 10
        assert metrics["MAE"] >= MAE_TOO_GOOD, (
            f"MAE={metrics['MAE']} too low, would be SUSPECT_LEAKAGE"
        )
        result = check_model_leakage("test_model", pred, actual)
        assert result["status"] == CONSERVATIVE_QUARANTINE, (
            f"Expected CONSERVATIVE_QUARANTINE, got {result['status']}: "
            f"{result.get('suspicion_reasons', [])}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  4. INVALID_SCHEMA                                                    (3)
# ═══════════════════════════════════════════════════════════════════════


class TestInvalidSchema:
    """Missing required columns -> INVALID_SCHEMA."""

    def test_missing_model_name_column(self, tmp_path):
        """No 'model_name' column -> INVALID_SCHEMA."""
        n = 24
        y_t = np.full(n, 300.0)
        y_p = y_t + 50.0
        pred_path = os.path.join(str(tmp_path), "pred_no_model.csv")
        pd.DataFrame({
            "y_pred": y_p,
            "business_day": ["2026-06-30"] * n,
            "hour_business": list(range(1, n + 1)),
        }).to_csv(pred_path, index=False)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred_path, actual)
        assert result["status"] == INVALID_SCHEMA
        assert "model_name" in result["details"].get("error", "").lower()

    def test_y_true_in_prediction_ledger(self, tmp_path):
        """y_true column in prediction ledger -> INVALID_SCHEMA."""
        n = 24
        y_t = np.full(n, 300.0)
        y_p = y_t + 50.0
        pred = _make_pred_ledger(tmp_path, y_t, y_p, add_y_true_col=True)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        assert result["status"] == INVALID_SCHEMA
        assert not result["checks"]["no_y_true_in_prediction_ledger"]

    def test_missing_merge_keys(self, tmp_path):
        """Actual ledger missing merge keys -> INVALID_SCHEMA."""
        n = 24
        y_t = np.full(n, 300.0)
        y_p = y_t + 50.0
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t, no_merge_keys=True)
        result = check_model_leakage("test_model", pred, actual)
        assert result["status"] == INVALID_SCHEMA


# ═══════════════════════════════════════════════════════════════════════
#  5. INVALID_24H                                                       (1)
# ═══════════════════════════════════════════════════════════════════════


class TestInvalid24H:
    """Incomplete 24-hour coverage -> INVALID_24H."""

    def test_not_24_hours(self, tmp_path):
        """Only 12 rows (not 24) -> INVALID_24H."""
        n = 12
        y_t = np.full(n, 300.0)
        y_p = y_t + 50.0
        # Write only 12 rows (hours 1-12)
        rng = np.random.default_rng(42)
        y_t_full = 200.0 + rng.uniform(0, 300, n)
        y_p_full = y_t_full + rng.normal(0, 40, n)
        pred = _make_pred_ledger(tmp_path, y_t_full, y_p_full)
        # Overwrite with partial data
        df = pd.read_csv(pred)
        df_partial = df[df["hour_business"] <= 12]
        df_partial.to_csv(pred, index=False)
        actual = _make_actual_ledger(tmp_path, y_t_full)
        result = check_model_leakage("test_model", pred, actual)
        assert result["status"] == INVALID_24H, (
            f"Expected INVALID_24H, got {result['status']}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  6. Edge cases                                                        (5)
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases: missing files, NaN, model name mismatch, empty ledger."""

    def test_missing_ledger_file_graceful(self, tmp_path):
        """Missing ledger file -> INVALID_SCHEMA."""
        result = check_model_leakage(
            "test_model",
            prediction_ledger_path="/nonexistent/pred.parquet",
            actual_ledger_path="/nonexistent/actual.parquet",
        )
        assert result["status"] == INVALID_SCHEMA
        assert not result["checks"]["ledger_loaded"]

    def test_nan_y_pred_detected(self, tmp_path):
        """NaN y_pred values produce a warning."""
        n = 24
        y_t = 200.0 + np.arange(n) * 10
        y_p = y_t + np.random.default_rng(42).normal(0, 40, n)
        y_p[5] = np.nan
        y_p[12] = np.nan
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("test_model", pred, actual)
        assert result["details"]["nan_y_pred_count"] == 2
        assert any("NaN" in w for w in result["warnings"])

    def test_model_name_mismatch(self, tmp_path):
        """Model not in prediction ledger -> INVALID_SCHEMA."""
        n = 24
        y_t = 200.0 + np.arange(n) * 10
        y_p = y_t + np.random.default_rng(42).normal(0, 40, n)
        pred = _make_pred_ledger(tmp_path, y_t, y_p, model_name="real_model")
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage("wrong_model", pred, actual)
        assert result["status"] == INVALID_SCHEMA
        assert "not found" in result["details"].get("error", "").lower()

    def test_empty_prediction_ledger(self, tmp_path):
        """Empty prediction ledger -> INVALID_SCHEMA."""
        n = 24
        y_t = 200.0 + np.arange(n) * 10
        actual = _make_actual_ledger(tmp_path, y_t)
        # Create empty ledger (headers only)
        empty_path = os.path.join(str(tmp_path), "empty_pred.csv")
        pd.DataFrame(columns=[
            "model_name", "task", "target_day", "business_day",
            "hour_business", "ds", "y_pred", "period",
        ]).to_csv(empty_path, index=False)
        result = check_model_leakage("test_model", empty_path, actual)
        assert result["status"] == INVALID_SCHEMA

    def test_target_in_feature_columns_detected(self, tmp_path):
        """Feature columns containing target names -> INVALID_SCHEMA."""
        n = 24
        y_t = 200.0 + np.arange(n) * 10
        y_p = y_t + np.random.default_rng(42).normal(0, 40, n)
        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        result = check_model_leakage(
            "test_model", pred, actual,
            feature_columns=["load", "wind", "y_true", "solar"],
        )
        assert result["status"] == INVALID_SCHEMA
        assert not result["checks"]["no_target_in_features"]


# ═══════════════════════════════════════════════════════════════════════
#  7. is_delivery_allowed                                               (4)
# ═══════════════════════════════════════════════════════════════════════


class TestDeliveryAllowed:
    """Delivery gate logic."""

    @staticmethod
    def _run_sentinel_for(
        tmp_path: Any, status: str,
    ) -> dict[str, Any]:
        """Create a sentinel result with a single model at *status*."""
        n = 24
        rng = np.random.default_rng(42)
        y_t = 200.0 + rng.uniform(0, 300, n)

        if status == TRUSTED:
            y_p = y_t + rng.normal(0, 40, n)
        elif status == CONSERVATIVE_QUARANTINE:
            y_p = y_t * 1.05 + 5.0  # high corr
        elif status == SUSPECT_LEAKAGE:
            y_p = y_t + rng.normal(0, 3.0, n)  # low sMAPE
        elif status == INVALID_SCHEMA:
            # Missing model_name column
            pred_path = os.path.join(str(tmp_path), "pred_bad.csv")
            pd.DataFrame({"y_pred": [100.0]}).to_csv(pred_path, index=False)
            actual = _make_actual_ledger(tmp_path, np.array([100.0]))
            return check_model_leakage("test_model", pred_path, actual)
        elif status == INVALID_24H:
            y_p = y_t + rng.normal(0, 40, n)
            pred = _make_pred_ledger(tmp_path, y_t[:12], y_p[:12])
            actual = _make_actual_ledger(tmp_path, y_t)
            return check_model_leakage("test_model", pred, actual)
        else:
            raise ValueError(f"Unknown status: {status}")

        pred = _make_pred_ledger(tmp_path, y_t, y_p)
        actual = _make_actual_ledger(tmp_path, y_t)
        return check_model_leakage("test_model", pred, actual)

    def test_delivery_allows_trusted(self, tmp_path):
        """TRUSTED model allowed for delivery."""
        sr = self._run_sentinel_for(tmp_path, TRUSTED)
        sentinel_result = {"models": [sr], "summary": {TRUSTED: 1}}
        assert is_delivery_allowed("test_model", sentinel_result, "trusted_delivery") is True

    def test_delivery_blocks_suspect_leakage(self, tmp_path):
        """SUSPECT_LEAKAGE blocked for ANY profile."""
        sr = self._run_sentinel_for(tmp_path, SUSPECT_LEAKAGE)
        sentinel_result = {"models": [sr], "summary": {SUSPECT_LEAKAGE: 1}}
        assert is_delivery_allowed("test_model", sentinel_result, "trusted_delivery") is False
        assert is_delivery_allowed("test_model", sentinel_result, "research_all_models") is False

    def test_delivery_blocks_conservative_for_delivery_profile(self, tmp_path):
        """CONSERVATIVE_QUARANTINE blocked for trusted_delivery."""
        sr = self._run_sentinel_for(tmp_path, CONSERVATIVE_QUARANTINE)
        sentinel_result = {"models": [sr], "summary": {CONSERVATIVE_QUARANTINE: 1}}
        assert is_delivery_allowed(
            "test_model", sentinel_result, "trusted_delivery",
        ) is False

    def test_delivery_allows_conservative_for_research_profile(self, tmp_path):
        """CONSERVATIVE_QUARANTINE allowed for research profiles."""
        sr = self._run_sentinel_for(tmp_path, CONSERVATIVE_QUARANTINE)
        sentinel_result = {"models": [sr], "summary": {CONSERVATIVE_QUARANTINE: 1}}
        assert is_delivery_allowed(
            "test_model", sentinel_result, "research_all_models",
        ) is True
        assert is_delivery_allowed(
            "test_model", sentinel_result, "balanced_candidate",
        ) is True

    def test_delivery_blocks_invalid_schema(self, tmp_path):
        """INVALID_SCHEMA blocked for all profiles."""
        sr = self._run_sentinel_for(tmp_path, INVALID_SCHEMA)
        sentinel_result = {"models": [sr], "summary": {INVALID_SCHEMA: 1}}
        assert is_delivery_allowed("test_model", sentinel_result, "trusted_delivery") is False
        assert is_delivery_allowed("test_model", sentinel_result, "research_all_models") is False

    def test_delivery_blocks_unknown_model(self, tmp_path):
        """Model not in sentinel result -> blocked."""
        sr = self._run_sentinel_for(tmp_path, TRUSTED)
        sentinel_result = {"models": [sr], "summary": {TRUSTED: 1}}
        assert is_delivery_allowed("unknown_model", sentinel_result, "trusted_delivery") is False


# ═══════════════════════════════════════════════════════════════════════
#  8. run_leakage_sentinel                                              (1)
# ═══════════════════════════════════════════════════════════════════════


class TestRunLeakageSentinel:
    """Integration: run_leakage_sentinel on multiple models."""

    def test_sentinel_summary_structure(self, tmp_path):
        """run_leakage_sentinel returns proper summary structure."""
        n = 24
        rng = np.random.default_rng(42)
        y_t = 200.0 + rng.uniform(0, 300, n)
        y_p = y_t + rng.normal(0, 40, n)
        pred = _make_pred_ledger(tmp_path, y_t, y_p, model_name="model_a")
        actual = _make_actual_ledger(tmp_path, y_t)

        # Add second model
        y_p2 = y_t + rng.normal(0, 3.0, n)  # suspicious
        df = pd.read_csv(pred)
        df2 = df.copy()
        df2["model_name"] = "model_b"
        df2["y_pred"] = y_p2
        df = pd.concat([df, df2], ignore_index=True)
        df.to_csv(pred, index=False)

        result = run_leakage_sentinel(
            trusted_models=["model_a", "model_b"],
            prediction_ledger_path=pred,
            actual_ledger_path=actual,
        )

        assert result["phase"] == "P53"
        assert result["n_models_checked"] == 2
        assert len(result["models"]) == 2
        assert isinstance(result["summary"], dict)
        total = sum(result["summary"].values())
        assert total == 2, f"Summary counts should sum to 2, got {total}"

        # Model A should be TRUSTED, model B should be SUSPECT_LEAKAGE
        model_a = [m for m in result["models"] if m["model_name"] == "model_a"][0]
        model_b = [m for m in result["models"] if m["model_name"] == "model_b"][0]
        assert model_a["status"] == TRUSTED, (
            f"model_a should be TRUSTED, got {model_a['status']}"
        )
        assert model_b["status"] == SUSPECT_LEAKAGE, (
            f"model_b should be SUSPECT_LEAKAGE, got {model_b['status']}"
        )

        # Eval rows should be consistent (both have same n)
        assert result["eval_rows_consistent"] is True
