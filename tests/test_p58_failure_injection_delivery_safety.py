"""
tests/test_p58_failure_injection_delivery_safety.py — P58: Failure Injection Tests
for the P52-P57 Safety Supervisor Pipeline.

Each test injects a specific failure mode into the production modules and
verifies that safety checks correctly catch it.  All assertions use clear
messages.  All temporary files use tmp_path.

Tests are self-contained and import the ACTUAL production modules (no mocks).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
#  Constants re-exported from production modules for readability
# ======================================================================
from safety.leakage_sentinel import (
    CONSERVATIVE_QUARANTINE,
    INVALID_SCHEMA,
    INVALID_24H,
    SUSPECT_LEAKAGE,
    TRUSTED,
)
# Adaptive training days status values (not exported as constants, use str literals)
_ADAPTIVE_COMPLETE_30D = "COMPLETE_30D"
_ADAPTIVE_DEGRADED_MIN_DAYS = "DEGRADED_MIN_DAYS"
_ADAPTIVE_INSUFFICIENT_DAYS = "INSUFFICIENT_DAYS"
_ADAPTIVE_NO_VALID_DAYS = "NO_VALID_DAYS"
from fusion.trust_gated_regime_bgew import (
    ALLOWED_TRUST_STATES,
    ALLOWED_TRUST_STATES_BALANCED,
    TRUST_STATE_CONSERVATIVE_QUARANTINE,
    TRUST_STATE_SUSPECT_LEAKAGE,
)

# ======================================================================
#  Helpers — reusable test-data builders
# ======================================================================


def _make_stage3_pred_ledger(
    tmp_path: pytest.TempPathFactory,
    model_name: str = "stage3",
) -> str:
    """Create a prediction ledger where *model_name* has suspiciously
    accurate predictions that should trigger SUSPECT_LEAKAGE."""
    n = 24
    rng = np.random.default_rng(20260705)
    y_t = 200.0 + rng.uniform(0, 300, n)
    # Very tight noise — sMAPE << 2%  and  MAE << 10 CNY
    y_p = y_t + rng.normal(0, 1.5, n)
    rows: list[dict] = []
    target_dt = pd.Timestamp("2026-07-05")
    for i in range(n):
        h = (i % 24) + 1
        if h == 24:
            ts = target_dt + pd.Timedelta(days=1)
        else:
            ts = target_dt + pd.Timedelta(hours=h)
        rows.append({
            "model_name": model_name,
            "task": "dayahead",
            "target_day": "2026-07-05",
            "business_day": "2026-07-04",
            "hour_business": h,
            "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "y_pred": float(y_p[i]),
            "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
        })
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), "pred_ledger.csv")
    df.to_csv(path, index=False)
    return path


def _make_actual_ledger(
    tmp_path: pytest.TempPathFactory,
    y_true: np.ndarray | None = None,
    target_day: str = "2026-07-05",
    business_day: str = "2026-07-04",
    filename: str = "actual_ledger.csv",
) -> str:
    """Create a standard 24-row actual ledger CSV."""
    if y_true is None:
        rng = np.random.default_rng(20260705)
        y_true = 200.0 + rng.uniform(0, 300, 24)
    n = len(y_true)
    rows: list[dict] = []
    target_dt = pd.Timestamp(target_day)
    for i in range(n):
        h = (i % 24) + 1
        if h == 24:
            ts = target_dt + pd.Timedelta(days=1)
        else:
            ts = target_dt + pd.Timedelta(hours=h)
        rows.append({
            "task": "dayahead",
            "target_day": target_day,
            "business_day": business_day,
            "hour_business": h,
            "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "y_true": float(y_true[i]),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False)
    return path


def _make_output_csv(
    tmp_path: pytest.TempPathFactory,
    n_rows: int = 24,
    nan_hour: int | None = None,
    extra_cols: list[str] | None = None,
    filename: str = "final_output.csv",
    target_date: str = "2026-07-05",
) -> str:
    """Create a delivery-output-style CSV with configurable failures."""
    rows: list[dict] = []
    for i in range(n_rows):
        h = (i % 24) + 1
        val = None if (nan_hour is not None and h == nan_hour) else 100.0 + h * 0.5
        row: dict = {
            "business_day": target_date,
            "hour_business": h,
            "y_pred": val,
            "ds": f"{target_date} {h:02d}:00:00",
        }
        if extra_cols:
            for col in extra_cols:
                row[col] = "dummy"
        rows.append(row)
    # Hour-24 convention: ds = D+1 00:00:00
    if n_rows >= 24:
        rows[23]["ds"] = "2026-07-06 00:00:00"
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False)
    return path


def _make_prediction_parquet_ledger(
    tmp_path: pytest.TempPathFactory,
    days: list[str],
    models: list[str],
    *,
    seed: int = 42,
    add_y_true_col: bool = False,
) -> str:
    """Write a prediction ledger parquet with 24-hour data per (day, model)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for day in days:
        day_dt = pd.Timestamp(day)
        for model in models:
            for h in range(1, 25):
                if h == 24:
                    ts = day_dt + pd.Timedelta(days=1)
                else:
                    ts = day_dt + pd.Timedelta(hours=h)
                row: dict = {
                    "task": "dayahead",
                    "model_name": model,
                    "target_day": day_dt,
                    "business_day": day_dt,
                    "ds": ts,
                    "hour_business": h,
                    "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
                    "y_pred": float(rng.uniform(80, 200)),
                }
                if add_y_true_col:
                    row["y_true"] = float(rng.uniform(80, 200))
                rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), "prediction_ledger.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_actual_parquet_ledger(
    tmp_path: pytest.TempPathFactory,
    days: list[str],
    *,
    seed: int = 42,
) -> str:
    """Write an actual ledger parquet with 24-hour data per day."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for day in days:
        day_dt = pd.Timestamp(day)
        for h in range(1, 25):
            if h == 24:
                ts = day_dt + pd.Timedelta(days=1)
            else:
                ts = day_dt + pd.Timedelta(hours=h)
            rows.append({
                "task": "dayahead",
                "target_day": day_dt,
                "business_day": day_dt,
                "ds": ts,
                "hour_business": h,
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
                "y_true": float(rng.uniform(80, 200)),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), "actual_ledger.parquet")
    df.to_parquet(path, index=False)
    return path


def _make_profile_def(delivery_allowed: bool = True) -> dict:
    """Sample profile definition for postflight tests."""
    return {
        "delivery_allowed": delivery_allowed,
        "allowed_models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
        "excluded_models": {"stage3": SUSPECT_LEAKAGE},
    }


# ======================================================================
#  1. test_stage3_blocked
# ======================================================================


class TestStage3Blocked:
    """Stage3 model must be blocked by leakage sentinel (SUSPECT_LEAKAGE)."""

    def test_stage3_suspect_leakage(self, tmp_path):
        """stage3 model with suspiciously close predictions -> SUSPECT_LEAKAGE."""
        from safety.leakage_sentinel import check_model_leakage

        pred = _make_stage3_pred_ledger(tmp_path, model_name="stage3")
        actual = _make_actual_ledger(tmp_path)
        result = check_model_leakage("stage3", pred, actual)

        assert result["status"] == SUSPECT_LEAKAGE, (
            f"Expected SUSPECT_LEAKAGE for stage3, got {result['status']}: "
            f"{result.get('suspicion_reasons', [])}"
        )

    def test_stage3_blocked_by_delivery_gate(self, tmp_path):
        """stage3 SUSPECT_LEAKAGE model must be blocked by delivery gate."""
        from safety.leakage_sentinel import (
            check_model_leakage,
            is_delivery_allowed,
            run_leakage_sentinel,
        )

        pred = _make_stage3_pred_ledger(tmp_path, model_name="stage3")
        actual = _make_actual_ledger(tmp_path)
        sentinel_result = run_leakage_sentinel(
            trusted_models=["stage3"],
            prediction_ledger_path=pred,
            actual_ledger_path=actual,
        )

        assert sentinel_result["models"][0]["status"] == SUSPECT_LEAKAGE
        allowed = is_delivery_allowed(
            "stage3", sentinel_result, "trusted_delivery",
        )
        assert allowed is False, "stage3 must be blocked in trusted_delivery"

        allowed_research = is_delivery_allowed(
            "stage3", sentinel_result, "research_all_models",
        )
        assert allowed_research is False, "stage3 must be blocked even in research profile"


# ======================================================================
#  2. test_y_true_leakage_detected
# ======================================================================


class TestYTrueLeakageDetected:
    """Predictions suspiciously close to actuals -> SUSPECT_LEAKAGE."""

    def test_y_true_leakage_detected(self, tmp_path):
        """Correlation > 0.995 + sMAPE < 2% + MAE < 10 -> SUSPECT_LEAKAGE."""
        from safety.leakage_sentinel import check_model_leakage

        n = 24
        rng = np.random.default_rng(42)
        # y_true with a strong trend to ensure high correlation
        y_t = 100.0 + np.arange(n) * 20  # 100..560
        # Nearly identical predictions (tiny noise) -> corr ~1.0, sMAPE << 2%, MAE << 10
        y_p = y_t + rng.normal(0, 1.0, n)

        pred = os.path.join(str(tmp_path), "pred_leak.csv")
        rows: list[dict] = []
        for i in range(n):
            h = (i % 24) + 1
            if h == 24:
                ts = pd.Timestamp("2025-07-06 00:00:00")
            else:
                ts = pd.Timestamp("2025-07-05") + pd.Timedelta(hours=h)
            rows.append({
                "model_name": "leaky_model",
                "task": "dayahead",
                "target_day": "2026-07-05",
                "business_day": "2026-07-04",
                "hour_business": h,
                "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "y_pred": float(y_p[i]),
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
            })
        pd.DataFrame(rows).to_csv(pred, index=False)

        actual = _make_actual_ledger(tmp_path, y_true=y_t)

        result = check_model_leakage("leaky_model", pred, actual)
        metrics = result["details"].get("metrics", {})
        suspicion_reasons = result.get("suspicion_reasons", [])

        # At least one of the SUSPECT_LEAKAGE triggers must fire
        assert result["status"] == SUSPECT_LEAKAGE, (
            f"Expected SUSPECT_LEAKAGE, got {result['status']}. "
            f"Metrics: corr={metrics.get('corr_y_pred_y_true')}, "
            f"sMAPE={metrics.get('sMAPE_floor50')}, "
            f"MAE={metrics.get('MAE')}, "
            f"within_1pct={metrics.get('within_1pct_ratio')}. "
            f"Suspicion reasons: {suspicion_reasons}"
        )

        # Verify that correlation is indeed very high (> 0.995)
        corr = metrics.get("corr_y_pred_y_true", 0)
        assert corr > 0.995, (
            f"Expected corr > 0.995 but got {corr}"
        )


# ======================================================================
#  3. test_23_rows_instead_of_24_rejected
# ======================================================================


class Test23RowsRejected:
    """Output CSV with 23 rows must be rejected by postflight."""

    def test_23_rows_rejected(self, tmp_path):
        """Postflight twenty_four_rows check fails for 23-row output."""
        from delivery.postflight import run_postflight

        csv_path = _make_output_csv(tmp_path, n_rows=23)
        result = run_postflight(
            output_path=csv_path,
            target_date="2026-07-05",
            profile_name="trusted_delivery",
        )

        check = result["checks"].get("twenty_four_rows", {})
        assert check.get("passed") is False, (
            f"Expected twenty_four_rows check to fail, got: {check}"
        )
        assert "23" in check.get("detail", ""), (
            f"Expected detail to mention '23 rows', got: {check.get('detail')}"
        )

    def test_24_rows_pass(self, tmp_path):
        """24-row output passes the twenty_four_rows check."""
        from delivery.postflight import run_postflight

        csv_path = _make_output_csv(tmp_path, n_rows=24)
        result = run_postflight(
            output_path=csv_path,
            target_date="2026-07-05",
            profile_name="trusted_delivery",
        )

        check = result["checks"].get("twenty_four_rows", {})
        assert check.get("passed") is True, (
            f"Expected twenty_four_rows check to pass, got: {check}"
        )


# ======================================================================
#  4. test_nan_predictions_detected
# ======================================================================


class TestNanPredictionsDetected:
    """NaN in prediction column must be detected by postflight."""

    def test_nan_predictions_detected(self, tmp_path):
        """Postflight no_nan_in_predictions check fails with NaN in y_pred."""
        from delivery.postflight import run_postflight

        csv_path = _make_output_csv(tmp_path, n_rows=24, nan_hour=12)
        result = run_postflight(
            output_path=csv_path,
            target_date="2026-07-05",
            profile_name="trusted_delivery",
        )

        check = result["checks"].get("no_nan_in_predictions", {})
        assert check.get("passed") is False, (
            f"Expected no_nan_in_predictions check to fail, got: {check}"
        )

    def test_nan_detected_by_leakage_sentinel(self, tmp_path):
        """NaN y_pred values produce a warning in the sentinel result."""
        from safety.leakage_sentinel import check_model_leakage

        n = 24
        y_t = 200.0 + np.arange(n) * 10
        y_p = y_t + np.random.default_rng(42).normal(0, 40, n)
        y_p[5] = np.nan
        y_p[12] = np.nan

        rows: list[dict] = []
        for i in range(n):
            h = (i % 24) + 1
            if h == 24:
                ts = pd.Timestamp("2025-07-06 00:00:00")
            else:
                ts = pd.Timestamp("2025-07-05") + pd.Timedelta(hours=h)
            rows.append({
                "model_name": "nan_model",
                "task": "dayahead",
                "target_day": "2026-07-05",
                "business_day": "2026-07-04",
                "hour_business": h,
                "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "y_pred": float(y_p[i]) if not np.isnan(y_p[i]) else "",
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
            })
        pred_path = os.path.join(str(tmp_path), "pred_nan.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        actual_path = _make_actual_ledger(tmp_path, y_true=y_t)

        result = check_model_leakage("nan_model", pred_path, actual_path)
        assert result["details"].get("nan_y_pred_count", 0) >= 2, (
            f"Expected at least 2 NaN y_pred values, got: "
            f"{result['details'].get('nan_y_pred_count')}"
        )
        assert any("NaN" in w for w in result.get("warnings", [])), (
            "Expected a warning about NaN values in sentinel result"
        )


# ======================================================================
#  5. test_no_training_days_insufficient
# ======================================================================


class TestNoTrainingDaysInsufficient:
    """select_complete_training_days must return NO_VALID_DAYS or
    INSUFFICIENT_DAYS when no or very few complete training days exist."""

    def test_missing_ledgers_no_valid_days(self, tmp_path):
        """Non-existent prediction ledger -> NO_VALID_DAYS."""
        from fusion.adaptive_training_days import select_complete_training_days

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=["model_a"],
            prediction_ledger_path=os.path.join(str(tmp_path), "nonexistent.parquet"),
            actual_ledger_path=os.path.join(str(tmp_path), "nonexistent.parquet"),
        )

        assert result["status"] == _ADAPTIVE_NO_VALID_DAYS, (
            f"Expected {_ADAPTIVE_NO_VALID_DAYS}, got {result['status']}: {result.get('errors', [])}"
        )
        assert len(result.get("errors", [])) >= 1

    def test_empty_prediction_ledger_no_valid_days(self, tmp_path):
        """Empty prediction ledger -> NO_VALID_DAYS."""
        from fusion.adaptive_training_days import select_complete_training_days

        pred_path = os.path.join(str(tmp_path), "empty_pred.parquet")
        act_path = _make_actual_parquet_ledger(tmp_path, ["2026-07-04"])
        pd.DataFrame(columns=[
            "task", "model_name", "target_day", "business_day",
            "hour_business", "y_pred",
        ]).to_parquet(pred_path, index=False)

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=["model_a"],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        assert result["status"] == _ADAPTIVE_NO_VALID_DAYS, (
            f"Expected {_ADAPTIVE_NO_VALID_DAYS} for empty prediction ledger, got {result['status']}"
        )

    def test_insufficient_days_with_few_days(self, tmp_path):
        """Only 3 clean training days -> INSUFFICIENT_DAYS."""
        from fusion.adaptive_training_days import select_complete_training_days

        days = ["2026-07-01", "2026-07-02", "2026-07-03"]
        pred_path = _make_prediction_parquet_ledger(tmp_path, days, ["model_a"])
        act_path = _make_actual_parquet_ledger(tmp_path, days)

        result = select_complete_training_days(
            target_date="2026-07-05",
            trusted_models=["model_a"],
            prediction_ledger_path=pred_path,
            actual_ledger_path=act_path,
        )

        # 3 days < 7 (min_days_for_degraded) -> INSUFFICIENT_DAYS
        assert result["status"] == _ADAPTIVE_INSUFFICIENT_DAYS, (
            f"Expected {_ADAPTIVE_INSUFFICIENT_DAYS} for 3 days, got {result['status']}: "
            f"selected_count={result.get('selected_count')}"
        )
        assert result["selected_count"] == 3


# ======================================================================
#  6. test_forbidden_phrases_in_output
# ======================================================================


class TestForbiddenPhrasesInOutput:
    """Forbidden column names in the prediction ledger must be caught."""

    def test_y_true_column_in_prediction_ledger_detected(self, tmp_path):
        """y_true column in prediction ledger -> INVALID_SCHEMA."""
        from safety.leakage_sentinel import check_model_leakage

        n = 24
        rng = np.random.default_rng(42)
        y_t = 200.0 + rng.uniform(0, 300, n)
        y_p = y_t + rng.normal(0, 40, n)

        rows: list[dict] = []
        for i in range(n):
            h = (i % 24) + 1
            if h == 24:
                ts = pd.Timestamp("2025-07-06 00:00:00")
            else:
                ts = pd.Timestamp("2025-07-05") + pd.Timedelta(hours=h)
            rows.append({
                "model_name": "test_model",
                "task": "dayahead",
                "target_day": "2026-07-05",
                "business_day": "2026-07-04",
                "hour_business": h,
                "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "y_pred": float(y_p[i]),
                "y_true": float(y_t[i]),  # forbidden column
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
            })
        pred_path = os.path.join(str(tmp_path), "pred_forbidden.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        actual_path = _make_actual_ledger(tmp_path, y_true=y_t)

        result = check_model_leakage("test_model", pred_path, actual_path)

        assert result["status"] == INVALID_SCHEMA, (
            f"Expected INVALID_SCHEMA for y_true column in prediction ledger, "
            f"got {result['status']}"
        )
        assert result["checks"].get("no_y_true_in_prediction_ledger") is False, (
            "Expected no_y_true_in_prediction_ledger check to fail"
        )

    def test_target_column_in_prediction_ledger_detected(self, tmp_path):
        """"target" column in prediction ledger -> INVALID_SCHEMA."""
        from safety.leakage_sentinel import check_model_leakage

        n = 24
        rng = np.random.default_rng(42)
        y_t = 200.0 + rng.uniform(0, 300, n)
        y_p = y_t + rng.normal(0, 40, n)

        rows: list[dict] = []
        for i in range(n):
            h = (i % 24) + 1
            if h == 24:
                ts = pd.Timestamp("2025-07-06 00:00:00")
            else:
                ts = pd.Timestamp("2025-07-05") + pd.Timedelta(hours=h)
            rows.append({
                "model_name": "test_model",
                "task": "dayahead",
                "target_day": "2026-07-05",
                "business_day": "2026-07-04",
                "hour_business": h,
                "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "y_pred": float(y_p[i]),
                "target": float(y_t[i]),  # forbidden column
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
            })
        pred_path = os.path.join(str(tmp_path), "pred_target_col.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        actual_path = _make_actual_ledger(tmp_path, y_true=y_t)

        result = check_model_leakage("test_model", pred_path, actual_path)

        assert result["status"] == INVALID_SCHEMA, (
            f"Expected INVALID_SCHEMA for 'target' column in prediction ledger, "
            f"got {result['status']}"
        )


# ======================================================================
#  7. test_non_delivery_profile_blocks
# ======================================================================


class TestNonDeliveryProfileBlocks:
    """Non-delivery profile must be blocked by postflight."""

    def test_non_delivery_profile_rejected(self, tmp_path):
        """delivery_allowed=False profile -> profile_delivery_allowed check fails."""
        from delivery.postflight import run_postflight

        csv_path = _make_output_csv(tmp_path, n_rows=24)
        profile_def = _make_profile_def(delivery_allowed=False)
        result = run_postflight(
            output_path=csv_path,
            target_date="2026-07-05",
            profile_name="balanced_candidate",
            profile_def=profile_def,
        )

        check = result["checks"].get("profile_delivery_allowed", {})
        assert check.get("passed") is False, (
            f"Expected profile_delivery_allowed check to fail, got: {check}"
        )

    def test_delivery_profile_allowed(self, tmp_path):
        """delivery_allowed=True profile -> profile_delivery_allowed check passes."""
        from delivery.postflight import run_postflight

        csv_path = _make_output_csv(tmp_path, n_rows=24)
        profile_def = _make_profile_def(delivery_allowed=True)
        result = run_postflight(
            output_path=csv_path,
            target_date="2026-07-05",
            profile_name="trusted_delivery",
            profile_def=profile_def,
        )

        check = result["checks"].get("profile_delivery_allowed", {})
        assert check.get("passed") is True, (
            f"Expected profile_delivery_allowed check to pass, got: {check}"
        )


# ======================================================================
#  8. test_conservative_quarantine_allowed_balanced
# ======================================================================


class TestConservativeQuarantineAllowedBalanced:
    """CONSERVATIVE_QUARANTINE models allowed in balanced_candidate profile,
    but blocked in trusted_delivery."""

    def test_conservative_quarantine_blocked_in_trusted_delivery(self, tmp_path):
        """CONSERVATIVE_QUARANTINE blocked in trusted_delivery profile."""
        from safety.leakage_sentinel import check_model_leakage, is_delivery_allowed, run_leakage_sentinel

        n = 24
        y_t = 100.0 + np.arange(n) * 20
        noise = np.random.default_rng(42).normal(0, 2.0, n)
        y_p = y_t * 1.05 + 5.0 + noise

        rows: list[dict] = []
        for i in range(n):
            h = (i % 24) + 1
            if h == 24:
                ts = pd.Timestamp("2025-07-06 00:00:00")
            else:
                ts = pd.Timestamp("2025-07-05") + pd.Timedelta(hours=h)
            rows.append({
                "model_name": "quarantine_model",
                "task": "dayahead",
                "target_day": "2026-07-05",
                "business_day": "2026-07-04",
                "hour_business": h,
                "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "y_pred": float(y_p[i]),
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
            })
        pred_path = os.path.join(str(tmp_path), "pred_quarantine.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        actual_path = _make_actual_ledger(tmp_path, y_true=y_t)

        sr = check_model_leakage("quarantine_model", pred_path, actual_path)
        assert sr["status"] == CONSERVATIVE_QUARANTINE, (
            f"Expected CONSERVATIVE_QUARANTINE, got {sr['status']}: "
            f"{sr.get('suspicion_reasons', [])}"
        )

        sentinel_result = {"models": [sr], "summary": {CONSERVATIVE_QUARANTINE: 1}}
        allowed_trusted = is_delivery_allowed(
            "quarantine_model", sentinel_result, "trusted_delivery",
        )
        assert allowed_trusted is False, (
            "CONSERVATIVE_QUARANTINE should be blocked in trusted_delivery"
        )

    def test_conservative_quarantine_allowed_in_balanced_candidate(self, tmp_path):
        """CONSERVATIVE_QUARANTINE allowed in balanced_candidate profile."""
        from safety.leakage_sentinel import check_model_leakage, is_delivery_allowed

        n = 24
        y_t = 100.0 + np.arange(n) * 20
        noise = np.random.default_rng(42).normal(0, 2.0, n)
        y_p = y_t * 1.05 + 5.0 + noise

        rows: list[dict] = []
        for i in range(n):
            h = (i % 24) + 1
            if h == 24:
                ts = pd.Timestamp("2025-07-06 00:00:00")
            else:
                ts = pd.Timestamp("2025-07-05") + pd.Timedelta(hours=h)
            rows.append({
                "model_name": "quarantine_model",
                "task": "dayahead",
                "target_day": "2026-07-05",
                "business_day": "2026-07-04",
                "hour_business": h,
                "ds": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "y_pred": float(y_p[i]),
                "period": "1_8" if h <= 8 else "9_16" if h <= 16 else "17_24",
            })
        pred_path = os.path.join(str(tmp_path), "pred_quarantine2.csv")
        pd.DataFrame(rows).to_csv(pred_path, index=False)
        actual_path = _make_actual_ledger(tmp_path, y_true=y_t)

        sr = check_model_leakage("quarantine_model", pred_path, actual_path)
        assert sr["status"] == CONSERVATIVE_QUARANTINE, (
            f"Expected CONSERVATIVE_QUARANTINE, got {sr['status']}"
        )

        sentinel_result = {"models": [sr], "summary": {CONSERVATIVE_QUARANTINE: 1}}
        allowed_balanced = is_delivery_allowed(
            "quarantine_model", sentinel_result, "balanced_candidate",
        )
        assert allowed_balanced is True, (
            "CONSERVATIVE_QUARANTINE should be allowed in balanced_candidate profile"
        )

    def test_trust_gate_allows_conservative_quarantine_in_balanced(self):
        """The trust gate permits CONSERVATIVE_QUARANTINE in balanced_candidate."""
        from fusion.trust_gated_regime_bgew import _apply_trust_gate

        allowed, blocked, warnings = _apply_trust_gate(
            model_names=["model_a", "model_b"],
            trusted_models=["model_a"],
            profile_name="balanced_candidate",
            model_trust_states={
                "model_b": TRUST_STATE_CONSERVATIVE_QUARANTINE,
            },
        )

        assert "model_b" in allowed, (
            f"CONSERVATIVE_QUARANTINE model_b should be allowed in "
            f"balanced_candidate, but was blocked: {blocked}"
        )
        assert "model_b" not in blocked, (
            f"model_b should not be in blocked list: {blocked}"
        )

    def test_trust_gate_blocks_conservative_quarantine_in_trusted(self):
        """The trust gate blocks CONSERVATIVE_QUARANTINE in trusted_delivery."""
        from fusion.trust_gated_regime_bgew import _apply_trust_gate

        allowed, blocked, warnings = _apply_trust_gate(
            model_names=["model_a", "model_b"],
            trusted_models=["model_a"],
            profile_name="trusted_delivery",
            model_trust_states={
                "model_b": TRUST_STATE_CONSERVATIVE_QUARANTINE,
            },
        )

        assert "model_b" in blocked, (
            f"CONSERVATIVE_QUARANTINE model_b should be blocked in "
            f"trusted_delivery, but was allowed: {allowed}"
        )
        assert "model_b" not in allowed


# ======================================================================
#  9. test_regime_bgew_fallback_on_low_data
# ======================================================================


class TestRegimeBgewFallbackOnLowData:
    """run_trust_gated_regime_bgew falls back from regime_bgew when
    training data is limited."""

    def _make_training_ledgers(
        self,
        n_training_days: int,
        target_date: str = "2026-07-05",
        models: list[str] | None = None,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create prediction and actual ledgers with *n_training_days*
        of historical data plus the target date predictions.

        Returns (prediction_ledger, actual_ledger) as DataFrames.
        """
        if models is None:
            models = ["cfg05", "boost_model"]

        rng = np.random.default_rng(seed)
        target_dt = pd.Timestamp(target_date)

        # Build training days: n_training_days consecutive days ending
        # the day before target_date
        train_end = target_dt - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=n_training_days - 1)
        train_days = [
            (train_start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n_training_days)
        ]

        pred_rows: list[pd.DataFrame] = []
        actual_rows: list[pd.DataFrame] = []

        for day_str in train_days:
            day_dt = pd.Timestamp(day_str)
            timestamps = pd.date_range(
                f"{day_str} 01:00", periods=24, freq="h",
            )
            # 24-hour timestamps: hour 24 is next day 00:00
            ts_list = list(timestamps)
            ts_list[-1] = ts_list[-1] + pd.Timedelta(hours=1)  # hour 24 -> next day

            for model in models:
                prices = rng.uniform(80, 250, 24)
                pred_rows.append(pd.DataFrame({
                    "task": ["dayahead"] * 24,
                    "model_name": [model] * 24,
                    "target_day": [day_dt] * 24,
                    "business_day": [day_dt] * 24,
                    "ds": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in ts_list],
                    "hour_business": list(range(1, 25)),
                    "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                    "y_pred": prices,
                }))

            actual_prices = rng.uniform(80, 250, 24)
            actual_rows.append(pd.DataFrame({
                "task": ["dayahead"] * 24,
                "target_day": [day_dt] * 24,
                "business_day": [day_dt] * 24,
                "ds": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in ts_list],
                "hour_business": list(range(1, 25)),
                "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                "y_true": actual_prices,
            }))

        # Add target date predictions (needed for target_preds lookup)
        target_ts_list = list(
            pd.date_range(f"{target_date} 01:00", periods=24, freq="h"),
        )
        target_ts_list[-1] = target_ts_list[-1] + pd.Timedelta(hours=1)
        for model in models:
            target_prices = rng.uniform(80, 250, 24)
            pred_rows.append(pd.DataFrame({
                "task": ["dayahead"] * 24,
                "model_name": [model] * 24,
                "target_day": [target_dt] * 24,
                "business_day": [target_dt] * 24,
                "ds": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in target_ts_list],
                "hour_business": list(range(1, 25)),
                "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                "y_pred": target_prices,
            }))

        pred_df = pd.concat(pred_rows, ignore_index=True)
        act_df = pd.concat(actual_rows, ignore_index=True)

        return pred_df, act_df

    def test_fallback_to_period_bgew_with_6_days(self):
        """With 6 training days (< 10 for regime, >= 5 for period),
        method should be 'period_bgew' (not 'regime_bgew')."""
        from fusion.trust_gated_regime_bgew import run_trust_gated_regime_bgew

        pred_df, act_df = self._make_training_ledgers(n_training_days=6)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-05",
            trusted_models=["cfg05", "boost_model"],
            prediction_ledger_path="",  # not used when DataFrames provided
            actual_ledger_path="",      # not used when DataFrames provided
            prediction_ledger=pred_df,
            actual_ledger=act_df,
        )

        assert result["method"] != "regime_bgew", (
            f"Expected fallback from regime_bgew, got method={result['method']}. "
            f"Training days used: {result.get('training_days_used')}. "
            f"Fallback chain: {result.get('fallback_chain')}"
        )
        # With 6 days (>= min_training_days_for_period=5) we expect period_bgew
        assert result["method"] == "period_bgew", (
            f"Expected period_bgew with 6 training days, got {result['method']}. "
            f"Training days: {result.get('training_days_used')}. "
            f"Fallback chain: {result.get('fallback_chain')}"
        )
        assert result["success"] is True, (
            f"Expected success=True, got: {result.get('errors', [])}"
        )

    def test_fallback_to_equal_weight_with_3_days(self):
        """With only 3 training days (< 5 for period, < 10 for regime),
        method should be 'equal_weight'."""
        from fusion.trust_gated_regime_bgew import run_trust_gated_regime_bgew

        pred_df, act_df = self._make_training_ledgers(n_training_days=3)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-05",
            trusted_models=["cfg05", "boost_model"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=pred_df,
            actual_ledger=act_df,
        )

        assert result["method"] != "regime_bgew", (
            f"Expected fallback from regime_bgew, got method={result['method']}. "
            f"Training days used: {result.get('training_days_used')}"
        )
        # With 3 days (< min_training_days_for_period=5) expect equal_weight
        assert result["method"] == "equal_weight", (
            f"Expected equal_weight with 3 training days, got {result['method']}. "
            f"Training days: {result.get('training_days_used')}. "
            f"Fallback chain: {result.get('fallback_chain')}"
        )

    def test_regime_bgew_works_with_sufficient_data(self):
        """With 15 training days (>= 10 for regime), method should be
        'regime_bgew'."""
        from fusion.trust_gated_regime_bgew import run_trust_gated_regime_bgew

        pred_df, act_df = self._make_training_ledgers(n_training_days=15)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-05",
            trusted_models=["cfg05", "boost_model"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=pred_df,
            actual_ledger=act_df,
        )

        assert result["method"] == "regime_bgew", (
            f"Expected regime_bgew with 15 training days, got {result['method']}. "
            f"Training days used: {result.get('training_days_used')}. "
            f"Fallback chain: {result.get('fallback_chain')}"
        )
        assert result["success"] is True


# ======================================================================
#  10. test_claim_guard_violations
# ======================================================================


class TestClaimGuardViolations:
    """Claim guard detects forbidden delivery claims in reports."""

    def test_claim_guard_detects_production_smape_claim(self, tmp_path):
        """Forbidden '2.97% production' claim detected as violation."""
        from scripts.validate_delivery_claims import run_claim_guard

        report_dir = os.path.join(str(tmp_path), "reports")
        os.makedirs(report_dir, exist_ok=True)

        # Write a report with a forbidden claim (no caveat)
        report_path = os.path.join(report_dir, "results.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Delivery Results\n\n")
            f.write("Our model achieves 2.97% production sMAPE on the test set.\n")

        result = run_claim_guard(
            report_dir=report_dir,
            readme_path=os.path.join(str(tmp_path), "README.md"),
            profiles_path=os.path.join(str(tmp_path), "profiles.yaml"),
        )

        assert len(result["violations"]) >= 1, (
            f"Expected at least 1 violation, got {len(result['violations'])}: "
            f"{result.get('violations', [])}"
        )
        violation_labels = [v["label"] for v in result["violations"]]
        assert "production_sMAPE_2_97" in violation_labels, (
            f"Expected production_sMAPE_2_97 violation, got: {violation_labels}"
        )

    def test_claim_guard_detects_stage3_production_claim(self, tmp_path):
        """Forbidden 'stage3 production ready' claim detected as violation."""
        from scripts.validate_delivery_claims import run_claim_guard

        report_dir = os.path.join(str(tmp_path), "reports")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, "stage3_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Stage3 Evaluation\n\n")
            f.write("stage3 is production ready for delivery.\n")

        result = run_claim_guard(
            report_dir=report_dir,
            readme_path=os.path.join(str(tmp_path), "README.md"),
            profiles_path=os.path.join(str(tmp_path), "profiles.yaml"),
        )

        assert len(result["violations"]) >= 1, (
            f"Expected at least 1 violation, got {len(result['violations'])}"
        )
        violation_labels = [v["label"] for v in result["violations"]]
        assert "stage3_production_readiness" in violation_labels, (
            f"Expected stage3_production_readiness violation, got: {violation_labels}"
        )

    def test_claim_guard_caveated_claim_is_warning(self, tmp_path):
        """Forbidden claim with research caveat becomes a warning, not violation."""
        from scripts.validate_delivery_claims import run_claim_guard

        report_dir = os.path.join(str(tmp_path), "reports")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, "caveated_results.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Research Results\n\n")
            f.write("Our model achieves 2.97% production sMAPE (research only, ")
            f.write("not delivery ready).\n")

        result = run_claim_guard(
            report_dir=report_dir,
            readme_path=os.path.join(str(tmp_path), "README.md"),
            profiles_path=os.path.join(str(tmp_path), "profiles.yaml"),
        )

        # The claim appears but with "research only" caveat -> should be warning
        violations = result.get("violations", [])
        warnings = result.get("warnings", [])

        # With "research only" caveat, the claim severity drops to warning
        assert len(violations) == 0, (
            f"Expected 0 violations for caveated claim, got: {violations}"
        )
        claim_warnings = [w for w in warnings if w.get("label") == "production_sMAPE_2_97"]
        assert len(claim_warnings) >= 1, (
            f"Expected production_sMAPE_2_97 warning, got warnings: "
            f"{[w.get('label') for w in warnings]}"
        )

    def test_claim_guard_summary_reflects_status(self, tmp_path):
        """When violations exist, summary status is P46_CLAIM_GUARD_FAILED."""
        from scripts.validate_delivery_claims import run_claim_guard

        report_dir = os.path.join(str(tmp_path), "reports")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, "delivery_claim.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Claims\n\n")
            f.write("69.96% production improvement confirmed.\n")

        result = run_claim_guard(
            report_dir=report_dir,
            readme_path=os.path.join(str(tmp_path), "README.md"),
            profiles_path=os.path.join(str(tmp_path), "profiles.yaml"),
        )

        assert result["summary"]["p46_status"] == "P46_CLAIM_GUARD_FAILED", (
            f"Expected P46_CLAIM_GUARD_FAILED, got {result['summary']['p46_status']}"
        )
        assert result["summary"]["total_violations"] >= 1

    def test_claim_guard_no_false_positives_for_clean_reports(self, tmp_path):
        """Clean reports without forbidden phrases produce no violations."""
        from scripts.validate_delivery_claims import run_claim_guard

        report_dir = os.path.join(str(tmp_path), "reports")
        os.makedirs(report_dir, exist_ok=True)

        report_path = os.path.join(report_dir, "clean_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Clean Report\n\n")
            f.write("Model evaluation in progress. Results pending.\n")

        result = run_claim_guard(
            report_dir=report_dir,
            readme_path=os.path.join(str(tmp_path), "README.md"),
            profiles_path=os.path.join(str(tmp_path), "profiles.yaml"),
        )

        assert len(result["violations"]) == 0, (
            f"Expected 0 violations for clean report, got: {result['violations']}"
        )
        # With 0 violations and possibly 0 warnings, status should be PASS
        assert result["summary"]["p46_status"] == "P46_CLAIM_GUARD_PASS", (
            f"Expected P46_CLAIM_GUARD_PASS, got {result['summary']['p46_status']}"
        )
