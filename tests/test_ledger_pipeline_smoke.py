"""
tests/test_ledger_pipeline_smoke.py — Full synthetic ledger smoke test.

Validates the end-to-end flow:
    prediction output → corrected output → corrected ledger → actual ledger
    → ledger_fusion → fusion ledger → weight ledger

All data is synthetic.  No files are committed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.business_day import add_business_time_columns
from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    CORRECTED_LEDGER_COLUMNS,
    ACTUAL_LEDGER_COLUMNS,
    FUSION_LEDGER_COLUMNS,
    WEIGHT_LEDGER_COLUMNS,
)
from fusion.engine import READY_DRY_RUN
from ledgers.prediction_ledger import (
    append_predictions_to_ledger,
    append_corrected_predictions_to_ledger,
)
from ledgers.actual_ledger import append_actuals_to_ledger
from pipelines.ledger_fusion import run_ledger_fusion


def _synthetic_p2_output(n_hours: int = 24) -> pd.DataFrame:
    """Generate synthetic P2 prediction output."""
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    rows = []
    for model in ["cfg05", "best_two_average"]:
        df = pd.DataFrame({
            "task": "dayahead",
            "model_name": model,
            "target_day": "2026-07-04",
            "ds": ts,
            "y_pred": rng.uniform(80, 200, n_hours),
            "source_confidence": 0.9,
            "model_version": "1.0.0",
        })
        df = add_business_time_columns(df, timestamp_col="ds")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _synthetic_p3_corrected(p2_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic P3 corrected output from P2 output."""
    rng = np.random.default_rng(42)
    df = p2_df.copy()
    df["y_pred_raw"] = df["y_pred"]
    correction = rng.uniform(-5, 5, len(df))
    df["y_pred_corrected"] = df["y_pred"] + correction
    df["residual_delta"] = correction
    df["correction_applied"] = True
    df["correction_module"] = "p5m_residual_noop"
    df["risk_source"] = "NONE"
    df["reason_codes"] = "NO_CORRECTION_NEEDED"
    df["correction_version"] = "1.0.0"
    df = df.drop(columns=["y_pred"])
    return df


def _synthetic_actuals(n_hours: int = 24) -> pd.DataFrame:
    """Generate synthetic actuals for past days."""
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    rows = []
    for day in ["2026-07-01", "2026-07-02", "2026-07-03"]:
        ts = pd.date_range(f"{day} 01:00", periods=n_hours, freq="h")
        df = pd.DataFrame({
            "task": "dayahead",
            "target_day": day,
            "ds": ts,
            "y_true": rng.uniform(80, 200, n_hours),
            "actual_source": "market_feed",
        })
        df = add_business_time_columns(df, timestamp_col="ds")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


class TestLedgerPipelineSmoke:
    """Full synthetic ledger smoke test."""

    def test_end_to_end_ledger_flow(self):
        """Complete prediction→corrected→actual→fusion→weight flow."""
        # Step 1: P2 predictions → prediction ledger
        p2 = _synthetic_p2_output(24)
        pred_ledger = append_predictions_to_ledger(p2, run_id="smoke_pred")
        assert list(pred_ledger.columns) == PREDICTION_LEDGER_COLUMNS
        assert len(pred_ledger) == 48  # 2 models * 24 hours

        # Step 2: P2 predictions → P3 corrected → corrected ledger
        p3 = _synthetic_p3_corrected(p2)
        corrected_ledger = append_corrected_predictions_to_ledger(p3, run_id="smoke_corrected")
        assert list(corrected_ledger.columns) == CORRECTED_LEDGER_COLUMNS
        assert len(corrected_ledger) == 48

        # Step 3: Actuals → actual ledger
        actuals = _synthetic_actuals(24)
        actual_ledger = append_actuals_to_ledger(actuals, run_id="smoke_actuals")
        assert list(actual_ledger.columns) == ACTUAL_LEDGER_COLUMNS
        assert len(actual_ledger) == 72  # 3 days * 24 hours

        # Step 4: Ledger fusion → fusion ledger + weight ledger
        result = run_ledger_fusion(
            corrected_ledger_df=corrected_ledger,
            actual_ledger_df=actual_ledger,
            method="equal_weight",
            allow_dry_run=True,
            run_id="smoke_fusion",
            readiness_status={
                "cfg05": READY_DRY_RUN,
                "best_two_average": READY_DRY_RUN,
            },
        )

        # Check summary
        assert result["fusion_rows"] == 24
        assert result["weight_rows"] == 48  # 24 hours * 2 models
        assert result["run_id"] == "smoke_fusion"

        # Check fusion ledger
        assert result["fusion_ledger_size"] >= 24
        assert result["weight_ledger_size"] >= 48

    def test_ledger_fusion_without_actuals(self):
        """Fusion works without actuals (falls back to equal_weight)."""
        p2 = _synthetic_p2_output(24)
        p3 = _synthetic_p3_corrected(p2)
        corrected_ledger = append_corrected_predictions_to_ledger(p3, run_id="test")

        result = run_ledger_fusion(
            corrected_ledger_df=corrected_ledger,
            actual_ledger_df=None,
            method="equal_weight",
            allow_dry_run=True,
            run_id="fusion_no_actuals",
            readiness_status={
                "cfg05": READY_DRY_RUN,
                "best_two_average": READY_DRY_RUN,
            },
        )

        assert result["fusion_rows"] == 24
        assert result["weight_rows"] == 48

    def test_backfill_with_tmp_path(self, tmp_path):
        """run_ledger_backfill with tmp_path saves to disk."""
        from pipelines.ledger_backfill import run_ledger_backfill

        p2 = _synthetic_p2_output(24)
        p3 = _synthetic_p3_corrected(p2)
        actuals = _synthetic_actuals(24)

        summary = run_ledger_backfill(
            prediction_df=p2,
            corrected_df=p3,
            actuals_df=actuals,
            ledger_dir=str(tmp_path),
            run_id="backfill_test",
        )

        assert summary["prediction_rows"] == 48
        assert summary["corrected_rows"] == 48
        assert summary["actual_rows"] == 72

        # Check files exist
        assert (tmp_path / "prediction_ledger.csv").exists()
        assert (tmp_path / "corrected_ledger.csv").exists()
        assert (tmp_path / "actual_ledger.csv").exists()

    def test_dedup_after_backfill(self, tmp_path):
        """Backfill twice produces same size ledger (dedup)."""
        from pipelines.ledger_backfill import run_ledger_backfill

        p2 = _synthetic_p2_output(24)
        p3 = _synthetic_p3_corrected(p2)

        # First backfill
        s1 = run_ledger_backfill(
            prediction_df=p2,
            corrected_df=p3,
            ledger_dir=str(tmp_path),
            run_id="first",
        )

        # Second backfill with same data
        s2 = run_ledger_backfill(
            prediction_df=p2,
            corrected_df=p3,
            ledger_dir=str(tmp_path),
            run_id="second",
        )

        # Sizes should be the same
        assert s2["prediction_ledger_size"] == s1["prediction_ledger_size"]
        assert s2["corrected_ledger_size"] == s1["corrected_ledger_size"]
