"""
tests/test_multi_day_backfill_smoke.py — Multi-day backfill smoke tests.

Validates:
    1. 3-day smoke returns PASS quickly
    2. 30-day smoke row counts match expected values
    3. corrected ledger has no duplicate keys
    4. fusion ledger has no duplicate keys
    5. weight ledger has no duplicate keys
    6. Running same smoke twice is idempotent
    7. All validators pass
    8. Weight rows sum to 1 for every hour
    9. Synthetic actual ledger validates
    10. filter_actuals_for_training never returns business_day >= target_day
    11. bgew_skeleton with insufficient actuals falls back safely
    12. No REAL label without verified artifacts
    13. Forbidden files check passes
    14. CLI exits 0 with tmp_path ledger_dir
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from pipelines.multi_day_backfill_smoke import run_multi_day_backfill_smoke
from ledgers.store import load_ledger, validate_ledger_keys
from data.schema import (
    CORRECTED_LEDGER_KEY,
    FUSION_LEDGER_KEY,
    WEIGHT_LEDGER_KEY,
    ACTUAL_LEDGER_KEY,
)


class TestMultiDayBackfillSmoke:
    """Contract: run_multi_day_backfill_smoke basic behavior."""

    def test_3_day_smoke_returns_pass(self):
        """3-day smoke returns PASS quickly."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        assert summary["overall_status"] == "PASS"

    def test_30_day_row_counts(self):
        """30-day smoke row counts match expected values."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=30, generate_synthetic_actuals=True,
        )
        # Predictions: 30 * 2 models * 24 hours = 1440
        assert summary["prediction_rows_total"] == 1440
        # Corrected: same as predictions
        assert summary["corrected_rows_total"] == 1440
        # Fusion: 30 * 24 = 720
        assert summary["fusion_rows_total"] == 720
        # Weights: 30 * 2 * 24 = 1440
        assert summary["weight_rows_total"] == 1440
        # Final: 30 * 24 = 720
        assert summary["final_rows_total"] == 720

    def test_corrected_ledger_no_duplicate_keys(self):
        """Corrected ledger has no duplicate keys."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=5, generate_synthetic_actuals=True,
        )
        assert summary["key_uniqueness_check"]["corrected_ledger"] == "PASS"

    def test_fusion_ledger_no_duplicate_keys(self):
        """Fusion ledger has no duplicate keys."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=5, generate_synthetic_actuals=True,
        )
        assert summary["key_uniqueness_check"]["fusion_ledger"] == "PASS"

    def test_weight_ledger_no_duplicate_keys(self):
        """Weight ledger has no duplicate keys."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=5, generate_synthetic_actuals=True,
        )
        assert summary["key_uniqueness_check"]["weight_ledger"] == "PASS"

    def test_actual_ledger_no_duplicate_keys(self):
        """Actual ledger has no duplicate keys."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=5, generate_synthetic_actuals=True,
        )
        assert summary["key_uniqueness_check"]["actual_ledger"] == "PASS"

    def test_idempotent_dedup(self):
        """Running same smoke twice produces same ledger sizes (dedup works)."""
        # Run 3-day smoke once
        summary1 = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        assert summary1["idempotency_check"] == "PASS"

    def test_all_validators_pass(self):
        """All validators pass."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        expected = {"prediction_validator", "residual_validator",
                     "fusion_validator", "final_output_validator",
                     "actual_ledger_validator"}
        assert expected.issubset(set(summary["validators_passed"]))

    def test_weights_sum_to_1(self):
        """Weight rows sum to 1 for every task/target_day/hour."""
        # Re-construct ledgers by running the pipeline
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_multi_day_backfill_smoke(
                start_day="2026-07-01", n_days=3,
                ledger_dir=tmp, generate_synthetic_actuals=True,
            )
            wl_path = os.path.join(tmp, "weight_ledger.csv")
            assert os.path.isfile(wl_path)
            wl = load_ledger(wl_path, columns=None)
            if len(wl) > 0:
                groups = wl.groupby(["task", "target_day", "hour_business"])
                for (task, td, hb), group in groups:
                    total = group["weight"].sum()
                    assert abs(total - 1.0) < 1e-4, (
                        f"({task}, {td}, hb={hb}) weights sum to {total}"
                    )

    def test_actual_ledger_validates(self):
        """Synthetic actual ledger validates successfully."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        assert "actual_ledger_validator" in summary["validators_passed"]
        assert summary["actual_ledger_rows"] == 3 * 24  # 3 days * 24 hours

    def test_no_leakage(self):
        """filter_actuals_for_training never returns business_day >= target_day."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        assert summary["no_leakage_check"] == "PASS"

    def test_bgew_fallback_safe(self):
        """bgew_skeleton with insufficient actuals falls back safely."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3,
            fusion_method="bgew_skeleton",
            generate_synthetic_actuals=True,
        )
        # Should still pass (bgew_skeleton will see some actuals for
        # later days, fall back gracefully for earlier days)
        assert summary["overall_status"] == "PASS"

    def test_no_real_label(self):
        """No REAL label appears without verified artifacts."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        assert "REAL" not in summary["mode_label"]

    def test_forbidden_files_check_pass(self):
        """Forbidden files check passes by default."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3,
        )
        assert summary["forbidden_files_check"] == "PASS"

    def test_mode_labels_correct(self):
        """Default mode_label contains expected labels."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3, generate_synthetic_actuals=True,
        )
        assert "DRY_RUN" in summary["mode_label"]
        assert "STRUCTURAL_ONLY" in summary["mode_label"]
        assert "DATA_MISSING" in summary["mode_label"]
        assert "RULE_FALLBACK" in summary["mode_label"]

    def test_without_actuals_still_passes(self):
        """Smoke without synthetic actuals still passes."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3,
            generate_synthetic_actuals=False,
        )
        assert summary["overall_status"] == "PASS"
        assert summary["actual_ledger_rows"] == 0

    def test_ledger_writes_to_tmp_path(self, tmp_path):
        """Ledgers are written to tmp_path correctly."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-01", n_days=3,
            ledger_dir=str(tmp_path),
            generate_synthetic_actuals=True,
        )
        assert summary["overall_status"] == "PASS"
        assert os.path.isfile(os.path.join(str(tmp_path), "corrected_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "fusion_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "weight_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "actual_ledger.csv"))


class TestMultiDayBackfillCLI:
    """Contract: CLI entry point."""

    def test_cli_3_day_exits_0(self):
        """CLI 3-day smoke exits 0."""
        from scripts.run_multi_day_backfill_smoke import main
        exit_code = main([
            "--start-day", "2026-07-01", "--n-days", "3",
            "--no-production", "--no-generate-synthetic-actuals",
        ])
        assert exit_code == 0

    def test_cli_with_tmp_path_ledger(self, tmp_path):
        """CLI with tmp_path ledger_dir exits 0."""
        from scripts.run_multi_day_backfill_smoke import main
        exit_code = main([
            "--start-day", "2026-07-01", "--n-days", "3",
            "--ledger-dir", str(tmp_path),
            "--no-production",
        ])
        assert exit_code == 0
        assert os.path.isfile(os.path.join(str(tmp_path), "corrected_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "fusion_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "weight_ledger.csv"))


class TestMultiDayBackfillEdgeCases:
    """Contract: edge cases."""

    def test_1_day_minimal(self):
        """1-day smoke with no actuals passes."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-07-04", n_days=1,
            generate_synthetic_actuals=False,
        )
        assert summary["overall_status"] == "PASS"
        assert summary["prediction_rows_total"] == 48  # 1d * 2 models * 24h
        assert summary["fusion_rows_total"] == 24

    def test_actual_ledger_30_day_size(self):
        """30-day actual ledger has 720 rows."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=30, generate_synthetic_actuals=True,
        )
        assert summary["actual_ledger_rows"] == 720  # 30 * 24

    def test_corrected_ledger_30_day_size(self):
        """30-day corrected ledger has 1440 rows."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=30, generate_synthetic_actuals=True,
        )
        assert summary["corrected_ledger_rows"] == 1440  # 30 * 2 * 24

    def test_fusion_ledger_30_day_size(self):
        """30-day fusion ledger has 720 rows."""
        summary = run_multi_day_backfill_smoke(
            start_day="2026-06-01", n_days=30, generate_synthetic_actuals=True,
        )
        assert summary["fusion_ledger_rows"] == 720  # 30 * 24
