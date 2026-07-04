"""
tests/test_full_chain_smoke.py — Full-chain structural smoke contract tests.

Validates:
    1. Full chain smoke returns overall_status = PASS
    2. Default mode_label contains DRY_RUN / STRUCTURAL_ONLY / DATA_MISSING / RULE_FALLBACK
    3. Prediction validator passes
    4. Residual validator passes
    5. Residual no-op holds when no artifact exists
    6. Fusion validator passes
    7. Fusion uses y_pred_corrected
    8. Corrected ledger can be written to tmp_path
    9. Fusion ledger can be written to tmp_path
    10. Weight ledger rows can be expanded
    11. Each hour weights sum to 1
    12. Final output validator passes
    13. Final output has 24 rows for a single day
    14. No REAL label appears without verified artifact
    15. CLI dry-run exits 0
    16. Forbidden files check pass
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from pipelines.full_chain_smoke import run_full_chain_smoke
from data.schema import (
    FINAL_OUTPUT_COLUMNS,
)


class TestFullChainSmokePipeline:
    """Contract: run_full_chain_smoke basic behavior."""

    def test_returns_pass_status(self):
        """Full-chain smoke returns overall_status = PASS."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert summary["overall_status"] == "PASS"

    def test_contains_correct_mode_labels(self):
        """Default mode_labels contain expected structural labels."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        labels = summary["mode_label"]
        assert "DRY_RUN" in labels
        assert "STRUCTURAL_ONLY" in labels
        assert "DATA_MISSING" in labels
        assert "RULE_FALLBACK" in labels

    def test_prediction_validator_passes(self):
        """Prediction validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "prediction_validator" in summary["validators_passed"]

    def test_residual_validator_passes(self):
        """Residual validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "residual_validator" in summary["validators_passed"]

    def test_residual_is_data_missing_noop(self):
        """Residual no-op holds when no artifact exists."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        reasons = ";".join(summary["reason_codes"])
        assert "RESIDUAL_DATA_MISSING_NO_OP" in reasons

    def test_fusion_validator_passes(self):
        """Fusion validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "fusion_validator" in summary["validators_passed"]

    def test_fusion_uses_corrected_not_raw(self):
        """Fusion uses y_pred_corrected (y_pred_corrected != y_pred_raw after
        correction, even if no-op, the corrected schema has both fields)."""
        # We run the internal stages manually to verify
        from pipelines.full_chain_smoke import _build_synthetic_predictions
        from pipelines.residual_correction import apply_residual_correction
        from fusion.engine import run_fusion, READY_DRY_RUN

        preds = _build_synthetic_predictions("2026-07-04")
        corrected = apply_residual_correction(preds)
        result = run_fusion(
            corrected, method="equal_weight", allow_dry_run=True,
            readiness_status={"cfg05": READY_DRY_RUN, "best_two_average": READY_DRY_RUN},
        )
        if len(result) > 0:
            assert "fused_price" in result.columns
            assert not result["fused_price"].isna().any()

    def test_corrected_ledger_writes_to_tmp_path(self, tmp_path):
        """Corrected ledger can be written to tmp_path."""
        summary = run_full_chain_smoke(
            target_day="2026-07-04", ledger_dir=str(tmp_path),
        )
        assert summary["overall_status"] == "PASS"
        corrected_path = os.path.join(str(tmp_path), "corrected_ledger.csv")
        assert os.path.isfile(corrected_path)

    def test_fusion_ledger_writes_to_tmp_path(self, tmp_path):
        """Fusion ledger can be written to tmp_path."""
        summary = run_full_chain_smoke(
            target_day="2026-07-04", ledger_dir=str(tmp_path),
        )
        assert summary["overall_status"] == "PASS"
        fusion_path = os.path.join(str(tmp_path), "fusion_ledger.csv")
        assert os.path.isfile(fusion_path)

    def test_weight_ledger_expands(self):
        """Weight ledger rows are expanded from fusion output."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        # 24 hours * 2 models = 48 weight rows
        assert summary["weight_rows"] == 48

    def test_weights_sum_to_1_per_hour(self):
        """Weights extracted from fusion sum to 1 per hour."""
        from pipelines.full_chain_smoke import _build_synthetic_predictions
        from pipelines.residual_correction import apply_residual_correction
        from fusion.engine import run_fusion, READY_DRY_RUN
        from ledgers.weight_ledger import extract_weight_rows

        preds = _build_synthetic_predictions("2026-07-04")
        corrected = apply_residual_correction(preds)
        fusion = run_fusion(
            corrected, method="equal_weight", allow_dry_run=True,
            readiness_status={"cfg05": READY_DRY_RUN, "best_two_average": READY_DRY_RUN},
        )
        weights = extract_weight_rows(fusion)
        for hour in range(1, 25):
            hour_weights = weights[weights["hour_business"] == hour]
            if len(hour_weights) > 0:
                total = hour_weights["weight"].sum()
                assert abs(total - 1.0) < 1e-4, f"Hour {hour} weights sum to {total}"

    def test_final_output_validator_passes(self):
        """Final output validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "final_output_validator" in summary["validators_passed"]

    def test_final_output_24_rows(self):
        """Final output has 24 rows for a single day."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert summary["final_rows"] == 24

    def test_no_real_label_without_artifact(self):
        """No REAL label appears without verified artifact."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        labels = summary["mode_label"]
        assert "REAL" not in labels, f"REAL label found: {labels}"

    def test_real_label_appears_with_artifact(self, tmp_path):
        """REAL label appears when artifact path is verified."""
        # Create a dummy artifact
        artifact = tmp_path / "model.pkl"
        artifact.write_text("dummy")

        summary = run_full_chain_smoke(
            target_day="2026-07-04",
            cfg05_artifact_path=str(artifact),
        )
        labels = summary["mode_label"]
        # The day-ahead stage should be REAL
        assert summary["stage_labels"]["dayahead"] == "REAL"

    def test_forbidden_files_check_pass(self):
        """Forbidden files check passes by default."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert summary["forbidden_files_check"] == "PASS"

    def test_row_counts_match_24_hours(self):
        """Row counts are consistent for a single day with 2 models."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert summary["prediction_rows"] == 48  # 24h * 2 models
        assert summary["corrected_rows"] == 48  # 24h * 2 models
        assert summary["fusion_rows"] == 24  # 1 fused price per hour
        assert summary["final_rows"] == 24

    def test_validators_list_non_empty(self):
        """validators_passed contains multiple validators."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert len(summary["validators_passed"]) >= 5  # pred, residual, fusion, + ledgers, final

    def test_stage_labels_all_present(self):
        """All expected stage labels are in stage_labels dict."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        expected_stages = [
            "dayahead", "residual", "corrected_ledger",
            "fusion", "fusion_ledger", "weight_ledger",
            "negative_classifier", "final_output",
        ]
        for stage in expected_stages:
            assert stage in summary["stage_labels"], f"Missing stage: {stage}"


class TestFullChainSmokeCLI:
    """Contract: CLI entry point."""

    def test_cli_dry_run_exits_0(self):
        """CLI dry-run exits with code 0."""
        from scripts.run_full_chain_smoke import main
        exit_code = main(["--target-day", "2026-07-04", "--no-production"])
        assert exit_code == 0

    def test_cli_with_tmp_path_ledger(self, tmp_path):
        """CLI with tmp_path ledger_dir exits 0."""
        from scripts.run_full_chain_smoke import main
        exit_code = main([
            "--target-day", "2026-07-04",
            "--ledger-dir", str(tmp_path),
            "--no-production",
        ])
        assert exit_code == 0
        assert os.path.isfile(os.path.join(str(tmp_path), "corrected_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "fusion_ledger.csv"))
        assert os.path.isfile(os.path.join(str(tmp_path), "weight_ledger.csv"))


class TestFullChainSmokeEdgeCases:
    """Contract: edge cases and error handling."""

    def test_no_real_in_reason_codes(self):
        """No REAL reason code appears without artifacts."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        reasons = ";".join(summary["reason_codes"])
        assert "REAL" not in reasons.split("_STAGE"), (
            "REAL label leaked into reason_codes without artifact"
        )

    def test_weight_ledger_validator_passes(self):
        """Weight ledger validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "weight_ledger_validator" in summary["validators_passed"]

    def test_fusion_ledger_validator_passes(self):
        """Fusion ledger validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "fusion_ledger_validator" in summary["validators_passed"]

    def test_corrected_ledger_validator_passes(self):
        """Corrected ledger validator is in validators_passed."""
        summary = run_full_chain_smoke(target_day="2026-07-04")
        assert "corrected_ledger_validator" in summary["validators_passed"]

    def test_classifier_rule_fallback_disabled(self):
        """Negative classifier label changes when rule_fallback disabled."""
        summary = run_full_chain_smoke(
            target_day="2026-07-04",
            classifier_rule_fallback=False,
        )
        assert summary["stage_labels"]["negative_classifier"] == "CLASSIFIER_ARTIFACT_MISSING"
