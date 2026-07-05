"""
P89 — 2.5 Style Chain Parity Audit tests.

Verifies that 3.0 has parity with (and exceeds) the 2.5 chain:
  - 3.0 has ledger_predict (prediction ledgers) -> True
  - 3.0 has ledger_weight (weight learner) -> True
  - 3.0 has ledger_fuse (fusion engine) -> True
  - 3.0 has ledger_classifier -> True
  - 3.0 has final_outputs -> True
  - 3.0 has postflight -> True
  - 3.0 has manifest -> True
  - 3.0 has delivery report -> True
  - 3.0 has NORMAL/DEGRADED_DELIVERED/FAILED_NO_DELIVERY statuses -> True
  - 3.0 supports one-click main.py -> True
  - Realtime deep model: fallback (not real)
  - Residual: no-op (not real)
  - Classifier: rule fallback (not real ML)
  - Adaptive learner: dimensional (task x period x regime) -> True (3.0 innovation)
  - Safety supervisor: enforced -> True (3.0 innovation)
"""
from __future__ import annotations

import importlib
import os

import pytest


# ── Helper: check module importability ──────────────────────────────────────

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


def _can_import(module_path: str) -> bool:
    """Check if a module can be imported (structural check)."""
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False


def _module_has_attr(module_path: str, attr: str) -> bool:
    """Check if a module has a specific attribute."""
    try:
        mod = importlib.import_module(module_path)
        return hasattr(mod, attr)
    except ImportError:
        return False


# ── Tests ───────────────────────────────────────────────────────────────────


class TestLedgerPredict:
    """3.0 has prediction ledgers (ledger_predict)."""

    def test_prediction_ledger_module_exists(self):
        assert _can_import("ledgers.prediction_ledger")

    def test_prediction_ledger_has_append(self):
        assert _module_has_attr(
            "ledgers.prediction_ledger",
            "append_corrected_predictions_to_ledger",
        )


class TestLedgerWeight:
    """3.0 has weight learner (ledger_weight)."""

    def test_weight_ledger_module_exists(self):
        assert _can_import("ledgers.weight_ledger")

    def test_unified_weight_learner_exists(self):
        assert _can_import("fusion.unified_weight_learner")


class TestLedgerFuse:
    """3.0 has fusion engine (ledger_fuse)."""

    def test_fusion_ledger_module_exists(self):
        assert _can_import("ledgers.fusion_ledger")

    def test_unified_fusion_engine_exists(self):
        assert _can_import("fusion.unified_fusion_engine")


class TestLedgerClassifier:
    """3.0 has classifier layer."""

    def test_classifier_module_exists(self):
        assert _can_import("classifiers")

    def test_classifier_has_rule_fallback_constant(self):
        assert _module_has_attr("classifiers", "CLASSIFIER_RULE_FALLBACK")


class TestFinalOutputs:
    """3.0 has final output builder."""

    def test_final_output_builder_exists(self):
        assert _can_import("delivery.final_output_builder")

    def test_final_output_columns_defined(self):
        from delivery.final_output_builder import FINAL_OUTPUT_COLUMNS
        assert len(FINAL_OUTPUT_COLUMNS) == 17


class TestPostflight:
    """3.0 has postflight validation."""

    def test_postflight_module_exists(self):
        assert _can_import("delivery.postflight")

    def test_postflight_has_run(self):
        assert _module_has_attr("delivery.postflight", "run_postflight")


class TestManifest:
    """3.0 has delivery manifest."""

    def test_manifest_module_exists(self):
        assert _can_import("delivery.manifest")

    def test_manifest_has_create(self):
        assert _module_has_attr("delivery.manifest", "create_manifest")


class TestDeliveryReport:
    """3.0 has delivery report generation."""

    def test_report_module_exists(self):
        assert _can_import("delivery.report")

    def test_report_has_generate(self):
        assert _module_has_attr("delivery.report", "generate_delivery_report")


class TestDeliveryStatuses:
    """3.0 has NORMAL/DEGRADED_DELIVERED/FAILED_NO_DELIVERY statuses."""

    def test_failed_no_delivery_status(self):
        from delivery.fallback_ladder import FALLBACK_LEVEL_NAMES
        assert 6 in FALLBACK_LEVEL_NAMES
        assert FALLBACK_LEVEL_NAMES[6] == "FAILED_NO_DELIVERY"


class TestOneClickMain:
    """3.0 supports one-click main.py entry point."""

    def test_main_py_exists(self):
        main_path = os.path.join(REPO_ROOT, "main.py")
        assert os.path.isfile(main_path)

    def test_main_delegates_to_run_full_chain(self):
        import main
        import inspect
        source = inspect.getsource(main.main)
        assert "run_full_chain" in source


class TestComponentMaturity:
    """Audit which components are real vs fallback in 3.0."""

    def test_residual_is_noop(self):
        from residuals import RESIDUAL_NO_OP_FALLBACK
        assert RESIDUAL_NO_OP_FALLBACK == "RESIDUAL_NO_OP_FALLBACK"

    def test_classifier_is_rule_fallback(self):
        from classifiers import CLASSIFIER_RULE_FALLBACK
        assert CLASSIFIER_RULE_FALLBACK == "CLASSIFIER_RULE_FALLBACK"


class TestInnovations30:
    """3.0 innovations beyond 2.5 parity."""

    def test_adaptive_learner_dimensional(self):
        from fusion.unified_weight_learner import (
            PERIODS,
            REGIMES,
            LEARNER_FULL_DIMENSIONAL,
        )
        assert len(PERIODS) == 3
        assert len(REGIMES) == 4
        assert LEARNER_FULL_DIMENSIONAL == "LEARNER_FULL_DIMENSIONAL"

    def test_safety_supervisor_enforced(self):
        from safety.full_chain_safety_supervisor import (
            run_full_chain_safety,
            FULL_CHAIN_SAFETY_PASS,
        )
        result = run_full_chain_safety()
        assert "status" in result
        assert "checks" in result
        assert result["status"] == FULL_CHAIN_SAFETY_PASS
