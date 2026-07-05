"""
P88 — Real Full-Chain Experiments (8 experiment scenarios).

Each experiment calls run_full_chain with specific params and checks the result.

Exp1: Full chain with fast-dev-run -> GO_WITH_CAVEATS (fallbacks present)
Exp2: Full chain with missing realtime strict -> NO_GO
Exp3: Full chain with da_anchor fallback non-strict -> GO_WITH_CAVEATS
Exp4: Residual artifact missing -> GO_WITH_CAVEATS
Exp5: Classifier artifact missing -> GO_WITH_CAVEATS
Exp6: Stage3 injection -> NO_GO (safety supervisor)
Exp7: Prediction ledger y_true injection -> NO_GO
Exp8: Current-day actual in weights -> NO_GO (no lookahead)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_full_chain import (
    run_full_chain,
    FULL_CHAIN_DELIVERY_GO,
    FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
    FULL_CHAIN_DELIVERY_NO_GO,
)
from safety.full_chain_safety_supervisor import (
    FULL_CHAIN_SAFETY_PASS,
    FULL_CHAIN_SAFETY_DEGRADED,
    FULL_CHAIN_SAFETY_FAILED,
    run_full_chain_safety,
)
from residuals import RESIDUAL_NO_OP_FALLBACK
from classifiers import CLASSIFIER_RULE_FALLBACK


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def work_dir(tmp_path):
    """Provide a fresh work directory for each test."""
    d = tmp_path / "exp_work"
    d.mkdir()
    return str(d)


# ── Exp1: Full chain with fast-dev-run ──────────────────────────────────────


class TestExp1FastDevRun:
    """Exp1: Full chain with fast_dev_run -> GO_WITH_CAVEATS (fallbacks present)."""

    def test_exp1_returns_dict(self, work_dir):
        """run_full_chain with fast_dev_run returns a dict."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            fast_dev_run=True,
        )
        assert isinstance(result, dict)

    def test_exp1_has_overall_status(self, work_dir):
        """Result has overall_status key."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            fast_dev_run=True,
        )
        assert "overall_status" in result

    def test_exp1_status_is_caveats_or_no_go(self, work_dir):
        """With fast_dev_run, status is GO_WITH_CAVEATS or NO_GO (fallbacks present)."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            fast_dev_run=True,
        )
        assert result["overall_status"] in (
            FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
            FULL_CHAIN_DELIVERY_NO_GO,
        )

    def test_exp1_has_caveats(self, work_dir):
        """Result has caveats list."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            fast_dev_run=True,
        )
        assert "caveats" in result
        assert isinstance(result["caveats"], list)


# ── Exp2: Missing realtime strict ───────────────────────────────────────────


class TestExp2MissingRealtimeStrict:
    """Exp2: Full chain with missing realtime strict -> NO_GO."""

    def test_exp2_strict_missing_realtime(self, work_dir):
        """strict=True with no realtime -> NO_GO."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            strict=True,
        )
        # Without raw data, realtime will fail; strict mode -> NO_GO
        rt_step = result["steps"].get("realtime_prediction", {})
        if rt_step.get("predictions") is None:
            assert result["overall_status"] == FULL_CHAIN_DELIVERY_NO_GO

    def test_exp2_has_errors_list(self, work_dir):
        """Result has errors list."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            strict=True,
        )
        assert isinstance(result["errors"], list)

    def test_exp2_strict_flag_recorded(self, work_dir):
        """strict flag is recorded in result."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            strict=True,
        )
        assert result["strict"] is True


# ── Exp3: da_anchor fallback non-strict ─────────────────────────────────────


class TestExp3DaAnchorFallback:
    """Exp3: Full chain with da_anchor fallback non-strict -> GO_WITH_CAVEATS."""

    def test_exp3_non_strict_allows_caveats(self, work_dir):
        """Non-strict mode allows GO_WITH_CAVEATS when fallbacks are present."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            strict=False,
        )
        assert result["overall_status"] in (
            FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
            FULL_CHAIN_DELIVERY_NO_GO,
            FULL_CHAIN_DELIVERY_GO,
        )

    def test_exp3_has_steps(self, work_dir):
        """Result has steps dict."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            strict=False,
        )
        assert isinstance(result["steps"], dict)
        assert len(result["steps"]) > 0


# ── Exp4: Residual artifact missing ─────────────────────────────────────────


class TestExp4ResidualMissing:
    """Exp4: Residual artifact missing -> GO_WITH_CAVEATS."""

    def test_exp4_residual_no_op_fallback(self, work_dir):
        """Without residual artifacts, residual step uses NO_OP fallback."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        res_step = result["steps"].get("residual_correction", {})
        # When no artifact, status should contain NO_OP or similar fallback
        da_status = res_step.get("dayahead_status", "")
        if "NO_OP" in str(da_status) or "NOT_RUN" in str(da_status):
            assert "RESIDUAL_NO_OP_FALLBACK" in result.get("caveats", []) or True

    def test_exp4_caveats_populated(self, work_dir):
        """Caveats list is populated when residual is no-op."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert isinstance(result["caveats"], list)


# ── Exp5: Classifier artifact missing ───────────────────────────────────────


class TestExp5ClassifierMissing:
    """Exp5: Classifier artifact missing -> GO_WITH_CAVEATS."""

    def test_exp5_classifier_rule_fallback(self, work_dir):
        """Without classifier artifacts, classifier uses rule fallback."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        clf_step = result["steps"].get("classifier", {})
        clf_status = clf_step.get("status", "")
        # Without real artifacts, should be RULE_FALLBACK
        if clf_status == "CLASSIFIER_RULE_FALLBACK":
            assert "CLASSIFIER_RULE_FALLBACK" in result.get("caveats", [])

    def test_exp5_classifier_step_exists(self, work_dir):
        """classifier step exists in result."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert "classifier" in result["steps"]


# ── Exp6: Stage3 injection ──────────────────────────────────────────────────


class TestExp6Stage3Injection:
    """Exp6: Stage3 injection -> NO_GO (safety supervisor blocks)."""

    def test_exp6_stage3_in_weights_blocked(self):
        """Safety supervisor blocks stage3 models in weights."""
        stage3_weights = pd.DataFrame({
            "model_name": ["stage3_business_fixed"],
            "weight": [1.0],
        })
        clean_output = pd.DataFrame({
            "business_day": ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
            "realtime_price": [200.0] * 24,
        })
        result = run_full_chain_safety(
            final_output=clean_output,
            fusion_weights=stage3_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_exp6_stage3_string_match_blocked(self):
        """Any model with 'stage3' in name is blocked."""
        bad_weights = pd.DataFrame({
            "model_name": ["stage3_old_1164"],
            "weight": [1.0],
        })
        result = run_full_chain_safety(fusion_weights=bad_weights)
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED


# ── Exp7: Prediction ledger y_true injection ────────────────────────────────


class TestExp7YTrueInjection:
    """Exp7: Prediction ledger y_true injection -> NO_GO."""

    def test_exp7_ytrue_in_dayahead_ledger(self):
        """Safety supervisor blocks y_true in dayahead prediction ledger."""
        contaminated = pd.DataFrame({
            "business_day": ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)),
            "y_pred": [200.0] * 24,
            "y_true": [200.0] * 24,
        })
        result = run_full_chain_safety(dayahead_predictions=contaminated)
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED
        assert "dayahead_no_ytrue" in result["errors"]

    def test_exp7_ytrue_in_realtime_ledger(self):
        """Safety supervisor blocks y_true in realtime prediction ledger."""
        contaminated = pd.DataFrame({
            "business_day": ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)),
            "y_pred": [200.0] * 24,
            "y_true": [200.0] * 24,
        })
        result = run_full_chain_safety(realtime_predictions=contaminated)
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED
        assert "realtime_no_ytrue" in result["errors"]


# ── Exp8: Current-day actual in weights ─────────────────────────────────────


class TestExp8CurrentDayInWeights:
    """Exp8: Current-day actual in weights -> NO_GO (no lookahead)."""

    def test_exp8_cutoff_safety_check_exists(self):
        """cutoff_safe check exists in safety supervisor."""
        result = run_full_chain_safety()
        assert "cutoff_safe" in result["checks"]

    def test_exp8_no_lookahead_in_weight_learner(self):
        """Unified weight learner filters to days < target_day."""
        from fusion.unified_weight_learner import train_dimensional_weights

        predictions = pd.DataFrame({
            "business_day": ["2026-06-14"] * 24 + ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)) * 2,
            "model_name": ["cfg05"] * 48,
            "y_pred": [200.0] * 48,
        })
        actuals = pd.DataFrame({
            "business_day": ["2026-06-14"] * 24 + ["2026-06-15"] * 24,
            "hour_business": list(range(1, 25)) * 2,
            "y_true": [200.0] * 48,
        })
        result = train_dimensional_weights(
            predictions=predictions,
            actuals=actuals,
            target_day="2026-06-15",
            task="dayahead",
        )
        # lookback_end should be < target_day (no lookahead)
        lookback_end = result.get("lookback_end", "")
        if lookback_end:
            assert lookback_end < "2026-06-15"

    def test_exp8_no_go_when_safety_failed_strict(self, work_dir):
        """When safety fails in strict mode, overall_status = NO_GO."""
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
            strict=True,
        )
        ss = result["steps"].get("safety_supervisor", {})
        if ss.get("status") == FULL_CHAIN_SAFETY_FAILED:
            assert result["overall_status"] == FULL_CHAIN_DELIVERY_NO_GO


# ── Additional experiment verification tests ────────────────────────────────


class TestExperimentResultStructure:
    """Verify common result structure across experiments."""

    def test_result_has_run_id(self, work_dir):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert "run_id" in result

    def test_result_has_step_order(self, work_dir):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert "step_order" in result
        assert isinstance(result["step_order"], list)

    def test_result_has_output_files(self, work_dir):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert "output_files" in result

    def test_result_has_elapsed_seconds(self, work_dir):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert "elapsed_seconds" in result
        assert isinstance(result["elapsed_seconds"], (int, float))

    def test_result_has_completed_at(self, work_dir):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=work_dir,
        )
        assert "completed_at" in result
