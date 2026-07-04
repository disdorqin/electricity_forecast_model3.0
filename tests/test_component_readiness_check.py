"""
tests/test_component_readiness_check.py — Component readiness gate contract tests.

Validates:
    1. run_all_checks returns results for all 4 components
    2. cfg05 without real artifact is not READY_REAL
    3. SGDFNet without weights is READY_STUB
    4. P5M without canonical pack is DATA_MISSING or READY_STUB
    5. Each result has component, state, details keys
    6. All states are valid readiness states
    7. CLI exits 0 when no NOT_READY components
"""

from __future__ import annotations

import pytest

from scripts.component_readiness_check import (
    check_cfg05_dayahead,
    check_realtime_assist,
    check_sgdfnet,
    check_p5m_residual,
    run_all_checks,
    READY_REAL,
    READY_DRY_RUN,
    READY_STUB,
    DATA_MISSING,
    NOT_READY,
)

VALID_STATES = {READY_REAL, READY_DRY_RUN, READY_STUB, DATA_MISSING, NOT_READY}


class TestComponentReadinessCheck:
    """Contract: component readiness checks."""

    def test_run_all_checks_returns_4_results(self):
        """run_all_checks returns results for all 4 components."""
        results = run_all_checks()
        assert len(results) == 4

    def test_each_result_has_required_keys(self):
        """Each result has component, state, and details keys."""
        results = run_all_checks()
        for r in results:
            assert "component" in r
            assert "state" in r
            assert "details" in r

    def test_each_result_has_valid_state(self):
        """Each result state is a valid readiness state."""
        results = run_all_checks()
        for r in results:
            assert r["state"] in VALID_STATES, (
                f"Invalid state {r['state']} for {r['component']}"
            )

    def test_cfg05_not_readily_real(self):
        """cfg05 without real artifact is not READY_REAL."""
        result = check_cfg05_dayahead()
        assert result["component"] == "cfg05_dayahead_model_zoo"
        assert result["state"] != READY_REAL
        # Must be at least dry-run capable
        assert result["state"] in (READY_DRY_RUN, READY_STUB)

    def test_sgdfnet_is_at_most_stub(self):
        """SGDFNet without weights is READY_STUB at most."""
        result = check_sgdfnet()
        assert result["component"] == "sgdfnet_2_5"
        assert result["state"] in (READY_STUB, NOT_READY), (
            f"SGDFNet should be stub or not ready, got {result['state']}"
        )

    def test_p5m_is_not_readily_real(self):
        """P5M without canonical pack is not READY_REAL."""
        result = check_p5m_residual()
        assert result["component"] == "p5m_residual_plugin"
        assert result["state"] != READY_REAL
        assert result["state"] in (READY_STUB, DATA_MISSING, READY_DRY_RUN)

    def test_cfg05_details_are_populated(self):
        """cfg05 check has populated details list."""
        result = check_cfg05_dayahead()
        assert len(result["details"]) >= 2  # at least adapter + dry-run

    def test_realtime_assist_not_not_ready(self):
        """Realtime assist is at least runnable."""
        result = check_realtime_assist()
        assert result["component"] == "realtime_da_safe_assist"
        assert result["state"] != NOT_READY, (
            f"Realtime assist should not be NOT_READY: {result['details']}"
        )


class TestCLI:
    """Contract: CLI behavior."""

    def test_cli_exit_code_zero(self):
        """CLI exits 0 when no NOT_READY components."""
        from scripts.component_readiness_check import main
        exit_code = main([])
        assert exit_code == 0
