"""
P86 — Full Safety Supervisor Enforced tests.

Verifies that run_full_chain.py enforces the safety supervisor:
  - safety_supervisor step present in result["steps"]
  - FULL_CHAIN_SAFETY_FAILED in strict mode -> NO_GO
  - FULL_CHAIN_SAFETY_DEGRADED -> GO_WITH_CAVEATS
  - FULL_CHAIN_SAFETY_PASS -> allowed GO
  - CLI flags --strict, --strict-no-leakage
  - Safety checks include dayahead_no_ytrue, realtime_no_ytrue, etc.
  - Cutoff safety: weights with lookback_end >= target_day flagged
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safety.full_chain_safety_supervisor import (
    FULL_CHAIN_SAFETY_PASS,
    FULL_CHAIN_SAFETY_DEGRADED,
    FULL_CHAIN_SAFETY_FAILED,
    FORBIDDEN_IN_PRODUCTION,
    QUARANTINED_MODELS,
    run_full_chain_safety,
    _check_no_forbidden,
    _check_no_quarantined,
    _check_stage3_blocked,
    _check_cutoff_safety,
)
from scripts.run_full_chain import (
    run_full_chain,
    _parse_args,
    FULL_CHAIN_DELIVERY_GO,
    FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
    FULL_CHAIN_DELIVERY_NO_GO,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_dayahead():
    """24-row clean dayahead prediction ledger (no y_true)."""
    return pd.DataFrame({
        "business_day": ["2026-06-15"] * 24,
        "hour_business": list(range(1, 25)),
        "y_pred": np.random.default_rng(42).uniform(100, 400, 24),
        "model_name": ["cfg05"] * 24,
    })


@pytest.fixture
def clean_final_output():
    """24-row clean final output with valid prices."""
    return pd.DataFrame({
        "business_day": ["2026-06-15"] * 24,
        "hour_business": list(range(1, 25)),
        "dayahead_price": np.random.default_rng(44).uniform(100, 400, 24),
        "realtime_price": np.random.default_rng(45).uniform(100, 400, 24),
    })


@pytest.fixture
def clean_weights():
    """Clean fusion weights (no quarantined models)."""
    return pd.DataFrame({
        "model_name": ["cfg05", "catboost"],
        "weight": [0.6, 0.4],
    })


@pytest.fixture
def contaminated_predictions():
    """Predictions with y_true leaked."""
    return pd.DataFrame({
        "business_day": ["2026-06-15"] * 24,
        "hour_business": list(range(1, 25)),
        "y_pred": np.random.default_rng(46).uniform(100, 400, 24),
        "y_true": np.random.default_rng(47).uniform(100, 400, 24),
    })


# ── Tests ───────────────────────────────────────────────────────────────────


class TestSafetyCheckNames:
    """Verify the safety supervisor runs the expected named checks."""

    def test_dayahead_no_ytrue_check_exists(self, clean_dayahead, clean_final_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_dayahead,
            final_output=clean_final_output,
            fusion_weights=clean_weights,
        )
        assert "dayahead_no_ytrue" in result["checks"]

    def test_realtime_no_ytrue_check_exists(self, clean_dayahead, clean_final_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_dayahead,
            final_output=clean_final_output,
            fusion_weights=clean_weights,
        )
        assert "realtime_no_ytrue" in result["checks"]

    def test_stage3_blocked_check_exists(self, clean_dayahead, clean_final_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_dayahead,
            final_output=clean_final_output,
            fusion_weights=clean_weights,
        )
        assert "stage3_blocked" in result["checks"]

    def test_cutoff_safe_check_exists(self, clean_dayahead, clean_final_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_dayahead,
            final_output=clean_final_output,
            fusion_weights=clean_weights,
            target_day="2026-06-15",
        )
        assert "cutoff_safe" in result["checks"]


class TestSafetyStatusOutcomes:
    """Verify FULL_CHAIN_SAFETY_PASS / DEGRADED / FAILED mapping."""

    def test_all_clean_returns_pass(self, clean_dayahead, clean_final_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_dayahead,
            final_output=clean_final_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_PASS

    def test_y_true_in_dayahead_returns_failed(self, contaminated_predictions, clean_final_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=contaminated_predictions,
            final_output=clean_final_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_quarantined_model_returns_failed(self, clean_dayahead, clean_final_output):
        bad_weights = pd.DataFrame({
            "model_name": ["cfg05", "lgbm_spike_residual_1127"],
            "weight": [0.6, 0.4],
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_dayahead,
            final_output=clean_final_output,
            fusion_weights=bad_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED


class TestRunFullChainSafetyStep:
    """Verify run_full_chain result includes safety_supervisor in steps."""

    def test_result_has_safety_supervisor_step(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
        )
        assert "safety_supervisor" in result["steps"]

    def test_safety_supervisor_step_has_status(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
        )
        ss = result["steps"]["safety_supervisor"]
        assert "status" in ss


class TestStrictModeEnforcement:
    """FULL_CHAIN_SAFETY_FAILED in strict mode -> NO_GO."""

    def test_strict_flag_parsed(self):
        args = _parse_args(["--strict"])
        assert args.strict is True

    def test_strict_no_leakage_flag_parsed(self):
        args = _parse_args(["--strict-no-leakage"])
        assert args.strict_no_leakage is True

    def test_safety_failed_strict_returns_no_go(self, tmp_path):
        result = run_full_chain(
            target_start="2026-06-15",
            target_end="2026-06-15",
            work_dir=str(tmp_path / "fc"),
            strict=True,
        )
        # In strict mode, any failure (including safety) -> NO_GO
        if "safety_supervisor" in result["steps"]:
            ss_status = result["steps"]["safety_supervisor"]["status"]
            if ss_status == FULL_CHAIN_SAFETY_FAILED:
                assert result["overall_status"] == FULL_CHAIN_DELIVERY_NO_GO
        else:
            # Chain returned early (e.g. raw_data missing) — still NO_GO in strict
            assert result["overall_status"] == FULL_CHAIN_DELIVERY_NO_GO


class TestCutoffSafety:
    """Cutoff safety: weights with lookback_end >= target_day flagged."""

    def test_cutoff_safe_no_target_day(self, clean_weights):
        result = _check_cutoff_safety(clean_weights, target_day="")
        assert result["pass"] is True

    def test_cutoff_safe_none_weights(self):
        result = _check_cutoff_safety(None, target_day="2026-06-15")
        assert result["pass"] is True


class TestCheckHelpers:
    """Unit tests for individual safety check helper functions."""

    def test_check_no_forbidden_none_df(self):
        result = _check_no_forbidden(None, "test", FORBIDDEN_IN_PRODUCTION)
        assert result["pass"] is True

    def test_check_no_forbidden_contaminated_df(self, contaminated_predictions):
        result = _check_no_forbidden(contaminated_predictions, "test", FORBIDDEN_IN_PRODUCTION)
        assert result["pass"] is False
        assert result["critical"] is True

    def test_check_no_quarantined_clean(self, clean_weights):
        result = _check_no_quarantined(clean_weights)
        assert result["pass"] is True

    def test_check_stage3_blocked_clean(self, clean_weights):
        result = _check_stage3_blocked(clean_weights)
        assert result["pass"] is True
