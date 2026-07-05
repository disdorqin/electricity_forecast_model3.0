"""
tests/test_p62_delivery_experiments.py — P62: End-to-end delivery experiment tests.

Verifies that all 6 experiments can run without crashing and produce
the expected result structure.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from scripts.run_p62_delivery_experiments import (
    experiment_a_fresh_strict_run,
    experiment_b_regime_bgew_run,
    experiment_c_stage3_injection,
    experiment_d_missing_hour_injection,
    experiment_e_nan_y_pred_injection,
    experiment_f_no_training_days,
    run_all_experiments,
    main,
)


class TestExperimentA:
    """Experiment A: Fresh strict run (period_bgew)."""

    def test_runs_without_crash(self, tmp_path):
        result = experiment_a_fresh_strict_run(str(tmp_path))
        assert isinstance(result, dict)
        assert "overall_status" in result

    def test_has_steps(self, tmp_path):
        result = experiment_a_fresh_strict_run(str(tmp_path))
        assert "steps" in result

    def test_has_metrics(self, tmp_path):
        result = experiment_a_fresh_strict_run(str(tmp_path))
        assert "metrics" in result


class TestExperimentB:
    """Experiment B: Regime BGEW fusion."""

    def test_runs_without_crash(self, tmp_path):
        result = experiment_b_regime_bgew_run(str(tmp_path))
        assert isinstance(result, dict)
        assert "overall_status" in result

    def test_has_fusion_engine(self, tmp_path):
        result = experiment_b_regime_bgew_run(str(tmp_path))
        assert result.get("fusion_engine") == "regime_bgew"

    def test_has_steps(self, tmp_path):
        result = experiment_b_regime_bgew_run(str(tmp_path))
        assert "steps" in result


class TestExperimentC:
    """Experiment C: Stage3 injection."""

    def test_runs_without_crash(self, tmp_path):
        result = experiment_c_stage3_injection(str(tmp_path))
        assert isinstance(result, dict)
        assert "overall_status" in result

    def test_reports_errors_or_blocked(self, tmp_path):
        result = experiment_c_stage3_injection(str(tmp_path))
        # Either errors list is populated or steps show blocking
        has_errors = len(result.get("errors", [])) > 0
        has_blocked = any(
            s.get("blocked_models")
            for s in result.get("steps", {}).values()
        )
        assert has_errors or has_blocked, (
            "Stage3 should produce errors or blocked_models"
        )


class TestExperimentD:
    """Experiment D: Missing hour injection."""

    def test_runs_without_crash(self, tmp_path):
        result = experiment_d_missing_hour_injection(str(tmp_path))
        assert isinstance(result, dict)
        assert "overall_status" in result

    def test_has_level_or_steps(self, tmp_path):
        result = experiment_d_missing_hour_injection(str(tmp_path))
        assert "steps" in result


class TestExperimentE:
    """Experiment E: NaN y_pred injection."""

    def test_runs_without_crash(self, tmp_path):
        result = experiment_e_nan_y_pred_injection(str(tmp_path))
        assert isinstance(result, dict)

    def test_has_steps(self, tmp_path):
        result = experiment_e_nan_y_pred_injection(str(tmp_path))
        assert "steps" in result


class TestExperimentF:
    """Experiment F: No complete training days."""

    def test_runs_without_crash(self, tmp_path):
        result = experiment_f_no_training_days(str(tmp_path))
        assert isinstance(result, dict)

    def test_has_steps(self, tmp_path):
        result = experiment_f_no_training_days(str(tmp_path))
        assert "steps" in result


class TestRunAll:
    """Test run_all_experiments orchestrator."""

    def test_run_all_does_not_crash(self, tmp_path):
        result = run_all_experiments(base_dir=str(tmp_path))
        assert isinstance(result, dict)
        assert "experiments" in result

    def test_run_all_has_6_experiments(self, tmp_path):
        result = run_all_experiments(base_dir=str(tmp_path))
        assert len(result["experiments"]) == 6

    def test_run_all_json_serializable(self, tmp_path):
        result = run_all_experiments(base_dir=str(tmp_path))
        dumped = json.dumps(result, default=str)
        assert isinstance(dumped, str)


class TestMain:
    """Test main entry point."""

    def test_main_returns_int(self, tmp_path):
        ret = main(["--base-dir", str(tmp_path)])
        assert isinstance(ret, int)

    def test_main_json_flag(self, tmp_path):
        ret = main(["--base-dir", str(tmp_path), "--json"])
        assert isinstance(ret, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
