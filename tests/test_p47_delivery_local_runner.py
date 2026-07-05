"""
tests/test_p47_delivery_local_runner.py — P47 tests (15+).

Tests: profile loading, artifact validation, runner CLI, dry-run,
missing data handling, profile validation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")


# ──────────────────────────────────────────────────────
# Profile loading (4)
# ──────────────────────────────────────────────────────


class TestProfileLoading:
    """Tests for profile loading in run_delivery_local_chain.py."""

    def test_load_trusted_delivery_profile(self):
        from scripts.run_delivery_local_chain import _load_profile_models
        profile = _load_profile_models("trusted_delivery")
        assert profile.get("delivery_allowed") is True
        assert "lightgbm_cfg05_dayahead" in profile.get("allowed_models", [])
        assert "catboost_spike_residual" in profile.get("allowed_models", [])

    def test_trusted_models_from_profile(self):
        from scripts.run_delivery_local_chain import _load_profile_models, _trusted_models_from_profile
        profile = _load_profile_models("trusted_delivery")
        models = _trusted_models_from_profile(profile)
        assert "lightgbm_cfg05_dayahead" in models
        assert "stage3_business_fixed" not in models

    def test_load_balanced_profile(self):
        from scripts.run_delivery_local_chain import _load_profile_models
        profile = _load_profile_models("balanced_candidate")
        assert profile.get("delivery_allowed") is False
        assert "best_two_average" in profile.get("allowed_models", [])

    def test_load_research_profile(self):
        from scripts.run_delivery_local_chain import _load_profile_models
        profile = _load_profile_models("research_all_models")
        assert profile.get("delivery_allowed") is False
        assert "stage3_business_fixed" in profile.get("allowed_models", [])


# ──────────────────────────────────────────────────────
# Artifact validation (4)
# ──────────────────────────────────────────────────────


class TestArtifactValidation:
    """Tests for artifact validation helpers."""

    def test_file_hash_non_existent(self):
        from scripts.run_delivery_local_chain import _file_hash
        assert _file_hash("/nonexistent/path.csv") == ""

    def test_file_hash_existing_file(self):
        from scripts.run_delivery_local_chain import _file_hash
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"a,b,c\n1,2,3\n")
            fname = f.name
        try:
            h = _file_hash(fname)
            assert len(h) == 16
            assert h != ""
        finally:
            os.unlink(fname)

    def test_artifact_valid_non_existent(self):
        from scripts.run_delivery_local_chain import _artifact_valid
        assert _artifact_valid("/nonexistent.csv") is False

    def test_artifact_valid_with_min_rows(self):
        from scripts.run_delivery_local_chain import _artifact_valid, _csv_row_count
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col\n1\n2\n3\n4\n5\n")
            fname = f.name
        try:
            assert _csv_row_count(fname) == 5
            assert _artifact_valid(fname, min_rows=3) is True
            assert _artifact_valid(fname, min_rows=10) is False
        finally:
            os.unlink(fname)


# ──────────────────────────────────────────────────────
# Runner logic (7)
# ──────────────────────────────────────────────────────


class TestRunnerLogic:
    """Tests for runner step functions."""

    def test_raw_data_step_missing(self):
        from scripts.run_delivery_local_chain import step_raw_data_check
        with tempfile.TemporaryDirectory() as tmp:
            result = step_raw_data_check("/nonexistent.csv", tmp, force=True)
            assert result["status"] == "FAILED"

    def test_source_repo_step_missing(self):
        from scripts.run_delivery_local_chain import step_source_repo_check
        with tempfile.TemporaryDirectory() as tmp:
            result = step_source_repo_check("/nonexistent", tmp, force=True)
            assert result["status"] == "FAILED"

    def test_step_forbidden_file_check(self):
        from scripts.run_delivery_local_chain import step_forbidden_file_check
        with tempfile.TemporaryDirectory() as tmp:
            result = step_forbidden_file_check(tmp)
            assert result["status"] == "PASSED"

    def test_step_forbidden_file_detected(self):
        from scripts.run_delivery_local_chain import step_forbidden_file_check
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.pkl"), "w") as f:
                f.write("dummy")
            result = step_forbidden_file_check(tmp)
            assert result["status"] == "WARNING"
            assert len(result["forbidden_found"]) >= 1

    def test_trust_gate_step_override(self):
        from scripts.run_delivery_local_chain import step_load_or_run_trust_gate
        with tempfile.TemporaryDirectory() as tmp:
            result = step_load_or_run_trust_gate(tmp, force=True, trusted_pool_override=["cfg05"])
            assert result["status"] == "OVERRIDDEN"

    def test_trust_gate_step_cached(self):
        from scripts.run_delivery_local_chain import step_load_or_run_trust_gate
        with tempfile.TemporaryDirectory() as tmp:
            result1 = step_load_or_run_trust_gate(tmp, force=True, trusted_pool_override=["cfg05"])
            result2 = step_load_or_run_trust_gate(tmp, force=False)
            assert result2["status"] == "CACHED"

    def test_trusted_models_from_default_profile(self):
        from scripts.run_delivery_local_chain import _load_profile_models, _trusted_models_from_profile
        profile = _load_profile_models("trusted_delivery")
        models = _trusted_models_from_profile(profile)
        assert len(models) >= 2


# ──────────────────────────────────────────────────────
# Runner dry-run (1)
# ──────────────────────────────────────────────────────


class TestRunnerDryRun:
    """Tests for full runner dry-run with minimal work_dir."""

    def test_runner_dry_run_no_data(self):
        """Runner should not crash when called with missing data."""
        from scripts.run_delivery_local_chain import run_delivery_chain
        with tempfile.TemporaryDirectory() as tmp:
            result = run_delivery_chain(
                raw_data="/nonexistent.csv",
                source_repo="/nonexistent",
                profile="trusted_delivery",
                work_dir=tmp,
                force=True,
            )
            assert result["phase"] == "P47"
            assert result["profile"] == "trusted_delivery"
            # Should have steps recorded
            assert len(result.get("step_order", [])) > 0
            # Some steps should fail, but the runner should complete
            assert result["overall_status"] in (
                "P47_DELIVERY_CHAIN_PASS", "P47_DELIVERY_CHAIN_FAILED"
            )


# ──────────────────────────────────────────────────────
# Full run via main (1)
# ──────────────────────────────────────────────────────


class TestP47FullRun:
    def test_p47_main_json(self):
        from scripts.run_delivery_local_chain import main
        exit_code = main(["--json"])
        assert exit_code == 0
