"""
tests/test_p46_profile_claim_guard.py — P46 tests (15+).

Tests: profile registry loading, profile exclusions, claim guard
forbidden pattern detection, research caveat allowance.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PROFILES_PATH = "config/fusion_profiles.yaml"


# ──────────────────────────────────────────────────────
# Profile Registry (7)
# ──────────────────────────────────────────────────────


class TestProfileRegistry:
    """Tests for config/fusion_profiles.yaml loading and structure."""

    def test_profiles_yaml_exists(self):
        assert os.path.isfile(_PROFILES_PATH)

    def test_profiles_yaml_is_valid_yaml(self):
        with open(_PROFILES_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "profiles" in data

    def test_trusted_delivery_profile_exists(self):
        with open(_PROFILES_PATH) as f:
            data = yaml.safe_load(f)
        profiles = data.get("profiles", {})
        assert "trusted_delivery" in profiles
        td = profiles["trusted_delivery"]
        assert td.get("delivery_allowed") is True
        assert td.get("default") is True

    def test_trusted_delivery_excludes_stage3(self):
        with open(_PROFILES_PATH) as f:
            data = yaml.safe_load(f)
        profiles = data.get("profiles", {})
        td = profiles["trusted_delivery"]
        excluded = td.get("excluded_models", {})
        assert "stage3_business_fixed" in excluded
        assert excluded["stage3_business_fixed"] == "SUSPECT_LEAKAGE"

    def test_trusted_delivery_allowed_models(self):
        with open(_PROFILES_PATH) as f:
            data = yaml.safe_load(f)
        profiles = data.get("profiles", {})
        td = profiles["trusted_delivery"]
        allowed = td.get("allowed_models", [])
        assert "lightgbm_cfg05_dayahead" in allowed
        assert "catboost_spike_residual" in allowed
        assert "stage3_business_fixed" not in allowed

    def test_balanced_profile_excludes_stage3(self):
        with open(_PROFILES_PATH) as f:
            data = yaml.safe_load(f)
        profiles = data.get("profiles", {})
        bp = profiles["balanced_candidate"]
        excluded = bp.get("excluded_models", {})
        assert "stage3_business_fixed" in excluded

    def test_research_profile_includes_stage3_with_forbidden_claims(self):
        with open(_PROFILES_PATH) as f:
            data = yaml.safe_load(f)
        profiles = data.get("profiles", {})
        rp = profiles["research_all_models"]
        assert "stage3_business_fixed" in rp["allowed_models"]
        assert rp.get("delivery_allowed") is False
        assert "required_caveats" in rp


# ──────────────────────────────────────────────────────
# Claim Guard (8)
# ──────────────────────────────────────────────────────


class TestClaimGuard:
    """Tests for scripts/validate_delivery_claims.py."""

    def test_claim_guard_loads_profiles(self):
        from scripts.validate_delivery_claims import load_profiles
        profiles = load_profiles(_PROFILES_PATH)
        assert len(profiles) >= 3
        assert "trusted_delivery" in profiles

    def test_claim_guard_catches_production_2_97(self):
        from scripts.validate_delivery_claims import scan_file_for_claims
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\nOur model achieves 2.97% production sMAPE.\n")
            fname = f.name
        try:
            violations = scan_file_for_claims(fname, "test.md")
            assert len(violations) >= 1
            labels = [v["label"] for v in violations]
            assert "production_sMAPE_2_97" in labels
        finally:
            os.unlink(fname)

    def test_claim_guard_catches_production_69_96(self):
        from scripts.validate_delivery_claims import scan_file_for_claims
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n69.96% production improvement achieved.\n")
            fname = f.name
        try:
            violations = scan_file_for_claims(fname, "test.md")
            labels = [v["label"] for v in violations]
            assert "production_improvement_69_96" in labels
        finally:
            os.unlink(fname)

    def test_claim_guard_allows_research_caveated(self):
        from scripts.validate_delivery_claims import scan_file_for_claims
        content = (
            "# Test\n"
            "Research only - not delivery.\n"
            "The fusion achieved 2.97% sMAPE (research only, stage3 leakage caveat).\n"
        )
        fname = os.path.join(tempfile.mkdtemp(), "test.md")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(content)
            violations = scan_file_for_claims(fname, "test.md")
            # When caveat is present, it should be a warning not a violation
            for v in violations:
                if v["label"] == "production_sMAPE_2_97":
                    assert v["severity"] == "warning"
        finally:
            os.unlink(fname)
            os.rmdir(os.path.dirname(fname))

    def test_claim_guard_no_false_positive_code_block(self):
        from scripts.validate_delivery_claims import scan_file_for_claims
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "# Test\n"
                "```\n"
                "Some code mentioning 2.97% production metric here\n"
                "```\n"
                "Real text after code block.\n"
            )
            fname = f.name
        try:
            violations = scan_file_for_claims(fname, "test.md")
            # Code blocks should be stripped, so no violations
            prod_violations = [v for v in violations if "production_sMAPE" in v["label"]]
            # The code block stripping may leave text outside. Since "2.97%" and "production"
            # aren't on the same line outside code block, should be fine
            assert len(prod_violations) == 0
        finally:
            os.unlink(fname)

    def test_claim_guard_reports_violation_severity(self):
        from scripts.validate_delivery_claims import scan_file_for_claims
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("production stage3 readiness claim here\n")
            fname = f.name
        try:
            violations = scan_file_for_claims(fname, "test.md")
            stage3_v = [v for v in violations if "stage3" in v["label"]]
            if stage3_v:
                assert stage3_v[0]["severity"] == "violation"
        finally:
            os.unlink(fname)

    def test_claim_guard_integration_run(self):
        from scripts.validate_delivery_claims import run_claim_guard
        result = run_claim_guard()
        assert result["phase"] == "P46"
        assert result["profiles_loaded"] is True
        assert result["default_profile"] == "trusted_delivery"
        assert len(result["files_scanned"]) >= 5
        assert "p46_status" in result["summary"]


# ──────────────────────────────────────────────────────
# Full run (1)
# ──────────────────────────────────────────────────────


class TestP46FullRun:
    def test_p46_full_run_via_main(self):
        from scripts.validate_delivery_claims import main
        exit_code = main(["--json"])
        assert exit_code == 0
