"""
tests/test_p49_final_delivery_audit.py — P49 tests (13+).

Tests: final audit checks, runner imports, profile checks,
README/runbook consistency, forbidden file detection.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFinalAuditChecks:
    """Tests for individual audit functions."""

    def test_audit_runs_and_returns_expected_keys(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["phase"] == "P49"
        assert "checks" in result
        assert "summary" in result

    def test_audit_checks_all_major_items(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        check_names = list(result["checks"].keys())
        assert "profile_registry_exists" in check_names
        assert "readme_exists" in check_names
        assert "runbook_exists" in check_names
        assert "runner_cli_exists" in check_names
        assert "delivery_status_exists" in check_names

    def test_audit_profile_checks(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["checks"]["trusted_delivery_profile"]["passed"] is True
        assert result["checks"]["balanced_candidate_profile"]["passed"] is True
        assert result["checks"]["research_all_models_profile"]["passed"] is True

    def test_audit_stage3_quarantined(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["checks"]["stage3_quarantined_in_trusted"]["passed"] is True
        assert result["checks"]["stage3_quarantined_label"]["passed"] is True

    def test_audit_readme_no_forbidden_claims(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["checks"]["readme_no_forbidden_production_claims"]["passed"] is True

    def test_audit_no_csv_committed(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["checks"]["no_csv_committed"]["passed"] is True

    def test_audit_no_forbidden_files(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["checks"]["no_forbidden_files_in_repo"]["passed"] is True

    def test_audit_runbook_refs_trusted_delivery(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["checks"]["runbook_refs_trusted_delivery"]["passed"] is True

    def test_audit_overall_passes(self):
        from scripts.run_p49_final_delivery_audit import run_final_audit
        result = run_final_audit()
        assert result["summary"]["p49_status"] == "P49_FINAL_AUDIT_PASS"
        assert result["summary"]["failed"] == 0

    def test_audit_main_function(self):
        from scripts.run_p49_final_delivery_audit import main
        exit_code = main(["--json"])
        assert exit_code == 0


class TestAuditHelperFunctions:
    """Tests for helper functions used by the audit."""

    def test_file_contains_found(self):
        from scripts.run_p49_final_delivery_audit import _file_contains
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Hello world\n")
            fname = f.name
        try:
            assert _file_contains(fname, "hello") is True
            assert _file_contains(fname, "world") is True
            assert _file_contains(fname, "nope") is False
        finally:
            os.unlink(fname)

    def test_has_trusted_delivery_ref(self):
        from scripts.run_p49_final_delivery_audit import _has_trusted_delivery_ref
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Using trusted_delivery profile\n")
            fname = f.name
        try:
            assert _has_trusted_delivery_ref(fname) is True
        finally:
            os.unlink(fname)

    def test_has_forbidden_claim_detects(self):
        from scripts.run_p49_final_delivery_audit import _has_forbidden_claim
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Our model achieves 2.97% production sMAPE\n")
            fname = f.name
        try:
            claims = _has_forbidden_claim(fname)
            assert len(claims) >= 1
        finally:
            os.unlink(fname)

    def test_has_forbidden_claim_with_caveat(self):
        from scripts.run_p49_final_delivery_audit import _has_forbidden_claim
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Research only, not delivery. 2.97% production sMAPE (research only).\n")
            fname = f.name
        try:
            claims = _has_forbidden_claim(fname)
            assert len(claims) == 0  # caveated
        finally:
            os.unlink(fname)
