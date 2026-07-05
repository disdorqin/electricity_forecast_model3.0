"""
tests/test_p51_25_mechanism_alignment.py — P51 tests (8+).

Tests: alignment audit completeness, action item coverage,
mechanism absorption decisions, forbidden copy detection.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestAlignmentReportExists:
    """Verify the audit report was created."""

    def test_report_exists(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        assert os.path.isfile(path), f"Report not found: {path}"

    def test_report_has_key_sections(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        required_sections = [
            "2.5 Five-Stage Pipeline",
            "Adaptive Complete Training Days",
            "Delivery Status Three-Tier",
            "Postflight / Fallback / Manifest / Report",
            "Cannot Be Copied",
            "3.0 Innovation",
            "Action Items for P52-P60",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"


class TestAbsorptionDecisions:
    """Verify that absorption decisions are correct."""

    def test_realtime_classifier_not_absorbed(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        # Must NOT recommend absorbing realtime classifier
        assert "ledger_classifier" in content
        assert "cannot absorb" in content or "skip" in content

    def test_delivery_status_absorbed(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        assert "NORMAL" in content
        assert "DEGRADED_DELIVERED" in content
        assert "FAILED_NO_DELIVERY" in content


class TestActionItems:
    """Verify action items cover all P52-P60 phases."""

    def test_all_phases_covered(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        for phase in ["P52", "P53", "P54", "P55", "P56", "P57", "P58", "P59", "P60"]:
            assert phase in content, f"Phase {phase} not covered in action items"

    def test_forbidden_claims_referenced(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        assert "claim guard" in content


class TestNoForbiddenClaims:
    """Ensure the audit report itself doesn't contain forbidden claims."""

    def test_no_forbidden_production_claims(self):
        path = "docs/reports/p51_25_mechanism_alignment_audit.md"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().lower()
        forbidden = [
            "2.97% production",
            "69.96% production",
        ]
        for claim in forbidden:
            if claim in content:
                # Must be caveated
                assert "research only" in content or "not delivery" in content, \
                    f"Found uncaveated forbidden claim: {claim}"
