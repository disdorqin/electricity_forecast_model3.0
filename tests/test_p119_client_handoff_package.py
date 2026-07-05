"""P119: Client handoff package freeze tests."""
from __future__ import annotations
import os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN_CLAIMS = [
    "CLIENT_DELIVERY_READY_PRODUCTION",
    "SGDFNet production ready",
    "Full P5M ready",
    "ML classifier production ready",
    "Realtime deep model beats DA",
]

# These may appear in caveats context but not as standalone production claims
CAVEAT_ALLOWED = [
    "FINAL_REAL_INTEGRATED_GO",
]

class TestClientDocsExist:
    def test_delivery_note(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_DELIVERY_NOTE.md"))

    def test_runbook(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_RUNBOOK.md"))

    def test_caveats(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_CAVEATS.md"))

    def test_demo_commands(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_DEMO_COMMANDS.md"))

    def test_acceptance_checklist(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_ACCEPTANCE_CHECKLIST.md"))

class TestNoForbiddenClaims:
    def test_delivery_note_no_fake_go(self):
        path = os.path.join(REPO_ROOT, "docs", "CLIENT_DELIVERY_NOTE.md")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # FINAL_REAL_INTEGRATED_GO (without CAVEATS) is forbidden as a standalone production claim
        assert "FINAL_REAL_INTEGRATED_GO\n" not in content
        assert "FINAL_REAL_INTEGRATED_GO " not in content
        for claim in FORBIDDEN_CLAIMS:
            assert claim not in content

    def test_runbook_no_fake_go(self):
        path = os.path.join(REPO_ROOT, "docs", "CLIENT_RUNBOOK.md")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for claim in FORBIDDEN_CLAIMS:
            assert claim not in content

    def test_caveats_mentions_blocked_claims(self):
        path = os.path.join(REPO_ROOT, "docs", "CLIENT_CAVEATS.md")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert any(claim.lower() in content.lower() for claim in FORBIDDEN_CLAIMS)

    def test_demo_commands_no_fake_go(self):
        path = os.path.join(REPO_ROOT, "docs", "CLIENT_DEMO_COMMANDS.md")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        for claim in FORBIDDEN_CLAIMS:
            assert claim not in content

class TestVersion:
    def test_version_is_rc(self):
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            v = f.read().strip()
        assert "rc" in v
        assert "3.0.0" in v
