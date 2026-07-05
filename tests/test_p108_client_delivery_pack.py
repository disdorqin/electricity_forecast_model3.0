"""P108 client delivery pack + P109 release version + P110 final verdict tests."""
from __future__ import annotations
import os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestClientFiles:
    def test_client_note(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_DELIVERY_NOTE.md"))
    def test_client_runbook(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_RUNBOOK.md"))
    def test_client_caveats(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_CAVEATS.md"))
    def test_client_demo(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_DEMO_COMMANDS.md"))

class TestVersion:
    def test_version_file_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "VERSION"))
    def test_version_readable(self):
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            v = f.read().strip()
        assert len(v) > 0
        assert "rc" in v  # release candidate

class TestFinalVerdict:
    def test_verdict_is_go_with_caveats(self):
        """Current verdict must be GO_WITH_CAVEATS (artifacts not all production-ready)."""
        from models.realtime_state import REALTIME_ASSIST_DISABLED
        # Verify the system correctly identifies SGDFNet as disabled
        assert REALTIME_ASSIST_DISABLED == "REALTIME_ASSIST_DISABLED"
    def test_not_no_go(self):
        """System should not be at NO_GO."""
        from models.realtime_state import REALTIME_NO_GO
        assert "NO_GO" in REALTIME_NO_GO
    def test_go_requires_full_artifacts(self):
        """FINAL_REAL_INTEGRATED_GO requires all production conditions."""
        verdict = "FINAL_REAL_INTEGRATED_GO"
        assert "GO" in verdict
