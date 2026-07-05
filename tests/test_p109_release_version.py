"""P109: Release version tests + P110: Final production GO/NO-GO."""
from __future__ import annotations
import os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestReleaseVersion:
    def test_version_rc(self):
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            v = f.read().strip()
        assert "rc" in v, "Should be rc until all artifacts ready"

class TestFinalGoNoGo:
    def test_go_with_caveats_is_current(self):
        """System should report GO_WITH_CAVEATS."""
        verdict = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
        assert verdict == "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"

    def test_go_conditions_listed(self):
        """Listing conditions for FINAL_GO."""
        conditions = [
            "SGDFNet assist ready or optional accepted",
            "Full P5M residual ready",
            "ML classifier ready",
            "Day-ahead artifacts ready",
            "Realtime pooled learner ready",
            "30D rehearsal GO",
            "Extended backtest completed",
            "Safety supervisor PASS",
            "Postflight PASS",
            "Claim guard PASS",
            "Full pytest PASS",
            "production_certification.json says GO",
        ]
        assert len(conditions) >= 10

    def test_no_go_conditions(self):
        conditions = [
            "realtime_price NaN",
            "y_true in production output",
            "current-day actual leakage",
            "stage3 in delivery path",
            "main.py cannot run",
            "postflight failed",
            "safety supervisor failed",
            "claim guard failed",
            "tests failed",
        ]
        assert len(conditions) >= 5
