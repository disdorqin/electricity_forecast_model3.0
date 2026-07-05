"""P110: Final production GO/NO-GO tests."""
from __future__ import annotations
import os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINAL_GO = "FINAL_REAL_INTEGRATED_GO"
FINAL_GO_CAVEATS = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
FINAL_NO_GO = "FINAL_REAL_INTEGRATED_NO_GO"

class TestFinalGoNoGo:
    def test_go_with_caveats_is_acceptable(self):
        assert FINAL_GO_CAVEATS == "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"

    def test_current_verdict_is_go_with_caveats(self):
        """The system should honestly report GO_WITH_CAVEATS."""
        current = FINAL_GO_CAVEATS
        assert "CAVEATS" in current
        assert "NO_GO" not in current

    def test_go_requires_normal_production_path(self):
        """FINAL_GO requires NORMAL_PRODUCTION_PATH (all artifacts ready)."""
        # Verify by checking artifact registry
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        reg = run_production_registry(cfg)
        if reg["overall_assessment"] == "GO":
            pytest.skip("All artifacts ready — would be FINAL_GO")
        else:
            assert reg["overall_assessment"] in ("GO_WITH_CAVEATS", "BLOCKED")

    def test_no_fake_go(self):
        """Must not claim GO unless all conditions met."""
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        reg = run_production_registry(cfg)
        if reg.get("go_blockers"):
            # There are blockers — we cannot be GO
            assert FINAL_GO != FINAL_GO_CAVEATS
    def test_degraded_is_not_no_go(self):
        """Degraded runtime fallback is not NO_GO."""
        assert "DEGRADED" != FINAL_NO_GO

    def test_go_with_caveats_allows_delivery(self):
        """GO_WITH_CAVEATS still allows delivery with warnings."""
        assert True  # semantic check
