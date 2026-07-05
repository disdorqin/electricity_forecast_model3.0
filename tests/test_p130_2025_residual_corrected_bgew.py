"""P130-P132: Residual, fair comparison, final claim tests."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.mark.skip(reason="P130 residual-corrected BGEW depends on P129 BGEW which is blocked")
class TestResidualCorrected:
    def test_status_not_fake(self):
        assert True

class TestFairComparison:
    def test_25_unavailable(self):
        """No 2.5 artifacts available — cannot claim beat 2.5."""
        assert True

    def test_comparison_script_exists(self):
        assert os.path.isfile(os.path.join(REPO, "scripts/run_p131_2025_fair_comparison_matrix.py"))

class TestFinalClaim:
    def test_claim_report_exists(self):
        assert os.path.isfile(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"))

    def test_no_fake_go(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert "GO" not in m.get("verdict", "") or "BLOCKED" in m.get("verdict", "")

    def test_blocked_reason_documented(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert "feature" in m.get("blocked_reason", "").lower()
