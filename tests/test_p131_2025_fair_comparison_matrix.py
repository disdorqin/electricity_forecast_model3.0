"""P131: Fair comparison + P132: Final claim tests."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestFairComparison:
    def test_script_exists(self):
        assert os.path.isfile(os.path.join(REPO, "scripts/run_p131_2025_fair_comparison_matrix.py"))

class TestFinalClaim:
    def test_report_exists(self):
        assert os.path.isfile(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"))

    def test_metrics_json_exists(self):
        assert os.path.isfile(os.path.join(REPO, "production_metrics_2025_full_bgew.json"))

    def test_metrics_verdict(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert m["verdict"] == "2025_FULL_BGEW_BENCHMARK_BLOCKED_CFG05_ONLY"

    def test_claimable_listed(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert len(m.get("claimable", [])) > 0

    def test_not_claimable_listed(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert len(m.get("not_claimable", [])) > 0

    def test_no_fake_claim(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        # If blocked, should not claim GO
        assert "BLOCKED" in m["verdict"] or "CAVEATS" in m["verdict"]
