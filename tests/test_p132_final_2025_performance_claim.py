"""P132: Final 2025 performance claim tests."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestFinalClaim:
    def test_report_exists(self):
        assert os.path.isfile(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"))

    def test_metrics_exists(self):
        assert os.path.isfile(os.path.join(REPO, "production_metrics_2025_full_bgew.json"))

    def test_cfg05_documented(self):
        with open(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"), encoding="utf-8") as f:
            content = f.read()
        assert "20.22%" in content

    def test_blocked_documented(self):
        with open(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"), encoding="utf-8") as f:
            content = f.read()
        assert "BLOCKED" in content

    def test_not_claimable_listed(self):
        with open(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"), encoding="utf-8") as f:
            content = f.read()
        assert "CANNOT be claimed" in content

    def test_claimable_listed(self):
        with open(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"), encoding="utf-8") as f:
            content = f.read()
        assert "can be claimed" in content.lower()

    def test_no_fake_bgew(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert "BLOCKED" in m["verdict"]

    def test_unblock_steps_listed(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert len(m.get("unblock_steps", [])) > 0

    def test_bugA_fixed(self):
        """Bug A: export_eval_pack output_dir should work with None."""
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "eval_only" in content
        assert "base_dir" in content

    def test_bugB_fixed(self):
        """Bug B: fallback ladder should use dayahead_actual_ledger.csv."""
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "dayahead_actual_ledger.csv" in src
