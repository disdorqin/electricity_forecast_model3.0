"""P129-P132: BGEW benchmark, residual, fair comparison, final claim tests."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestBgewBenchmark:
    def test_benchmark_script_exists(self):
        assert os.path.isfile(os.path.join(REPO, "scripts/run_p129_2025_trusted_bgew_benchmark.py"))

    def test_metrics_json_exists(self):
        path = os.path.join(REPO, "production_metrics_2025_full_bgew.json")
        assert os.path.isfile(path)

    def test_metrics_verdict_blocked(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert "BLOCKED" in m["verdict"]

    def test_cfg05_smape_documented(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert m["dayahead"]["cfg05_only_sMAPE_floor50"] == 20.22

    def test_local_bgew_documented(self):
        with open(os.path.join(REPO, "production_metrics_2025_full_bgew.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert m["dayahead"]["local_jun2026_trusted_bgew"] == 9.23

class TestResidual:
    def test_residual_script_exists(self):
        path = os.path.join(REPO, "scripts/run_p130_2025_residual_corrected_bgew.py")
        assert os.path.isfile(path)

class TestFairComparison:
    def test_fair_comparison_script_exists(self):
        path = os.path.join(REPO, "scripts/run_p131_2025_fair_comparison_matrix.py")
        assert os.path.isfile(path)

class TestClaimReport:
    def test_claim_report_exists(self):
        path = os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md")
        assert os.path.isfile(path)

    def test_claim_report_has_cfg05(self):
        with open(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"), encoding="utf-8") as f:
            content = f.read()
        assert "20.22%" in content

    def test_claim_report_no_fake_bgew(self):
        with open(os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md"), encoding="utf-8") as f:
            content = f.read()
        assert "BLOCKED" in content
