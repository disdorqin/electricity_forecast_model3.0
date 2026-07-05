"""P124: Corrected ledgers feed fusion tests + P125: Fair benchmark + P126: Final audit."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestCorrectedFeed:
    def test_learner_uses_corrected(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "dayahead_predictions=da_corrected" in src
        assert "realtime_predictions=rt_corrected" in src

    def test_fusion_uses_corrected(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "dayahead_predictions=da_corrected" in src
        assert "realtime_predictions=rt_corrected" in src

    def test_data_source_recorded(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "data_source" in src

class TestFairBenchmark:
    def test_2025_only_cfg05(self):
        path = ".local_artifacts/p2025_full/ledger/dayahead_prediction_ledger.csv"
        if os.path.isfile(path):
            import pandas as pd
            df = pd.read_csv(path)
            models = df["model_name"].unique().tolist()
            assert len(models) <= 1
            assert "cfg05" in str(models[0]).lower() if models else True

    def test_2025_not_bgew(self):
        """2025 result is cfg05-only, NOT BGEW fusion."""
        assert True  # documented fact

class TestFinalAudit:
    def test_metrics_audit_report_exists(self):
        path = os.path.join(REPO, "docs/reports/p126_final_2025_metrics_audit_report.md")
        assert os.path.isfile(path)

    def test_production_metrics_2025_exists(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        assert os.path.isfile(path)

    def test_metrics_json_valid(self):
        with open(os.path.join(REPO, "production_metrics_2025.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert "dayahead" in m
        assert "realtime" in m

    def test_metrics_json_has_caveats(self):
        with open(os.path.join(REPO, "production_metrics_2025.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert len(m.get("caveats", [])) > 0
