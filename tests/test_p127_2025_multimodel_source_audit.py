"""P127-P132: 2025 multi-model tests."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestMultiModelSource:
    def test_p31_models_exist(self):
        base = os.path.join(REPO, ".local_artifacts", "p31_p40_multimodel_fusion", "models")
        dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        assert len(dirs) >= 3, f"Expected >=3 model dirs, got {len(dirs)}: {dirs}"

    def test_catboost_spike_exists(self):
        path = os.path.join(REPO, ".local_artifacts/p31_p40_multimodel_fusion/models/catboost_spike_residual/catboost_spike_residual.cbm")
        assert os.path.isfile(path)

class TestMultiModelLedger:
    def test_ledger_exists(self):
        path = os.path.join(REPO, ".local_artifacts/p128_2025_multimodel/dayahead_prediction_ledger_2025.csv")
        if not os.path.isfile(path):
            pytest.skip("Ledger not yet generated")
        import pandas as pd
        df = pd.read_csv(path)
        models = df["model_name"].unique()
        assert len(models) >= 2, f"Expected >=2 models, got {len(models)}: {models}"

    def test_ledger_no_y_true(self):
        path = os.path.join(REPO, ".local_artifacts/p128_2025_multimodel/dayahead_prediction_ledger_2025.csv")
        if not os.path.isfile(path):
            pytest.skip("Ledger not yet generated")
        import pandas as pd
        df = pd.read_csv(path)
        for col in ["y_true", "actual", "label"]:
            assert col not in df.columns

class TestBgewBenchmark:
    def test_bgew_metrics_exist(self):
        path = os.path.join(REPO, ".local_artifacts/p129_2025_bgew/2025_trusted_bgew_metrics.json")
        if not os.path.isfile(path):
            pytest.skip("Bgew metrics not yet generated")
        with open(path) as f:
            m = json.load(f)
        assert "cfg05_only" in m
        assert "trusted_bgew" in m

    def test_cfg05_metrics_reasonable(self):
        path = os.path.join(REPO, ".local_artifacts/p129_2025_bgew/2025_trusted_bgew_metrics.json")
        if not os.path.isfile(path):
            pytest.skip("Bgew metrics not yet generated")
        with open(path) as f:
            m = json.load(f)
        smape = m.get("cfg05_only", {}).get("sMAPE_floor50")
        if smape:
            assert 10 < smape < 50

class TestFinalClaim:
    def test_claim_report_exists(self):
        path = os.path.join(REPO, "docs/reports/p132_final_2025_performance_claim_report.md")
        assert os.path.isfile(path)

    def test_production_metrics_2025_full_exists(self):
        path = os.path.join(REPO, "production_metrics_2025_full_bgew.json")
        assert os.path.isfile(path)
