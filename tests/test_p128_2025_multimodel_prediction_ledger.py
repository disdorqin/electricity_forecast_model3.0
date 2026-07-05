"""P128-P129 multi-model ledger + BGEW benchmark tests."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestLedger:
    def test_ledger_file(self):
        path = ".local_artifacts/p128_2025_multimodel/dayahead_prediction_ledger_2025.csv"
        if not os.path.isfile(path):
            pytest.skip("Ledger not generated yet")
        import pandas as pd
        df = pd.read_csv(path)
        assert "model_name" in df.columns
        assert "y_pred" in df.columns
        assert "y_true" not in df.columns

    def test_multiple_models(self):
        path = ".local_artifacts/p128_2025_multimodel/dayahead_prediction_ledger_2025.csv"
        if not os.path.isfile(path):
            pytest.skip("Ledger not generated yet")
        import pandas as pd
        df = pd.read_csv(path)
        assert df["model_name"].nunique() >= 2

class TestBenchmark:
    def test_bgew_metrics(self):
        path = ".local_artifacts/p129_2025_bgew/2025_trusted_bgew_metrics.json"
        if not os.path.isfile(path):
            pytest.skip("Bgew metrics not generated yet")
        with open(path) as f:
            m = json.load(f)
        for k in ("cfg05_only", "catboost_spike", "trusted_bgew"):
            assert k in m
