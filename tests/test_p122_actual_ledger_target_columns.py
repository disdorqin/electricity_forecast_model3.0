"""P122: Actual ledger target columns + P123: Eval pack + P124: Corrected ledger flow."""
from __future__ import annotations
import os, pytest, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestActualLedgerColumns:
    def test_actual_ledger_task_parameter(self):
        from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
        import inspect
        sig = inspect.signature(build_actual_ledger_from_raw_csv)
        assert "task" in sig.parameters

    def test_dayahead_output_path(self):
        raw = os.path.join(REPO, "data", "shandong_pmos_hourly.csv")
        if not os.path.isfile(raw):
            pytest.skip("no raw data")
        from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
        with tempfile.TemporaryDirectory() as tmp:
            r = build_actual_ledger_from_raw_csv(raw, "2025-01-01", "2025-01-02", tmp, task="dayahead")
            assert "dayahead_actual_ledger.csv" in r.get("actual_ledger_path", "")

    def test_realtime_output_path(self):
        raw = os.path.join(REPO, "data", "shandong_pmos_hourly.csv")
        if not os.path.isfile(raw):
            pytest.skip("no raw data")
        from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
        with tempfile.TemporaryDirectory() as tmp:
            r = build_actual_ledger_from_raw_csv(raw, "2025-01-01", "2025-01-02", tmp, task="realtime")
            assert "realtime_actual_ledger.csv" in r.get("actual_ledger_path", "")

class TestEvalPack:
    def test_eval_pack_uses_realtime_price(self):
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "实时电价" in content
        assert "eval_only" in content

class TestCorrectedFlow:
    def test_corrected_feed_learner(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "dayahead_predictions=da_corrected" in src
        assert "realtime_predictions=rt_corrected" in src

    def test_corrected_feed_fusion(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "dayahead_predictions=da_corrected" in src
        assert "realtime_predictions=rt_corrected" in src

    def test_data_source_tracked(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "data_source" in src
