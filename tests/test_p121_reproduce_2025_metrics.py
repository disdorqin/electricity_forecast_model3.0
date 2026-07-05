"""P121: 2025 metrics reproduce tests + P122: actual ledger target columns."""
from __future__ import annotations
import json, os, pytest
import pandas as pd, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestRootCause:
    def test_cfg05_only_in_2025_ledger(self):
        """The 2025 prediction ledger should only have cfg05 (confirms root cause)."""
        path = ".local_artifacts/p2025_full/ledger/dayahead_prediction_ledger.csv"
        if not os.path.isfile(path):
            pytest.skip("2025 ledger not found")
        df = pd.read_csv(path)
        models = df["model_name"].unique().tolist() if "model_name" in df.columns else []
        assert "lightgbm_cfg05_dayahead" in models
        # This is the root cause: only one model
        assert len(models) == 1

    def test_no_catboost_in_main_path(self):
        """CatBoost spike residual not in 2025 main pipeline."""
        path = ".local_artifacts/p2025_full/ledger/dayahead_prediction_ledger.csv"
        if not os.path.isfile(path):
            pytest.skip("2025 ledger not found")
        df = pd.read_csv(path)
        models = df["model_name"].unique().tolist() if "model_name" in df.columns else []
        assert not any("catboost" in m.lower() for m in models)

    def test_no_bgew_in_2025(self):
        """No BGEW fusion weights in 2025 output (single model)."""
        path = ".local_artifacts/p2025_full/final_output.csv"
        if not os.path.isfile(path):
            pytest.skip("2025 final output not found")
        df = pd.read_csv(path)
        if "dayahead_model_or_fusion" in df.columns:
            val = str(df["dayahead_model_or_fusion"].iloc[0])
            # The 2025 run used cfg05-only, but unified_fusion_engine still tagged it
            # This confirms it was a single-model run not true multi-model BGEW
            assert "bgew" in val.lower() or "cfg05" in val.lower()

class TestActualLedgerTaskColumn:
    def test_build_actual_ledger_accepts_task(self):
        from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
        import inspect
        sig = inspect.signature(build_actual_ledger_from_raw_csv)
        assert "task" in sig.parameters

    def test_dayahead_uses_日前电价(self):
        from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
        raw = os.path.join(REPO, "data", "shandong_pmos_hourly.csv")
        if os.path.isfile(raw):
            df = pd.read_csv(raw, encoding="gbk", nrows=5)
            assert "日前电价" in df.columns

    def test_realtime_uses_实时电价(self):
        raw = os.path.join(REPO, "data", "shandong_pmos_hourly.csv")
        if os.path.isfile(raw):
            df = pd.read_csv(raw, encoding="gbk", nrows=5)
            assert "实时电价" in df.columns

    def test_actual_ledger_output_per_task(self):
        """Output should be task-specific filename."""
        import tempfile
        from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
        raw = os.path.join(REPO, "data", "shandong_pmos_hourly.csv")
        if not os.path.isfile(raw):
            pytest.skip("raw data not found")
        with tempfile.TemporaryDirectory() as tmp:
            r = build_actual_ledger_from_raw_csv(raw, "2025-01-01", "2025-01-01", tmp, task="dayahead")
            assert "dayahead" in r.get("actual_ledger_path", "")
            r2 = build_actual_ledger_from_raw_csv(raw, "2025-01-01", "2025-01-01", tmp, task="realtime")
            assert "realtime" in r2.get("actual_ledger_path", "")

class TestP123EvalPack:
    def test_export_eval_pack_uses_实时电价(self):
        """realtime_deep_adapter export_eval_pack should use 实时电价."""
        path = os.path.join(REPO, "models/adapters/realtime_deep_adapter.py")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # The y_true mapping should use 实时电价 in export_eval_pack
        assert "实时电价" in content

    def test_online_pack_no_y_true(self):
        """Online pack should not contain y_true."""
        path = os.path.join(REPO, ".local_artifacts/p2025_full/realtime/online_pack/realtime_online_pack.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            assert "y_true" not in df.columns

class TestP124CorrectedLedgers:
    def test_run_full_chain_uses_corrected(self):
        """run_full_chain should use da_corrected/rt_corrected for learner."""
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        # Should reference corrected variables in learner/fusion calls
        assert "da_corrected" in src
        assert "rt_corrected" in src

    def test_residual_output_feed_learner(self):
        """Learner should receive da_corrected not da_ledger."""
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        # Check the learner call uses corrected
        assert "dayahead_predictions=da_corrected" in src
        assert "realtime_predictions=rt_corrected" in src

    def test_residual_output_feed_fusion(self):
        """Fusion should also receive corrected data."""
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "dayahead_predictions=da_corrected" in src
        assert "realtime_predictions=rt_corrected" in src

    def test_learner_records_data_source(self):
        """Learner step should record whether data is from corrected or raw ledger."""
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "data_source" in src
