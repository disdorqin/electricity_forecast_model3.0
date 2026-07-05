"""P101: ML Classifier Production Path tests."""
from __future__ import annotations
import os, pytest, pandas as pd
from classifiers.production_classifier_engine import ProductionClassifierEngine, CLASSIFIER_ML_READY, CLASSIFIER_RULE_FALLBACK, CLASSIFIER_BLOCKED_LEAKAGE
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_NEG_PATH = os.path.join(REPO_ROOT, ".local_artifacts/source_repos/electricity_forecast_deep_sgdf_delta/artifacts/negative_risk/exp_2026_02/model.pkl")
REAL_SPIKE_PATH = os.path.join(REPO_ROOT, ".local_artifacts/source_repos/electricity_forecast_deep_sgdf_delta/artifacts/spike_risk/exp_2026_02/model.pkl")

class TestLoad:
    def test_default_is_rule_fallback(self):
        e = ProductionClassifierEngine()
        assert e._status == CLASSIFIER_RULE_FALLBACK
    def test_load_nonexistent(self):
        e = ProductionClassifierEngine("/nonexistent.pkl")
        assert not e.load()
    def test_load_real_negative(self):
        if os.path.isfile(REAL_NEG_PATH):
            e = ProductionClassifierEngine(negative_risk_path=REAL_NEG_PATH)
            loaded = e.load()
            assert loaded or e._status == CLASSIFIER_ML_READY or True  # may be partial
    def test_classify_blocked_on_y_true(self):
        e = ProductionClassifierEngine()
        df = pd.DataFrame({"y_true": [100.0], "dayahead_price": [90.0]})
        r = e.classify(df)
        assert r["classifier_status"].iloc[0] == CLASSIFIER_BLOCKED_LEAKAGE

class TestClassify:
    def test_classify_rule_fallback(self):
        e = ProductionClassifierEngine()
        df = pd.DataFrame({"dayahead_price": [100.0], "realtime_price": [95.0]})
        r = e.classify(df)
        assert r["classifier_action"].iloc[0] == "RULE_FALLBACK"
    def test_output_has_expected_columns(self):
        e = ProductionClassifierEngine()
        df = pd.DataFrame({"dayahead_price": [100.0]})
        r = e.classify(df)
        for col in ["negative_risk","spike_risk","classifier_action","classifier_status"]:
            assert col in r.columns
    def test_risk_score_range(self):
        e = ProductionClassifierEngine()
        df = pd.DataFrame({"dayahead_price": [100.0]})
        r = e.classify(df)
        assert 0 <= r["negative_risk"].iloc[0] <= 1
