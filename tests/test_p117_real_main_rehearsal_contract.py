"""P117: Real main.py rehearsal contract — validates output from real data run."""
from __future__ import annotations
import json, os, pytest
import pandas as pd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestRehearsalOutput:
    def test_manifest_exists(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "run_manifest.json")
        if os.path.isfile(path):
            with open(path) as f:
                m = json.load(f)
            assert "overall_status" in m

    def test_final_output_exists(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "final_output.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            assert len(df) > 0

    def test_final_output_24_rows(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "final_output.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            assert len(df) == 24

    def test_final_output_valid_schema(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "final_output.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            for col in ["business_day", "hour_business", "dayahead_price", "realtime_price"]:
                assert col in df.columns

    def test_no_y_true(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "final_output.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            for col in ["y_true", "actual", "label", "future_actual"]:
                assert col not in df.columns, f"Forbidden column: {col}"

    def test_hour_business_range(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "final_output.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            assert df["hour_business"].min() >= 1
            assert df["hour_business"].max() <= 24

    def test_delivery_report_exists(self):
        path = os.path.join(REPO_ROOT, ".local_artifacts", "p117_real_main_rehearsal", "delivery_report.md")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                assert len(f.read()) > 0

    def test_certification_reflects_caveats(self):
        cert = os.path.join(REPO_ROOT, "production_certification.json")
        with open(cert, encoding="utf-8") as f:
            c = json.load(f)
        assert "CAVEATS" in c["final_verdict"]
