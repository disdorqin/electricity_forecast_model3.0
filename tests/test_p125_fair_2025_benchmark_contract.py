"""P125: Fair 2025 benchmark contract + P126 final audit."""
from __future__ import annotations
import json, os, pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestFairBenchmark:
    def test_dayahead_metrics_documented(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert "sMAPE_floor50" in m.get("dayahead", {})
        assert "MAE" in m.get("dayahead", {})

    def test_realtime_metrics_documented(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert "sMAPE_floor50" in m.get("realtime", {})
        assert "actual_source" in m.get("realtime", {})

    def test_realtime_uses_实时电价(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert m["realtime"]["actual_source"] == "实时电价"

    def test_cfg05_only_documented(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert "cfg05" in str(m.get("dayahead", {}).get("models", []))

    def test_no_bgew_fusion(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert m["dayahead"]["model_count"] == 1

class TestFinalAuditVerdict:
    def test_verdict_is_caveats(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert "CAVEATS" in m.get("verdict", "")

    def test_root_causes_documented(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert len(m.get("root_causes", [])) > 0

    def test_no_fake_claim(self):
        path = os.path.join(REPO, "production_metrics_2025.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        assert "GO" not in m.get("verdict", "") or "CAVEATS" in m.get("verdict", "")
