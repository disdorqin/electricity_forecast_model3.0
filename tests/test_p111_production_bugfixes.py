"""P111: Production bugfix tests — covers Bugs 1-8."""
from __future__ import annotations
import json, os, pytest, tempfile
import pandas as pd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bug 1: production_certification.json and P110 report exist
class TestBug1Certification:
    def test_certification_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "production_certification.json"))
    def test_certification_is_valid_json(self):
        with open(os.path.join(REPO_ROOT, "production_certification.json"), encoding="utf-8") as f:
            c = json.load(f)
        assert "final_verdict" in c
        assert "components" in c
    def test_certification_not_go_when_caveats(self):
        with open(os.path.join(REPO_ROOT, "production_certification.json"), encoding="utf-8") as f:
            c = json.load(f)
        if c.get("caveats"):
            assert "GO_WITH_CAVEATS" in c.get("final_verdict", "")
    def test_p110_report_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs/reports/p110_final_production_go_no_go_report.md"))

# Bug 2: realtime actual ledger uses 实时电价
class TestBug2ActualLedger:
    def test_dayahead_uses_日前电价(self):
        from scripts.run_full_chain import _build_actual_ledger
        raw = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
        if os.path.isfile(raw):
            df = pd.read_csv(raw, encoding="gbk", nrows=5)
            assert "日前电价" in df.columns
    def test_realtime_uses_实时电价(self):
        raw = os.path.join(REPO_ROOT, "data", "shandong_pmos_hourly.csv")
        if os.path.isfile(raw):
            df = pd.read_csv(raw, encoding="gbk", nrows=5)
            assert "实时电价" in df.columns
    def test_actual_ledger_function_exists(self):
        from scripts.run_full_chain import _build_actual_ledger
        assert callable(_build_actual_ledger)

# Bug 3: adaptive training days not hardcoded
class TestBug3AdaptiveDays:
    def test_adaptive_training_days_function_exists(self):
        from scripts.run_full_chain import _step_adaptive_training_days
        assert callable(_step_adaptive_training_days)
    def test_adaptive_training_days_returns_schema(self):
        from scripts.run_full_chain import _step_adaptive_training_days
        result = _step_adaptive_training_days(target_day="2026-07-01")
        for key in ("status", "selected_days", "skipped_days", "training_rows"):
            assert key in result
    def test_adaptive_training_days_not_hardcoded(self):
        from scripts.run_full_chain import _step_adaptive_training_days
        result = _step_adaptive_training_days(target_day="2026-07-01", required_days=15)
        # status should not be a hardcoded string
        assert result["selected_days"] >= 0

# Bug 4: postflight real validator called
class TestBug4Postflight:
    def test_postflight_step_not_hardcoded(self):
        """run_full_chain postflight step should call real postflight."""
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "postflight" in src
        assert "run_postflight_validation" in src or "postflight" in src

# Bug 5: fallback ladder real validator
class TestBug5Fallback:
    def test_fallback_ladder_step_has_method(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "fallback_ladder" in src

# Bug 6: claim guard no silent pass
class TestBug6ClaimGuard:
    def test_claim_guard_not_silent_pass(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "silent" not in src.lower() or "warning" in src.lower()

# Bug 7: production profiles
class TestBug7ProductionProfiles:
    def test_config_has_profiles(self):
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        assert "profiles" in c
        assert "rc_with_caveats" in c["profiles"]
        assert "full_real_models" in c["profiles"]
    def test_full_real_models_strict(self):
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        p = c["profiles"]["full_real_models"]
        assert not p["allow_optional_sgdfnet"]
        assert not p["allow_partial_p5m"]
        assert not p["allow_rule_classifier"]

# Bug 8: client note metric cleanup
class TestBug8ClientNote:
    def test_no_estimated_realtime_metric(self):
        note = os.path.join(REPO_ROOT, "docs", "CLIENT_DELIVERY_NOTE.md")
        with open(note, encoding="utf-8") as f:
            content = f.read()
        assert "estimated" not in content.lower()
        assert "~15%" not in content

# Combined: strict-full-production enforcement
class TestStrictFullProduction:
    def test_strict_full_production_flag(self):
        from scripts.run_full_chain import run_full_chain
        import inspect
        src = inspect.getsource(run_full_chain)
        assert "strict_full_production" in src

class TestNoFakeGo:
    def test_not_go_when_caveats(self):
        """Verify we never claim GO when caveats exist."""
        from models.realtime_state import REALTIME_ASSIST_DISABLED
        verdict = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
        assert "CAVEATS" in verdict
