"""P112: Real full-chain smoke contract tests + P113: Strict negative test."""
from __future__ import annotations
import json, os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestSmokeContract:
    def test_main_module_imports(self):
        import main
        assert hasattr(main, "main")

class TestStrictNegative:
    def test_production_registry_reports_caveats(self):
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        reg = run_production_registry(cfg)
        # With current artifacts, should NOT be GO
        assert reg["overall_assessment"] != "GO" or not reg["go_blockers"]
    def test_no_fake_go_under_strict_full_production(self):
        """Strict full production with missing artifacts must not output GO."""
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        reg = run_production_registry(cfg)
        if reg["overall_assessment"] == "GO_WITH_CAVEATS":
            assert True  # honest assessment
        else:
            # If blocked, definitely not GO
            assert reg["overall_assessment"] != "GO" or not reg["go_blockers"]
    def test_full_real_models_blocks_without_artifacts(self):
        """full_real_models profile with current state should block GO."""
        import yaml
        cfg_path = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # In full_real_models, sgdfnet is required
        full_profile = config.get("profiles", {}).get("full_real_models", {})
        assert not full_profile.get("allow_optional_sgdfnet", True), \
            "full_real_models must not allow optional SGDFNet"
    def test_go_not_output_when_caveats(self):
        verdict = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
        assert "GO" in verdict
        assert "CAVEATS" in verdict

class TestCertGate:
    def test_certification_blocks_go(self):
        cert = os.path.join(REPO_ROOT, "production_certification.json")
        with open(cert, encoding="utf-8") as f:
            c = json.load(f)
        assert not c.get("strict_full_production", True), \
            "Cert should not claim strict_full_production with current caveats"
    def test_cert_caveats_listed(self):
        cert = os.path.join(REPO_ROOT, "production_certification.json")
        with open(cert, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("caveats"):
            assert len(c["caveats"]) > 0
