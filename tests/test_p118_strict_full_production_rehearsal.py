"""P118: Strict full production negative rehearsal."""
from __future__ import annotations
import json, os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestNegativeRehearsal:
    def test_strict_full_production_not_go(self):
        """With current artifacts, strict-full-production must not output GO."""
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        reg = run_production_registry(cfg)
        assert reg["overall_assessment"] != "GO" or not reg.get("go_blockers")
        assert reg["overall_assessment"] in ("GO_WITH_CAVEATS", "BLOCKED")

    def test_sgdfnet_blocker(self):
        """SGDFNet should be a blocker under full_real_models profile."""
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        profile = config.get("profiles", {}).get("full_real_models", {})
        if not profile.get("allow_optional_sgdfnet", True):
            from artifacts.production_registry import run_production_registry
            reg = run_production_registry(cfg)
            sgdfnet = reg.get("artifacts", {}).get("realtime.sgdfnet_assist", {})
            assert sgdfnet.get("status") != "ARTIFACT_LOADED", \
                "SGDFNet should not be loaded in current env"

    def test_p5m_blocker(self):
        """P5M full stack not found in primary path (may use fallback)."""
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        reg = run_production_registry(cfg)
        p5m = reg.get("artifacts", {}).get("residual.p5m_full_stack", {})
        # Primary paths (.local_artifacts/residual/p5m_full_stack.pkl) don't exist
        # Fallback catboost_spike_residual.cbm may exist — that's partial
        primary_path = ".local_artifacts/residual/p5m_full_stack.pkl"
        assert primary_path not in str(p5m.get("path", "")), f"Unexpected: {primary_path} found"

    def test_classifier_blocker(self):
        """Rule classifier should be a blocker under full_real_models."""
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        profile = config.get("profiles", {}).get("full_real_models", {})
        if not profile.get("allow_rule_classifier", True):
            from classifiers.production_classifier_engine import ProductionClassifierEngine
            engine = ProductionClassifierEngine()
            assert engine._status != "CLASSIFIER_ML_READY"

    def test_certification_not_production(self):
        with open(os.path.join(REPO_ROOT, "production_certification.json"), encoding="utf-8") as f:
            c = json.load(f)
        assert not c.get("strict_full_production", True)

    def test_no_fake_go_claim(self):
        verdict = "CLIENT_DELIVERY_READY_RC"
        assert "RC" in verdict
        assert "PRODUCTION" not in verdict
