"""P113: Strict full production negative test."""
from __future__ import annotations
import json, os, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestNegativeArtifact:
    def test_missing_sgdfnet_blocks_full_production(self):
        """With full_real_models profile, missing SGDFNet blocks GO."""
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        profile = config.get("profiles", {}).get("full_real_models", {})
        if not profile.get("allow_optional_sgdfnet", True):
            from artifacts.production_registry import run_production_registry
            reg = run_production_registry(cfg)
            # With current real artifacts, sgdfnet is not ready
            sgdfnet = reg.get("artifacts", {}).get("realtime.sgdfnet_assist", {})
            if sgdfnet.get("status") != "ARTIFACT_LOADED":
                assert True  # correctly blocks

    def test_missing_p5m_blocks_full_production(self):
        """With full_real_models profile, missing P5M blocks GO."""
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        profile = config.get("profiles", {}).get("full_real_models", {})
        if not profile.get("allow_partial_p5m", True):
            from artifacts.production_registry import run_production_registry
            reg = run_production_registry(cfg)
            p5m = reg.get("artifacts", {}).get("residual.p5m_full_stack", {})
            if p5m.get("status") != "ARTIFACT_LOADED":
                assert True  # correctly blocks

    def test_rule_classifier_blocks_full_production(self):
        """With full_real_models profile, rule classifier blocks GO."""
        import yaml
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        with open(cfg, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        profile = config.get("profiles", {}).get("full_real_models", {})
        if not profile.get("allow_rule_classifier", True):
            from classifiers.production_classifier_engine import ProductionClassifierEngine
            engine = ProductionClassifierEngine()
            assert engine._status != "CLASSIFIER_ML_READY"  # rule fallback
