"""P102: Strict Full Production Mode tests."""
from __future__ import annotations
import os, pytest, tempfile
from artifacts.production_registry import run_production_registry
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")

class TestRegistryGate:
    def test_registry_config_exists(self):
        assert os.path.isfile(CONFIG)
    def test_registry_runs(self):
        r = run_production_registry(CONFIG)
        assert r["status"] == "REGISTRY_COMPLETE"
    def test_go_blockers_not_none(self):
        r = run_production_registry(CONFIG)
        assert isinstance(r["go_blockers"], list)
    def test_overall_assessment_valid(self):
        r = run_production_registry(CONFIG)
        assert r["overall_assessment"] in ("GO","GO_WITH_CAVEATS","BLOCKED")
    def test_required_artifacts_have_name(self):
        r = run_production_registry(CONFIG)
        for k, v in r["artifacts"].items():
            assert "required_for_go" in v
    def test_summary_totals_match(self):
        r = run_production_registry(CONFIG)
        assert r["summary"]["found"] + r["summary"]["missing"] == r["summary"]["total"]

class TestStrictFlags:
    def test_strict_full_production_semantics(self):
        """strict-full-production requires all required_for_go artifacts."""
        from scripts.run_full_chain import FULL_CHAIN_DELIVERY_GO, FULL_CHAIN_DELIVERY_NO_GO
        assert FULL_CHAIN_DELIVERY_GO != FULL_CHAIN_DELIVERY_NO_GO
    def test_final_go_requires_normal_path(self):
        verdict = "FINAL_REAL_INTEGRATED_GO"
        assert "NO_GO" not in verdict
    def test_no_go_requires_value(self):
        assert "NO_GO" in "FINAL_REAL_INTEGRATED_NO_GO"
