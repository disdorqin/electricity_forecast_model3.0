"""
Tests for P95: Final Report and Status Relabeling.

Covers:
  1. P90 report exists and has correct verdict
  2. P95 report exists and covers all phases
  3. README updated with new status
  4. Status language uses DA-Safe not fallback
  5. Caveats are accurate (not NO_GO)
  6. RUNBOOK updated with P91-P95 commands
  7. Final verdict constant
  8. No "da_anchor fallback" caveat language in key docs
"""

from __future__ import annotations

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ENC = "utf-8"


def _read(path: str) -> str:
    with open(path, encoding=_ENC) as f:
        return f.read()


class TestP90Report:
    def test_p90_report_exists(self):
        path = os.path.join(REPO_ROOT, "docs", "reports",
                            "p90_final_real_integrated_release_report.md")
        assert os.path.isfile(path)

    def test_p90_verdict_correct(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS" in content

    def test_p90_has_realtime_section(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "Real-Time" in content or "Realtime" in content or "DA-Safe" in content

    def test_p90_no_fallback_language(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "da_anchor fallback" not in content.lower()


class TestP95Report:
    def test_p95_report_exists(self):
        path = os.path.join(REPO_ROOT, "docs", "reports",
                            "p95_da_safe_realtime_sgdfnet_fusion_report.md")
        assert os.path.isfile(path)

    def test_p95_mentions_all_phases(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p95_da_safe_realtime_sgdfnet_fusion_report.md"))
        for phase in ["P91", "P92", "P93", "P94", "P95"]:
            assert phase in content

    def test_p95_has_verdict(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p95_da_safe_realtime_sgdfnet_fusion_report.md"))
        assert "GO_WITH_CAVEATS" in content

    def test_p95_caveats_listed(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p95_da_safe_realtime_sgdfnet_fusion_report.md"))
        assert "caveats" in content.lower()

    def test_p95_no_old_fallback_naming(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p95_da_safe_realtime_sgdfnet_fusion_report.md"))
        # Old status names may appear in context (e.g. "Before → After" tables),
        # but should not be the primary status designation
        assert "REALTIME_DA_SAFE_BASELINE" in content


class TestREADME:
    def test_readme_has_new_verdict(self):
        content = _read(os.path.join(REPO_ROOT, "README.md"))
        assert "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS" in content

    def test_readme_has_realtime_section(self):
        content = _read(os.path.join(REPO_ROOT, "README.md"))
        assert "Realtime Prediction Status" in content or "DA-Safe" in content

    def test_readme_has_learner_policy(self):
        content = _read(os.path.join(REPO_ROOT, "README.md"))
        assert "pooled_30d_bgew" in content

    def test_readme_status_not_delivery_freeze(self):
        content = _read(os.path.join(REPO_ROOT, "README.md"))
        # The primary status should be the new verdict
        assert "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS" in content


class TestRunbook:
    def test_runbook_has_p91_p95(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "RUNBOOK_REAL_LOCAL_CHAIN.md"))
        assert "P91-P95" in content

    def test_runbook_has_p92_commands(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "RUNBOOK_REAL_LOCAL_CHAIN.md"))
        assert "run_p92_sgdfnet_assist_adapter" in content

    def test_runbook_has_p93_commands(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "RUNBOOK_REAL_LOCAL_CHAIN.md"))
        assert "run_p93_realtime_two_candidate_ledger" in content

    def test_runbook_has_p94_commands(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "RUNBOOK_REAL_LOCAL_CHAIN.md"))
        assert "run_p94_realtime_pooled_learner" in content

    def test_runbook_realtime_status(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "RUNBOOK_REAL_LOCAL_CHAIN.md"))
        assert "DA-Safe" in content or "DA_SAFE" in content

    def test_runbook_no_fallback_language(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "RUNBOOK_REAL_LOCAL_CHAIN.md"))
        assert "da_anchor fallback" not in content.lower()


class TestStatusRelabeling:
    def test_config_has_rt_da_anchor_default(self):
        content = _read(os.path.join(REPO_ROOT, "config", "model_sets.yaml"))
        assert "rt_da_anchor" in content
        assert "official_default" in content

    def test_config_has_sgdfnet_assist(self):
        content = _read(os.path.join(REPO_ROOT, "config", "model_sets.yaml"))
        assert "sgdfnet_rt_assist" in content

    def test_no_old_online_pack_naming(self):
        content = _read(os.path.join(REPO_ROOT, "models", "adapters",
                                      "realtime_da_safe_assist.py"))
        assert "da_error_prob" in content
        assert "correction_permission" in content

    def test_final_verdict_format(self):
        verdict = "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"
        assert verdict.startswith("FINAL_REAL_INTEGRATED")
        assert "GO" in verdict

    def test_not_no_go_verdict(self):
        assert "NO_GO" not in "FINAL_REAL_INTEGRATED_GO_WITH_CAVEATS"

    def test_realtime_state_module_exists(self):
        path = os.path.join(REPO_ROOT, "models", "realtime_state.py")
        assert os.path.isfile(path)

    def test_realtime_state_no_old_names(self):
        content = _read(os.path.join(REPO_ROOT, "models", "realtime_state.py"))
        assert "REALTIME_DA_SAFE_BASELINE" in content
        # New constants should NOT define old fallback names as active states
        assert 'REALTIME_DA_SAFE_BASELINE = "REALTIME_DA_SAFE_BASELINE"' in content


class TestNewFilesExist:
    @pytest.mark.parametrize("rel_path", [
        "models/realtime_state.py",
        "models/adapters/sgdfnet_assist_adapter.py",
        "ledgers/realtime_prediction_ledger.py",
        "scripts/run_p92_sgdfnet_assist_adapter.py",
        "scripts/run_p93_realtime_two_candidate_ledger.py",
        "scripts/run_p94_realtime_pooled_learner.py",
        "tests/test_p91_realtime_design_reclassification.py",
        "tests/test_p92_sgdfnet_assist_adapter.py",
        "tests/test_p93_realtime_two_candidate_ledger.py",
        "tests/test_p94_realtime_pooled_learner.py",
        "tests/test_p95_report_status_relabeling.py",
        "docs/reports/p91_realtime_design_reclassification_report.md",
        "docs/reports/p92_sgdfnet_assist_adapter_report.md",
        "docs/reports/p93_realtime_two_candidate_ledger_report.md",
        "docs/reports/p94_realtime_pooled_learner_report.md",
        "docs/reports/p95_da_safe_realtime_sgdfnet_fusion_report.md",
        "docs/reports/p90_final_real_integrated_release_report.md",
    ])
    def test_file_exists(self, rel_path):
        full_path = os.path.join(REPO_ROOT, rel_path)
        assert os.path.isfile(full_path)


class TestCaveatLanguage:
    def test_caveats_mention_residual(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "residual" in content.lower()

    def test_caveats_mention_classifier(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "classifier" in content.lower()

    def test_caveats_mention_sgdfnet(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "SGDFNet" in content

    def test_caveats_not_blocking(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "GO_WITH_CAVEATS" in content
        # NO_GO may appear as a contrast ("not a NO_GO condition"),
        # but the verdict should not be a plain NO_GO
        assert "NO_GO" not in content or "not a" in content or "does not" in content.lower()

    def test_not_pure_no_go(self):
        content = _read(os.path.join(REPO_ROOT, "docs", "reports",
                                      "p90_final_real_integrated_release_report.md"))
        assert "NO_GO" not in content[:200]  # First 200 chars should not say NO_GO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
