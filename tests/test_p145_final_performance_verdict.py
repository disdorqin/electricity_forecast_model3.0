"""
tests/test_p145_final_performance_verdict.py — Tests for P145 Final Performance Verdict.

Validates verdict logic for each status, including:
  - PERFORMANCE_UNLOCKED_GO requires all conditions
  - PERFORMANCE_BLOCKED when single model
  - PERFORMANCE_NO_IMPROVEMENT when BGEW = cfg05
  - Verdict reads actual artifact data
  - Report format
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_p145_final_performance_verdict import (
    BASELINE_CFG05_ONLY_SMAPE,
    BASELINE_REALTIME_DA_SAFE_SMAPE,
    LOCAL_2026_BGEW_SMAPE,
    PERFORMANCE_TARGETS,
    determine_verdict,
    extract_key_metrics,
    read_all_phase_artifacts,
    run_p145_final_verdict,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def empty_artifacts_dir():
    """Create an empty artifacts directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def artifacts_bgew_unlocked():
    """Artifacts dir with BGEW that unlocks GO verdict."""
    with tempfile.TemporaryDirectory() as d:
        # P138: BGEW with 2 models, improving
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump({
                "sMAPE_floor50": 14.0,
                "model_count": 2,
                "models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
            }, f)

        # P140: Realtime improving
        p140_dir = os.path.join(d, "p140_realtime_unblock")
        os.makedirs(p140_dir)
        with open(os.path.join(p140_dir, "realtime_metrics.json"), "w") as f:
            json.dump({"sMAPE_floor50": 28.0}, f)

        yield d


@pytest.fixture
def artifacts_bgew_improved_caveats():
    """Artifacts dir with BGEW improved but partial."""
    with tempfile.TemporaryDirectory() as d:
        # P138: BGEW with only 1 model (blocked for full claim)
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump({
                "sMAPE_floor50": 18.0,
                "model_count": 1,
                "models": ["lightgbm_cfg05_dayahead"],
            }, f)
        yield d


@pytest.fixture
def artifacts_bgew_blocked():
    """Artifacts dir with no BGEW (blocked feature pipeline)."""
    with tempfile.TemporaryDirectory() as d:
        # No P138 directory at all
        yield d


@pytest.fixture
def artifacts_bgew_no_improvement():
    """Artifacts dir with BGEW that doesn't improve."""
    with tempfile.TemporaryDirectory() as d:
        # P138: BGEW with 2 models but worse than cfg05
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump({
                "sMAPE_floor50": 22.0,
                "model_count": 2,
                "models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
            }, f)
        yield d


# ── Test 1: Verdict logic for PERFORMANCE_UNLOCKED_GO ──────────────


class TestVerdictUnlockedGo:
    """Test PERFORMANCE_UNLOCKED_GO requires all conditions."""

    def test_unlocked_go_with_all_conditions(self, artifacts_bgew_unlocked):
        """UNLOCKED_GO when BGEW exists, model_count >= 2, improves, no fakes."""
        phases = read_all_phase_artifacts(artifacts_bgew_unlocked)
        metrics = extract_key_metrics(phases)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_UNLOCKED_GO"

    def test_unlocked_go_requires_bgew_exists(self):
        """UNLOCKED_GO requires BGEW to exist."""
        metrics = _make_metrics(bgew_2025_smape=None, bgew_model_count=2)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] != "PERFORMANCE_UNLOCKED_GO"

    def test_unlocked_go_requires_model_count_gte_2(self):
        """UNLOCKED_GO requires model_count >= 2."""
        metrics = _make_metrics(bgew_2025_smape=14.0, bgew_model_count=1)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] != "PERFORMANCE_UNLOCKED_GO"

    def test_unlocked_go_requires_bgew_improves(self):
        """UNLOCKED_GO requires BGEW improves vs cfg05."""
        metrics = _make_metrics(
            bgew_2025_smape=22.0,  # worse than 20.22
            bgew_model_count=2,
            bgew_improves_vs_cfg05=False,
        )
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] != "PERFORMANCE_UNLOCKED_GO"

    def test_unlocked_go_conditions_all_true(self, artifacts_bgew_unlocked):
        """All conditions should be True for UNLOCKED_GO."""
        phases = read_all_phase_artifacts(artifacts_bgew_unlocked)
        metrics = extract_key_metrics(phases)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["conditions"]["bgew_2025_exists"] is True
        assert verdict_data["conditions"]["bgew_model_count_gte_2"] is True
        assert verdict_data["conditions"]["bgew_improves_vs_cfg05"] is True
        assert verdict_data["conditions"]["no_fake_claims"] is True


# ── Test 2: PERFORMANCE_BLOCKED when single model ──────────────────


class TestVerdictBlocked:
    """Test PERFORMANCE_BLOCKED_FEATURE_PIPELINE when single model."""

    def test_blocked_when_no_bgew_artifacts(self, artifacts_bgew_blocked):
        """BLOCKED when P138 artifacts don't exist."""
        phases = read_all_phase_artifacts(artifacts_bgew_blocked)
        metrics = extract_key_metrics(phases)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_BLOCKED_FEATURE_PIPELINE"

    def test_blocked_when_single_model(self):
        """BLOCKED when model_count < 2."""
        metrics = _make_metrics(bgew_2025_smape=None, bgew_model_count=1)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_BLOCKED_FEATURE_PIPELINE"

    def test_blocked_reason_mentions_feature_pipeline(self, artifacts_bgew_blocked):
        """BLOCKED verdict mentions feature pipeline."""
        phases = read_all_phase_artifacts(artifacts_bgew_blocked)
        metrics = extract_key_metrics(phases)
        verdict_data = determine_verdict(metrics)
        reasons_text = " ".join(verdict_data["reasons"]).lower()
        assert "feature pipeline" in reasons_text or "2nd trusted model" in reasons_text


# ── Test 3: PERFORMANCE_NO_IMPROVEMENT when BGEW = cfg05 ───────────


class TestVerdictNoImprovement:
    """Test PERFORMANCE_NO_IMPROVEMENT when BGEW doesn't improve."""

    def test_no_improvement_when_bgew_worse(self, artifacts_bgew_no_improvement):
        """NO_IMPROVEMENT when BGEW sMAPE >= cfg05."""
        phases = read_all_phase_artifacts(artifacts_bgew_no_improvement)
        metrics = extract_key_metrics(phases)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_NO_IMPROVEMENT"

    def test_no_improvement_when_bgew_equal_to_cfg05(self):
        """NO_IMPROVEMENT when BGEW sMAPE = cfg05 (exactly equal)."""
        metrics = _make_metrics(
            bgew_2025_smape=20.22,  # exactly equal
            bgew_model_count=2,
            bgew_improves_vs_cfg05=False,
        )
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_NO_IMPROVEMENT"


# ── Test 4: PERFORMANCE_IMPROVED_WITH_CAVEATS ─────────────────────


class TestVerdictImprovedWithCaveats:
    """Test PERFORMANCE_IMPROVED_WITH_CAVEATS verdict."""

    def test_improved_with_caveats_when_partial(self, artifacts_bgew_improved_caveats):
        """IMPROVED_WITH_CAVEATS when BGEW improves but model_count < 2."""
        phases = read_all_phase_artifacts(artifacts_bgew_improved_caveats)
        metrics = extract_key_metrics(phases)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_IMPROVED_WITH_CAVEATS"

    def test_improved_with_caveats_when_only_realtime(self):
        """IMPROVED_WITH_CAVEATS when only realtime improves."""
        metrics = _make_metrics(
            bgew_2025_smape=None,
            bgew_model_count=0,
            bgew_improves_vs_cfg05=False,
            realtime_improved_smape=28.0,
            realtime_delta_improves=True,
        )
        verdict_data = determine_verdict(metrics)
        assert verdict_data["verdict"] == "PERFORMANCE_IMPROVED_WITH_CAVEATS"


# ── Test 5: Verdict reads actual artifact data ─────────────────────


class TestVerdictReadsArtifacts:
    """Test that verdict reads actual artifact data."""

    def test_reads_p138_artifact(self, artifacts_bgew_unlocked):
        """Verdict reads P138 BGEW metrics."""
        phases = read_all_phase_artifacts(artifacts_bgew_unlocked)
        assert phases["p138_rolling_bgew"]["exists"] is True
        assert phases["p138_rolling_bgew"]["data"]["sMAPE_floor50"] == 14.0

    def test_reads_p140_artifact(self, artifacts_bgew_unlocked):
        """Verdict reads P140 realtime metrics."""
        phases = read_all_phase_artifacts(artifacts_bgew_unlocked)
        assert phases["p140_realtime"]["exists"] is True

    def test_extracts_bgew_smape(self, artifacts_bgew_unlocked):
        """Metrics extraction gets BGEW sMAPE from P138."""
        phases = read_all_phase_artifacts(artifacts_bgew_unlocked)
        metrics = extract_key_metrics(phases)
        assert metrics["bgew_2025_smape"] == 14.0

    def test_extracts_model_count(self, artifacts_bgew_unlocked):
        """Metrics extraction gets model_count from P138."""
        phases = read_all_phase_artifacts(artifacts_bgew_unlocked)
        metrics = extract_key_metrics(phases)
        assert metrics["bgew_model_count"] == 2


# ── Test 6: Report format ──────────────────────────────────────────


class TestReportFormat:
    """Test the generated report format."""

    def test_verdict_json_created(self, artifacts_bgew_unlocked):
        """verdict.json is created in .local_artifacts/p145_final_verdict/."""
        result = run_p145_final_verdict(artifacts_bgew_unlocked)
        assert os.path.isfile(result["output_files"]["verdict_json"])

    def test_report_md_created(self, artifacts_bgew_unlocked):
        """Report markdown is created in docs/reports/."""
        result = run_p145_final_verdict(artifacts_bgew_unlocked)
        assert os.path.isfile(result["output_files"]["report_md"])

    def test_verdict_json_contains_verdict(self, artifacts_bgew_unlocked):
        """verdict.json contains the verdict string."""
        result = run_p145_final_verdict(artifacts_bgew_unlocked)
        with open(result["output_files"]["verdict_json"], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "verdict" in data
        assert data["verdict"] == "PERFORMANCE_UNLOCKED_GO"

    def test_report_contains_verdict_string(self, artifacts_bgew_unlocked):
        """Report markdown contains the verdict string."""
        result = run_p145_final_verdict(artifacts_bgew_unlocked)
        with open(result["output_files"]["report_md"], "r", encoding="utf-8") as f:
            content = f.read()
        assert "PERFORMANCE_UNLOCKED_GO" in content

    def test_report_contains_metrics_table(self, artifacts_bgew_unlocked):
        """Report markdown contains metrics table."""
        result = run_p145_final_verdict(artifacts_bgew_unlocked)
        with open(result["output_files"]["report_md"], "r", encoding="utf-8") as f:
            content = f.read()
        assert "cfg05-only 2025 sMAPE" in content
        assert "20.22" in content


# ── Test 7: Graceful degradation ───────────────────────────────────


class TestGracefulDegradation:
    """Test graceful handling of missing artifacts."""

    def test_no_crash_with_empty_dir(self, empty_artifacts_dir):
        """No crash when artifacts dir is empty."""
        result = run_p145_final_verdict(empty_artifacts_dir)
        assert result is not None
        assert "verdict" in result

    def test_empty_dir_gives_blocked_or_no_improvement(self, empty_artifacts_dir):
        """Empty artifacts dir gives BLOCKED or NO_IMPROVEMENT."""
        result = run_p145_final_verdict(empty_artifacts_dir)
        assert result["verdict"] in (
            "PERFORMANCE_BLOCKED_FEATURE_PIPELINE",
            "PERFORMANCE_NO_IMPROVEMENT",
        )


# ── Test 8: Target classification in verdict ───────────────────────


class TestTargetClassification:
    """Test that verdict correctly classifies performance targets."""

    def test_target_met_stretch(self):
        """sMAPE 9.0% meets stretch target."""
        metrics = _make_metrics(bgew_2025_smape=9.0, bgew_model_count=2,
                                bgew_improves_vs_cfg05=True)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["target_met"] == "stretch"

    def test_target_met_strong(self):
        """sMAPE 11.0% meets strong target."""
        metrics = _make_metrics(bgew_2025_smape=11.0, bgew_model_count=2,
                                bgew_improves_vs_cfg05=True)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["target_met"] == "strong"

    def test_target_met_reasonable(self):
        """sMAPE 14.0% meets reasonable target."""
        metrics = _make_metrics(bgew_2025_smape=14.0, bgew_model_count=2,
                                bgew_improves_vs_cfg05=True)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["target_met"] == "reasonable"

    def test_target_met_minimum(self):
        """sMAPE 19.0% meets minimum target."""
        metrics = _make_metrics(bgew_2025_smape=19.0, bgew_model_count=2,
                                bgew_improves_vs_cfg05=True)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["target_met"] == "minimum"

    def test_target_none_when_above_minimum(self):
        """sMAPE 25.0% doesn't meet any target."""
        metrics = _make_metrics(bgew_2025_smape=25.0, bgew_model_count=2,
                                bgew_improves_vs_cfg05=False)
        verdict_data = determine_verdict(metrics)
        assert verdict_data["target_met"] == "none"


# ── Helper: create metrics dict ────────────────────────────────────


def _make_metrics(**overrides) -> dict:
    """Create a metrics dict with defaults, overridden by kwargs."""
    defaults = {
        "cfg05_only_smape": BASELINE_CFG05_ONLY_SMAPE,
        "realtime_da_safe_smape": BASELINE_REALTIME_DA_SAFE_SMAPE,
        "local_2026_bgew_smape": LOCAL_2026_BGEW_SMAPE,
        "bgew_2025_smape": None,
        "bgew_model_count": 0,
        "bgew_improves_vs_cfg05": False,
        "residual_smape": None,
        "residual_is_noop": True,
        "realtime_improved_smape": None,
        "realtime_delta_improves": False,
        "p143_verdict": None,
        "p143_claims_count": 0,
        "p143_blocked_count": 0,
    }
    defaults.update(overrides)
    return defaults
