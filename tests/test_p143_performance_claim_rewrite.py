"""
tests/test_p143_performance_claim_rewrite.py — Tests for P143 Performance Claim Rewrite.

Validates that performance claims are built ONLY from actual artifact data,
with no fake numbers, correct labeling, and proper blocking logic.
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

from scripts.run_p143_performance_claim_rewrite import (
    BASELINE_CFG05_ONLY_SMAPE,
    BASELINE_REALTIME_DA_SAFE_SMAPE,
    LOCAL_2026_BGEW_SMAPE,
    PERFORMANCE_TARGETS,
    build_performance_claims,
    generate_client_caveats,
    generate_client_delivery_note,
    generate_metrics_json,
    generate_report_md,
    read_p138_rolling_bgew,
    read_p139_residual,
    read_p140_realtime,
    read_p141_audit,
    read_p142_comparison,
    run_p143_performance_claims,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def empty_artifacts_dir():
    """Create an empty artifacts directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def artifacts_with_p138():
    """Create artifacts dir with P138 BGEW data (improving)."""
    with tempfile.TemporaryDirectory() as d:
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        metrics = {
            "sMAPE_floor50": 15.5,
            "model_count": 2,
            "models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
            "status": "COMPLETE",
        }
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump(metrics, f)
        yield d


@pytest.fixture
def artifacts_with_p138_blocked():
    """Create artifacts dir with P138 BGEW data (blocked, single model)."""
    with tempfile.TemporaryDirectory() as d:
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        metrics = {
            "sMAPE_floor50": 20.22,
            "model_count": 1,
            "models": ["lightgbm_cfg05_dayahead"],
            "status": "BLOCKED_INSUFFICIENT_MODELS",
        }
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump(metrics, f)
        yield d


@pytest.fixture
def artifacts_with_p138_no_improvement():
    """Create artifacts dir with P138 BGEW data (no improvement)."""
    with tempfile.TemporaryDirectory() as d:
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        metrics = {
            "sMAPE_floor50": 22.0,
            "model_count": 2,
            "models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
            "status": "COMPLETE_NO_IMPROVEMENT",
        }
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump(metrics, f)
        yield d


@pytest.fixture
def artifacts_full():
    """Create artifacts dir with P138-P142 data."""
    with tempfile.TemporaryDirectory() as d:
        # P138
        p138_dir = os.path.join(d, "p138_rolling_bgew")
        os.makedirs(p138_dir)
        with open(os.path.join(p138_dir, "bgew_2025_metrics.json"), "w") as f:
            json.dump({
                "sMAPE_floor50": 14.0,
                "model_count": 2,
                "models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
            }, f)

        # P139
        p139_dir = os.path.join(d, "p139_residual_corrected")
        os.makedirs(p139_dir)
        with open(os.path.join(p139_dir, "residual_metrics.json"), "w") as f:
            json.dump({"sMAPE_floor50": 13.5, "status": "COMPLETE"}, f)

        # P140
        p140_dir = os.path.join(d, "p140_realtime_unblock")
        os.makedirs(p140_dir)
        with open(os.path.join(p140_dir, "realtime_metrics.json"), "w") as f:
            json.dump({"sMAPE_floor50": 28.0}, f)

        # P141
        p141_dir = os.path.join(d, "p141_negative_spike")
        os.makedirs(p141_dir)
        with open(os.path.join(p141_dir, "audit_summary.json"), "w") as f:
            json.dump({"status": "AUDIT_PASS"}, f)

        # P142
        p142_dir = os.path.join(d, "p142_fair_comparison")
        os.makedirs(p142_dir)
        with open(os.path.join(p142_dir, "comparison_metrics.json"), "w") as f:
            json.dump({"status": "COMPARISON_GENERATED"}, f)

        yield d


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── Test 1: Claims match actual metrics ────────────────────────────


class TestClaimsMatchActuals:
    """Test that claims match actual metrics from artifacts."""

    def test_baseline_claims_always_present(self, empty_artifacts_dir):
        """cfg05-only and realtime baselines are always claimed."""
        result = build_performance_claims(empty_artifacts_dir)
        claim_ids = [c["id"] for c in result["claims"]]
        assert "cfg05_only_2025" in claim_ids
        assert "realtime_da_safe_2025" in claim_ids

    def test_cfg05_value_matches_baseline(self, empty_artifacts_dir):
        """cfg05-only claim has exact baseline value."""
        result = build_performance_claims(empty_artifacts_dir)
        cfg05_claim = next(c for c in result["claims"] if c["id"] == "cfg05_only_2025")
        assert cfg05_claim["value"] == BASELINE_CFG05_ONLY_SMAPE
        assert cfg05_claim["value"] == 20.22

    def test_realtime_value_matches_baseline(self, empty_artifacts_dir):
        """Realtime claim has exact baseline value."""
        result = build_performance_claims(empty_artifacts_dir)
        rt_claim = next(c for c in result["claims"] if c["id"] == "realtime_da_safe_2025")
        assert rt_claim["value"] == BASELINE_REALTIME_DA_SAFE_SMAPE
        assert rt_claim["value"] == 33.03

    def test_bgew_claim_matches_artifact(self, artifacts_with_p138):
        """BGEW claim matches the actual P138 artifact value."""
        result = build_performance_claims(artifacts_with_p138)
        bgew_claims = [c for c in result["claims"] if c["id"] == "bgew_2025_rolling"]
        assert len(bgew_claims) == 1
        assert bgew_claims[0]["value"] == 15.5


# ── Test 2: No fake BGEW claim when blocked ────────────────────────


class TestNoFakeBGEWWhenBlocked:
    """Test that BGEW claim is blocked when artifacts are missing or insufficient."""

    def test_no_bgew_claim_when_artifacts_missing(self, empty_artifacts_dir):
        """No BGEW claim when P138 artifacts don't exist."""
        result = build_performance_claims(empty_artifacts_dir)
        bgew_claims = [c for c in result["claims"] if c["id"] == "bgew_2025_rolling"]
        assert len(bgew_claims) == 0

    def test_bgew_blocked_when_single_model(self, artifacts_with_p138_blocked):
        """BGEW blocked when model_count < 2."""
        result = build_performance_claims(artifacts_with_p138_blocked)
        bgew_claims = [c for c in result["claims"] if c["id"] == "bgew_2025_rolling"]
        assert len(bgew_claims) == 0
        blocked_ids = [bc["id"] for bc in result["blocked_claims"]]
        assert "bgew_2025_rolling" in blocked_ids

    def test_no_bgew_claim_when_no_improvement(self, artifacts_with_p138_no_improvement):
        """No BGEW claim when sMAPE doesn't improve vs cfg05."""
        result = build_performance_claims(artifacts_with_p138_no_improvement)
        bgew_claims = [c for c in result["claims"] if c["id"] == "bgew_2025_rolling"]
        assert len(bgew_claims) == 0


# ── Test 3: Local 2026 labeled correctly ───────────────────────────


class TestLocal2026Labeling:
    """Test that local 2026 BGEW 9.23% is labeled as local window."""

    def test_local_2026_claim_exists(self, empty_artifacts_dir):
        """Local 2026 claim always present."""
        result = build_performance_claims(empty_artifacts_dir)
        local_claims = [c for c in result["claims"] if c["id"] == "local_2026_bgew"]
        assert len(local_claims) == 1

    def test_local_2026_value_is_923(self, empty_artifacts_dir):
        """Local 2026 value is exactly 9.23."""
        result = build_performance_claims(empty_artifacts_dir)
        local_claim = next(c for c in result["claims"] if c["id"] == "local_2026_bgew")
        assert local_claim["value"] == 9.23
        assert local_claim["value"] == LOCAL_2026_BGEW_SMAPE

    def test_local_2026_has_local_window_caveat(self, empty_artifacts_dir):
        """Local 2026 claim has caveat about local window."""
        result = build_performance_claims(empty_artifacts_dir)
        local_claim = next(c for c in result["claims"] if c["id"] == "local_2026_bgew")
        assert local_claim["caveat"] is not None
        assert "local window" in local_claim["caveat"].lower() or "NOT full year" in local_claim["caveat"]

    def test_local_2026_period_not_full_year(self, empty_artifacts_dir):
        """Local 2026 period is labeled as local window, not full year."""
        result = build_performance_claims(empty_artifacts_dir)
        local_claim = next(c for c in result["claims"] if c["id"] == "local_2026_bgew")
        assert "local window" in local_claim["period"].lower() or "NOT full year" in local_claim["period"]


# ── Test 4: Improvement calculation matches actual ─────────────────


class TestImprovementCalculation:
    """Test that improvement percentages are calculated correctly."""

    def test_bgew_improvement_calculation(self, artifacts_with_p138):
        """BGEW improvement vs cfg05 is calculated correctly."""
        result = build_performance_claims(artifacts_with_p138)
        bgew_claim = next(c for c in result["claims"] if c["id"] == "bgew_2025_rolling")
        expected_improvement = BASELINE_CFG05_ONLY_SMAPE - 15.5
        assert bgew_claim["improvement_vs_cfg05"] == round(expected_improvement, 2)
        expected_pct = round(expected_improvement / BASELINE_CFG05_ONLY_SMAPE * 100, 1)
        assert bgew_claim["improvement_pct"] == expected_pct

    def test_realtime_improvement_calculation(self, artifacts_full):
        """Realtime improvement vs baseline is calculated correctly."""
        result = build_performance_claims(artifacts_full)
        rt_claims = [c for c in result["claims"] if c["id"] == "realtime_improved_2025"]
        assert len(rt_claims) == 1
        expected_improvement = BASELINE_REALTIME_DA_SAFE_SMAPE - 28.0
        assert rt_claims[0]["improvement_vs_baseline"] == round(expected_improvement, 2)


# ── Test 5: Output files created ───────────────────────────────────


class TestOutputFilesCreated:
    """Test that all output files are created."""

    def test_metrics_json_created(self, artifacts_full, tmp_output_dir):
        """production_metrics_2025_performance.json is created."""
        result = run_p143_performance_claims(artifacts_full, tmp_output_dir)
        assert os.path.isfile(result["output_files"]["metrics_json"])

    def test_report_md_created(self, artifacts_full, tmp_output_dir):
        """p143_performance_claim_update_report.md is created."""
        result = run_p143_performance_claims(artifacts_full, tmp_output_dir)
        assert os.path.isfile(result["output_files"]["report_md"])

    def test_client_delivery_note_created(self, artifacts_full, tmp_output_dir):
        """CLIENT_DELIVERY_NOTE.md is created."""
        result = run_p143_performance_claims(artifacts_full, tmp_output_dir)
        assert os.path.isfile(result["output_files"]["client_delivery_note"])

    def test_client_caveats_created(self, artifacts_full, tmp_output_dir):
        """CLIENT_CAVEATS.md is created."""
        result = run_p143_performance_claims(artifacts_full, tmp_output_dir)
        assert os.path.isfile(result["output_files"]["client_caveats"])


# ── Test 6: Report contains actual numbers ─────────────────────────


class TestReportContainsActualNumbers:
    """Test that generated report contains actual metric values."""

    def test_report_contains_cfg05_number(self, artifacts_full, tmp_output_dir):
        """Report contains the cfg05-only sMAPE number."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        report_path = os.path.join(tmp_output_dir, "docs", "reports",
                                   "p143_performance_claim_update_report.md")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "20.22" in content

    def test_report_contains_realtime_number(self, artifacts_full, tmp_output_dir):
        """Report contains the realtime sMAPE number."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        report_path = os.path.join(tmp_output_dir, "docs", "reports",
                                   "p143_performance_claim_update_report.md")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "33.03" in content

    def test_report_contains_local_2026_number(self, artifacts_full, tmp_output_dir):
        """Report contains the local 2026 BGEW number."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        report_path = os.path.join(tmp_output_dir, "docs", "reports",
                                   "p143_performance_claim_update_report.md")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "9.23" in content

    def test_report_contains_local_window_label(self, artifacts_full, tmp_output_dir):
        """Report labels local 2026 as local window."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        report_path = os.path.join(tmp_output_dir, "docs", "reports",
                                   "p143_performance_claim_update_report.md")
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "LOCAL WINDOW" in content or "local window" in content.lower()


# ── Test 7: Verdict logic ──────────────────────────────────────────


class TestVerdictLogic:
    """Test verdict determination."""

    def test_blocked_when_no_artifacts(self, empty_artifacts_dir):
        """Verdict is NO_IMPROVEMENT when no P138-P142 artifacts exist."""
        result = build_performance_claims(empty_artifacts_dir)
        assert result["verdict"] == "PERFORMANCE_NO_IMPROVEMENT"

    def test_blocked_when_single_model(self, artifacts_with_p138_blocked):
        """Verdict is BLOCKED when model_count < 2."""
        result = build_performance_claims(artifacts_with_p138_blocked)
        assert result["verdict"] == "PERFORMANCE_BLOCKED_FEATURE_PIPELINE"

    def test_improved_with_caveats_when_only_bgew(self, artifacts_with_p138):
        """Verdict is IMPROVED_WITH_CAVEATS when only BGEW improves."""
        result = build_performance_claims(artifacts_with_p138)
        assert result["verdict"] == "PERFORMANCE_IMPROVED_WITH_CAVEATS"

    def test_unlocked_go_when_both_improve(self, artifacts_full):
        """Verdict is UNLOCKED_GO when both BGEW and realtime improve."""
        result = build_performance_claims(artifacts_full)
        assert result["verdict"] == "PERFORMANCE_UNLOCKED_GO"


# ── Test 8: Metrics JSON structure ─────────────────────────────────


class TestMetricsJsonStructure:
    """Test the structure of the output metrics JSON."""

    def test_metrics_json_has_verdict(self, artifacts_full, tmp_output_dir):
        """Metrics JSON contains verdict."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        metrics_path = os.path.join(tmp_output_dir, "production_metrics_2025_performance.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "verdict" in data

    def test_metrics_json_has_claims(self, artifacts_full, tmp_output_dir):
        """Metrics JSON contains claims list."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        metrics_path = os.path.join(tmp_output_dir, "production_metrics_2025_performance.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "claims" in data
        assert isinstance(data["claims"], list)
        assert len(data["claims"]) > 0

    def test_metrics_json_has_rules(self, artifacts_full, tmp_output_dir):
        """Metrics JSON contains rules section."""
        run_p143_performance_claims(artifacts_full, tmp_output_dir)
        metrics_path = os.path.join(tmp_output_dir, "production_metrics_2025_performance.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "rules" in data
        assert data["rules"]["no_fake_numbers"] is True


# ── Test 9: Graceful degradation ───────────────────────────────────


class TestGracefulDegradation:
    """Test that the script handles missing artifacts gracefully."""

    def test_no_crash_with_empty_dir(self, empty_artifacts_dir):
        """No crash when artifacts dir is empty."""
        result = build_performance_claims(empty_artifacts_dir)
        assert result is not None
        assert "claims" in result

    def test_no_crash_with_partial_artifacts(self, artifacts_with_p138):
        """No crash when only P138 exists."""
        result = build_performance_claims(artifacts_with_p138)
        assert result is not None
        assert "claims" in result

    def test_read_p138_returns_empty_when_missing(self, empty_artifacts_dir):
        """read_p138 returns empty dict when artifacts missing."""
        result = read_p138_rolling_bgew(empty_artifacts_dir)
        assert result["exists"] is False
        assert result["smape"] is None

    def test_read_p139_returns_empty_when_missing(self, empty_artifacts_dir):
        """read_p139 returns empty dict when artifacts missing."""
        result = read_p139_residual(empty_artifacts_dir)
        assert result["exists"] is False

    def test_read_p140_returns_empty_when_missing(self, empty_artifacts_dir):
        """read_p140 returns empty dict when artifacts missing."""
        result = read_p140_realtime(empty_artifacts_dir)
        assert result["exists"] is False
