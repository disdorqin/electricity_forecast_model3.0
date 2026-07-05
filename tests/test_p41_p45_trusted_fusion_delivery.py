"""
tests/test_p41_p45_trusted_fusion_delivery.py — P41-P45 tests (35+ tests).

Tests the trust gate, trusted fusion backtest, rolling validation,
delivery packager, and integration consistency.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Allow importing from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")
LEDGER_DIR = os.path.join(WORK_DIR, "ledger")

_TRUSTED_ONLY = ["lightgbm_cfg05_dayahead", "catboost_spike_residual"]

# ──────────────────────────────────────────────────────
# P41: Model Trust Gate (8)
# ──────────────────────────────────────────────────────


class TestP41TrustGate:
    """Tests for scripts/run_p41_model_trust_gate.py."""

    def test_p41_gate_runs_and_returns_complete(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        assert result["phase"] == "P41"
        assert result["summary"]["p41_status"] == "P41_GATE_COMPLETE"

    def test_p41_stage3_suspect_leakage(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        m = result["models"].get("stage3_business_fixed", {})
        assert m.get("trust_label") == "SUSPECT_LEAKAGE"
        reasons = m.get("suspicion_reasons", [])
        assert any("within_1pct_ratio" in r for r in reasons)
        assert any("sMAPE" in r for r in reasons)
        assert any("MAE" in r for r in reasons)
        assert any("corr" in r for r in reasons)

    def test_p41_best_two_average_suspect_corr(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        m = result["models"].get("best_two_average", {})
        assert m.get("trust_label") == "SUSPECT_LEAKAGE"
        assert m.get("corr_y_pred_y_true", 0) > 0.995

    def test_p41_catboost_sota_suspect_corr(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        m = result["models"].get("catboost_sota", {})
        assert m.get("trust_label") == "SUSPECT_LEAKAGE"
        assert m.get("corr_y_pred_y_true", 0) > 0.995

    def test_p41_cfg05_trusted(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        m = result["models"].get("lightgbm_cfg05_dayahead", {})
        assert m.get("trust_label") == "TRUSTED"

    def test_p41_catboost_spike_residual_trusted(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        m = result["models"].get("catboost_spike_residual", {})
        assert m.get("trust_label") == "TRUSTED"

    def test_p41_profiles_structure(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        result = run_trust_gate(work_dir=WORK_DIR)
        profiles = result.get("profiles", {})
        assert "research_all_models" in profiles
        assert "trusted_no_stage3" in profiles
        assert profiles["trusted_no_stage3"]["delivery_allowed"] is True
        assert profiles["research_all_models"]["delivery_allowed"] is False

    def test_p41_missing_data_returns_data_missing(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        with tempfile.TemporaryDirectory() as tmp:
            result = run_trust_gate(work_dir=tmp)
        assert result["summary"]["p41_status"] == "P41_DATA_MISSING"


# ──────────────────────────────────────────────────────
# P42: Trusted Fusion Backtest (10)
# ──────────────────────────────────────────────────────


class TestP42TrustedFusion:
    """Tests for scripts/run_p42_trusted_fusion_backtest.py."""

    def test_p42_runs_and_returns_metrics(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR)
        assert result["phase"] == "P42"
        assert "cfg05_metrics" in result
        assert result["cfg05_metrics"]["sMAPE_floor50"] > 0

    def test_p42_with_trusted_only_pool(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(
            work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY
        )
        assert result["phase"] == "P42"
        assert result["trusted_models"] == _TRUSTED_ONLY
        assert result["best_single_model"] in _TRUSTED_ONLY

    def test_p42_cfg05_metrics_structure(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        cfg = result.get("cfg05_metrics", {})
        for key in ("sMAPE_floor50", "MAE", "RMSE", "n"):
            assert key in cfg, f"Missing key: {key}"
        assert cfg["n"] > 0

    def test_p42_best_single_identified(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        assert "best_single_model" in result
        assert "best_single_metrics" in result
        bs = result["best_single_metrics"]
        assert bs["sMAPE_floor50"] > 0

    def test_p42_equal_weight_fusion_computed(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        ew = result.get("equal_weight_metrics", {})
        assert ew.get("sMAPE_floor50", 0) > 0
        assert ew.get("n", 0) > 0

    def test_p42_bgew_fusion_computed(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        bf = result.get("fusion_metrics", {})
        assert bf.get("sMAPE_floor50", 0) > 0
        assert bf.get("n", 0) > 0

    def test_p42_fusion_weights_structure(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        weights = result.get("fusion_weights", {})
        assert "1_8" in weights
        assert "9_16" in weights
        assert "17_24" in weights
        for period, w in weights.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 0.01, f"{period} weights sum to {total}"

    def test_p42_fusion_deltas_computed(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        summary = result["summary"]
        assert summary["fusion_vs_cfg05_delta"] is not None
        assert summary["fusion_vs_best_single_delta"] is not None

    def test_p42_per_period_metrics(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        pm = result.get("period_metrics", {})
        assert len(pm) > 0
        for period, m in pm.items():
            assert "cfg05_sMAPE" in m
            assert "fusion_sMAPE" in m

    def test_p42_missing_data_returns_data_missing(self):
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        with tempfile.TemporaryDirectory() as tmp:
            result = run_trusted_fusion_backtest(work_dir=tmp)
        assert result["summary"]["p42_status"] == "P42_DATA_MISSING"


# ──────────────────────────────────────────────────────
# P43: Rolling Weight Validation (8)
# ──────────────────────────────────────────────────────


class TestP43RollingValidation:
    """Tests for scripts/run_p43_rolling_weight_fusion_validation.py."""

    def test_p43_runs_and_returns_complete(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        assert result["phase"] == "P43"
        assert result["summary"]["p43_status"] == "P43_VALIDATION_COMPLETE"

    def test_p43_full_period_metrics(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        fp = result.get("full_period", {})
        assert fp.get("sMAPE", 0) > 0
        assert fp.get("n", 0) > 0

    def test_p43_split_metrics(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        sp = result.get("split", {})
        assert sp.get("fusion_sMAPE", 0) > 0
        assert sp.get("cfg05_sMAPE", 0) > 0
        assert sp.get("train_days", 0) > 0
        assert sp.get("test_days", 0) > 0
        assert sp.get("n", 0) > 0

    def test_p43_split_fusion_beats_cfg05(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        sp = result.get("split", {})
        assert sp.get("fusion_sMAPE", 999) < sp.get("cfg05_sMAPE", 0)

    def test_p43_rolling_metrics(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        rl = result.get("rolling", {})
        metrics = rl.get("metrics", {})
        assert metrics.get("n_days", 0) > 0
        assert metrics.get("fusion_sMAPE", 0) > 0
        assert metrics.get("cfg05_sMAPE", 0) > 0

    def test_p43_rolling_fusion_beats_cfg05(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        rl = result.get("rolling", {}).get("metrics", {})
        assert rl.get("fusion_sMAPE", 999) < rl.get("cfg05_sMAPE", 0)

    def test_p43_rolling_days_info(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        days = result.get("rolling", {}).get("days_info", [])
        assert len(days) >= 20
        for d in days:
            assert "target_day" in d
            assert "past_days" in d
            assert d["n_fused"] > 0

    def test_p43_missing_data_returns_data_missing(self):
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        with tempfile.TemporaryDirectory() as tmp:
            result = run_rolling_validation(work_dir=tmp)
        assert result["summary"]["p43_status"] == "P43_DATA_MISSING"


# ──────────────────────────────────────────────────────
# P44: Delivery Readiness Packager (6)
# ──────────────────────────────────────────────────────


class TestP44DeliveryPackager:
    """Tests for scripts/run_p44_delivery_readiness_packager.py."""

    def test_p44_runs_and_packages(self):
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        assert result["phase"] == "P44"
        assert result["p44_status"] == "P44_DELIVERY_READINESS_PACKAGED"

    def test_p44_trusted_pool_matches_p41(self):
        from scripts.run_p41_model_trust_gate import run_trust_gate
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        gate = run_trust_gate(work_dir=WORK_DIR)
        p44 = run_delivery_packager(work_dir=WORK_DIR)
        assert p44["trusted_model_pool"] == gate["summary"]["trusted_models"]

    def test_p44_quarantined_no_duplicates(self):
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        q = result["quarantined_models"]
        assert len(q) == len(set(q)), f"Duplicates found in quarantined_models: {q}"

    def test_p44_recommended_default_is_valid(self):
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        valid = {"cfg05", "best_single_trusted_model", "trusted_bgew_fusion"}
        assert result["recommended_default"] in valid

    def test_p44_forbidden_claims_present(self):
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        assert len(result["forbidden_claims"]) >= 3
        assert any("2.97%" in c for c in result["forbidden_claims"])
        assert any("69.96%" in c for c in result["forbidden_claims"])

    def test_p44_known_caveats_present(self):
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        assert len(result["known_caveats"]) >= 3
        assert any("stage3" in c for c in result["known_caveats"])
        assert any("Quarantined" in c for c in result["known_caveats"])


# ──────────────────────────────────────────────────────
# Integration & Consistency (6)
# ──────────────────────────────────────────────────────


class TestP41P45Integration:
    """Cross-phase consistency tests."""

    def test_integration_p41_p42_trusted_pool_consistency(self):
        """P42 with P41's trusted pool should have matching model list."""
        from scripts.run_p41_model_trust_gate import run_trust_gate
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        gate = run_trust_gate(work_dir=WORK_DIR)
        trusted = gate["summary"]["trusted_models"]
        p42 = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=trusted)
        assert p42["trusted_models"] == trusted

    def test_integration_p41_p43_trusted_pool_consistency(self):
        """P43 with P41's trusted pool should have matching model list."""
        from scripts.run_p41_model_trust_gate import run_trust_gate
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        gate = run_trust_gate(work_dir=WORK_DIR)
        trusted = gate["summary"]["trusted_models"]
        p43 = run_rolling_validation(work_dir=WORK_DIR, trusted_models=trusted)
        assert p43["trusted_models"] == trusted

    def test_integration_p42_bgew_smape_reasonable(self):
        """BGEW sMAPE should be in a reasonable range (2-15%)."""
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        result = run_trusted_fusion_backtest(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        fus = result["fusion_metrics"]["sMAPE_floor50"]
        assert 2.0 <= fus <= 15.0, f"BGEW sMAPE={fus}% outside expected range"

    def test_integration_p43_reason_codes_populated(self):
        """P43 should have detailed reason codes."""
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        result = run_rolling_validation(work_dir=WORK_DIR, trusted_models=_TRUSTED_ONLY)
        codes = result.get("reason_codes", [])
        assert len(codes) >= 4
        assert any("DAYS" in c for c in codes)
        assert any("SPLIT" in c for c in codes)
        assert any("ROLLING" in c for c in codes)

    def test_integration_p44_includes_all_phases(self):
        """P44 should include results from P41, P42, P43."""
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        assert result.get("trust_gate_status") == "P41_GATE_COMPLETE"
        assert "p42_status" in result
        assert result.get("rolling_validation", {}).get("status") == "P43_VALIDATION_COMPLETE"

    def test_integration_p44_delivery_commands_listed(self):
        """P44 should list delivery commands."""
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        cmds = result.get("delivery_commands", [])
        assert len(cmds) >= 3
        assert any("p41" in c.lower() for c in cmds)

    def test_integration_p44_fusion_weights_normalized(self):
        """P44 fusion weights should sum to ~1.0 per period."""
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        result = run_delivery_packager(work_dir=WORK_DIR)
        weights = result["delivery_metrics"].get("fusion_weights", {})
        for period, w in weights.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 0.01, f"{period} weights sum to {total}"
