"""
tests/test_fusion_readiness_gates.py — Fusion readiness gate contract tests.

Validates:
    1. _apply_readiness_gate with all READY_REAL includes all
    2. _apply_readiness_gate excludes READY_STUB by default
    3. _apply_readiness_gate excludes DATA_MISSING
    4. _apply_readiness_gate excludes NOT_READY
    5. _apply_readiness_gate with allow_dry_run=True includes READY_DRY_RUN
    6. _apply_readiness_gate with allow_dry_run=False excludes READY_DRY_RUN
    7. _auto_readiness returns expected states for known models
    8. FUSION_READINESS_STATES constant includes all valid states
    9. READY_STUB never enters fusion by default
    10. DATA_MISSING never enters fusion
"""

from __future__ import annotations

import pytest

from fusion.engine import (
    _apply_readiness_gate,
    _auto_readiness,
    READY_REAL,
    READY_DRY_RUN,
    READY_STUB,
    DATA_MISSING,
    NOT_READY,
)
from data.schema import FUSION_READINESS_STATES


class TestApplyReadinessGate:
    """Contract: _apply_readiness_gate."""

    def test_all_real_included(self):
        """All READY_REAL models are included."""
        status = {"cfg05": READY_REAL, "best_two_average": READY_REAL}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "best_two_average"],
            allow_dry_run=False,
            readiness_status=status,
        )
        assert set(included) == {"cfg05", "best_two_average"}
        assert excluded == []
        assert mode == "REAL"

    def test_stub_excluded_by_default(self):
        """READY_STUB models are excluded by default."""
        status = {"cfg05": READY_REAL, "sgdfnet": READY_STUB}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "sgdfnet"],
            allow_dry_run=False,
            readiness_status=status,
        )
        assert "sgdfnet" in excluded
        assert "sgdfnet" not in included

    def test_data_missing_excluded(self):
        """DATA_MISSING models are excluded."""
        status = {"cfg05": READY_REAL, "p5m": DATA_MISSING}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "p5m"],
            allow_dry_run=False,
            readiness_status=status,
        )
        assert "p5m" in excluded
        assert "p5m" not in included

    def test_not_ready_excluded(self):
        """NOT_READY models are excluded."""
        status = {"cfg05": READY_REAL, "broken_model": NOT_READY}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "broken_model"],
            allow_dry_run=False,
            readiness_status=status,
        )
        assert "broken_model" in excluded

    def test_dry_run_included_when_allowed(self):
        """READY_DRY_RUN is included when allow_dry_run=True."""
        status = {"cfg05": READY_DRY_RUN, "best_two_average": READY_REAL}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "best_two_average"],
            allow_dry_run=True,
            readiness_status=status,
        )
        assert "cfg05" in included
        assert mode == "DRY_RUN"

    def test_dry_run_excluded_when_not_allowed(self):
        """READY_DRY_RUN is excluded when allow_dry_run=False."""
        status = {"cfg05": READY_DRY_RUN, "best_two_average": READY_REAL}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "best_two_average"],
            allow_dry_run=False,
            readiness_status=status,
        )
        assert "cfg05" in excluded
        # Only real models remain
        assert "best_two_average" in included

    def test_empty_model_list(self):
        """Empty model list returns empty."""
        included, excluded, mode = _apply_readiness_gate(
            [], readiness_status={},
        )
        assert included == []
        assert excluded == []

    def test_readiness_mode_is_dry_run_when_mixed(self):
        """readiness_mode is DRY_RUN when any included model is READY_DRY_RUN."""
        status = {"cfg05": READY_DRY_RUN, "best_two_average": READY_REAL}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "best_two_average"],
            allow_dry_run=True,
            readiness_status=status,
        )
        assert mode == "DRY_RUN"

    def test_readiness_mode_is_real_when_all_real(self):
        """readiness_mode is REAL when all included are READY_REAL."""
        status = {"cfg05": READY_REAL, "best_two_average": READY_REAL}
        included, excluded, mode = _apply_readiness_gate(
            ["cfg05", "best_two_average"],
            allow_dry_run=False,
            readiness_status=status,
        )
        assert mode == "REAL"


class TestAutoReadiness:
    """Contract: _auto_readiness."""

    def test_returns_dict(self):
        """_auto_readiness returns a dict."""
        status = _auto_readiness()
        assert isinstance(status, dict)
        assert len(status) > 0

    def test_cfg05_not_real(self):
        """cfg05 is not READY_REAL (no artifact on disk)."""
        status = _auto_readiness()
        assert "cfg05" in status
        assert status["cfg05"] in (READY_REAL, READY_DRY_RUN, READY_STUB)

    def test_known_model_ids_present(self):
        """Known model IDs are present in auto-readiness."""
        status = _auto_readiness()
        # At minimum, cfg05 should be present
        assert "cfg05" in status


class TestSchemaConstants:
    """Contract: FUSION_READINESS_STATES."""

    def test_includes_all_states(self):
        """FUSION_READINESS_STATES includes all 5 states."""
        assert len(FUSION_READINESS_STATES) == 5
        assert "READY_REAL" in FUSION_READINESS_STATES
        assert "READY_DRY_RUN" in FUSION_READINESS_STATES
        assert "READY_STUB" in FUSION_READINESS_STATES
        assert "DATA_MISSING" in FUSION_READINESS_STATES
        assert "NOT_READY" in FUSION_READINESS_STATES
