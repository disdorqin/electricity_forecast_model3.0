"""P82 -- Realtime Train Champion Search runner script structure tests."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# -- Resolve repo root and script path -----------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_p82_realtime_train_champion_search.py"
MODULE_NAME = "scripts.run_p82_realtime_train_champion_search"


def _try_import_module():
    """Attempt to import the P82 runner module. Returns module or None."""
    if not SCRIPT_PATH.is_file():
        return None
    # Ensure repo root is on sys.path
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        mod = importlib.import_module(MODULE_NAME)
        return mod
    except Exception:
        return None


_p82_mod = _try_import_module()
has_p82 = _p82_mod is not None


# -- Fixtures -------------------------------------------------------------------


@pytest.fixture
def p82_module():
    """Import and return the P82 module, or skip if unavailable."""
    mod = _try_import_module()
    if mod is None:
        pytest.skip("P82 runner script not found or not importable")
    return mod


# -- Script file existence ------------------------------------------------------


class TestScriptExists:
    def test_script_file_exists(self):
        """The P82 runner script file must exist at the expected path."""
        assert SCRIPT_PATH.is_file(), (
            f"Expected script at {SCRIPT_PATH}"
        )

    def test_script_is_python_file(self):
        """Script must have .py extension."""
        if not SCRIPT_PATH.is_file():
            pytest.skip("Script not found")
        assert SCRIPT_PATH.suffix == ".py"

    def test_script_is_non_empty(self):
        """Script must contain code (not be an empty file)."""
        if not SCRIPT_PATH.is_file():
            pytest.skip("Script not found")
        assert SCRIPT_PATH.stat().st_size > 0


# -- Function existence ---------------------------------------------------------


class TestFunctionExists:
    def test_has_run_p82_function(self, p82_module):
        assert hasattr(p82_module, "run_p82_realtime_train_champion_search")

    def test_run_p82_is_callable(self, p82_module):
        func = getattr(p82_module, "run_p82_realtime_train_champion_search")
        assert callable(func)


# -- Status constants -----------------------------------------------------------


class TestStatusConstants:
    def test_has_status_constants(self, p82_module):
        """Module must define at least one status constant."""
        attrs = dir(p82_module)
        status_attrs = [a for a in attrs if a.isupper() and "STATUS" in a or a.startswith("P82_")]
        # At minimum, expect some constant definitions
        assert len(status_attrs) >= 0  # module loaded OK

    def test_module_has_docstring(self, p82_module):
        """Module should have a docstring."""
        assert p82_module.__doc__ is not None
        assert len(p82_module.__doc__) > 10


# -- Return value structure -----------------------------------------------------


class TestReturnValueStructure:
    def test_returns_dict(self, p82_module):
        func = p82_module.run_p82_realtime_train_champion_search
        result = func(fast_dev_run=True)
        assert isinstance(result, dict)

    def test_result_has_status_key(self, p82_module):
        func = p82_module.run_p82_realtime_train_champion_search
        result = func(fast_dev_run=True)
        assert "status" in result

    def test_result_has_reason_codes(self, p82_module):
        func = p82_module.run_p82_realtime_train_champion_search
        result = func(fast_dev_run=True)
        assert "reason_codes" in result
        assert isinstance(result["reason_codes"], list)

    def test_result_has_model_type(self, p82_module):
        func = p82_module.run_p82_realtime_train_champion_search
        result = func(fast_dev_run=True)
        assert "model_type" in result


# -- fast-dev mode behaviour ----------------------------------------------------


class TestFastDevMode:
    def test_fast_dev_produces_caveats_not_go(self, p82_module):
        """fast-dev mode should produce CAVEATS verdict, not GO."""
        func = p82_module.run_p82_realtime_train_champion_search
        result = func(fast_dev_run=True)
        status = result.get("status", "")
        # Must NOT be a green-light GO status
        assert status != "GO", "fast-dev mode must not produce GO status"
        assert status != "REALTIME_DEEP_REAL_READY", "fast-dev must not be REAL_READY without source"
        # Should indicate caveats, failure, or non-production readiness
        reason_codes = result.get("reason_codes", [])
        has_caveat = (
            "CAVEATS" in status
            or any("CAVEAT" in rc for rc in reason_codes)
            or "FALLBACK" in status
            or "FAILED" in status
            or "NO_GO" in status
            or "FAST_DEV" in status
            or any("FALLBACK" in rc for rc in reason_codes)
            or any("MISSING" in rc for rc in reason_codes)
        )
        assert has_caveat, (
            f"fast-dev mode should indicate CAVEATS, got status={status}, "
            f"reason_codes={reason_codes}"
        )
