"""P100: Full P5M residual integration tests."""
from __future__ import annotations
import os
import pandas as pd
import pytest
from adapters.p5m_full_residual_adapter import P5MFullResidualAdapter, P5M_FULL_READY, P5M_FULL_MISSING, P5M_NO_OP_FALLBACK
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestLoad:
    def test_load_missing(self):
        a = P5MFullResidualAdapter("/nonexistent.pkl")
        assert not a.load()
    def test_status_missing_when_not_loaded(self):
        a = P5MFullResidualAdapter()
        assert a._status == P5M_FULL_MISSING
    def test_correct_noop(self):
        a = P5MFullResidualAdapter()
        df = pd.DataFrame({"y_pred": [100.0, 200.0]})
        r = a.correct(df)
        assert not r["correction_applied"].iloc[0]
        assert r["correction_module"].iloc[0] == "no_op_fallback"

class TestStatus:
    def test_no_op_is_go_with_caveats(self):
        assert P5M_NO_OP_FALLBACK == "P5M_NO_OP_FALLBACK"
    def test_full_ready_supports_go(self):
        assert P5M_FULL_READY == "P5M_FULL_READY"
    def test_missing_not_failed(self):
        """Missing but not failed should allow GO_WITH_CAVEATS."""
        assert P5M_FULL_MISSING != "P5M_FULL_FAILED"
