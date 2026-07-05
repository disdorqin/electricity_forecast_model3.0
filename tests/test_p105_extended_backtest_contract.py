"""P105: Extended backtest contract tests."""
from __future__ import annotations
import os, pytest
import numpy as np, pandas as pd
from fusion.unified_weight_learner import compute_bgew_weights

class TestExtendedMetrics:
    def test_smape_floor50(self):
        y_true = np.array([100,200,300,400,500])
        y_pred = np.array([110,190,310,390,510])
        smape = np.mean(2*np.abs(y_true-y_pred)/(np.abs(y_true)+np.abs(y_pred)+1e-8))*100
        assert 0 < smape < 20
    def test_bgew_weights(self):
        w = compute_bgew_weights({"a":5.0,"b":15.0}, alpha=0.05)
        assert abs(sum(w.values())-1.0) < 1e-6
        assert w["a"] > w["b"]  # lower sMAPE gets higher weight
    def test_bgew_min_weight(self):
        w = compute_bgew_weights({"a":2.0,"b":50.0}, alpha=0.05, min_weight=0.05)
        assert all(v >= 0.05 for v in w.values())
    def test_bgew_max_weight(self):
        w = compute_bgew_weights({"a":8.0,"b":12.0}, alpha=0.05, max_weight=0.9)
        assert all(v <= 0.9 + 1e-6 for v in w.values())
