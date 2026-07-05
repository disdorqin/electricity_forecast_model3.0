"""
adapters/p5m_full_residual_adapter.py — P100: Full P5M Residual Stack.

Integrates the P5M (5-model) residual correction stack from 2.0
experiment repository into the production adapter contract.

Status flow:
    P5M_FULL_READY / P5M_FULL_TRAINED → supports FINAL_GO
    P5M_CATBOOST_PARTIAL → GO_WITH_CAVEATS
    P5M_NO_OP_FALLBACK → GO_WITH_CAVEATS
    P5M_FULL_FAILED → NO_GO if strict_full_production
"""
from __future__ import annotations
import json, logging, os
from typing import Any, Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

P5M_FULL_READY = "P5M_FULL_READY"
P5M_FULL_TRAINED = "P5M_FULL_TRAINED"
P5M_FULL_MISSING = "P5M_FULL_MISSING"
P5M_FULL_FAILED = "P5M_FULL_FAILED"
P5M_CATBOOST_PARTIAL = "P5M_CATBOOST_PARTIAL"
P5M_NO_OP_FALLBACK = "P5M_NO_OP_FALLBACK"

class P5MFullResidualAdapter:
    """Production adapter for P5M full residual stack."""
    def __init__(self, artifact_path: str = ""):
        self.artifact_path = artifact_path
        self._stack = None
        self._manifest = {}
        self._status = P5M_FULL_MISSING

    def load(self) -> bool:
        if self.artifact_path and os.path.isfile(self.artifact_path):
            try:
                import joblib
                self._stack = joblib.load(self.artifact_path)
                self._status = P5M_FULL_READY
                return True
            except Exception as e:
                logger.warning("P5M load failed: %s", e)
                self._status = P5M_FULL_FAILED
                return False
        self._status = P5M_FULL_MISSING
        return False

    def correct(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if self._status in (P5M_FULL_READY, P5M_FULL_TRAINED) and self._stack is not None:
            try:
                result["residual_delta"] = self._stack.predict(df)
                result["y_pred_corrected"] = df.get("y_pred", df.get("fused_price", 0)) + result["residual_delta"]
                result["correction_applied"] = True
                result["correction_module"] = "p5m_full_stack"
                result["reason_codes"] = "P5M_FULL_CORRECTION_APPLIED"
            except Exception as e:
                logger.warning("P5M correction failed: %s", e)
                result = self._fallback_correct(result)
        else:
            result = self._fallback_correct(result)
        return result

    def _fallback_correct(self, df: pd.DataFrame) -> pd.DataFrame:
        df["residual_delta"] = 0.0
        df["y_pred_corrected"] = df.get("y_pred", df.get("fused_price", 0))
        df["correction_applied"] = False
        df["correction_module"] = "no_op_fallback"
        df["reason_codes"] = self._status
        return df
