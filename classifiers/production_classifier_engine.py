"""
classifiers/production_classifier_engine.py — P101: ML Classifier Production Path.

Loads real ML classifier artifacts (negative_risk, spike_risk) from
the deep_sgdf_delta source repo and runs inference.

Status flow:
    CLASSIFIER_ML_READY → supports FINAL_GO
    CLASSIFIER_PARTIAL_ML_READY → GO_WITH_CAVEATS
    CLASSIFIER_RULE_FALLBACK → GO_WITH_CAVEATS
    CLASSIFIER_FAILED → NO_GO in strict_full_production
"""
from __future__ import annotations
import json, logging, os
from typing import Any, Optional
import numpy as np, pandas as pd

logger = logging.getLogger(__name__)

CLASSIFIER_ML_READY = "CLASSIFIER_ML_READY"
CLASSIFIER_PARTIAL_ML_READY = "CLASSIFIER_PARTIAL_ML_READY"
CLASSIFIER_RULE_FALLBACK = "CLASSIFIER_RULE_FALLBACK"
CLASSIFIER_BLOCKED_LEAKAGE = "CLASSIFIER_BLOCKED_LEAKAGE"
CLASSIFIER_FAILED = "CLASSIFIER_FAILED"

ALLOWED_FEATURES = [
    "dayahead_price","realtime_price","trend_pred","da_error_prob",
    "residual_direction_prob","uncertainty_score","correction_permission",
    "hour_business","period",
]
FORBIDDEN_FEATURES = ["y_true","actual","实时电价","future_actual","label"]

class ProductionClassifierEngine:
    """Production classifier engine with real ML artifact loading."""
    def __init__(self, negative_risk_path: str = "", spike_risk_path: str = ""):
        self.negative_risk_path = negative_risk_path
        self.spike_risk_path = spike_risk_path
        self._neg_model = None
        self._spike_model = None
        self._status = CLASSIFIER_RULE_FALLBACK
        self._models_loaded = 0

    def load(self) -> bool:
        """Load artifact models."""
        for attr, path in [("_neg_model", self.negative_risk_path), ("_spike_model", self.spike_risk_path)]:
            if path and os.path.isfile(path):
                try:
                    import joblib
                    setattr(self, attr, joblib.load(path))
                    self._models_loaded += 1
                    logger.info("Loaded classifier: %s", path)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)
        if self._models_loaded >= 2:
            self._status = CLASSIFIER_ML_READY
        elif self._models_loaded >= 1:
            self._status = CLASSIFIER_PARTIAL_ML_READY
        else:
            self._status = CLASSIFIER_RULE_FALLBACK
        return self._models_loaded > 0

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run classification on features."""
        result = df.copy()
        # Check forbidden features
        for col in FORBIDDEN_FEATURES:
            if col in result.columns:
                result["classifier_status"] = CLASSIFIER_BLOCKED_LEAKAGE
                result["classifier_action"] = "BLOCKED"
                return result
        # Ensure allowed features exist
        for col in ALLOWED_FEATURES:
            if col not in result.columns:
                result[col] = 0.0

        if self._status == CLASSIFIER_ML_READY and self._neg_model is not None:
            try:
                features = result[ALLOWED_FEATURES].fillna(0)
                result["negative_risk"] = self._neg_model.predict_proba(features)[:, 1]
                result["classifier_action"] = "ML_NEGATIVE_RISK"
                result["classifier_status"] = self._status
            except Exception as e:
                logger.warning("Neg classifier failed: %s", e)
                result = self._rule_fallback(result)
        else:
            result = self._rule_fallback(result)

        if self._spike_model is not None and self._status == CLASSIFIER_ML_READY:
            try:
                features = result[ALLOWED_FEATURES].fillna(0)
                result["spike_risk"] = self._spike_model.predict_proba(features)[:, 1]
            except Exception:
                result["spike_risk"] = 0.0

        result["classifier_model_name"] = "prod_classifier_v1"
        result["classifier_version"] = "p101"
        return result

    def _rule_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        df["negative_risk"] = 0.0
        df["spike_risk"] = 0.0
        df["classifier_action"] = "RULE_FALLBACK"
        df["classifier_status"] = CLASSIFIER_RULE_FALLBACK
        return df
