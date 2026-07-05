"""
classifiers/final_classifier_engine.py — P72: Final Classifier Engine.

Classifies each prediction hour into risk categories:
  - negative price risk
  - high spike risk
  - normal trend
  - uncertainty score

If real classifier artifacts exist, uses them.
Otherwise, uses rule-based fallback with explicit status labeling.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────
CLASSIFIER_ML_READY = "CLASSIFIER_ML_READY"
CLASSIFIER_RULE_FALLBACK = "CLASSIFIER_RULE_FALLBACK"
CLASSIFIER_BLOCKED = "CLASSIFIER_BLOCKED"

# ── Thresholds ────────────────────────────────────────────────────────
NEGATIVE_PRICE_THRESHOLD = 0.0
HIGH_SPIKE_THRESHOLD = 500.0
HIGH_UNCERTAINTY_THRESHOLD = 0.7


def run_final_classifier(
    dayahead_fused: Optional[pd.DataFrame] = None,
    realtime_fused: Optional[pd.DataFrame] = None,
    work_dir: str = "",
) -> dict[str, Any]:
    """Run classification on fused predictions.

    Parameters
    ----------
    dayahead_fused : DataFrame, optional
        Fused day-ahead predictions.
    realtime_fused : DataFrame, optional
        Fused realtime predictions.
    work_dir : str
        Working directory for artifacts.

    Returns
    -------
    dict with classified outputs for both tasks.
    """
    result: dict[str, Any] = {
        "dayahead": {"status": "NOT_RUN", "output": None},
        "realtime": {"status": "NOT_RUN", "output": None},
        "classifier_status": CLASSIFIER_RULE_FALLBACK,
        "reason_codes": [],
    }

    # Check for ML classifier artifact
    classifier_artifact = os.path.join(work_dir, "classifier_model.pkl") if work_dir else ""
    use_ml = os.path.isfile(classifier_artifact) if classifier_artifact else False

    if use_ml:
        result["classifier_status"] = CLASSIFIER_ML_READY
        result["reason_codes"].append("ML_CLASSIFIER_AVAILABLE")
    else:
        result["classifier_status"] = CLASSIFIER_RULE_FALLBACK
        result["reason_codes"].append("NO_ML_CLASSIFIER_USING_RULES")

    # Classify dayahead
    if dayahead_fused is not None and len(dayahead_fused) > 0:
        da_classified = _classify_predictions(
            dayahead_fused,
            price_col="dayahead_price",
            use_ml=use_ml,
            classifier_artifact=classifier_artifact,
        )
        result["dayahead"] = {
            "status": "CLASSIFIED",
            "output": da_classified,
            "rows": len(da_classified),
        }

    # Classify realtime
    if realtime_fused is not None and len(realtime_fused) > 0:
        rt_classified = _classify_predictions(
            realtime_fused,
            price_col="realtime_price",
            use_ml=use_ml,
            classifier_artifact=classifier_artifact,
        )
        result["realtime"] = {
            "status": "CLASSIFIED",
            "output": rt_classified,
            "rows": len(rt_classified),
        }

    return result


def _classify_predictions(
    predictions: pd.DataFrame,
    price_col: str = "dayahead_price",
    use_ml: bool = False,
    classifier_artifact: str = "",
) -> pd.DataFrame:
    """Apply classification to a prediction DataFrame."""
    classified = predictions.copy()

    # Get price column
    if price_col not in classified.columns:
        # Try alternatives
        for alt in ["y_pred_corrected", "y_pred", "trend_pred"]:
            if alt in classified.columns:
                price_col = alt
                break

    if price_col not in classified.columns:
        classified["classifier_action"] = "UNKNOWN"
        classified["negative_risk"] = 0.0
        classified["spike_risk"] = 0.0
        classified["normal_trend_flag"] = 0
        classified["uncertainty_score"] = 1.0
        classified["delivery_warning_level"] = "HIGH"
        classified["classifier_model_name"] = "none"
        classified["classifier_status"] = CLASSIFIER_BLOCKED
        return classified

    prices = classified[price_col].fillna(0).values

    if use_ml and classifier_artifact:
        try:
            import pickle
            with open(classifier_artifact, "rb") as f:
                clf = pickle.load(f)
            X = prices.reshape(-1, 1)
            labels = clf.predict(X)
            classified["classifier_action"] = labels
            classified["classifier_model_name"] = "ml_classifier"
            classified["classifier_status"] = CLASSIFIER_ML_READY
        except Exception:
            use_ml = False

    if not use_ml:
        # Rule-based classification
        classified["negative_risk"] = np.where(
            prices < NEGATIVE_PRICE_THRESHOLD + 50, 0.3, 0.0
        )
        classified["spike_risk"] = np.where(
            prices > HIGH_SPIKE_THRESHOLD, 0.8,
            np.where(prices > HIGH_SPIKE_THRESHOLD * 0.7, 0.3, 0.0)
        )
        classified["normal_trend_flag"] = np.where(
            (prices >= NEGATIVE_PRICE_THRESHOLD) & (prices <= HIGH_SPIKE_THRESHOLD),
            1, 0
        )
        classified["uncertainty_score"] = np.where(
            (prices > HIGH_SPIKE_THRESHOLD * 0.7) | (prices < NEGATIVE_PRICE_THRESHOLD + 50),
            0.6, 0.3
        )

        # Determine action
        conditions = [
            prices < NEGATIVE_PRICE_THRESHOLD,
            prices > HIGH_SPIKE_THRESHOLD,
            (prices >= NEGATIVE_PRICE_THRESHOLD) & (prices <= HIGH_SPIKE_THRESHOLD),
        ]
        choices = ["NEGATIVE_DETECTED", "SPIKE_DETECTED", "NORMAL"]
        classified["classifier_action"] = np.select(conditions, choices, default="UNKNOWN")
        classified["classifier_model_name"] = "rule_fallback"
        classified["classifier_status"] = CLASSIFIER_RULE_FALLBACK

        # Warning level
        classified["delivery_warning_level"] = np.where(
            (prices < NEGATIVE_PRICE_THRESHOLD) | (prices > HIGH_SPIKE_THRESHOLD),
            "HIGH",
            np.where(
                (prices < 50) | (prices > 350),
                "MEDIUM",
                "LOW"
            )
        )

    return classified
