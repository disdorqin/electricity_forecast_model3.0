"""
adapters/negative_classifier_adapter.py — Negative/Spike Risk Classifier Adapter.

Integrates real ML classifier artifacts from the deep_sgdf_delta source repo
into the 3.0 pipeline.

Available real artifacts (in source_repo_path):
- artifacts/negative_risk/exp_2026_02/model.pkl — negative risk classifier
  (precision 0.81, recall 0.69, ROC-AUC 0.94)
- artifacts/spike_risk/exp_2026_02/model.pkl — spike risk classifier
  (precision 0.38, recall 0.24, ROC-AUC 0.79)

Safety invariants
-----------------
- Never uses y_true from the current target_day as input feature.
- If models not loaded, falls back to rule-based classification.
- Tracks classifier_status in every output DataFrame.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────
CLASSIFIER_ML_READY = "CLASSIFIER_ML_READY"
CLASSIFIER_RULE_FALLBACK = "CLASSIFIER_RULE_FALLBACK"

# ── Thresholds ────────────────────────────────────────────────────────
NEGATIVE_PRICE_THRESHOLD = 0.0
HIGH_SPIKE_THRESHOLD = 500.0
NEGATIVE_RISK_PROB_THRESHOLD = 0.5
SPIKE_RISK_PROB_THRESHOLD = 0.5


class NegativeClassifierAdapter:
    """Integrate real negative_risk and spike_risk classifiers from deep_sgdf_delta.

    The adapter searches for trained classifier artifacts in the source repo,
    loads them when available, and applies them to produce risk probabilities
    and classification labels.  Falls back to rule-based classification when
    no real models are found.

    Parameters
    ----------
    source_repo_path : str
        Path to the deep_sgdf_delta source repo containing ``artifacts/``.
        Typical values: ``"../deep_sgdf_delta"`` or
        ``".local_artifacts/source_repos/deep_sgdf_delta"``.
    work_dir : str
        Working directory for the current 3.0 run.
    """

    def __init__(self, source_repo_path: str = "", work_dir: str = "") -> None:
        self.source_repo_path = source_repo_path
        self.work_dir = work_dir

        self._artifacts: dict[str, Any] = {}
        self._negative_model: Any = None
        self._spike_model: Any = None
        self._status: str = CLASSIFIER_RULE_FALLBACK
        self._model_names: dict[str, str] = {
            "negative_risk": "none",
            "spike_risk": "none",
        }

    # ── Properties ────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        """Current adapter status (one of the ``CLASSIFIER_*`` constants)."""
        return self._status

    # ── Artifact discovery ────────────────────────────────────────────

    def find_artifacts(self) -> dict:
        """Search for negative_risk and spike_risk model.pkl files.

        Searches in priority order:
          a. ``source_repo_path/artifacts/``
          b. ``source_repo_path/`` (recursive)
          c. ``work_dir/artifacts/``
          d. ``work_dir/`` (recursive)

        Known artifact layout inside the source repo:
          - ``artifacts/negative_risk/exp_2026_02/model.pkl``
          - ``artifacts/spike_risk/exp_2026_02/model.pkl``

        Returns
        -------
        dict
            Keys: ``negative_risk_pkl``, ``spike_risk_pkl``.
            Values are paths (str) or *None*.
        """
        artifacts: dict[str, Optional[str]] = {
            "negative_risk_pkl": None,
            "spike_risk_pkl": None,
        }

        search_bases: list[Path] = []
        if self.source_repo_path:
            search_bases.append(Path(self.source_repo_path) / "artifacts")
            search_bases.append(Path(self.source_repo_path))
        if self.work_dir:
            search_bases.append(Path(self.work_dir) / "artifacts")
            search_bases.append(Path(self.work_dir))

        # ── (a) negative_risk model.pkl ───────────────────────────────
        for base in search_bases:
            if not base.exists():
                continue
            # Try known path first (fast path)
            known = base / "negative_risk" / "exp_2026_02" / "model.pkl"
            if known.is_file():
                artifacts["negative_risk_pkl"] = str(known)
                logger.info("Found negative_risk model (known path): %s", known)
                break
            # Fall back to recursive search for model.pkl under negative_risk/
            found = self._find_model_pkl(base, "negative_risk")
            if found:
                artifacts["negative_risk_pkl"] = found
                logger.info("Found negative_risk model (search): %s", found)
                break

        # ── (b) spike_risk model.pkl ──────────────────────────────────
        for base in search_bases:
            if not base.exists():
                continue
            known = base / "spike_risk" / "exp_2026_02" / "model.pkl"
            if known.is_file():
                artifacts["spike_risk_pkl"] = str(known)
                logger.info("Found spike_risk model (known path): %s", known)
                break
            found = self._find_model_pkl(base, "spike_risk")
            if found:
                artifacts["spike_risk_pkl"] = found
                logger.info("Found spike_risk model (search): %s", found)
                break

        self._artifacts = artifacts
        return artifacts

    # ── Model loading ─────────────────────────────────────────────────

    def load_models(self) -> dict:
        """Load negative_risk and spike_risk pickle models.

        Priority:
          1. Load both models if both artifact paths are found.
          2. Load whichever is available if only one is found.
          3. Fall back to CLASSIFIER_RULE_FALLBACK if neither is found.

        Returns
        -------
        dict
            Keys: ``status``, ``negative_model``, ``spike_model``,
            ``model_names``.
        """
        if not self._artifacts:
            self.find_artifacts()

        loaded_any = False

        # ── Load negative_risk model ──────────────────────────────────
        neg_pkl = self._artifacts.get("negative_risk_pkl")
        if neg_pkl:
            try:
                with open(neg_pkl, "rb") as f:
                    self._negative_model = pickle.load(f)
                self._model_names["negative_risk"] = "negative_risk_ml"
                logger.info("Loaded negative_risk model from %s", neg_pkl)
                loaded_any = True
            except Exception as exc:
                logger.warning("Failed to load negative_risk model from %s: %s", neg_pkl, exc)
                self._negative_model = None

        # ── Load spike_risk model ─────────────────────────────────────
        spk_pkl = self._artifacts.get("spike_risk_pkl")
        if spk_pkl:
            try:
                with open(spk_pkl, "rb") as f:
                    self._spike_model = pickle.load(f)
                self._model_names["spike_risk"] = "spike_risk_ml"
                logger.info("Loaded spike_risk model from %s", spk_pkl)
                loaded_any = True
            except Exception as exc:
                logger.warning("Failed to load spike_risk model from %s: %s", spk_pkl, exc)
                self._spike_model = None

        if loaded_any:
            self._status = CLASSIFIER_ML_READY
        else:
            self._status = CLASSIFIER_RULE_FALLBACK

        return {
            "status": self._status,
            "negative_model": self._negative_model,
            "spike_model": self._spike_model,
            "model_names": dict(self._model_names),
        }

    # ── Classification ────────────────────────────────────────────────

    def classify(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply real models (or rule fallback) to classify predictions.

        Input: DataFrame with price prediction columns.
        Output: adds columns:
          - ``classifier_action``: one of NEGATIVE_DETECTED / SPIKE_DETECTED / NORMAL / UNKNOWN
          - ``negative_risk``: probability or score of negative price risk
          - ``spike_risk``: probability or score of spike risk
          - ``normal_trend_flag``: 1 if normal, 0 if risk detected
          - ``uncertainty_score``: 0.0–1.0 uncertainty estimate
          - ``classifier_model_name``: name of the model(s) used
          - ``classifier_status``: CLASSIFIER_ML_READY or CLASSIFIER_RULE_FALLBACK

        Safety: never uses y_true as input feature.

        Parameters
        ----------
        predictions : pd.DataFrame
            DataFrame with price prediction columns.

        Returns
        -------
        pd.DataFrame
            Classified DataFrame with added columns.
        """
        classified = predictions.copy()

        # Guard: empty input
        if classified.empty:
            classified["classifier_action"] = "UNKNOWN"
            classified["negative_risk"] = 0.0
            classified["spike_risk"] = 0.0
            classified["normal_trend_flag"] = 0
            classified["uncertainty_score"] = 1.0
            classified["classifier_model_name"] = "none"
            classified["classifier_status"] = CLASSIFIER_RULE_FALLBACK
            return classified

        # Resolve price column (never y_true)
        price_col = self._resolve_price_col(classified)

        if price_col is None:
            classified["classifier_action"] = "UNKNOWN"
            classified["negative_risk"] = 0.0
            classified["spike_risk"] = 0.0
            classified["normal_trend_flag"] = 0
            classified["uncertainty_score"] = 1.0
            classified["classifier_model_name"] = "none"
            classified["classifier_status"] = CLASSIFIER_RULE_FALLBACK
            return classified

        # Lazy-load models if not yet attempted
        if self._negative_model is None and self._spike_model is None:
            self.load_models()

        prices = classified[price_col].fillna(0).values

        if self._status == CLASSIFIER_ML_READY:
            classified = self._apply_ml_classification(classified, price_col, prices)
        else:
            classified = self._apply_rule_fallback(classified, prices)

        return classified

    # ── Internal: ML classification ───────────────────────────────────

    def _apply_ml_classification(
        self,
        classified: pd.DataFrame,
        price_col: str,
        prices: np.ndarray,
    ) -> pd.DataFrame:
        """Apply loaded ML models to compute risk probabilities.

        Builds feature matrix from price column only — never y_true.
        Uses predict_proba() when available; falls back to predict().
        """
        # Feature matrix: price predictions only — y_true is NEVER used
        X = prices.reshape(-1, 1)

        # ── Negative risk probability ─────────────────────────────────
        neg_probs = self._predict_risk_probability(self._negative_model, X, len(prices))

        # ── Spike risk probability ────────────────────────────────────
        spk_probs = self._predict_risk_probability(self._spike_model, X, len(prices))

        classified["negative_risk"] = neg_probs
        classified["spike_risk"] = spk_probs

        # ── Determine action ──────────────────────────────────────────
        neg_flag = neg_probs >= NEGATIVE_RISK_PROB_THRESHOLD
        spk_flag = spk_probs >= SPIKE_RISK_PROB_THRESHOLD

        conditions = [neg_flag, spk_flag]
        choices = ["NEGATIVE_DETECTED", "SPIKE_DETECTED"]
        classified["classifier_action"] = np.select(conditions, choices, default="NORMAL")

        classified["normal_trend_flag"] = np.where(~neg_flag & ~spk_flag, 1, 0)

        # Uncertainty: higher when max risk probability is in ambiguous range
        max_prob = np.maximum(neg_probs, spk_probs)
        classified["uncertainty_score"] = np.where(
            max_prob > 0.8, 0.2,
            np.where(max_prob > 0.5, 0.5, 0.8)
        )

        # Model name: report which models were actually used
        model_parts = []
        if self._negative_model is not None:
            model_parts.append("negative_risk_ml")
        if self._spike_model is not None:
            model_parts.append("spike_risk_ml")
        classified["classifier_model_name"] = (
            "+".join(model_parts) if model_parts else "ml_classifier"
        )
        classified["classifier_status"] = CLASSIFIER_ML_READY

        return classified

    # ── Internal: rule-based fallback ──────────────────────────────────

    def _apply_rule_fallback(
        self,
        classified: pd.DataFrame,
        prices: np.ndarray,
    ) -> pd.DataFrame:
        """Rule-based fallback when ML models are not available."""
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
            (prices > HIGH_SPIKE_THRESHOLD * 0.7)
            | (prices < NEGATIVE_PRICE_THRESHOLD + 50),
            0.6, 0.3
        )

        conditions = [
            prices < NEGATIVE_PRICE_THRESHOLD,
            prices > HIGH_SPIKE_THRESHOLD,
            (prices >= NEGATIVE_PRICE_THRESHOLD) & (prices <= HIGH_SPIKE_THRESHOLD),
        ]
        choices = ["NEGATIVE_DETECTED", "SPIKE_DETECTED", "NORMAL"]
        classified["classifier_action"] = np.select(conditions, choices, default="UNKNOWN")
        classified["classifier_model_name"] = "rule_fallback"
        classified["classifier_status"] = CLASSIFIER_RULE_FALLBACK

        return classified

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _predict_risk_probability(
        model: Any,
        X: np.ndarray,
        expected_len: int,
    ) -> np.ndarray:
        """Predict risk probability from a single model.

        Uses predict_proba()[:, 1] when available; falls back to predict().
        Returns zeros on any failure.
        """
        if model is None:
            return np.zeros(expected_len)

        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                return np.asarray(proba, dtype=float).ravel()
            elif hasattr(model, "predict"):
                return np.asarray(model.predict(X), dtype=float).ravel()
        except Exception as exc:
            logger.warning("Model prediction failed: %s", exc)

        return np.zeros(expected_len)

    @staticmethod
    def _resolve_price_col(df: pd.DataFrame) -> Optional[str]:
        """Resolve price column — never returns y_true or actuals."""
        forbidden = {"y_true", "actual_price", "actual", "target"}
        candidates = [
            "y_pred", "dayahead_price", "realtime_price",
            "y_pred_corrected", "trend_pred", "price",
        ]
        for col in candidates:
            if col in df.columns and col not in forbidden:
                return col
        return None

    @staticmethod
    def _find_model_pkl(base: Path, keyword: str) -> Optional[str]:
        """Recursively search base for model.pkl under a directory matching keyword.

        Looks for paths like ``{base}/**/{keyword}/**/model.pkl``.
        """
        for match in base.rglob("model.pkl"):
            if keyword in str(match):
                return str(match)
        return None
