"""
adapters/residual_p5m_adapter.py — P5M Residual Correction Adapter (2.0 -> 3.0).

Integrates the 2.0/P5M residual correction stack into the 3.0 pipeline.
Searches for serialized models, CatBoost spike artifacts, and residual
stack code in the source repo.  Provides a unified, safety-guarded
interface for applying residual corrections.

Key facts
---------
- The P5M residual stack code lives in
  ``../electricity_forecast_model2.0_exp/residual_stack/``
  (orchestrator.py, priority.py, metrics.py, etc.)
- No serialized ``p5m_residual_model.pkl`` is guaranteed to exist —
  the residual system may be code-only.
- ``--residual-source-repo`` points to
  ``../electricity_forecast_model2.0_exp`` or
  ``.local_artifacts/source_repos/electricity_forecast_model2.0_exp``.
- CatBoost spike residual model exists at
  ``.local_artifacts/p31_p40_multimodel_fusion/models/catboost_spike_residual/catboost_spike_residual.cbm``.

Safety invariants
-----------------
- Never uses y_true from the current target_day for correction.
- If correction produces NaN, falls back to no-op.
- Tracks model_name, version, and source in every result dict.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Status constants ──────────────────────────────────────────────────

RESIDUAL_P5M_REAL_APPLIED = "RESIDUAL_P5M_REAL_APPLIED"
"""A real p5m_residual_model.pkl was loaded and applied."""

RESIDUAL_P5M_CATBOOST_APPLIED = "RESIDUAL_P5M_CATBOOST_APPLIED"
"""A CatBoost spike residual .cbm model was loaded and applied."""

RESIDUAL_P5M_CODE_ONLY = "RESIDUAL_P5M_CODE_ONLY"
"""P5M stack code was found but no serialised weights are available."""

RESIDUAL_P5M_NO_OP = "RESIDUAL_P5M_NO_OP"
"""No P5M artifacts found; correction is a no-op pass-through."""


# ── Adapter ───────────────────────────────────────────────────────────


class ResidualP5MAdapter:
    """Integrate the 2.0/P5M residual correction stack into 3.0.

    The adapter searches for P5M artifacts in the source repo and work
    directory, loads correction models when available, and applies
    corrections with safety guardrails.

    Parameters
    ----------
    residual_source_repo : str
        Path to the 2.0 source repo.  Typical values:
        ``"../electricity_forecast_model2.0_exp"`` or
        ``".local_artifacts/source_repos/electricity_forecast_model2.0_exp"``.
    work_dir : str
        Working directory for the current 3.0 run.
    strict : bool
        If *True*, raise on errors instead of falling back to no-op.
    """

    def __init__(
        self,
        residual_source_repo: str = "",
        work_dir: str = "",
        strict: bool = False,
    ) -> None:
        self.residual_source_repo = residual_source_repo
        self.work_dir = work_dir
        self.strict = strict

        self._artifacts: dict[str, Any] = {}
        self._model: Any = None
        self._status: str = RESIDUAL_P5M_NO_OP
        self._model_info: dict[str, str] = {
            "model_name": "none",
            "version": "0.0.0",
            "source": "none",
        }

    # ── Properties ────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        """Current adapter status (one of the ``RESIDUAL_P5M_*`` constants)."""
        return self._status

    # ── Artifact discovery ────────────────────────────────────────────

    def find_artifacts(self) -> dict:
        """Search for P5M artifacts in work_dir and source repo.

        Searches for:
          a. ``p5m_residual_model.pkl`` in work_dir or source repo
          b. ``catboost_spike_residual.cbm`` under ``.local_artifacts/``
          c. P5M residual stack code in source repo
             (``residual_stack/orchestrator.py``)
          d. P5M reports (``reports/local/p5m_residual_stack/``)

        Returns
        -------
        dict
            Keys: ``p5m_pkl``, ``catboost_cbm``, ``stack_code``, ``reports``.
            Values are paths (str) or *None*.
        """
        artifacts: dict[str, Any] = {
            "p5m_pkl": None,
            "catboost_cbm": None,
            "stack_code": None,
            "reports": None,
        }

        # Collect base directories to search
        search_bases: list[tuple[str, Path]] = []
        if self.work_dir:
            search_bases.append(("work_dir", Path(self.work_dir)))
        if self.residual_source_repo:
            search_bases.append(("source_repo", Path(self.residual_source_repo)))

        # ── (a) p5m_residual_model.pkl ───────────────────────────────
        for _label, base in search_bases:
            found = self._find_file(base, "p5m_residual_model.pkl")
            if found:
                artifacts["p5m_pkl"] = found
                logger.info("Found p5m_residual_model.pkl: %s", found)
                break

        # ── (b) catboost_spike_residual.cbm ──────────────────────────
        cbm_search_bases: list[Path] = []
        if self.work_dir:
            cbm_search_bases.append(Path(self.work_dir))
            cbm_search_bases.append(Path(self.work_dir).parent / ".local_artifacts")
        # Always check CWD-relative .local_artifacts
        cbm_search_bases.append(Path(".local_artifacts"))
        # Also check relative to source repo
        if self.residual_source_repo:
            cbm_search_bases.append(Path(self.residual_source_repo) / ".local_artifacts")

        for base in cbm_search_bases:
            if not base.exists():
                continue
            found = self._find_file(base, "catboost_spike_residual.cbm")
            if found:
                artifacts["catboost_cbm"] = found
                logger.info("Found catboost_spike_residual.cbm: %s", found)
                break

        # ── (c) P5M residual stack code ──────────────────────────────
        if self.residual_source_repo:
            orchestrator = (
                Path(self.residual_source_repo)
                / "residual_stack"
                / "orchestrator.py"
            )
            if orchestrator.is_file():
                artifacts["stack_code"] = str(orchestrator)
                logger.info("Found P5M residual stack code: %s", orchestrator)

        # ── (d) P5M reports ──────────────────────────────────────────
        if self.residual_source_repo:
            reports_dir = (
                Path(self.residual_source_repo)
                / "reports"
                / "local"
                / "p5m_residual_stack"
            )
            if reports_dir.is_dir():
                artifacts["reports"] = str(reports_dir)
                logger.info("Found P5M reports directory: %s", reports_dir)

        self._artifacts = artifacts
        return artifacts

    # ── Model loading ─────────────────────────────────────────────────

    def load_correction_model(self) -> dict:
        """Try to load any found artifact as a correction model.

        Priority order:
          1. ``p5m_residual_model.pkl``  (real P5M pickle model)
          2. ``catboost_spike_residual.cbm``  (CatBoost spike model)
          3. Code only  (stack code found but no weights)
          4. No-op  (nothing found)

        Returns
        -------
        dict
            Keys: ``status``, ``model``, ``model_info``.
        """
        if not self._artifacts:
            self.find_artifacts()

        # ── Priority 1: p5m pkl ──────────────────────────────────────
        if self._artifacts.get("p5m_pkl"):
            try:
                import pickle

                pkl_path = self._artifacts["p5m_pkl"]
                with open(pkl_path, "rb") as f:
                    self._model = pickle.load(f)
                self._status = RESIDUAL_P5M_REAL_APPLIED
                self._model_info = {
                    "model_name": "p5m_residual",
                    "version": "2.0",
                    "source": pkl_path,
                }
                logger.info("Loaded P5M real model from %s", pkl_path)
                return self._build_load_result()
            except Exception as exc:
                logger.warning("Failed to load p5m_residual_model.pkl: %s", exc)
                if self.strict:
                    raise

        # ── Priority 2: catboost cbm ─────────────────────────────────
        if self._artifacts.get("catboost_cbm"):
            cbm_path = self._artifacts["catboost_cbm"]
            try:
                from catboost import CatBoost

                self._model = CatBoost()
                self._model.load_model(cbm_path)
                self._status = RESIDUAL_P5M_CATBOOST_APPLIED
                self._model_info = {
                    "model_name": "catboost_spike_residual",
                    "version": "3.0",
                    "source": cbm_path,
                }
                logger.info(
                    "Loaded CatBoost spike residual model from %s", cbm_path
                )
                return self._build_load_result()
            except ImportError:
                logger.warning(
                    "catboost package not available — cannot load .cbm model"
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load catboost_spike_residual.cbm: %s", exc
                )
                if self.strict:
                    raise

        # ── Priority 3: code only ────────────────────────────────────
        if self._artifacts.get("stack_code"):
            self._status = RESIDUAL_P5M_CODE_ONLY
            self._model_info = {
                "model_name": "p5m_stack_code_only",
                "version": "2.0",
                "source": self._artifacts["stack_code"],
            }
            logger.warning(
                "P5M residual stack code found at %s but no serialized "
                "model weights — falling back to no-op.",
                self._artifacts["stack_code"],
            )
            return self._build_load_result()

        # ── Priority 4: nothing ──────────────────────────────────────
        self._status = RESIDUAL_P5M_NO_OP
        self._model_info = {
            "model_name": "none",
            "version": "0.0.0",
            "source": "none",
        }
        logger.info("No P5M artifacts found — using no-op fallback.")
        return self._build_load_result()

    # ── Correction ────────────────────────────────────────────────────

    def apply_correction(
        self,
        predictions: pd.DataFrame,
        task: str = "dayahead",
    ) -> dict:
        """Apply correction if a model is loaded, else no-op fallback.

        Safety invariants:
          - Never uses y_true from the current target_day.
          - If correction produces NaN, falls back to no-op.
          - Tracks model_name, version, source in the result.

        Parameters
        ----------
        predictions : pd.DataFrame
            Prediction ledger (must contain a price column).
        task : str
            ``"dayahead"`` or ``"realtime"``.

        Returns
        -------
        dict
            Keys: ``task``, ``status``, ``correction_applied``,
            ``model_info``, ``reason_codes``, ``output``, ``rows``.
        """
        result: dict[str, Any] = {
            "task": task,
            "status": RESIDUAL_P5M_NO_OP,
            "correction_applied": False,
            "model_info": dict(self._model_info),
            "reason_codes": [],
            "output": None,
        }

        # Guard: empty input
        if predictions is None or len(predictions) == 0:
            result["status"] = RESIDUAL_P5M_NO_OP
            result["reason_codes"].append("NO_PREDICTIONS")
            result["output"] = predictions
            return result

        # Lazy-load if not yet attempted
        if self._model is None and self._status in (
            RESIDUAL_P5M_NO_OP,
        ):
            self.load_correction_model()

        # ── Apply real model correction ──────────────────────────────
        if self._model is not None and self._status in (
            RESIDUAL_P5M_REAL_APPLIED,
            RESIDUAL_P5M_CATBOOST_APPLIED,
        ):
            try:
                corrected = self._apply_model_correction(predictions, task)

                # Safety: NaN check
                if "y_pred_corrected" in corrected.columns:
                    if corrected["y_pred_corrected"].isna().any():
                        nan_count = int(corrected["y_pred_corrected"].isna().sum())
                        logger.warning(
                            "Correction produced %d NaN value(s). "
                            "Falling back to no-op.",
                            nan_count,
                        )
                        result["reason_codes"].append("CORRECTION_PRODUCED_NAN")
                        corrected = self._apply_noop_fallback(predictions, task)
                        result["status"] = RESIDUAL_P5M_NO_OP
                        result["output"] = corrected
                        result["rows"] = len(corrected)
                        return result

                result["status"] = self._status
                result["correction_applied"] = True
                result["output"] = corrected
                result["rows"] = len(corrected)
                return result

            except Exception as exc:
                logger.warning(
                    "P5M correction failed: %s. Falling back to no-op.", exc
                )
                result["reason_codes"].append(f"P5M_CORRECTION_FAILED:{exc}")
                if self.strict:
                    raise

        # ── Code-only warning ────────────────────────────────────────
        if self._status == RESIDUAL_P5M_CODE_ONLY:
            logger.warning(
                "P5M stack code found but no model weights available. "
                "Using no-op fallback."
            )
            result["reason_codes"].append("CODE_ONLY_NO_WEIGHTS")

        # ── No-op fallback ───────────────────────────────────────────
        corrected = self._apply_noop_fallback(predictions, task)
        result["status"] = RESIDUAL_P5M_NO_OP
        result["output"] = corrected
        result["rows"] = len(corrected)
        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _apply_model_correction(
        self,
        predictions: pd.DataFrame,
        task: str,
    ) -> pd.DataFrame:
        """Apply model-based residual correction.

        The residual delta comes solely from the loaded model —
        y_true from the current target_day is never accessed.
        """
        corrected = predictions.copy()
        price_col = self._resolve_price_col(corrected)

        if price_col is None:
            corrected["y_pred_raw"] = np.nan
            corrected["residual_delta"] = 0.0
            corrected["y_pred_corrected"] = np.nan
            corrected["residual_model_name"] = self._model_info["model_name"]
            corrected["residual_version"] = self._model_info["version"]
            corrected["residual_source"] = self._model_info["source"]
            corrected["residual_status"] = RESIDUAL_P5M_NO_OP
            return corrected

        X = corrected[[price_col]].fillna(0).values

        # Compute residual delta from model only — never from y_true
        if hasattr(self._model, "predict"):
            try:
                residual_delta = np.asarray(
                    self._model.predict(X), dtype=float
                ).ravel()
            except Exception:
                residual_delta = np.zeros(len(X))
        else:
            residual_delta = np.zeros(len(X))

        # Ensure shapes match
        if len(residual_delta) != len(corrected):
            logger.warning(
                "Model output length mismatch (%d vs %d). "
                "Truncating/padding with zeros.",
                len(residual_delta),
                len(corrected),
            )
            padded = np.zeros(len(corrected))
            n = min(len(residual_delta), len(corrected))
            padded[:n] = residual_delta[:n]
            residual_delta = padded

        corrected["y_pred_raw"] = corrected[price_col]
        corrected["residual_delta"] = residual_delta
        corrected["y_pred_corrected"] = corrected[price_col] + residual_delta
        corrected["residual_model_name"] = self._model_info["model_name"]
        corrected["residual_version"] = self._model_info["version"]
        corrected["residual_source"] = self._model_info["source"]
        corrected["residual_status"] = self._status

        return corrected

    @staticmethod
    def _apply_noop_fallback(
        predictions: pd.DataFrame,
        task: str,
    ) -> pd.DataFrame:
        """No-op fallback: corrected equals original prediction."""
        corrected = predictions.copy()
        price_col = ResidualP5MAdapter._resolve_price_col(corrected)

        if price_col:
            corrected["y_pred_raw"] = corrected[price_col]
            corrected["residual_delta"] = 0.0
            corrected["y_pred_corrected"] = corrected[price_col]
        else:
            corrected["y_pred_raw"] = np.nan
            corrected["residual_delta"] = 0.0
            corrected["y_pred_corrected"] = np.nan

        corrected["residual_model_name"] = "noop_fallback"
        corrected["residual_version"] = "0.0.0"
        corrected["residual_source"] = "none"
        corrected["residual_status"] = RESIDUAL_P5M_NO_OP

        return corrected

    @staticmethod
    def _resolve_price_col(df: pd.DataFrame) -> Optional[str]:
        """Resolve the price column name from the DataFrame."""
        for col in ("y_pred", "dayahead_price", "trend_pred", "realtime_price"):
            if col in df.columns:
                return col
        return None

    def _build_load_result(self) -> dict:
        """Build the dict returned by load_correction_model()."""
        return {
            "status": self._status,
            "model": self._model,
            "model_info": dict(self._model_info),
        }

    @staticmethod
    def _find_file(base: Path, filename: str) -> Optional[str]:
        """Recursively search *base* for *filename*, returning first hit."""
        # Direct child first (fast path)
        direct = base / filename
        if direct.is_file():
            return str(direct)
        # Recursive search
        for match in base.rglob(filename):
            if match.is_file():
                return str(match)
        return None
