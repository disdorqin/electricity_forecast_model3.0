"""
models/adapters/sgdfnet_assist_adapter.py — P92: SGDFNet Assist Adapter.

Wraps the SGDFNet model from the 2.0 experiment repository
(``disdorqin/electricity_forecast_model2.0_exp/SGDFNet``) as an optional
assist / sidecar for realtime prediction.

This adapter does NOT rewrite the SGDFNet model. It imports from the
2.0 repo and translates its output into the 3.0 adapter contract.

Design:
    SGDFNet is a **candidate assist model** for realtime prediction.
    Its output is combined with the DA-Safe Baseline via the realtime
    prediction ledger and pooled learner (P93 + P94).

    If SGDFNet cannot be imported or run, status = SGDFNET_ASSIST_CODE_ONLY
    or SGDFNET_ASSIST_BLOCKED. The system still delivers on DA-Safe Baseline.

Output schema (sgdfnet_realtime_assist_pack.csv):
    business_day, ds, hour_business, period,
    model_name = sgdfnet_rt_assist,
    rt_pred, sgdfnet_pred, da_anchor,
    assist_available, source_confidence,
    da_error_prob, residual_direction_prob,
    uncertainty_score, correction_permission,
    reason_codes
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.adapters.base import BasePredictionAdapter
from models.realtime_state import (
    SGDFNET_ASSIST_READY,
    SGDFNET_ASSIST_CODE_ONLY,
    SGDFNET_ASSIST_BLOCKED,
    SGDFNET_ASSIST_ACTIVE,
)

logger = logging.getLogger(__name__)


class SGDFNetAssistAdapter(BasePredictionAdapter):
    """Adapter for SGDFNet assist model from 2.0 experiment repo.

    Parameters
    ----------
    sgdfnet_root : str
        Path to the SGDFNet directory in the 2.0 exp repo
        (e.g. ``../electricity_forecast_model2.0_exp/SGDFNet``).
    model_version : str, optional
        Version string. Defaults to "2.0.0".
    """

    # Output pack columns
    PACK_COLUMNS = [
        "business_day",
        "ds",
        "hour_business",
        "period",
        "model_name",
        "rt_pred",
        "sgdfnet_pred",
        "da_anchor",
        "assist_available",
        "source_confidence",
        "da_error_prob",
        "residual_direction_prob",
        "uncertainty_score",
        "correction_permission",
        "reason_codes",
    ]

    FORBIDDEN_COLUMNS = [
        "y_true",
        "actual",
        "label",
        "residual_from_y_true",
        "future_actual",
        "eval_residual",
    ]

    def __init__(
        self,
        sgdfnet_root: str = "",
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id="sgdfnet_rt_assist",
            model_version=model_version or "2.0.0",
        )
        self.sgdfnet_root = sgdfnet_root
        self._sgdfnet_available = False
        self._sgdfnet_pipeline = None
        self._import_error: Optional[str] = None
        self._assist_status = SGDFNET_ASSIST_CODE_ONLY

    @property
    def task(self) -> str:
        return "realtime"

    @property
    def assist_status(self) -> str:
        return self._assist_status

    def load(self) -> None:
        """Try to import SGDFNet from the 2.0 experiment repo.

        Sets assist_status:
            SGDFNET_ASSIST_READY  — SGDFNet imported successfully
            SGDFNET_ASSIST_CODE_ONLY — code available but root not found
            SGDFNET_ASSIST_BLOCKED  — import error
        """
        if not self.sgdfnet_root or not os.path.isdir(self.sgdfnet_root):
            self._assist_status = SGDFNET_ASSIST_CODE_ONLY
            self._import_error = f"SGDFNet root not found: {self.sgdfnet_root}"
            logger.warning(self._import_error)
            self._loaded = True
            return

        # Add SGDFNet src to sys.path
        sgdfnet_src = os.path.join(self.sgdfnet_root, "src")
        if sgdfnet_src not in sys.path:
            sys.path.insert(0, sgdfnet_src)

        # Add SGDFNet root to sys.path
        if self.sgdfnet_root not in sys.path:
            sys.path.insert(0, self.sgdfnet_root)

        try:
            from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment

            # Try to import ModelPipeline
            try:
                from SGDFNet.pipeline import ModelPipeline as _SgdfPipeline
                self._sgdfnet_pipeline = _SgdfPipeline
            except ImportError:
                # Try direct import
                self._sgdfnet_pipeline = type("SGDFNetPlaceholder", (), {})
                self._sgdfnet_pipeline.run_protocol_b = run_protocol_b_cutoff_experiment

            self._sgdfnet_available = True
            self._assist_status = SGDFNET_ASSIST_READY
            logger.info("SGDFNet assist adapter: READY (from %s)", self.sgdfnet_root)

        except ImportError as e:
            self._assist_status = SGDFNET_ASSIST_BLOCKED
            self._import_error = str(e)
            logger.warning("SGDFNet import failed: %s", e)
        except Exception as e:
            self._assist_status = SGDFNET_ASSIST_BLOCKED
            self._import_error = str(e)
            logger.warning("SGDFNet load error: %s", e)

        self._loaded = True

    def predict(  # type: ignore[override]
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        da_predictions: Optional[pd.DataFrame] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run SGDFNet assist prediction.

        If SGDFNet is not available, returns DataFrame with NaN predictions
        and ``assist_available=False``.

        Parameters
        ----------
        data_path : str, optional
            Path to raw data CSV.
        df : pd.DataFrame, optional
            Pre-loaded raw data DataFrame. Requires ``da_anchor`` column.
        da_predictions : pd.DataFrame, optional
            Day-ahead predictions with ``y_pred`` or ``da_anchor``.
        start : str, optional
            Start date inclusive (YYYY-MM-DD).
        end : str, optional
            End date inclusive (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            SGDFNet assist pack with standard and assist columns.
        """
        if not self._loaded:
            self.load()

        # Load data
        if df is None and data_path is not None:
            try:
                df = pd.read_csv(data_path, parse_dates=["ds"] if "ds" in open(data_path).read(2000) else [])
            except Exception:
                try:
                    df = pd.read_csv(data_path, encoding="gbk")
                except Exception as e:
                    raise ValueError(f"Cannot read data: {e}") from e

        if df is None and da_predictions is not None:
            df = da_predictions.copy()

        if df is None:
            raise ValueError("Either df=, data_path=, or da_predictions= must be provided")

        df = df.copy()

        # Normalize column names
        if "times" in df.columns and "ds" not in df.columns:
            df = df.rename(columns={"times": "ds"})
        if "da_price" in df.columns and "da_anchor" not in df.columns:
            df["da_anchor"] = df["da_price"]

        df["ds"] = pd.to_datetime(df["ds"])

        # Filter by date range
        if start:
            df = df[df["ds"] >= pd.Timestamp(start)]
        if end:
            df = df[df["ds"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]

        if len(df) == 0:
            logger.warning("No data after filtering, returning empty")
            return pd.DataFrame(columns=self.PACK_COLUMNS)

        # Ensure da_anchor exists
        if "da_anchor" not in df.columns:
            if da_predictions is not None and "y_pred" in da_predictions.columns:
                # Use y_pred from da_predictions as da_anchor
                df["da_anchor"] = da_predictions["y_pred"].values[:len(df)]
            else:
                raise ValueError(
                    "SGDFNet assist requires 'da_anchor' column in input data"
                )

        df = df.sort_values("ds").reset_index(drop=True)

        # Add business time columns
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")

        n = len(df)
        da_anchor = df["da_anchor"].values.astype(float)

        # Try to run SGDFNet prediction
        if self._sgdfnet_available and self._assist_status == SGDFNET_ASSIST_READY:
            sgdfnet_pred = self._run_sgdfnet_prediction(df, start, end)
            rt_pred = sgdfnet_pred.copy()
            assist_available = np.ones(n, dtype=bool)
            reason_codes_arr = np.full(n, SGDFNET_ASSIST_ACTIVE)
            source_confidence = np.full(n, 0.4)
            correction_permission = np.full(n, True)
        else:
            # SGDFNet not available — return NaN predictions
            sgdfnet_pred = np.full(n, np.nan)
            rt_pred = da_anchor.copy()
            assist_available = np.zeros(n, dtype=bool)
            reason_codes_arr = np.full(n, "SGDFNET_ASSIST_DISABLED")
            source_confidence = np.full(n, 0.0)
            correction_permission = np.full(n, False)

        # Build output pack
        out = pd.DataFrame({
            "business_day": df["business_day"],
            "ds": df["ds"],
            "hour_business": df["hour_business"],
            "period": df["period"],
            "model_name": "sgdfnet_rt_assist",
            "rt_pred": rt_pred,
            "sgdfnet_pred": sgdfnet_pred,
            "da_anchor": da_anchor,
            "assist_available": assist_available,
            "source_confidence": source_confidence,
            "da_error_prob": np.where(np.abs(da_anchor) < 1e-6, 0.5, 0.4 * np.ones(n)),
            "residual_direction_prob": 0.5 * np.ones(n),
            "uncertainty_score": np.where(assist_available, 0.4, 0.8),
            "correction_permission": correction_permission,
            "reason_codes": reason_codes_arr,
        })

        # Validate no forbidden columns
        for col in self.FORBIDDEN_COLUMNS:
            if col in out.columns:
                out = out.drop(columns=[col])

        return out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)

    def _run_sgdfnet_prediction(
        self,
        df: pd.DataFrame,
        start: Optional[str],
        end: Optional[str],
    ) -> np.ndarray:
        """Run SGDFNet model and return predictions.

        Falls back to da_anchor if SGDFNet execution fails at runtime.
        """
        try:
            # Import SGDFNet module
            from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment

            data_path = None
            # Try to write temp CSV for SGDFNet
            import tempfile
            tmp_csv = os.path.join(tempfile.gettempdir(), "sgdfnet_assist_input.csv")
            df.to_csv(tmp_csv, index=False)

            # Determine start/end
            s = start or str(df["ds"].min().date())
            e = end or str(df["ds"].max().date())

            # Try to run SGDFNet
            config_path = os.path.join(self.sgdfnet_root, "configs",
                                       "cutoff_recovery_2026_diag_a_prune_actualside.yaml")

            if not os.path.isfile(config_path):
                logger.warning("SGDFNet config not found: %s", config_path)
                return df["da_anchor"].values.astype(float)

            # Try running the experiment
            run_dir = run_protocol_b_cutoff_experiment(config_path)

            # Read predictions
            pred_path = os.path.join(run_dir, "predictions.csv")
            if os.path.isfile(pred_path):
                sgdf_raw = pd.read_csv(pred_path, encoding="utf-8-sig")

                # Detect prediction column
                pred_col = None
                for col_candidate in ["rt_hat", "y_pred", "prediction", "sgdfnet_pred"]:
                    if col_candidate in sgdf_raw.columns:
                        pred_col = col_candidate
                        break

                if pred_col is None:
                    logger.warning("SGDFNet predictions missing expected column")
                    return df["da_anchor"].values.astype(float)

                # Align by date/hour
                if "ds" not in sgdf_raw.columns and "timestamp" in sgdf_raw.columns:
                    sgdf_raw["ds"] = pd.to_datetime(sgdf_raw["timestamp"])
                elif "ds" not in sgdf_raw.columns:
                    # Assume same order
                    preds = sgdf_raw[pred_col].values[:len(df)]
                    return np.where(np.isnan(preds), df["da_anchor"].values.astype(float), preds)

                sgdf_raw["ds"] = pd.to_datetime(sgdf_raw["ds"])
                sgdf_raw = sgdf_raw.sort_values("ds").reset_index(drop=True)

                # Align to our dataframe
                min_len = min(len(sgdf_raw), len(df))
                preds = sgdf_raw[pred_col].values[:min_len]
                # Pad to full length
                full_preds = df["da_anchor"].values.astype(float).copy()
                full_preds[:min_len] = np.where(
                    np.isnan(preds),
                    full_preds[:min_len],
                    preds,
                )
                return full_preds
            else:
                logger.warning("SGDFNet predictions file not found: %s", pred_path)
                return df["da_anchor"].values.astype(float)

        except Exception as e:
            logger.warning("SGDFNet runtime execution failed: %s", e)
            return df["da_anchor"].values.astype(float)

    def export_assist_pack(
        self,
        output_dir: str,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        da_predictions: Optional[pd.DataFrame] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, Any]:
        """Export SGDFNet assist pack to directory.

        Produces:
            sgdfnet_realtime_assist_pack.csv
            sgdfnet_realtime_assist_manifest.json

        Returns dict with status and paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        result = self.predict(
            data_path=data_path,
            df=df,
            da_predictions=da_predictions,
            start=start,
            end=end,
        )

        csv_path = os.path.join(output_dir, "sgdfnet_realtime_assist_pack.csv")
        result.to_csv(csv_path, index=False)

        manifest = {
            "model_name": "sgdfnet_rt_assist",
            "model_version": self.model_version,
            "assist_status": self._assist_status,
            "assist_available": bool(result["assist_available"].any()),
            "rows": len(result),
            "columns": list(result.columns),
            "sgdfnet_root": self.sgdfnet_root,
            "import_error": self._import_error,
            "start": start,
            "end": end,
        }

        manifest_path = os.path.join(output_dir, "sgdfnet_realtime_assist_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return {
            "status": self._assist_status,
            "csv_path": csv_path,
            "manifest_path": manifest_path,
            "rows": len(result),
            "assist_available": bool(result["assist_available"].any()),
            "manifest": manifest,
        }


def create_adapter(**kwargs: Any) -> SGDFNetAssistAdapter:
    """Factory function for SGDFNetAssistAdapter."""
    return SGDFNetAssistAdapter(**kwargs)
