"""
models/adapters/realtime_da_safe_assist.py — DA-Safe Realtime Assist adapter.

Wraps the DA-Safe Realtime Assist Model (RT-Assist-1) into the 3.0 adapter contract.

Source repository: disdorqin/electricity_forecast_deep_sgdf_delta (main)

Final model positioning:
    Primary prediction:  rt_pred = da_anchor (DA-only, most stable)
    Optional correction: da_anchor + alpha * residual_pred (default: disabled)
    Assist outputs:      da_error_prob, residual_direction, uncertainty, etc.

Usage:
    adapter = DASafeRealtimeAssistAdapter()
    adapter.load(model_dir="exported_models/rt_assist_pack")
    result = adapter.predict(
        data_path="data/preprocessed.csv",
        start="2026-03-01",
        end="2026-03-31",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.adapters.base import BasePredictionAdapter

logger = logging.getLogger(__name__)


class DASafeRealtimeAssistAdapter(BasePredictionAdapter):
    """Adapter for DA-Safe Realtime Assist Model.

    Parameters
    ----------
    model_version : str, optional
        Version string. Defaults to "1.0.0".
    enable_safe_correction : bool
        Whether to enable residual correction (default: False).
    alpha : float
        Correction strength when enabled (default: 1.0).
    clip_correction : float
        Max absolute correction per hour (default: 0 = no clip).
    """

    def __init__(
        self,
        model_version: Optional[str] = None,
        enable_safe_correction: bool = False,
        alpha: float = 1.0,
        clip_correction: float = 0.0,
    ) -> None:
        super().__init__(
            model_id="da_safe_realtime_assist",
            model_version=model_version or "1.0.0",
        )
        self.enable_safe_correction = enable_safe_correction
        self.alpha = alpha
        self.clip_correction = clip_correction
        self._manifest: dict = {}
        self._residual_model = None

    @property
    def task(self) -> str:
        return "realtime"

    def load(self) -> None:
        """Mark adapter as ready. Residual model loading is lazy."""
        self._loaded = True
        logger.info(
            f"DASafeRealtimeAssistAdapter ({self.model_version}): "
            f"safe_correction={'ON' if self.enable_safe_correction else 'OFF'}"
        )

    def load_model_pack(self, model_dir: str) -> None:
        """Load the RT-Assist model pack from an exported directory.

        Parameters
        ----------
        model_dir : str
            Path to ``exported_models/rt_assist_pack/``.
        """
        import json

        model_path = Path(model_dir)
        manifest_path = model_path / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest = json.load(f)
            logger.info(f"Loaded manifest from {manifest_path}")

        # Try to load residual model
        residual_path = model_path / "residual_model.pkl"
        if residual_path.exists():
            import pickle
            try:
                with open(residual_path, "rb") as f:
                    self._residual_model = pickle.load(f)
                logger.info(f"Loaded residual model from {residual_path}")
            except Exception as e:
                logger.warning(f"Could not load residual model: {e}")
                self._residual_model = None

        self._loaded = True

    def predict(  # type: ignore[override]
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run DA-Safe Realtime Assist prediction.

        The default behavior is ``rt_pred = da_anchor`` (DA-only).
        If ``enable_safe_correction`` is True and a residual model is loaded,
        a correction term is added.

        Parameters
        ----------
        data_path : str, optional
            Path to CSV with raw hourly data. Requires ``da_anchor`` column.
        df : pd.DataFrame, optional
            Pre-loaded DataFrame. Requires ``da_anchor`` column.
        start : str, optional
            Start date inclusive (YYYY-MM-DD).
        end : str, optional
            End date inclusive (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Standard schema prediction output.

        Raises
        ------
        ValueError
            If neither ``data_path`` nor ``df`` is provided, or ``da_anchor`` is missing.
        """
        if not self._loaded:
            self.load()

        # Load data
        if df is None:
            if data_path is None:
                raise ValueError("Either data_path= or df= must be provided")
            df = pd.read_csv(
                data_path,
                parse_dates=["ds"] if "ds" in open(data_path).read(2000) else ["times"],
            )
        else:
            df = df.copy()

        # Normalize column names
        if "times" in df.columns and "ds" not in df.columns:
            df = df.rename(columns={"times": "ds"})
        if "da_price" in df.columns and "da_anchor" not in df.columns:
            df["da_anchor"] = df["da_price"]
        if "rt_price" in df.columns and "rt_actual" not in df.columns:
            df["rt_actual"] = df["rt_price"]

        df["ds"] = pd.to_datetime(df["ds"])

        # Filter by date range
        if start:
            df = df[df["ds"] >= pd.Timestamp(start)]
        if end:
            df = df[df["ds"] <= pd.Timestamp(end) + pd.Timedelta(days=1)]

        if len(df) == 0:
            logger.warning("No data after filtering, returning empty")
            return pd.DataFrame(columns=self._get_output_columns())

        # Ensure da_anchor exists
        if "da_anchor" not in df.columns:
            raise ValueError(
                "DA-Safe Realtime Assist requires 'da_anchor' column in input data"
            )

        df = df.sort_values("ds").reset_index(drop=True)
        da_anchor = df["da_anchor"].values.astype(float)

        # Primary prediction: DA-only
        safe_correction = np.zeros(len(df), dtype=float)

        # Optional safe correction
        if self.enable_safe_correction and self._residual_model is not None:
            feature_cols = self._manifest.get("feature_columns", [])
            feature_cols = [c for c in feature_cols if c in df.columns]
            if len(feature_cols) > 0:
                X = df[feature_cols].values.astype(float)
                residual_pred = self._residual_model.predict(X) * self.alpha
                if self.clip_correction > 0:
                    residual_pred = np.clip(
                        residual_pred,
                        -self.clip_correction,
                        self.clip_correction,
                    )
                safe_correction = residual_pred

        rt_pred = da_anchor + safe_correction

        # Add business time columns
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")

        # Build output
        out = pd.DataFrame({
            "task": "realtime",
            "model_name": "da_safe_realtime_assist",
            "target_day": df["ds"].dt.date.astype(str),
            "business_day": df["business_day"],
            "ds": df["ds"],
            "hour_business": df["hour_business"],
            "period": df["period"],
            "y_pred": rt_pred,
            "source_confidence": np.full(len(df), 0.5),
            "model_version": self.model_version,
        })

        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


def create_adapter(**kwargs: Any) -> DASafeRealtimeAssistAdapter:
    """Factory function for DASafeRealtimeAssistAdapter.

    Usage:
        adapter = create_adapter(enable_safe_correction=False)
        adapter.load()
        result = adapter.predict(data_path="...", start="...", end="...")
    """
    return DASafeRealtimeAssistAdapter(**kwargs)
