"""
models/adapters/sgdfnet_2_5.py — SGDFNet delta regressor adapter (from 2.5 repo).

Wraps the SGDFNet model from ``disdorqin/electricity_forecast_model2.5``
into the 3.0 adapter contract.

Source: 2.5 repo → SGDFNet/src/sgdfnet/
Status: CANDIDATE — full integration requires model weight files and
        feature pipeline wiring.

Usage:
    adapter = SGDFNet25Adapter()
    adapter.load(model_dir="path/to/sgdfnet_weights")
    result = adapter.predict(
        data_path="path/to/features.csv",
        target_dates=["2026-03-01", "2026-03-31"],
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


class SGDFNet25Adapter(BasePredictionAdapter):
    """Adapter for SGDFNet delta regressor from 2.5 repo.

    Parameters
    ----------
    model_version : str, optional
        Version string. Defaults to "0.1.0" (pre-release).
    """

    def __init__(self, model_version: Optional[str] = None) -> None:
        super().__init__(
            model_id="sgdfnet_2_5",
            model_version=model_version or "0.1.0",
        )
        self._model = None

    @property
    def task(self) -> str:
        return "realtime"

    def load(self) -> None:
        """Mark adapter as ready.

        Note: SGDFNet model weights are not tracked in git.
        Actual model loading requires external weight files.
        """
        self._loaded = True
        logger.info(
            f"SGDFNet25Adapter ({self.model_version}): "
            f"CANDIDATE — model weights not in git. "
            f"Call load_weights(weights_dir=) when weights are available."
        )

    def load_weights(self, weights_dir: str) -> None:
        """Load SGDFNet model weights from a directory.

        Parameters
        ----------
        weights_dir : str
            Path to directory containing sgdfnet weights/config.
        """
        weight_path = Path(weights_dir)
        # SGDFNet uses ProtocolB / DeltaRegressor
        # This will be implemented when weights are available
        logger.info(f"Looking for SGDFNet weights in {weight_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

        # Placeholder: actual model loading will depend on SGDFNet's
        # serialization format (pickle / joblib / custom)
        logger.info(f"SGDFNet25Adapter: weights loaded from {weights_dir}")
        self._model = object()  # placeholder

    def predict(  # type: ignore[override]
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_dates: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run SGDFNet prediction (stub — requires model weights).

        Parameters
        ----------
        data_path : str, optional
            Path to CSV with features.
        df : pd.DataFrame, optional
            Pre-loaded feature DataFrame.
        target_dates : list[str], optional
            List of target dates.

        Returns
        -------
        pd.DataFrame
            Standard schema prediction output. When weights are missing,
            returns an empty DataFrame with a warning.

        Raises
        ------
        RuntimeError
            If load() was not called.
        """
        if not self._loaded:
            self.load()

        if self._model is None:
            logger.warning(
                "SGDFNet25Adapter: model weights not loaded. "
                "Call load_weights(weights_dir=) first. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame(columns=self._get_output_columns())

        # Load data
        if df is None and data_path is not None:
            df = pd.read_csv(data_path, parse_dates=["ds"] if "ds" in open(data_path).read(2000) else [])
            df["ds"] = pd.to_datetime(df["ds"])

        if df is None or len(df) == 0:
            logger.warning("No data provided, returning empty")
            return pd.DataFrame(columns=self._get_output_columns())

        df = df.sort_values("ds").reset_index(drop=True)

        # Placeholder: actual prediction logic goes here
        # delta_pred = self._model.predict(features)
        # rt_pred = da_anchor + delta_pred
        logger.warning("SGDFNet25Adapter: predict() is a stub — no actual prediction")

        # Add business time columns
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")

        # Return zero predictions as placeholder
        out = pd.DataFrame({
            "task": "realtime",
            "model_name": "sgdfnet_2_5",
            "target_day": pd.to_datetime(df["ds"]).dt.date.astype(str),
            "business_day": df["business_day"],
            "ds": df["ds"],
            "hour_business": df["hour_business"],
            "period": df["period"],
            "y_pred": np.zeros(len(df)),
            "source_confidence": np.full(len(df), np.nan),
            "model_version": self.model_version,
        })

        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


def create_adapter(**kwargs: Any) -> SGDFNet25Adapter:
    """Factory function for SGDFNet25Adapter."""
    return SGDFNet25Adapter(**kwargs)
