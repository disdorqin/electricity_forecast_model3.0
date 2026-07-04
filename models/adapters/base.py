"""
models/adapters/base.py — Abstract base contract for all prediction adapters.

Every prediction adapter in 3.0 must subclass ``BasePredictionAdapter``
and implement ``predict()``.

Contract rules:
    1. ``predict()`` returns a pandas DataFrame conforming to
       ``data.schema.PREDICTION_OUTPUT_COLUMNS``.
    2. Production predictions must NOT contain ``y_true``.
    3. ``predict()`` accepts an ``**kwargs`` catch-all for adapter-specific
       options (e.g. data_path, model_dir, date range).
    4. Adapters should fail fast with clear error messages if required
       resources (model weights, data files) are missing.
    5. Adapters must not silently return empty DataFrames when data is
       expected — raise or warn explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class BasePredictionAdapter(ABC):
    """Abstract base class for prediction adapters.

    Parameters
    ----------
    model_id : str
        Unique identifier for this model (matches registry).
    model_version : str, optional
        Version string (e.g. "1.0.0"). Defaults to "0.0.0".
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.model_version = model_version or "0.0.0"
        self._loaded: bool = False

    @property
    @abstractmethod
    def task(self) -> str:
        """Return the task type: 'dayahead' or 'realtime'."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model weights / resources. Called before predict().

        Implementations should set self._loaded = True on success,
        or raise a descriptive error on failure.
        """
        ...

    @abstractmethod
    def predict(
        self,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run prediction and return a standard-schema DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns matching
            ``data.schema.PREDICTION_OUTPUT_COLUMNS``.
            Must NOT contain ``y_true`` (production contract).

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called or failed.
        ValueError
            If input arguments are invalid.
        """
        ...

    def validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate prediction output against the standard schema.

        Parameters
        ----------
        df : pd.DataFrame
            Prediction output to validate.

        Returns
        -------
        pd.DataFrame
            The validated DataFrame (pass-through).

        Raises
        ------
        ValueError
            If required columns are missing or eval columns are present.
        """
        from data.schema import (
            PREDICTION_OUTPUT_COLUMNS,
            EVAL_ONLY_COLUMNS,
        )

        missing = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Adapter '{self.model_id}' output missing required columns: {missing}"
            )

        leaked = [c for c in EVAL_ONLY_COLUMNS if c in df.columns]
        if leaked:
            raise ValueError(
                f"Adapter '{self.model_id}' production output must NOT contain "
                f"eval-only columns: {leaked}"
            )

        # Check no NaN in y_pred
        nan_count = df["y_pred"].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"Adapter '{self.model_id}' output has {nan_count} NaN values in y_pred"
            )

        # Check hour_business range
        hb = df["hour_business"]
        if hb.min() < 1 or hb.max() > 24:
            raise ValueError(
                f"Adapter '{self.model_id}' output has hour_business outside [1,24]: "
                f"min={hb.min()}, max={hb.max()}"
            )

        return df
