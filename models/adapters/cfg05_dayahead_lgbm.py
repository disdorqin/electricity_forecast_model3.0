"""
models/adapters/cfg05_dayahead_lgbm.py — cfg05 day-ahead LightGBM champion adapter.

Wraps the frozen cfg05 day-ahead champion (LightGBM, 90d window, mae objective)
into the 3.0 adapter contract.

Source repository: disdorqin/epf-sota-experiment (main)

Usage:
    adapter = CFG05DayaheadAdapter()
    adapter.load(model_dir="path/to/weights")
    result = adapter.predict(data_path="path/to/data.csv", target_date="2026-03-01")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.adapters.base import BasePredictionAdapter

logger = logging.getLogger(__name__)

# cfg05 champion params (frozen)
CFG05_PARAMS = {
    "objective": "mae",
    "num_leaves": 191,
    "min_data_in_leaf": 30,
    "learning_rate": 0.015,
    "lambda_l1": 0.1,
    "lambda_l2": 5.0,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.95,
    "bagging_freq": 5,
    "n_estimators": 2000,
    "verbosity": -1,
}

# Feature columns expected by cfg05 (v3: 54 features)
# v2 base (40) + 2 extra lags (lag_48h, lag_168h) + 14 v3 features = 54
# v3 new: volatility, additional ranks, change, exact spring festival, interactions
CFG05_FEATURE_COLUMNS = [
    # ── Base (24) ──
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
    "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
    "morning_trend", "is_info_fresh",
    # ── Extended lags (5) ──
    "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
    # ── Same-hour stats (5) ──
    "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
    "same_hour_max_7d", "same_hour_min_7d",
    # ── Momentum + ranks (3) ──
    "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
    # ── Calendar (5) ──
    "is_spring_festival_window", "days_to_spring_festival",
    "days_after_spring_festival", "is_month_start", "is_month_end",
    # ── v3: Volatility (2) ──
    "price_volatility_24h", "price_volatility_168h",
    # ── v3: Additional ranks (2) ──
    "renewable_penetration_rank_30d", "load_ramp_rank_30d",
    # ── v3: Change features (3) ──
    "bidding_space_change_24h", "net_load_change_24h", "renewable_change_24h",
    # ── v3: Exact spring festival (3) ──
    "is_spring_festival_exact", "days_to_spring_festival_exact",
    "days_after_spring_festival_exact",
    # ── v3: Interaction features (4) ──
    "hour_x_bidding_space", "hour_x_net_load",
    "period_x_bidding_space", "period_x_renewable_penetration",
]


class CFG05DayaheadAdapter(BasePredictionAdapter):
    """Adapter for the frozen cfg05 day-ahead LightGBM champion.

    Parameters
    ----------
    model_version : str, optional
        Version string. Defaults to "1.0.0".
    """

    def __init__(self, model_version: Optional[str] = None) -> None:
        super().__init__(
            model_id="cfg05",
            model_version=model_version or "1.0.0",
        )
        self._model = None

    @property
    def task(self) -> str:
        return "dayahead"

    def load(self) -> None:
        """Load LightGBM model weights.

        LightGBM must be installed. Model weights are loaded from
        the path set via ``model_dir`` argument in ``predict()``.
        """
        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError:
            raise ImportError(
                "lightgbm is required for CFG05DayaheadAdapter. "
                "Install with: pip install lightgbm"
            )
        # Actual model loading happens in predict() when model_dir is given
        self._loaded = True
        logger.info(f"CFG05DayaheadAdapter ({self.model_version}): ready")

    def _load_model(self, model_dir: str) -> None:
        """Load the LightGBM booster from a directory.

        Parameters
        ----------
        model_dir : str
            Path to directory containing ``cfg05_model.txt`` or ``model.txt``.
        """
        import lightgbm as lgb

        model_path = Path(model_dir)
        candidates = [
            model_path / "cfg05_model.txt",
            model_path / "model.txt",
            model_path / "lightgbm_cfg05_dayahead.txt",
        ]
        for p in candidates:
            if p.exists():
                self._model = lgb.Booster(model_file=str(p))
                logger.info(f"Loaded model from {p}")
                return

        raise FileNotFoundError(
            f"No LightGBM model file found in {model_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from a raw data DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw hourly data with the columns needed by cfg05.

        Returns
        -------
        np.ndarray
            Feature matrix (n_samples, n_features).
        """
        available = [c for c in CFG05_FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in CFG05_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(
                f"Missing {len(missing)} feature columns for cfg05: {missing[:5]}..."
            )
        X = df[available].fillna(0).values
        return X

    def predict(  # type: ignore[override]
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_date: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run day-ahead prediction for a target date.

        Parameters
        ----------
        data_path : str, optional
            Path to CSV with raw hourly data. Required if ``df`` is None.
        df : pd.DataFrame, optional
            Pre-loaded raw hourly DataFrame. Required if ``data_path`` is None.
        target_date : str, optional
            Target day in YYYY-MM-DD format. If None, predicts all available days.
        model_dir : str, optional
            Path to model weight directory. Required on first call.

        Returns
        -------
        pd.DataFrame
            Standard schema prediction output.

        Raises
        ------
        ValueError
            If neither ``data_path`` nor ``df`` is provided.
        RuntimeError
            If model weights are not loaded.
        """
        if not self._loaded:
            self.load()

        if model_dir and self._model is None:
            self._load_model(model_dir)

        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Call predict() with model_dir= or call load_model() first."
            )

        # Load data
        if df is None:
            if data_path is None:
                raise ValueError("Either data_path= or df= must be provided")
            df = pd.read_csv(data_path, parse_dates=["ds"] if "ds" in open(data_path).read(1000) else [])
        else:
            df = df.copy()

        # Filter to target date (canonical 24-hour window: D+01:00 .. D+1 01:00)
        if target_date:
            from artifacts.dayahead_window import filter_dayahead
            df = filter_dayahead(df, target_date)

        if len(df) == 0:
            logger.warning(f"No data for target_date={target_date}, returning empty")
            return pd.DataFrame(columns=self._get_output_columns())

        # Ensure sorted by ds
        df = df.sort_values("ds").reset_index(drop=True)

        # Build features and predict
        X = self._build_features(df)
        y_pred = self._model.predict(X)
        df["y_pred"] = y_pred

        # Add business time columns
        from data.business_day import add_business_time_columns
        df = add_business_time_columns(df, timestamp_col="ds")

        # Build output
        out = pd.DataFrame({
            "task": "dayahead",
            "model_name": "lightgbm_cfg05_dayahead",
            "target_day": target_date if target_date else pd.to_datetime(df["ds"]).dt.date.astype(str),
            "business_day": df["business_day"],
            "ds": pd.to_datetime(df["ds"]),
            "hour_business": df["hour_business"],
            "period": df["period"],
            "y_pred": df["y_pred"],
            "source_confidence": np.nan,
            "model_version": self.model_version,
        })

        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


def create_adapter(**kwargs: Any) -> CFG05DayaheadAdapter:
    """Factory function for CFG05DayaheadAdapter.

    Usage:
        adapter = create_adapter()
        adapter.load()
        result = adapter.predict(data_path="...", target_date="...", model_dir="...")
    """
    return CFG05DayaheadAdapter(**kwargs)
