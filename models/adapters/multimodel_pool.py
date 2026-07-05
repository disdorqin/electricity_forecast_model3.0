"""
models/adapters/multimodel_pool.py — Multi-model adapter pool for P31-P40.

Wraps source-repo models into the 3.0 BasePredictionAdapter contract:

    best_two_average
    stage3_business_fixed
    catboost_sota
    catboost_spike_residual

Each adapter:
    - Can train from source repo
    - Save/load model artifacts
    - Predict with canonical 24-hour day-ahead window
    - Validate output schema
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

logger = logging.getLogger(__name__)

# ── Canonical model IDs ──
MODEL_BEST_TWO_AVERAGE = "best_two_average"
MODEL_STAGE3_FIXED = "stage3_business_fixed"
MODEL_CATBOOST_SOTA = "catboost_sota"
MODEL_CATBOOST_SPIKE_RESIDUAL = "catboost_spike_residual"

ALL_CANDIDATE_MODELS = [
    MODEL_BEST_TWO_AVERAGE,
    MODEL_STAGE3_FIXED,
    MODEL_CATBOOST_SOTA,
    MODEL_CATBOOST_SPIKE_RESIDUAL,
]

BANNED_MODELS = [
    "lgbm_spike_residual_1127",
    "stage3_old_1164",
    "lightgbm_90d_orig_1197",
]

# Readiness labels
REAL_24H_READY = "REAL_24H_READY"
TRAINED_BUT_NOT_24H = "TRAINED_BUT_NOT_24H"
DEP_MISSING = "DEP_MISSING"
SOURCE_SCRIPT_MISSING = "SOURCE_SCRIPT_MISSING"
MODEL_TRAIN_FAILED = "MODEL_TRAIN_FAILED"
INVALID_BANNED = "INVALID_BANNED"


# ── Source module loader ──


def _import_source_module(source_repo: str, module_name: str):
    """Import a module from the source repo by name."""
    import importlib as _il

    full_path = os.path.join(source_repo, *module_name.split(".")) + ".py"
    if not os.path.isfile(full_path):
        raise ImportError(f"Source module not found: {full_path}")

    if source_repo not in sys.path:
        sys.path.insert(0, source_repo)

    return _il.import_module(module_name)


# ══════════════════════════════════════════════════════════════════════════════
#  Model adapter implementations
# ══════════════════════════════════════════════════════════════════════════════


class BestTwoAverageAdapter(BasePredictionAdapter):
    """best_two_average: Simple average of two LightGBM trials.

    Trains two LightGBM models (trial_02 config and trial_24 config)
    and averages their predictions.
    """

    def __init__(self, model_version: Optional[str] = None) -> None:
        super().__init__(
            model_id=MODEL_BEST_TWO_AVERAGE,
            model_version=model_version or "1.0.0",
        )
        self._model_1 = None
        self._model_2 = None
        self._feature_cols: list[str] = []
        self._trained = False

    @property
    def task(self) -> str:
        return "dayahead"

    def _get_feature_cols(self) -> list[str]:
        """Get the feature columns from the source adapter."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        return list(CFG05_FEATURE_COLUMNS)

    def train(
        self,
        source_repo: str,
        df_feat: pd.DataFrame,
        target_day: str,
        train_window_days: int = 90,
    ) -> dict[str, Any]:
        """Train two LightGBM models with different configs.

        Parameters
        ----------
        source_repo : str
            Path to epf-sota-experiment.
        df_feat : pd.DataFrame
            Full feature DataFrame with 'ds', 'y', and feature columns.
        target_day : str
            Target day (YYYY-MM-DD).
        train_window_days : int
            Training window in days (default 90).

        Returns
        -------
        dict with training manifest.
        """
        import lightgbm as lgb
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

        self._feature_cols = list(CFG05_FEATURE_COLUMNS)
        target_dt = pd.Timestamp(target_day)
        train_start = target_dt - pd.Timedelta(days=train_window_days)
        train_end = target_dt - pd.Timedelta(hours=1)

        mask = (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
        train_df = df_feat[mask].copy()

        X_train = train_df[self._feature_cols].fillna(0).values
        y_train = train_df["y"].values

        # Config 1: "trial_02" style — mae objective, moderate leaves
        params_1 = {
            "objective": "mae",
            "num_leaves": 127,
            "min_data_in_leaf": 20,
            "learning_rate": 0.02,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.9,
            "bagging_freq": 3,
            "verbosity": -1,
        }
        # Config 2: "trial_24" style — rmse objective, more leaves
        params_2 = {
            "objective": "rmse",
            "num_leaves": 191,
            "min_data_in_leaf": 30,
            "learning_rate": 0.015,
            "lambda_l1": 0.1,
            "lambda_l2": 5.0,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbosity": -1,
        }

        manifest = {"configs": {}, "feature_cols": self._feature_cols}

        for cfg_name, params in [("trial_02", params_1), ("trial_24", params_2)]:
            booster = lgb.train(
                params,
                lgb.Dataset(X_train, y_train),
                num_boost_round=2000,
                callbacks=[lgb.log_evaluation(0)],
            )
            if cfg_name == "trial_02":
                self._model_1 = booster
            else:
                self._model_2 = booster

            manifest["configs"][cfg_name] = {
                "params": params,
                "best_iteration": booster.best_iteration,
                "n_train": len(X_train),
            }

        self._trained = True
        manifest["train_rows"] = len(X_train)
        manifest["target_day"] = target_day
        manifest["train_window_days"] = train_window_days
        return manifest

    def load(self) -> None:
        """Load saved .txt model files (two constituent models)."""
        import lightgbm as lgb

        self._loaded = True

    def _load_artifacts(self, model_dir: str) -> None:
        """Load two constituent LightGBM boosters from a directory."""
        import lightgbm as lgb

        model_path = Path(model_dir)
        p1 = model_path / "best_two_average_trial_02.txt"
        p2 = model_path / "best_two_average_trial_24.txt"

        if not p1.exists() or not p2.exists():
            raise FileNotFoundError(
                f"best_two_average artifacts not found in {model_dir}. "
                f"Need: {p1.name}, {p2.name}"
            )

        self._model_1 = lgb.Booster(model_file=str(p1))
        self._model_2 = lgb.Booster(model_file=str(p2))
        self._trained = True
        logger.info(f"best_two_average: loaded trial_02 and trial_24 from {model_dir}")

    def save_artifacts(self, model_dir: str) -> None:
        """Save two constituent model files."""
        if not self._trained:
            raise RuntimeError("No trained model to save.")
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        self._model_1.save_model(str(model_path / "best_two_average_trial_02.txt"))
        self._model_2.save_model(str(model_path / "best_two_average_trial_24.txt"))
        logger.info(f"best_two_average: saved artifacts to {model_dir}")

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix."""
        if not self._feature_cols:
            from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
            self._feature_cols = list(CFG05_FEATURE_COLUMNS)

        available = [c for c in self._feature_cols if c in df.columns]
        return df[available].fillna(0).values

    def predict(
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_date: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run day-ahead prediction (average of two LightGBM models)."""
        if not self._loaded:
            self.load()

        if model_dir and not self._trained:
            self._load_artifacts(model_dir)

        if not self._trained or self._model_1 is None or self._model_2 is None:
            raise RuntimeError(
                "best_two_average not trained. Call train() or load artifacts first."
            )

        # Load or copy data
        if df is None:
            if data_path is None:
                raise ValueError("Either data_path= or df= must be provided")
            df_in = pd.read_csv(data_path)
        else:
            df_in = df.copy()

        # Parse ds
        if "ds" in df_in.columns:
            df_in["ds"] = pd.to_datetime(df_in["ds"])

        # Filter to target day
        if target_date:
            from artifacts.dayahead_window import filter_dayahead
            df_day = filter_dayahead(df_in, target_date)
        else:
            df_day = df_in.sort_values("ds").reset_index(drop=True)

        if len(df_day) == 0:
            logger.warning(f"No data for target_date={target_date}")
            return pd.DataFrame(columns=self._get_output_columns())

        # Build features and predict with both models
        X = self._build_features(df_day)
        y_pred_1 = self._model_1.predict(X)
        y_pred_2 = self._model_2.predict(X)
        y_pred = (y_pred_1 + y_pred_2) / 2.0

        df_day["y_pred"] = y_pred

        # Build output
        from data.business_day import add_business_time_columns
        df_day = add_business_time_columns(df_day, timestamp_col="ds")

        from data.schema import PREDICTION_OUTPUT_COLUMNS

        out = pd.DataFrame({
            "task": "dayahead",
            "model_name": "best_two_average",
            "target_day": target_date or "",
            "business_day": df_day["business_day"],
            "ds": pd.to_datetime(df_day["ds"]),
            "hour_business": df_day["hour_business"],
            "period": df_day["period"],
            "y_pred": df_day["y_pred"],
            "source_confidence": np.nan,
            "model_version": self.model_version,
        })
        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


# ══════════════════════════════════════════════════════════════════════════════


class Stage3BusinessFixedAdapter(BasePredictionAdapter):
    """stage3_business_fixed: LightGBM Stage-3 with corrected business_day mapping.

    Uses the source repo's LightGBMDayaheadAdapter with stage3-optimized params.
    """

    def __init__(self, model_version: Optional[str] = None) -> None:
        super().__init__(
            model_id=MODEL_STAGE3_FIXED,
            model_version=model_version or "1.0.0",
        )
        self._lgb_adapter = None
        self._trained = False
        self._feature_cols: list[str] = []

    @property
    def task(self) -> str:
        return "dayahead"

    def train(
        self,
        source_repo: str,
        df_feat: pd.DataFrame,
        target_day: str,
        train_window_days: int = 90,
    ) -> dict[str, Any]:
        """Train Stage-3 style LightGBM model.

        Parameters
        ----------
        source_repo : str
            Path to epf-sota-experiment.
        df_feat : pd.DataFrame
            Full feature DataFrame.
        target_day : str
            Target day (YYYY-MM-DD).
        train_window_days : int
            Training window in days.

        Returns
        -------
        dict with training manifest.
        """
        # Use the source repo's LightGBMDayaheadAdapter
        try:
            mod = _import_source_module(
                source_repo, "src.models.lightgbm_dayahead_adapter"
            )
        except Exception as e:
            raise ImportError(f"Failed to import source LightGBM adapter: {e}")

        # Stage-3 optimized params
        stage3_params = {
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "min_data_in_leaf": 30,
            "objective": "rmse",
            "metric": "rmse",
            "verbosity": -1,
        }

        adapter = mod.LightGBMDayaheadAdapter(
            config_name="stage3_fixed",
            model_params=stage3_params,
        )
        self._lgb_adapter = adapter
        self._feature_cols = list(adapter.feature_cols)

        target_dt = pd.Timestamp(target_day)
        train_start = target_dt - pd.Timedelta(days=train_window_days)
        train_end = target_dt - pd.Timedelta(hours=1)

        mask = (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
        train_df = df_feat[mask].copy()

        manifest = adapter.train(train_df)
        self._trained = True
        manifest["target_day"] = target_day
        manifest["train_window_days"] = train_window_days
        return manifest

    def load(self) -> None:
        """Load adapter."""
        self._loaded = True

    def _load_artifacts(self, model_dir: str) -> None:
        """Load saved model from directory."""
        import lightgbm as lgb

        model_path = Path(model_dir)
        candidates = [
            model_path / "stage3_model.txt",
            model_path / "model.txt",
        ]
        for p in candidates:
            if p.exists():
                # Recreate source adapter — try multiple source repo paths
                source_candidates = [
                    os.path.join(".local_artifacts", "source_repos", "epf-sota-experiment"),
                    str(Path(model_dir).parent.parent / "source_repos" / "epf-sota-experiment"),
                ]
                mod = None
                for src in source_candidates:
                    try:
                        mod = _import_source_module(
                            src, "src.models.lightgbm_dayahead_adapter",
                        )
                        break
                    except Exception:
                        continue

                booster = lgb.Booster(model_file=str(p))
                # Handle both property and method forms of feature_name
                try:
                    booster_raw_names = list(booster.feature_name)
                except TypeError:
                    booster_raw_names = list(booster.feature_name())

                # If booster only has generic names (Column_0..N), use real columns
                if booster_raw_names and booster_raw_names[0].startswith("Column_"):
                    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
                    # Use intersection of CFG05 columns and source FEATURE_COLS (42)
                    booster_feature_names = [
                        "hour", "month", "day_of_week", "is_weekend",
                        "lag_price_target", "lag_price_week",
                        "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
                        "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
                        "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
                        "morning_trend", "is_info_fresh",
                        "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
                        "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
                        "same_hour_max_7d", "same_hour_min_7d",
                        "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
                        "is_spring_festival_window", "days_to_spring_festival",
                        "days_after_spring_festival", "is_month_start", "is_month_end",
                    ]
                else:
                    booster_feature_names = booster_raw_names

                if mod is None:
                    # Direct LightGBM load — track feature names from booster
                    self._lgb_adapter = _LGBMWrapper(booster, feature_names=booster_feature_names)
                else:
                    adapter = mod.LightGBMDayaheadAdapter(config_name="stage3_fixed")
                    adapter.model = booster
                    self._lgb_adapter = adapter

                self._feature_cols = booster_feature_names
                self._trained = True
                logger.info(
                    "stage3_fixed: loaded model from %s (%d features)",
                    p, len(booster_feature_names),
                )
                return

        raise FileNotFoundError(
            f"No stage3 model file found in {model_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def save_artifacts(self, model_dir: str) -> None:
        """Save model file."""
        if not self._trained or self._lgb_adapter is None:
            raise RuntimeError("No trained model to save.")
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self._lgb_adapter, "model") and self._lgb_adapter.model is not None:
            self._lgb_adapter.model.save_model(str(model_path / "stage3_model.txt"))
        else:
            raise RuntimeError("No booster available in adapter to save.")
        logger.info(f"stage3_fixed: saved model to {model_path}")

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix using stored feature columns."""
        if self._lgb_adapter is None:
            raise RuntimeError("Model not loaded.")

        if self._feature_cols:
            available = [c for c in self._feature_cols if c in df.columns]
            return df[available].fillna(0).values

        return self._lgb_adapter._prepare_X(df)

    def predict(
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_date: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run day-ahead prediction for stage3 model."""
        if not self._loaded:
            self.load()

        if model_dir and not self._trained:
            self._load_artifacts(model_dir)

        if not self._trained or self._lgb_adapter is None:
            raise RuntimeError("Model not trained. Call train() or load artifacts.")

        # Load data
        if df is None:
            if data_path is None:
                raise ValueError("Either data_path= or df= must be provided")
            df_in = pd.read_csv(data_path)
        else:
            df_in = df.copy()

        if "ds" in df_in.columns:
            df_in["ds"] = pd.to_datetime(df_in["ds"])

        # Filter to target day
        if target_date:
            from artifacts.dayahead_window import filter_dayahead
            df_day = filter_dayahead(df_in, target_date)
        else:
            df_day = df_in.sort_values("ds").reset_index(drop=True)

        if len(df_day) == 0:
            return pd.DataFrame(columns=self._get_output_columns())

        # Predict
        X = self._build_features(df_day)
        y_pred = self._lgb_adapter.model.predict(X)
        df_day["y_pred"] = y_pred

        # Build output
        from data.business_day import add_business_time_columns
        df_day = add_business_time_columns(df_day, timestamp_col="ds")

        out = pd.DataFrame({
            "task": "dayahead",
            "model_name": "stage3_business_fixed",
            "target_day": target_date or "",
            "business_day": df_day["business_day"],
            "ds": pd.to_datetime(df_day["ds"]),
            "hour_business": df_day["hour_business"],
            "period": df_day["period"],
            "y_pred": df_day["y_pred"],
            "source_confidence": np.nan,
            "model_version": self.model_version,
        })
        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


# ══════════════════════════════════════════════════════════════════════════════


class _LGBMWrapper:
    """Minimal wrapper for a LightGBM booster when source adapter unavailable."""
    def __init__(self, booster, feature_names=None):
        self.model = booster
        self._feature_names = feature_names or list(booster.feature_name)

    def _prepare_X(self, df):
        available = [c for c in self._feature_names if c in df.columns]
        return df[available].fillna(0).values


# ══════════════════════════════════════════════════════════════════════════════


class CatBoostSotaAdapter(BasePredictionAdapter):
    """catboost_sota: CatBoost SOTA baseline adapter.

    Uses the source repo's CatBoostAdapter class for training/inference.
    """

    def __init__(self, model_version: Optional[str] = None) -> None:
        super().__init__(
            model_id=MODEL_CATBOOST_SOTA,
            model_version=model_version or "1.0.0",
        )
        self._cb_adapter = None
        self._trained = False
        self._catboost_available = False

    @property
    def task(self) -> str:
        return "dayahead"

    def _check_catboost(self) -> bool:
        try:
            import catboost  # noqa: F401
            self._catboost_available = True
            return True
        except ImportError:
            self._catboost_available = False
            return False

    def train(
        self,
        source_repo: str,
        df_feat: pd.DataFrame,
        target_day: str,
        train_window_days: int = 90,
    ) -> dict[str, Any]:
        """Train CatBoost SOTA model.

        Parameters
        ----------
        source_repo : str
            Path to epf-sota-experiment.
        df_feat : pd.DataFrame
            Full feature DataFrame with 'ds', 'y', and feature columns.
        target_day : str
            Target day (YYYY-MM-DD).
        train_window_days : int
            Training window in days.

        Returns
        -------
        dict with training manifest.
        """
        if not self._check_catboost():
            raise ImportError("catboost is not installed")

        try:
            mod = _import_source_module(
                source_repo, "src.models.catboost_adapter"
            )
        except Exception as e:
            raise ImportError(f"Failed to import source CatBoost adapter: {e}")

        # Use the catboost_adapter.CatBoostAdapter for training
        adapter = mod.CatBoostAdapter(
            model_name="catboost_sota",
            task_type="CPU",
        )
        self._cb_adapter = adapter

        target_dt = pd.Timestamp(target_day)
        train_start = target_dt - pd.Timedelta(days=train_window_days)
        train_end = target_dt - pd.Timedelta(hours=1)

        mask = (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
        train_df = df_feat[mask].copy()

        manifest = adapter.train(train_df)
        self._trained = True
        manifest["target_day"] = target_day
        manifest["train_window_days"] = train_window_days
        return manifest

    def load(self) -> None:
        self._check_catboost()
        self._loaded = True

    def _load_artifacts(self, model_dir: str) -> None:
        """Load .cbm model file."""
        if not self._check_catboost():
            raise ImportError("catboost is not installed")
        import catboost

        model_path = Path(model_dir)
        candidates = [
            model_path / "catboost_sota_model.cbm",
            model_path / "model.cbm",
        ]
        for p in candidates:
            if p.exists():
                model = catboost.CatBoostRegressor()
                model.load_model(str(p))
                self._cb_adapter = _CatBoostWrapper(model)
                self._trained = True
                logger.info(f"catboost_sota: loaded model from {p}")
                return
        raise FileNotFoundError(
            f"No CatBoost model found in {model_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def save_artifacts(self, model_dir: str) -> None:
        """Save .cbm model file."""
        if not self._trained or self._cb_adapter is None:
            raise RuntimeError("No trained model to save.")
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self._cb_adapter, "save_model"):
            self._cb_adapter.save_model(str(model_path / "catboost_sota_model.cbm"))
        else:
            logger.warning("_cb_adapter has no save_model method, trying raw model")
            if hasattr(self._cb_adapter, "_model"):
                self._cb_adapter._model.save_model(str(model_path / "catboost_sota_model.cbm"))
            else:
                raise RuntimeError("Cannot save: no save mechanism on adapter")

        logger.info(f"catboost_sota: saved model to {model_path}")

    def predict(
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_date: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._loaded:
            self.load()

        if model_dir and not self._trained:
            self._load_artifacts(model_dir)

        if not self._trained or self._cb_adapter is None:
            raise RuntimeError("Model not trained. Call train() or load artifacts.")

        if not self._catboost_available:
            raise RuntimeError("catboost not available")

        # Load data
        if df is None:
            if data_path is None:
                raise ValueError("data_path= or df= required")
            df_in = pd.read_csv(data_path)
        else:
            df_in = df.copy()

        if "ds" in df_in.columns:
            df_in["ds"] = pd.to_datetime(df_in["ds"])

        # Filter to target day
        if target_date:
            from artifacts.dayahead_window import filter_dayahead
            df_day = filter_dayahead(df_in, target_date)
        else:
            df_day = df_in.sort_values("ds").reset_index(drop=True)

        if len(df_day) == 0:
            return pd.DataFrame(columns=self._get_output_columns())

        # Predict
        X = self._cb_adapter._prepare_X(df_day)
        y_pred = self._cb_adapter.predict(df_day)
        df_day["y_pred"] = y_pred

        # Build output
        from data.business_day import add_business_time_columns
        df_day = add_business_time_columns(df_day, timestamp_col="ds")

        out = pd.DataFrame({
            "task": "dayahead",
            "model_name": "catboost_sota",
            "target_day": target_date or "",
            "business_day": df_day["business_day"],
            "ds": pd.to_datetime(df_day["ds"]),
            "hour_business": df_day["hour_business"],
            "period": df_day["period"],
            "y_pred": df_day["y_pred"],
            "source_confidence": np.nan,
            "model_version": self.model_version,
        })
        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


class _CatBoostWrapper:
    """Minimal wrapper for a CatBoost model when source adapter unavailable."""
    def __init__(self, model):
        self._model = model

    def _prepare_X(self, df):
        """Return a DataFrame (not numpy) so CatBoost preserves categorical info."""
        if hasattr(self._model, "feature_names_") and self._model.feature_names_:
            feature_cols = list(self._model.feature_names_)
        else:
            # Fallback: use source CatBoost feature columns
            feature_cols = [
                "hour", "month", "day_of_week", "is_weekend",
                "lag_price_target", "lag_price_week",
                "load", "wind", "solar", "interconnect",
                "bidding_space", "space_ratio",
                "net_load", "solar_ratio", "net_load_sq",
                "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
                "morning_mean", "noon_min", "morning_std",
                "morning_trend", "is_info_fresh",
            ]
        available = [c for c in feature_cols if c in df.columns]
        X = df[available].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        # Convert categorical columns to string for CatBoost recognition
        for col in ["hour", "month", "day_of_week", "is_weekend"]:
            if col in X.columns:
                X[col] = X[col].astype(str)
        return X

    def predict(self, df):
        X = self._prepare_X(df)
        return self._model.predict(X)


# ══════════════════════════════════════════════════════════════════════════════


class CatBoostSpikeResidualAdapter(BasePredictionAdapter):
    """catboost_spike_residual: CatBoost with spike residual correction.

    Similar to catboost_sota but also trains a residual corrector for spike hours.
    For the initial implementation, this uses the same CatBoostAdapter but with
    different params tuned for spike robustness.
    """

    def __init__(self, model_version: Optional[str] = None) -> None:
        super().__init__(
            model_id=MODEL_CATBOOST_SPIKE_RESIDUAL,
            model_version=model_version or "1.0.0",
        )
        self._cb_adapter = None
        self._trained = False
        self._catboost_available = False

    @property
    def task(self) -> str:
        return "dayahead"

    def _check_catboost(self) -> bool:
        try:
            import catboost  # noqa: F401
            self._catboost_available = True
            return True
        except ImportError:
            self._catboost_available = False
            return False

    def train(
        self,
        source_repo: str,
        df_feat: pd.DataFrame,
        target_day: str,
        train_window_days: int = 90,
    ) -> dict[str, Any]:
        if not self._check_catboost():
            raise ImportError("catboost is not installed")

        import catboost

        target_dt = pd.Timestamp(target_day)
        train_start = target_dt - pd.Timedelta(days=train_window_days)
        train_end = target_dt - pd.Timedelta(hours=1)

        mask = (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
        train_df = df_feat[mask].copy()

        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

        feature_cols = [c for c in CFG05_FEATURE_COLUMNS if c in train_df.columns]
        X_train = train_df[feature_cols].copy()
        # Convert categorical
        for col in ["hour", "month", "day_of_week", "is_weekend"]:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(str)
        for col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)

        y_train = train_df["y"].values

        # Spike-residual tuned params (higher regularization, depth-limited)
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1500,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 10.0,
            "random_seed": 42,
            "od_type": "Iter",
            "od_wait": 100,
            "verbose": False,
            "allow_writing_files": False,
            "task_type": "CPU",
        }

        cat_indices = [
            i for i, c in enumerate(feature_cols)
            if c in ["hour", "month", "day_of_week", "is_weekend"]
        ]

        model = catboost.CatBoostRegressor(**params, cat_features=cat_indices)
        model.fit(X_train, y_train, plot=False)

        self._cb_adapter = _CatBoostSkateWrapper(model, feature_cols)
        self._trained = True

        return {
            "model_name": "catboost_spike_residual",
            "params": params,
            "train_rows": len(X_train),
            "feature_cols": feature_cols,
            "target_day": target_day,
            "train_window_days": train_window_days,
        }

    def load(self) -> None:
        self._check_catboost()
        self._loaded = True

    def save_artifacts(self, model_dir: str) -> None:
        if not self._trained or self._cb_adapter is None:
            raise RuntimeError("No trained model to save.")
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        self._cb_adapter._model.save_model(str(model_path / "catboost_spike_residual.cbm"))
        logger.info(f"catboost_spike_residual: saved to {model_path}")

    def _load_artifacts(self, model_dir: str) -> None:
        if not self._check_catboost():
            raise ImportError("catboost not installed")
        import catboost

        model_path = Path(model_dir)
        p = model_path / "catboost_spike_residual.cbm"
        if not p.exists():
            raise FileNotFoundError(f"catboost_spike_residual model not found: {p}")
        model = catboost.CatBoostRegressor()
        model.load_model(str(p))

        feature_cols = list(model.feature_names_)
        self._cb_adapter = _CatBoostSkateWrapper(model, feature_cols)
        self._trained = True
        logger.info(f"catboost_spike_residual: loaded from {p}")

    def predict(
        self,
        data_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        target_date: Optional[str] = None,
        model_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._loaded:
            self.load()

        if model_dir and not self._trained:
            self._load_artifacts(model_dir)

        if not self._trained or self._cb_adapter is None:
            raise RuntimeError("Model not trained.")

        if df is None:
            if data_path is None:
                raise ValueError("data_path= or df= required")
            df_in = pd.read_csv(data_path)
        else:
            df_in = df.copy()

        if "ds" in df_in.columns:
            df_in["ds"] = pd.to_datetime(df_in["ds"])

        if target_date:
            from artifacts.dayahead_window import filter_dayahead
            df_day = filter_dayahead(df_in, target_date)
        else:
            df_day = df_in.sort_values("ds").reset_index(drop=True)

        if len(df_day) == 0:
            return pd.DataFrame(columns=self._get_output_columns())

        y_pred = self._cb_adapter.predict(df_day)
        df_day["y_pred"] = y_pred

        from data.business_day import add_business_time_columns
        df_day = add_business_time_columns(df_day, timestamp_col="ds")

        out = pd.DataFrame({
            "task": "dayahead",
            "model_name": "catboost_spike_residual",
            "target_day": target_date or "",
            "business_day": df_day["business_day"],
            "ds": pd.to_datetime(df_day["ds"]),
            "hour_business": df_day["hour_business"],
            "period": df_day["period"],
            "y_pred": df_day["y_pred"],
            "source_confidence": np.nan,
            "model_version": self.model_version,
        })
        out = out.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
        return self.validate_output(out)

    def _get_output_columns(self) -> list[str]:
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        return list(PREDICTION_OUTPUT_COLUMNS)


class _CatBoostSkateWrapper:
    """Wrapper for a CatBoost model with feature column tracking."""
    def __init__(self, model, feature_cols):
        self._model = model
        self._feature_cols = feature_cols

    def predict(self, df):
        available = [c for c in self._feature_cols if c in df.columns]
        X = df[available].copy()
        for col in ["hour", "month", "day_of_week", "is_weekend"]:
            if col in X.columns:
                X[col] = X[col].astype(str)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        return self._model.predict(X)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════


def create_adapter(model_id: str, model_version: Optional[str] = None):
    """Factory: create an adapter for the given model_id.

    Parameters
    ----------
    model_id : str
        One of ALL_CANDIDATE_MODELS.
    model_version : str, optional

    Returns
    -------
    BasePredictionAdapter

    Raises
    ------
    ValueError
        If model_id is unknown.
    """
    mapping = {
        MODEL_BEST_TWO_AVERAGE: BestTwoAverageAdapter,
        MODEL_STAGE3_FIXED: Stage3BusinessFixedAdapter,
        MODEL_CATBOOST_SOTA: CatBoostSotaAdapter,
        MODEL_CATBOOST_SPIKE_RESIDUAL: CatBoostSpikeResidualAdapter,
    }
    if model_id not in mapping:
        raise ValueError(
            f"Unknown model_id={model_id}. "
            f"Known: {list(mapping.keys())}"
        )
    return mapping[model_id](model_version=model_version)
