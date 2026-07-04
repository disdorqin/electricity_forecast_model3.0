"""
fusion/learners/bgew.py — BGEW skeleton weight learner.

P4 only provides the skeleton / interface.  Real BGEW weight learning
requires the full 2.5 ledger chain migration and production actuals.

Current implementation delegates to ``fusion.weights.bgew_skeleton``
which computes rolling inverse-MAE weights from historical actuals.

Usage:
    from fusion.learners.bgew import BGEWLearner

    learner = BGEWLearner(window=30, min_history=7)
    weights = learner.compute_weights(
        model_names=["cfg05", "best_two_average"],
        corrected_df=corrected,
        actuals_df=actuals,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from fusion.weights import bgew_skeleton

logger = logging.getLogger(__name__)


class BGEWLearner:
    """BGEW skeleton weight learner.

    Parameters
    ----------
    window : int
        Rolling window size in business days (default 30).
    min_history : int
        Minimum business days of history required (default 7).
    version : str
        Learner version string (default ``"0.1.0-skeleton"``).
    """

    def __init__(
        self,
        window: int = 30,
        min_history: int = 7,
        version: str = "0.1.0-skeleton",
    ) -> None:
        self.window = window
        self.min_history = min_history
        self.version = version

    def compute_weights(
        self,
        model_names: list[str],
        corrected_df: pd.DataFrame,
        actuals_df: Optional[pd.DataFrame] = None,
    ) -> tuple[dict[str, float], list[str]]:
        """Compute BGEW weights for the given models.

        Delegates to ``fusion.weights.bgew_skeleton``.

        Returns
        -------
        tuple[dict[str, float], list[str]]
            ({model: weight, ...}, reason_codes).
        """
        return bgew_skeleton(
            model_names,
            corrected_df=corrected_df,
            actuals_df=actuals_df,
            window=self.window,
            min_history=self.min_history,
        )

    def get_version(self) -> str:
        return self.version
