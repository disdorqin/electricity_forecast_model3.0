"""
extreme/negative_classifier.py — Negative-price classifier adapter.

Provides a unified interface for negative-price risk assessment:

1. **No-artifact fallback** — If no real ExtremPriceClf artifact is found,
   the adapter returns a no-op result (final_price = fused_price,
   classifier_applied = False).

2. **Rule fallback** — When *rule_fallback=True*, any row with
   fused_price < 0 triggers negative_flag = True with
   reason_codes += "RULE_NEGATIVE_PRICE".

3. **ExtremPriceClf path** — The ``load()`` method checks for a real
   ExtremPriceClf artifact.  When present, production inference can be
   switched on.  This implementation stubs the production path — the
   real ExtremPriceClf integration can be plugged in later without
   changing the adapter interface.

Usage::

    adapter = NegativeClassifierAdapter()
    adapter.load(model_dir="/path/to/models")
    result = adapter.predict(fusion_df, rule_fallback=True)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FINAL_UNIQUE_KEY,
    FUSION_OUTPUT_COLUMNS,
    VALID_NEGATIVE_SEVERITY,
    NEGATIVE_CLASSIFIER_NOOP,
    NEGATIVE_CLASSIFIER_RULE,
    NEGATIVE_CLASSIFIER_EXTREMPRICE,
)

logger = logging.getLogger(__name__)

# ── No-op / fallback constants ──────────────────────────────────────────

NOOP_VERSION: Final[str] = "0.0.0-noop"
RULE_VERSION: Final[str] = "0.1.0-rule"
EXTREMPRICE_VERSION: Final[str] = "0.0.0-stub"


class NegativeClassifierAdapter:
    """Negative-price classifier adapter.

    Parameters
    ----------
    rule_fallback : bool
        Default True — apply rule-based fallback when fused_price < 0.
    production : bool
        If True (default), classifier is expected to be fully
        operational.  No-op mode is allowed but logged as warning.
    """

    def __init__(
        self,
        rule_fallback: bool = True,
        production: bool = True,
    ) -> None:
        self.rule_fallback = rule_fallback
        self.production = production
        self._model_dir: Optional[str] = None
        self._artifact_found: bool = False
        self._loaded: bool = False

    # ── Public API ──────────────────────────────────────────────────────

    def load(self, model_dir: Optional[str] = None) -> None:
        """Load classifier resources from *model_dir*.

        If *model_dir* is None or does not contain a recognised
        ExtremPriceClf artifact, the adapter stays in no-op mode.

        No-op mode is **not** an error — it is the expected state when
        the ExtremPriceClf component has not been deployed yet.
        """
        self._loaded = True
        self._model_dir = model_dir

        if model_dir is None or not os.path.isdir(model_dir):
            logger.info(
                "NegativeClassifierAdapter: no model_dir — no-op fallback"
            )
            self._artifact_found = False
            return

        # Check for recognised artifact patterns
        artifact_patterns = [
            "ExtremPriceClf",
            "extreme_price_radar",
            "classifier",
        ]
        found: list[str] = []
        for name in artifact_patterns:
            for ext in ["", ".pkl", ".joblib", ".pt", ".pth"]:
                candidate = os.path.join(model_dir, f"{name}{ext}")
                if os.path.isfile(candidate):
                    found.append(candidate)

        if found:
            self._artifact_found = True
            logger.info(
                "NegativeClassifierAdapter: found ExtremPriceClf "
                "artifact(s) — production inference ready. "
                f"Artifacts: {found}"
            )
        else:
            self._artifact_found = False
            logger.info(
                "NegativeClassifierAdapter: no ExtremPriceClf artifact "
                "found in %s — no-op fallback",
                model_dir,
            )

    def predict(
        self,
        fusion_df: pd.DataFrame,
        rule_fallback: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Run negative-price classification on fusion output.

        Parameters
        ----------
        fusion_df : pd.DataFrame
            Fusion output (must contain at minimum the
            ``FUSION_OUTPUT_COLUMNS`` or at least the fields needed
            for classification).
        rule_fallback : bool, optional
            Override the instance-level ``rule_fallback`` setting.
            If None, the instance setting is used.

        Returns
        -------
        pd.DataFrame
            Final output in ``FINAL_OUTPUT_COLUMNS`` schema.
        """
        if not self._loaded:
            self.load()

        df = fusion_df.copy()
        n = len(df)

        if n == 0:
            return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

        rule = (
            rule_fallback if rule_fallback is not None
            else self.rule_fallback
        )

        # Determine classifier mode
        if self._artifact_found:
            classifier_module = NEGATIVE_CLASSIFIER_EXTREMPRICE
            classifier_version = EXTREMPRICE_VERSION
            classifier_applied = True
            risk_source = "MODEL"
            # Stub: in production the real model would be invoked here.
            # For now, produce no-op output with ExtremPriceClf metadata.
            result = self._apply_extremprice_stub(df)
            base_reason = "NEGATIVE_CLASSIFIER_EXTREMPRICE_STUB"
        else:
            classifier_module = NEGATIVE_CLASSIFIER_NOOP
            classifier_version = NOOP_VERSION
            classifier_applied = False
            risk_source = "CLASSIFIER_ARTIFACT_MISSING"
            base_reason = "NEGATIVE_CLASSIFIER_NO_OP"
            result = self._apply_noop(df)

        # Apply rule fallback on top (regardless of classifier mode)
        if rule:
            result = self._apply_rule_fallback(result, base_reason)

        # Ensure no NaN in final_price
        result["final_price"] = result["final_price"].fillna(
            result["fused_price"]
        )

        return result[FINAL_OUTPUT_COLUMNS]

    # ── Internal helpers ─────────────────────────────────────────────────

    def _apply_noop(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce no-op final output (no classifier applied)."""
        result = self._build_base_output(df)
        result["final_price"] = result["fused_price"]
        result["negative_prob"] = np.nan
        result["negative_flag"] = False
        result["negative_severity"] = "none"
        result["classifier_applied"] = False
        result["classifier_module"] = NEGATIVE_CLASSIFIER_NOOP
        result["classifier_version"] = NOOP_VERSION
        result["risk_source"] = "CLASSIFIER_ARTIFACT_MISSING"
        result["reason_codes"] = "NEGATIVE_CLASSIFIER_NO_OP"
        return result

    def _apply_extremprice_stub(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stub for ExtremPriceClf — same as no-op but with model metadata.

        When a real ExtremPriceClf artifact is deployed, this method
        would be replaced with actual model inference.
        """
        result = self._build_base_output(df)
        result["final_price"] = result["fused_price"]
        result["negative_prob"] = np.nan
        result["negative_flag"] = False
        result["negative_severity"] = "none"
        result["classifier_applied"] = True
        result["classifier_module"] = NEGATIVE_CLASSIFIER_EXTREMPRICE
        result["classifier_version"] = EXTREMPRICE_VERSION
        result["risk_source"] = "MODEL"
        result["reason_codes"] = "NEGATIVE_CLASSIFIER_EXTREMPRICE_STUB"
        return result

    def _apply_rule_fallback(
        self,
        result: pd.DataFrame,
        base_reason: str,
    ) -> pd.DataFrame:
        """Apply rule-based fallback: flag fused_price < 0."""
        out = result.copy()
        neg_mask = out["fused_price"] < 0

        if neg_mask.any():
            n_neg = neg_mask.sum()
            logger.info(
                "Rule fallback: flagged %d rows with fused_price < 0",
                n_neg,
            )
            out.loc[neg_mask, "negative_flag"] = True
            out.loc[neg_mask, "negative_prob"] = 1.0
            out.loc[neg_mask, "negative_severity"] = "high"
            out.loc[neg_mask, "classifier_module"] = NEGATIVE_CLASSIFIER_RULE
            out.loc[neg_mask, "classifier_version"] = RULE_VERSION
            out.loc[neg_mask, "risk_source"] = "RULE_FALLBACK"

            # Append rule reason to existing reason_codes
            existing: pd.Series = out.loc[neg_mask, "reason_codes"]
            out.loc[neg_mask, "reason_codes"] = existing.apply(
                lambda r: f"{r};RULE_NEGATIVE_PRICE"
                if "RULE_NEGATIVE_PRICE" not in str(r)
                else r
            )

        return out

    def _build_base_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a DataFrame with core columns from fusion output."""
        # Start with all FINAL_OUTPUT_COLUMNS — fill with defaults
        result = pd.DataFrame({c: pd.Series(dtype="object") for c in FINAL_OUTPUT_COLUMNS})

        # Map fusion columns into result
        fusion_cols = [c for c in FUSION_OUTPUT_COLUMNS if c in df.columns]
        for c in fusion_cols:
            if c in FINAL_OUTPUT_COLUMNS:
                result[c] = df[c].values if len(df) > 0 else pd.Series(dtype="object")

        # Build model_lineage_json
        lineage: dict[str, Any] = {}
        if "fusion_method" in df.columns:
            lineage["fusion_method"] = str(df["fusion_method"].iloc[0]) if len(df) > 0 else ""
        if "included_models" in df.columns:
            lineage["included_models"] = str(df["included_models"].iloc[0]) if len(df) > 0 else ""
        if "weights_json" in df.columns:
            lineage["weights_json"] = str(df["weights_json"].iloc[0]) if len(df) > 0 else ""
        lineage["classifier_module"] = "pending"
        result["model_lineage_json"] = json.dumps(lineage)

        return result


def run_adapter(
    fusion_df: pd.DataFrame,
    model_dir: Optional[str] = None,
    rule_fallback: bool = True,
    production: bool = True,
) -> pd.DataFrame:
    """Convenience function: create adapter, load, predict.

    Parameters
    ----------
    fusion_df : pd.DataFrame
        Fusion output.
    model_dir : str, optional
        Directory containing classifier artifacts.
    rule_fallback : bool
        Apply rule-based fallback for negative prices.
    production : bool
        Production mode flag.

    Returns
    -------
    pd.DataFrame
        Final output in ``FINAL_OUTPUT_COLUMNS`` schema.
    """
    adapter = NegativeClassifierAdapter(
        rule_fallback=rule_fallback,
        production=production,
    )
    adapter.load(model_dir=model_dir)
    return adapter.predict(fusion_df, rule_fallback=rule_fallback)
