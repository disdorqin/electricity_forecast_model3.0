"""
tests/test_final_output_validator.py — Final output validator contract tests.

Validates:
    1. Valid final output passes all checks
    2. Missing columns detected
    3. NaN final_price detected
    4. NaN fused_price detected
    5. negative_prob outside [0, 1] detected
    6. Duplicate final key detected
    7. model_lineage_json invalid JSON detected
    8. hour_business outside 1..24 detected
    9. Invalid period detected
    10. Empty DataFrame handling
    11. Production mode: y_true rejected
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.schema import (
    FINAL_OUTPUT_COLUMNS,
    FINAL_UNIQUE_KEY,
)
from scripts.validate_final_output import validate_final_dataframe


def _valid_final_df(n_hours: int = 24) -> pd.DataFrame:
    """Build a valid final output DataFrame."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")
    prices = rng.uniform(80, 200, n_hours)

    rows: list[dict] = []
    for i in range(n_hours):
        lineage = json.dumps({
            "fusion_method": "equal_weight",
            "included_models": "cfg05;best_two_average",
            "classifier_module": "negative_classifier_noop",
        })
        rows.append({
            "task": "dayahead",
            "target_day": "2026-07-04",
            "business_day": pd.Timestamp("2026-07-04"),
            "ds": timestamps[i],
            "hour_business": i + 1,
            "period": "1_8" if i < 8 else ("9_16" if i < 16 else "17_24"),
            "fused_price": float(prices[i]),
            "final_price": float(prices[i]),
            "negative_prob": 0.0,
            "negative_flag": False,
            "negative_severity": "none",
            "classifier_applied": False,
            "classifier_module": "negative_classifier_noop",
            "classifier_version": "0.0.0-noop",
            "risk_source": "CLASSIFIER_ARTIFACT_MISSING",
            "reason_codes": "NEGATIVE_CLASSIFIER_NO_OP",
            "model_lineage_json": lineage,
        })

    return pd.DataFrame(rows)


class TestFinalOutputValidator:
    """Contract: validate_final_dataframe."""

    def test_valid_passes(self):
        """Valid final output passes all checks."""
        df = _valid_final_df(24)
        valid, errors = validate_final_dataframe(df, production=True)
        assert valid, f"Validation failed: {errors}"

    def test_missing_columns_detected(self):
        """Missing column is detected."""
        df = _valid_final_df(24)
        bad = df.drop(columns=["final_price"])
        valid, errors = validate_final_dataframe(bad, production=True)
        assert not valid
        assert any("final_price" in e for e in errors)

    def test_nan_final_price_detected(self):
        """NaN in final_price is detected."""
        df = _valid_final_df(24)
        df.loc[0, "final_price"] = np.nan
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("NaN final_price" in e for e in errors)

    def test_nan_fused_price_detected(self):
        """NaN in fused_price is detected."""
        df = _valid_final_df(24)
        df.loc[0, "fused_price"] = np.nan
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("NaN fused_price" in e for e in errors)

    def test_negative_prob_out_of_range_detected(self):
        """negative_prob > 1 is detected."""
        df = _valid_final_df(24)
        df.loc[0, "negative_prob"] = 1.5
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("outside" in e.lower() for e in errors)

    def test_negative_prob_negative_detected(self):
        """negative_prob < 0 is detected."""
        df = _valid_final_df(24)
        df.loc[0, "negative_prob"] = -0.1
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("outside" in e.lower() for e in errors)

    def test_negative_prob_nan_allowed(self):
        """NaN negative_prob is allowed."""
        df = _valid_final_df(24)
        df["negative_prob"] = np.nan
        valid, errors = validate_final_dataframe(df, production=True)
        assert valid

    def test_duplicate_key_detected(self):
        """Duplicate final key is detected."""
        df = _valid_final_df(24)
        # Duplicate the first row
        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        valid, errors = validate_final_dataframe(dup, production=True)
        assert not valid
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_json_lineage_detected(self):
        """Invalid model_lineage_json is detected."""
        df = _valid_final_df(24)
        df.loc[0, "model_lineage_json"] = "not-valid-json"
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("JSON" in e for e in errors)

    def test_invalid_hour_detected(self):
        """hour_business outside 1..24 is detected."""
        df = _valid_final_df(24)
        df.loc[0, "hour_business"] = 99
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("hour_business" in e for e in errors)

    def test_invalid_period_detected(self):
        """Invalid period is detected."""
        df = _valid_final_df(24)
        df.loc[0, "period"] = "invalid"
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("period" in e for e in errors)

    def test_production_rejects_y_true(self):
        """Production mode rejects y_true."""
        df = _valid_final_df(24)
        df["y_true"] = 100.0
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("y_true" in e for e in errors)

    def test_empty_allowed(self):
        """Empty DataFrame is valid when allow_empty=True."""
        empty = pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)
        valid, errors = validate_final_dataframe(empty, allow_empty=True, production=True)
        assert valid

    def test_empty_not_allowed(self):
        """Empty DataFrame is invalid when allow_empty=False."""
        empty = pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)
        valid, errors = validate_final_dataframe(empty, allow_empty=False, production=True)
        assert not valid

    def test_invalid_task_detected(self):
        """Invalid task value is detected."""
        df = _valid_final_df(24)
        df.loc[0, "task"] = "invalid_task"
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
        assert any("task" in e for e in errors)

    def test_non_boolean_negative_flag_detected(self):
        """non-boolean negative_flag is detected."""
        df = _valid_final_df(24)
        df.loc[0, "negative_flag"] = "maybe"
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid

    def test_non_boolean_classifier_applied_detected(self):
        """non-boolean classifier_applied is detected."""
        df = _valid_final_df(24)
        df.loc[0, "classifier_applied"] = "yes"
        valid, errors = validate_final_dataframe(df, production=True)
        assert not valid
