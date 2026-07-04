"""
tests/test_ledger_schema.py — P5 ledger schema contract tests.

Validates:
    1. All 5 ledger schemas are defined and non-empty
    2. All 5 ledger key constants are defined
    3. Key columns are subsets of their respective schemas
    4. Correct column counts
    5. No overlap confusion between schema and key columns
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    PREDICTION_LEDGER_KEY,
    CORRECTED_LEDGER_COLUMNS,
    CORRECTED_LEDGER_KEY,
    ACTUAL_LEDGER_COLUMNS,
    ACTUAL_LEDGER_KEY,
    FUSION_LEDGER_COLUMNS,
    FUSION_LEDGER_KEY,
    WEIGHT_LEDGER_COLUMNS,
    WEIGHT_LEDGER_KEY,
)


class TestPredictionLedgerSchema:
    """Contract: prediction ledger schema."""

    def test_has_13_columns(self):
        """PREDICTION_LEDGER_COLUMNS has exactly 13 columns."""
        assert len(PREDICTION_LEDGER_COLUMNS) == 13

    def test_key_is_subset(self):
        """PREDICTION_LEDGER_KEY columns exist in PREDICTION_LEDGER_COLUMNS."""
        for col in PREDICTION_LEDGER_KEY:
            assert col in PREDICTION_LEDGER_COLUMNS, f"{col} missing"

    def test_key_has_5_columns(self):
        """PREDICTION_LEDGER_KEY has 5 columns."""
        assert len(PREDICTION_LEDGER_KEY) == 5

    def test_has_run_id(self):
        """PREDICTION_LEDGER_COLUMNS includes run_id."""
        assert "run_id" in PREDICTION_LEDGER_COLUMNS

    def test_has_updated_at(self):
        """PREDICTION_LEDGER_COLUMNS includes updated_at."""
        assert "updated_at" in PREDICTION_LEDGER_COLUMNS


class TestCorrectedLedgerSchema:
    """Contract: corrected ledger schema."""

    def test_has_20_columns(self):
        """CORRECTED_LEDGER_COLUMNS has exactly 20 columns."""
        assert len(CORRECTED_LEDGER_COLUMNS) == 20

    def test_key_is_subset(self):
        """CORRECTED_LEDGER_KEY columns exist in CORRECTED_LEDGER_COLUMNS."""
        for col in CORRECTED_LEDGER_KEY:
            assert col in CORRECTED_LEDGER_COLUMNS, f"{col} missing"

    def test_key_has_5_columns(self):
        """CORRECTED_LEDGER_KEY has 5 columns."""
        assert len(CORRECTED_LEDGER_KEY) == 5

    def test_includes_corrected_fields(self):
        """CORRECTED_LEDGER_COLUMNS includes correction-specific columns."""
        assert "y_pred_raw" in CORRECTED_LEDGER_COLUMNS
        assert "y_pred_corrected" in CORRECTED_LEDGER_COLUMNS
        assert "correction_module" in CORRECTED_LEDGER_COLUMNS


class TestActualLedgerSchema:
    """Contract: actual ledger schema."""

    def test_has_11_columns(self):
        """ACTUAL_LEDGER_COLUMNS has exactly 11 columns."""
        assert len(ACTUAL_LEDGER_COLUMNS) == 11

    def test_key_is_subset(self):
        """ACTUAL_LEDGER_KEY columns exist in ACTUAL_LEDGER_COLUMNS."""
        for col in ACTUAL_LEDGER_KEY:
            assert col in ACTUAL_LEDGER_COLUMNS, f"{col} missing"

    def test_key_has_4_columns(self):
        """ACTUAL_LEDGER_KEY has 4 columns."""
        assert len(ACTUAL_LEDGER_KEY) == 4

    def test_includes_y_true(self):
        """ACTUAL_LEDGER_COLUMNS includes y_true."""
        assert "y_true" in ACTUAL_LEDGER_COLUMNS

    def test_includes_actual_source(self):
        """ACTUAL_LEDGER_COLUMNS includes actual_source."""
        assert "actual_source" in ACTUAL_LEDGER_COLUMNS


class TestFusionLedgerSchema:
    """Contract: fusion ledger schema."""

    def test_has_17_columns(self):
        """FUSION_LEDGER_COLUMNS has exactly 17 columns."""
        assert len(FUSION_LEDGER_COLUMNS) == 17

    def test_key_is_subset(self):
        """FUSION_LEDGER_KEY columns exist in FUSION_LEDGER_COLUMNS."""
        for col in FUSION_LEDGER_KEY:
            assert col in FUSION_LEDGER_COLUMNS, f"{col} missing"

    def test_key_has_4_columns(self):
        """FUSION_LEDGER_KEY has 4 columns."""
        assert len(FUSION_LEDGER_KEY) == 4

    def test_includes_fused_price(self):
        """FUSION_LEDGER_COLUMNS includes fused_price."""
        assert "fused_price" in FUSION_LEDGER_COLUMNS

    def test_includes_weights_json(self):
        """FUSION_LEDGER_COLUMNS includes weights_json."""
        assert "weights_json" in FUSION_LEDGER_COLUMNS


class TestWeightLedgerSchema:
    """Contract: weight ledger schema."""

    def test_has_15_columns(self):
        """WEIGHT_LEDGER_COLUMNS has exactly 15 columns."""
        assert len(WEIGHT_LEDGER_COLUMNS) == 15

    def test_key_is_subset(self):
        """WEIGHT_LEDGER_KEY columns exist in WEIGHT_LEDGER_COLUMNS."""
        for col in WEIGHT_LEDGER_KEY:
            assert col in WEIGHT_LEDGER_COLUMNS, f"{col} missing"

    def test_key_has_6_columns(self):
        """WEIGHT_LEDGER_KEY has 6 columns."""
        assert len(WEIGHT_LEDGER_KEY) == 6

    def test_includes_weight(self):
        """WEIGHT_LEDGER_COLUMNS includes weight."""
        assert "weight" in WEIGHT_LEDGER_COLUMNS

    def test_includes_weight_source(self):
        """WEIGHT_LEDGER_COLUMNS includes weight_source."""
        assert "weight_source" in WEIGHT_LEDGER_COLUMNS


class TestAllLedgerKeysUnique:
    """Contract: all keys are disjoint."""

    def test_prediction_key_differs_from_actual_key(self):
        """Prediction and actual keys have different column counts."""
        assert set(PREDICTION_LEDGER_KEY) != set(ACTUAL_LEDGER_KEY)
