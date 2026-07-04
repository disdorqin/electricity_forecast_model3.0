"""
tests/test_fusion_weight_ledgers.py — Fusion and weight ledger contract tests.

Validates:
    1. append_fusion_to_ledger from P4 output produces correct schema
    2. append_fusion_to_ledger deduplicates on key
    3. validate_fusion_ledger passes valid ledger
    4. extract_weight_rows expands weights_json into per-model rows
    5. Weight rows sum to 1 per fusion row
    6. append_weights_to_ledger deduplicates on key
    7. validate_weight_ledger passes valid ledger
    8. Empty input handling
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from data.schema import (
    FUSION_LEDGER_COLUMNS,
    FUSION_LEDGER_KEY,
    WEIGHT_LEDGER_COLUMNS,
    WEIGHT_LEDGER_KEY,
    FUSION_OUTPUT_COLUMNS,
)
from ledgers.fusion_ledger import (
    append_fusion_to_ledger,
    validate_fusion_ledger,
)
from ledgers.weight_ledger import (
    extract_weight_rows,
    append_weights_to_ledger,
    validate_weight_ledger,
)
from fusion.engine import run_fusion, READY_DRY_RUN


def _fusion_df(n_hours: int = 24) -> pd.DataFrame:
    """Synthetic P4 fusion output."""
    from data.business_day import add_business_time_columns

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-07-04 01:00", periods=n_hours, freq="h")

    rows: list[dict] = []
    for i in range(n_hours):
        w1 = round(rng.uniform(0.3, 0.7), 4)
        w2 = round(1.0 - w1, 4)
        rows.append({
            "task": "dayahead",
            "target_day": "2026-07-04",
            "ds": timestamps[i],
            "fused_price": rng.uniform(100, 150),
            "weights_json": json.dumps({"cfg05": w1, "best_two_average": w2}),
            "included_models": "cfg05;best_two_average",
            "excluded_models": "",
            "fusion_method": "equal_weight",
            "learner_version": "0.1.0-skeleton",
            "readiness_mode": "DRY_RUN",
            "reason_codes": "FUSION_EQUAL_WEIGHT",
        })

    df = pd.DataFrame(rows)
    df = add_business_time_columns(df, timestamp_col="ds")
    # Ensure all FUSION_OUTPUT_COLUMNS are present
    for c in FUSION_OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[FUSION_OUTPUT_COLUMNS]


class TestFusionLedger:
    """Contract: fusion ledger."""

    def test_appends_fusion_output(self):
        """append_fusion_to_ledger produces correct schema."""
        fusion = _fusion_df(24)
        result = append_fusion_to_ledger(fusion, run_id="test_run")
        assert list(result.columns) == FUSION_LEDGER_COLUMNS
        assert len(result) == 24

    def test_dedup_on_key(self):
        """append_fusion_to_ledger deduplicates on key."""
        fusion = _fusion_df(24)
        ledger = append_fusion_to_ledger(fusion, run_id="run1")

        time.sleep(0.02)
        fusion2 = fusion.copy()
        fusion2["fused_price"] = fusion["fused_price"] * 0.9
        result = append_fusion_to_ledger(fusion2, ledger_df=ledger, run_id="run2")

        assert len(result) == 24
        assert (result["run_id"] == "run2").all()

    def test_validate_passes(self):
        """validate_fusion_ledger passes valid ledger."""
        fusion = _fusion_df(24)
        ledger = append_fusion_to_ledger(fusion, run_id="test")
        valid, issues = validate_fusion_ledger(ledger)
        assert valid

    def test_empty_input(self):
        """Empty fusion input returns empty ledger."""
        result = append_fusion_to_ledger(pd.DataFrame(), run_id="test")
        assert list(result.columns) == FUSION_LEDGER_COLUMNS
        assert len(result) == 0


class TestWeightLedger:
    """Contract: weight ledger."""

    def test_extract_weight_rows_produces_2n_rows(self):
        """extract_weight_rows produces 2 rows per fusion row (2 models)."""
        fusion = _fusion_df(24)
        weights = extract_weight_rows(fusion)
        assert len(weights) == 48  # 24 hours * 2 models

    def test_weight_rows_have_correct_schema(self):
        """Weight rows conform to WEIGHT_LEDGER_COLUMNS."""
        fusion = _fusion_df(24)
        weights = extract_weight_rows(fusion)
        assert list(weights.columns) == WEIGHT_LEDGER_COLUMNS

    def test_weights_sum_to_1_per_hour(self):
        """Weights sum to 1 for each fusion row."""
        fusion = _fusion_df(24)
        weights = extract_weight_rows(fusion)

        for hour in range(1, 25):
            hour_weights = weights[weights["hour_business"] == hour]
            total = hour_weights["weight"].sum()
            assert abs(total - 1.0) < 1e-4, f"Hour {hour} weights sum to {total}"

    def test_appends_weights(self):
        """append_weights_to_ledger produces correct schema."""
        fusion = _fusion_df(24)
        weights = extract_weight_rows(fusion)
        result = append_weights_to_ledger(weights, run_id="test_run")
        assert list(result.columns) == WEIGHT_LEDGER_COLUMNS
        assert len(result) == 48

    def test_dedup_on_key(self):
        """append_weights_to_ledger deduplicates on key."""
        fusion = _fusion_df(24)
        weights = extract_weight_rows(fusion)
        ledger = append_weights_to_ledger(weights, run_id="run1")

        time.sleep(0.02)
        # Same fusion rows produce same weight keys
        weights2 = extract_weight_rows(fusion)
        result = append_weights_to_ledger(weights2, ledger_df=ledger, run_id="run2")

        assert len(result) == 48
        assert (result["run_id"] == "run2").all()

    def test_validate_passes(self):
        """validate_weight_ledger passes valid ledger."""
        fusion = _fusion_df(24)
        weights = extract_weight_rows(fusion)
        ledger = append_weights_to_ledger(weights, run_id="test")
        valid, issues = validate_weight_ledger(ledger)
        assert valid

    def test_empty_fusion_input(self):
        """Empty fusion input produces empty weight rows."""
        empty = pd.DataFrame(columns=FUSION_OUTPUT_COLUMNS)
        weights = extract_weight_rows(empty)
        assert len(weights) == 0
        assert list(weights.columns) == WEIGHT_LEDGER_COLUMNS

    def test_invalid_json_handling(self):
        """Invalid weights_json is handled without crashing."""
        fusion = _fusion_df(1)
        fusion["weights_json"] = "not-valid-json"
        weights = extract_weight_rows(fusion)
        assert len(weights) == 0
