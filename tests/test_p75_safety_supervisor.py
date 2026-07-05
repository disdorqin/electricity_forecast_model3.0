"""P75 — Full Chain Safety Supervisor unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from safety.full_chain_safety_supervisor import (
    FORBIDDEN_IN_PRODUCTION,
    FULL_CHAIN_SAFETY_DEGRADED,
    FULL_CHAIN_SAFETY_FAILED,
    FULL_CHAIN_SAFETY_PASS,
    QUARANTINED_MODELS,
    run_full_chain_safety,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "y_pred": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def contaminated_predictions():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "y_pred": np.random.uniform(100, 400, 24),
        "y_true": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def clean_output():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "dayahead_price": np.random.uniform(100, 400, 24),
        "realtime_price": np.random.uniform(100, 400, 24),
    })


@pytest.fixture
def clean_weights():
    return pd.DataFrame({
        "model_name": ["cfg05", "catboost"],
        "weight": [0.6, 0.4],
    })


# ── Constants ─────────────────────────────────────────────────────────────────


class TestSafetyConstants:
    def test_forbidden_columns(self):
        assert "y_true" in FORBIDDEN_IN_PRODUCTION
        assert len(FORBIDDEN_IN_PRODUCTION) == 6

    def test_quarantined_models(self):
        assert "lgbm_spike_residual_1127" in QUARANTINED_MODELS
        assert "stage3_business_fixed" in QUARANTINED_MODELS
        assert len(QUARANTINED_MODELS) == 4


# ── run_full_chain_safety ─────────────────────────────────────────────────────


class TestRunFullChainSafety:
    def test_all_none_passes(self):
        result = run_full_chain_safety()
        assert result["status"] == FULL_CHAIN_SAFETY_PASS

    def test_clean_inputs_pass(self, clean_predictions, clean_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            realtime_predictions=clean_predictions,
            final_output=clean_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_PASS

    def test_y_true_in_dayahead_fails(self, contaminated_predictions, clean_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=contaminated_predictions,
            realtime_predictions=contaminated_predictions,
            final_output=clean_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_y_true_in_realtime_fails(self, clean_predictions, contaminated_predictions, clean_output, clean_weights):
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            realtime_predictions=contaminated_predictions,
            final_output=clean_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_quarantined_model_fails(self, clean_predictions, clean_output):
        weights = pd.DataFrame({
            "model_name": ["cfg05", "lgbm_spike_residual_1127"],
            "weight": [0.6, 0.4],
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            final_output=clean_output,
            fusion_weights=weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_stage3_in_weights_fails(self, clean_predictions, clean_output):
        weights = pd.DataFrame({
            "model_name": ["stage3_business_fixed"],
            "weight": [1.0],
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            final_output=clean_output,
            fusion_weights=weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_incomplete_24h_fails(self, clean_output, clean_weights):
        incomplete = pd.DataFrame({
            "business_day": ["2026-06-01"] * 12,
            "hour_business": list(range(1, 13)),
            "y_pred": [200.0] * 12,
        })
        result = run_full_chain_safety(
            dayahead_predictions=incomplete,
            final_output=clean_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_nan_prices_in_output(self, clean_predictions, clean_weights):
        bad_output = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [np.nan] * 24,
            "realtime_price": [200.0] * 24,
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            final_output=bad_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_duplicate_hours_fails(self, clean_predictions, clean_weights):
        dup_output = pd.DataFrame({
            "business_day": ["2026-06-01"] * 25,
            "hour_business": list(range(1, 25)) + [1],
            "dayahead_price": [200.0] * 25,
            "realtime_price": [200.0] * 25,
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            final_output=dup_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_checks_dict_structure(self, clean_predictions):
        result = run_full_chain_safety(dayahead_predictions=clean_predictions)
        assert isinstance(result["checks"], dict)
        for check_name, check_result in result["checks"].items():
            assert "pass" in check_result
            assert isinstance(check_result["pass"], bool)

    def test_errors_list(self, contaminated_predictions):
        result = run_full_chain_safety(
            dayahead_predictions=contaminated_predictions,
        )
        assert isinstance(result["errors"], list)

    def test_warnings_list(self):
        result = run_full_chain_safety()
        assert isinstance(result["warnings"], list)

    def test_y_true_in_final_output_fails(self, clean_predictions, clean_weights):
        bad_output = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
            "y_true": [200.0] * 24,
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            final_output=bad_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED

    def test_online_pack_with_y_true_fails(self, clean_predictions, clean_output, clean_weights):
        bad_pack = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "y_true": [200.0] * 24,
        })
        result = run_full_chain_safety(
            dayahead_predictions=clean_predictions,
            online_pack=bad_pack,
            final_output=clean_output,
            fusion_weights=clean_weights,
        )
        assert result["status"] == FULL_CHAIN_SAFETY_FAILED
