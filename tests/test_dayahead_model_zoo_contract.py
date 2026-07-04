"""
tests/test_dayahead_model_zoo_contract.py — Day-ahead model registry contract tests.

Validates:
    1. Invalid model raises ValueError
    2. Unknown model raises KeyError
    3. DEFAULT_FUSION_POOL contains correct models
    4. Champion identity and sMAPE
    5. All pool models are valid
    6. is_valid_model / is_invalid_model helpers
"""

from __future__ import annotations

import pytest

from src.registry.dayahead_models import (
    CHAMPION_MODEL_ID,
    CHAMPION_SMAPE_FLOOR50,
    DEFAULT_FUSION_POOL,
    INVALID_MODELS,
    MODEL_CONFIGS,
    is_valid_model,
    is_invalid_model,
    get_model_config,
    list_valid_models,
)


class TestInvalidModels:
    """Contract: invalid models must raise ValueError."""

    @pytest.mark.parametrize("invalid_id", [
        "lgbm_spike_residual_1127",
        "stage3_old_1164",
        "lightgbm_90d_orig_1197",
    ])
    def test_invalid_model_raises_value_error(self, invalid_id):
        """get_model_config for an invalid model raises ValueError."""
        with pytest.raises(ValueError, match="INVALID"):
            get_model_config(invalid_id)

    def test_is_invalid_model_returns_true(self):
        """is_invalid_model returns True for known invalid models."""
        for mid in INVALID_MODELS:
            assert is_invalid_model(mid)

    def test_is_valid_model_returns_false_for_invalid(self):
        """is_valid_model returns False for invalid models."""
        for mid in INVALID_MODELS:
            assert not is_valid_model(mid)

    def test_invalid_models_have_reasons(self):
        """Every invalid model entry has a non-empty reason string."""
        for mid, reason in INVALID_MODELS.items():
            assert len(reason) > 10, f"{mid} reason too short: {reason}"


class TestUnknownModels:
    """Contract: unknown model IDs raise KeyError."""

    def test_unknown_model_raises_key_error(self):
        """get_model_config for unknown ID raises KeyError."""
        with pytest.raises(KeyError):
            get_model_config("nonexistent_model_xyz")

    def test_empty_string_raises_key_error(self):
        """get_model_config for empty string raises KeyError."""
        with pytest.raises(KeyError):
            get_model_config("")


class TestChampion:
    """Contract: champion identity and metrics."""

    def test_champion_is_cfg05(self):
        """CHAMPION_MODEL_ID is cfg05."""
        assert CHAMPION_MODEL_ID == "cfg05"

    def test_champion_smape_is_float(self):
        """CHAMPION_SMAPE_FLOOR50 is a positive float."""
        assert isinstance(CHAMPION_SMAPE_FLOOR50, float)
        assert CHAMPION_SMAPE_FLOOR50 > 0

    def test_champion_in_model_configs(self):
        """Champion exists in MODEL_CONFIGS and is marked champion."""
        cfg = get_model_config(CHAMPION_MODEL_ID)
        assert cfg.get("champion") is True

    def test_champion_is_valid(self):
        """Champion is a valid model."""
        assert is_valid_model(CHAMPION_MODEL_ID)


class TestDefaultFusionPool:
    """Contract: DEFAULT_FUSION_POOL structure and content."""

    def test_fusion_pool_has_5_models(self):
        """DEFAULT_FUSION_POOL contains exactly 5 entries."""
        assert len(DEFAULT_FUSION_POOL) == 5

    def test_fusion_pool_first_is_cfg05(self):
        """First entry in DEFAULT_FUSION_POOL is cfg05."""
        assert DEFAULT_FUSION_POOL[0]["model_id"] == "cfg05"

    def test_all_pool_entries_have_required_keys(self):
        """Each pool entry has model_id, formal_name, sMAPE_floor50, description."""
        for entry in DEFAULT_FUSION_POOL:
            assert "model_id" in entry
            assert "formal_name" in entry
            assert "sMAPE_floor50" in entry
            assert "description" in entry

    def test_all_pool_models_are_valid(self):
        """Every model in DEFAULT_FUSION_POOL is valid (not invalidated)."""
        for entry in DEFAULT_FUSION_POOL:
            assert is_valid_model(entry["model_id"]), (
                f"Pool model {entry['model_id']} is not valid"
            )

    def test_pool_sorted_by_rank(self):
        """Fusion pool entries are sorted by sMAPE_floor50 ascending."""
        smapes = [e["sMAPE_floor50"] for e in DEFAULT_FUSION_POOL]
        assert smapes == sorted(smapes), "Fusion pool not sorted by sMAPE"

    def test_pool_model_ids_in_configs(self):
        """Every pool model_id exists in MODEL_CONFIGS."""
        for entry in DEFAULT_FUSION_POOL:
            assert entry["model_id"] in MODEL_CONFIGS


class TestModelConfigs:
    """Contract: MODEL_CONFIGS entries."""

    def test_cfg05_params_complete(self):
        """cfg05 params contain all key hyperparameters."""
        cfg = get_model_config("cfg05")
        params = cfg["params"]
        assert params["objective"] == "mae"
        assert params["num_leaves"] == 191
        assert params["learning_rate"] == 0.015
        assert params["n_estimators"] == 2000

    def test_all_configs_have_task_dayahead(self):
        """Every MODEL_CONFIGS entry has task=dayahead."""
        for mid in list_valid_models():
            cfg = get_model_config(mid)
            assert cfg.get("task") == "dayahead", f"{mid} task is not dayahead"


class TestHelpers:
    """Contract: registry helper functions."""

    def test_list_valid_models(self):
        """list_valid_models returns only valid models."""
        valid = list_valid_models()
        assert CHAMPION_MODEL_ID in valid
        for mid in INVALID_MODELS:
            assert mid not in valid

    def test_is_valid_model_returns_true_for_champion(self):
        """is_valid_model returns True for cfg05."""
        assert is_valid_model("cfg05")

    def test_is_valid_model_returns_false_for_nonexistent(self):
        """is_valid_model returns False for unknown model."""
        assert not is_valid_model("fake_model")
