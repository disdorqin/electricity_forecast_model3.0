"""
src/registry/realtime_models.py — Realtime model registry.

Central registry of approved realtime models, their positioning,
and configuration. Single source of truth for realtime model identity in 3.0.

Usage:
    from src.registry.realtime_models import (
        get_realtime_model,
        list_realtime_models,
        DA_SAFE_ASSIST_ID,
        SGDFNET_2_5_ID,
    )
"""

from __future__ import annotations

from typing import Any

# ──────────────────────────────────────────────
# Model IDs
# ──────────────────────────────────────────────

DA_SAFE_ASSIST_ID: str = "da_safe_realtime_assist"
SGDFNET_2_5_ID: str = "sgdfnet_2_5"

# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

REALTIME_MODELS: dict[str, dict[str, Any]] = {
    DA_SAFE_ASSIST_ID: {
        "model_id": DA_SAFE_ASSIST_ID,
        "formal_name": "da_safe_realtime_assist",
        "model_type": "DA-Safe Assist (sidecar)",
        "task": "realtime",
        "positioning": "sidecar_assist",
        "default_prediction": "rt_pred = da_anchor",
        "description": "DA-Safe Realtime Assist Model. "
                       "Primary: rt_pred = da_anchor (DA-only). "
                       "Optional safe correction disabled by default. "
                       "Outputs assist signals (error_prob, direction, uncertainty).",
        "source_repo": "disdorqin/electricity_forecast_deep_sgdf_delta",
        "source_branch": "main",
        "status": "READY_FOR_CHAIN_HANDOFF",
        "output_fields": [
            "rt_pred",
            "da_error_prob",
            "residual_direction_prob",
            "uncertainty_score",
            "correction_permission",
            "reason_codes",
        ],
        "features": {
            "load_forecast": "MISSING",
            "wind_forecast": "MISSING",
            "solar_forecast": "MISSING",
            "net_load": "MISSING",
        },
        "risks": [
            "Does not guarantee beat DA anchor — DA-only is strongest baseline",
            "Sensitive to missing forecast-side features",
            "Small correction benefit insufficient for default enablement",
        ],
    },
    SGDFNET_2_5_ID: {
        "model_id": SGDFNET_2_5_ID,
        "formal_name": "sgdfnet_2_5",
        "model_type": "Delta Regressor (SGDFNet)",
        "task": "realtime",
        "positioning": "realtime_candidate",
        "description": "SGDFNet delta regressor from 2.5 repo. "
                       "Predicts rt - da delta. "
                       "Requires external model weights and feature pipeline.",
        "source_repo": "disdorqin/electricity_forecast_model2.5",
        "source_branch": "main",
        "status": "CANDIDATE — needs adapter wiring + model weights",
        "output_fields": [
            "delta_pred",
            "rt_pred",
        ],
        "features": {
            "load_forecast": "MISSING",
            "bidding_space_forecast": "MISSING",
            "renewable_forecast": "MISSING",
        },
        "notes": "Full integration pending — model weight files not in git. "
                 "Adapter skeleton only for P1.",
    },
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def get_realtime_model(model_id: str) -> dict[str, Any]:
    """Get the registry entry for a realtime model.

    Parameters
    ----------
    model_id : str
        The model identifier.

    Returns
    -------
    dict
        Model registry entry.

    Raises
    ------
    KeyError
        If model_id is not in the realtime registry.
    """
    if model_id not in REALTIME_MODELS:
        raise KeyError(f"Model '{model_id}' not found in realtime registry")
    return dict(REALTIME_MODELS[model_id])


def list_realtime_models() -> list[str]:
    """Return list of all realtime model IDs."""
    return list(REALTIME_MODELS.keys())


def is_ready_for_chain_handoff(model_id: str) -> bool:
    """Check if a realtime model is ready for chain handoff.

    Parameters
    ----------
    model_id : str
        The model identifier.

    Returns
    -------
    bool
        True if the model's status is READY_FOR_CHAIN_HANDOFF.
    """
    entry = get_realtime_model(model_id)
    return entry.get("status") == "READY_FOR_CHAIN_HANDOFF"
