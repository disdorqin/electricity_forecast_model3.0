"""
delivery/manifest.py — P55: Write and read delivery run manifests (JSON format).

Manifests capture all metadata about a delivery run for traceability,
auditing, and report generation.
"""

from __future__ import annotations

import json
import os
from typing import Any

_MANIFEST_FILENAME = "run_manifest.json"

_REQUIRED_MANIFEST_KEYS = [
    "run_id",
    "target_day",
    "profile",
    "started_at",
    "completed_at",
    "status",
    "delivery_status",
    "selected_training_days",
    "trusted_models",
    "quarantined_models",
    "fusion_method",
    "fallback",
    "postflight",
    "metrics",
    "warnings",
    "errors",
]


def create_manifest(
    run_id: str,
    target_day: str,
    profile: str,
    started_at: str,
    completed_at: str,
    status: str,
    delivery_status: str,
    selected_training_days: int,
    trusted_models: list[str],
    quarantined_models: list[str],
    fusion_method: str,
    fallback_used: bool,
    fallback_method: str,
    postflight: dict,
    metrics: dict,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    """Create a delivery manifest dict.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    target_day : str
        The target delivery date (YYYY-MM-DD).
    profile : str
        Fusion profile name used.
    started_at : str
        ISO-format start timestamp.
    completed_at : str
        ISO-format completion timestamp.
    status : str
        Overall run status (e.g. \"PASS\", \"FAIL\", \"WARN\").
    delivery_status : str
        Delivery-specific status label.
    selected_training_days : int
        Number of training days used.
    trusted_models : list[str]
        Models that passed the trust gate.
    quarantined_models : list[str]
        Models quarantined by the trust gate.
    fusion_method : str
        Fusion method used (e.g. \"BGEW\", \"equal_weight\").
    fallback_used : bool
        Whether fallback was triggered.
    fallback_method : str
        Fallback method name (empty string if not used).
    postflight : dict
        Postflight validation results dict.
    metrics : dict
        Key-value metrics from the delivery run.
    warnings : list[str]
        Warning messages.
    errors : list[str]
        Error messages.

    Returns
    -------
    dict
        Manifest dict conforming to the delivery manifest schema.
    """
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "target_day": target_day,
        "profile": profile,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "delivery_status": delivery_status,
        "selected_training_days": selected_training_days,
        "trusted_models": trusted_models,
        "quarantined_models": quarantined_models,
        "fusion_method": fusion_method,
        "fallback": {
            "fallback_used": fallback_used,
            "fallback_method": fallback_method,
        },
        "postflight": postflight,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
    }
    return manifest


def write_manifest(manifest: dict[str, Any], output_dir: str) -> str:
    """Write manifest to ``output_dir/run_manifest.json``.

    Parameters
    ----------
    manifest : dict
        Manifest dict to write.
    output_dir : str
        Directory to write the manifest file into.

    Returns
    -------
    str
        Absolute path to the written manifest file.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, _MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return os.path.abspath(manifest_path)


def read_manifest(manifest_path: str) -> dict[str, Any]:
    """Read manifest from a JSON file.

    Parameters
    ----------
    manifest_path : str
        Path to the manifest JSON file.

    Returns
    -------
    dict
        The parsed manifest dict.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains invalid JSON.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
        manifest = json.load(f)
    return manifest


def validate_manifest_keys(manifest: dict[str, Any]) -> list[str]:
    """Validate that a manifest dict contains all required keys.

    Parameters
    ----------
    manifest : dict
        The manifest dict to validate.

    Returns
    -------
    list[str]
        List of missing keys (empty if all required keys are present).
    """
    missing = []
    for key in _REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            missing.append(key)
    # Check nested fallback key
    if "fallback" in manifest and not isinstance(manifest["fallback"], dict):
        missing.append("fallback (must be dict)")
    return missing
