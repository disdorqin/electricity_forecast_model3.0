"""
scripts/run_p133_model_feature_schema_audit.py — P133: Model Feature Schema Audit.

Loads each model artifact (catboost_spike_residual.cbm, catboost_sota.cbm,
cfg05_model.txt), extracts feature names and counts, checks the registry for
trusted/quarantine status, and outputs a comprehensive schema report.

Output: .local_artifacts/p133_feature_schema/model_feature_schema.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Default artifact paths
DEFAULT_MODEL_DIR = os.path.join(
    REPO_ROOT, ".local_artifacts", "p31_p40_multimodel_fusion", "models"
)

MODEL_ARTIFACTS = {
    "cfg05": {
        "path": os.path.join("cfg05_dayahead_lgbm", "cfg05_model.txt"),
        "type": "lightgbm",
        "registry_id": "cfg05",
    },
    "catboost_spike_residual": {
        "path": os.path.join("catboost_spike_residual", "catboost_spike_residual.cbm"),
        "type": "catboost",
        "registry_id": "catboost_spike_residual",
    },
    "catboost_sota": {
        "path": os.path.join("catboost_sota", "catboost_sota_model.cbm"),
        "type": "catboost",
        "registry_id": "catboost_sota",
    },
}


def _load_lightgbm_features(model_path: str) -> list[str]:
    """Extract feature names from a LightGBM model file."""
    import lightgbm as lgb
    booster = lgb.Booster(model_file=model_path)
    return list(booster.feature_name())


def _load_catboost_features(model_path: str) -> list[str]:
    """Extract feature names from a CatBoost model file."""
    from catboost import CatBoost
    model = CatBoost()
    model.load_model(model_path)
    feature_names = model.feature_names_
    if feature_names is not None:
        return list(feature_names)
    # Fallback: use n_features_in_ or generic names
    try:
        n_features = model.n_features_in_
    except AttributeError:
        n_features = model.get_n_features_in()
    return [f"feature_{i}" for i in range(n_features)]


def _get_registry_status(registry_id: str) -> dict[str, Any]:
    """Look up a model in the dayahead registry and return trust/status info."""
    from src.registry.dayahead_models import (
        MODEL_CONFIGS,
        INVALID_MODELS,
        DEFAULT_FUSION_POOL,
    )

    result: dict[str, Any] = {
        "in_registry": registry_id in MODEL_CONFIGS,
        "is_invalid": registry_id in INVALID_MODELS,
        "invalid_reason": INVALID_MODELS.get(registry_id, ""),
        "in_fusion_pool": False,
        "trusted": False,
        "quarantined": False,
    }

    if registry_id in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[registry_id]
        result["model_type"] = cfg.get("model_type", "unknown")
        result["champion"] = cfg.get("champion", False)
        result["sMAPE_floor50"] = cfg.get("sMAPE_floor50", None)

    for entry in DEFAULT_FUSION_POOL:
        if entry.get("model_id") == registry_id:
            result["in_fusion_pool"] = True
            break

    # Trust rules: cfg05 is trusted champion; catboost_spike_residual is trusted;
    # catboost_sota is quarantined (not trusted)
    if registry_id == "cfg05":
        result["trusted"] = True
        result["quarantined"] = False
    elif registry_id == "catboost_spike_residual":
        result["trusted"] = True
        result["quarantined"] = False
    elif registry_id == "catboost_sota":
        result["trusted"] = False
        result["quarantined"] = True
    else:
        result["trusted"] = False
        result["quarantined"] = not result["in_fusion_pool"]

    return result


def _check_feature_alignment(
    model_features: list[str],
    registry_id: str,
) -> dict[str, Any]:
    """Check if model features align with the registry's expected feature list."""
    from src.registry.dayahead_models import MODEL_CONFIGS

    result: dict[str, Any] = {
        "registry_has_feature_list": False,
        "registry_feature_count": 0,
        "model_feature_count": len(model_features),
        "exact_match": False,
        "overlap_count": 0,
        "missing_from_model": [],
        "extra_in_model": [],
        "can_infer_with_current_builder": False,
    }

    cfg = MODEL_CONFIGS.get(registry_id, {})
    registry_features = cfg.get("feature_columns")

    if registry_features is not None:
        result["registry_has_feature_list"] = True
        result["registry_feature_count"] = len(registry_features)
        reg_set = set(registry_features)
        mod_set = set(model_features)
        result["overlap_count"] = len(reg_set & mod_set)
        result["missing_from_model"] = sorted(reg_set - mod_set)
        result["extra_in_model"] = sorted(mod_set - reg_set)
        result["exact_match"] = (reg_set == mod_set)

    # Check if current feature builder can produce these features
    # cfg05 and catboost_spike_residual both use 56 v3 features
    # catboost_sota uses 24 base features
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
    cfg05_set = set(CFG05_FEATURE_COLUMNS)
    mod_set = set(model_features)

    if mod_set == cfg05_set:
        result["can_infer_with_current_builder"] = True
        result["builder_match"] = "cfg05_v3_56"
    elif mod_set.issubset(cfg05_set):
        result["can_infer_with_current_builder"] = True
        result["builder_match"] = "cfg05_v3_subset"
    else:
        result["can_infer_with_current_builder"] = False
        result["builder_match"] = "none"

    return result


def audit_model(
    model_name: str,
    model_def: dict[str, Any],
    model_dir: str,
) -> dict[str, Any]:
    """Audit a single model artifact."""
    artifact_path = os.path.join(model_dir, model_def["path"])
    model_type = model_def["type"]
    registry_id = model_def["registry_id"]

    entry: dict[str, Any] = {
        "model_name": model_name,
        "artifact_path": artifact_path,
        "model_type": model_type,
        "registry_id": registry_id,
        "artifact_exists": os.path.isfile(artifact_path),
        "loaded_successfully": False,
        "feature_count": 0,
        "feature_names": [],
        "trusted_status": {},
        "feature_alignment": {},
        "reason_codes": [],
    }

    if not entry["artifact_exists"]:
        entry["reason_codes"].append("ARTIFACT_FILE_MISSING")
        return entry

    # Load model and extract features
    try:
        if model_type == "lightgbm":
            features = _load_lightgbm_features(artifact_path)
        elif model_type == "catboost":
            features = _load_catboost_features(artifact_path)
        else:
            entry["reason_codes"].append(f"UNKNOWN_MODEL_TYPE:{model_type}")
            return entry

        entry["loaded_successfully"] = True
        entry["feature_count"] = len(features)
        entry["feature_names"] = features
    except Exception as e:
        entry["reason_codes"].append(f"LOAD_FAILED:{e}")
        return entry

    # Registry status
    try:
        entry["trusted_status"] = _get_registry_status(registry_id)
    except Exception as e:
        entry["reason_codes"].append(f"REGISTRY_LOOKUP_FAILED:{e}")
        entry["trusted_status"] = {"in_registry": False, "error": str(e)}

    # Feature alignment
    try:
        entry["feature_alignment"] = _check_feature_alignment(features, registry_id)
    except Exception as e:
        entry["reason_codes"].append(f"ALIGNMENT_CHECK_FAILED:{e}")

    return entry


def run_p133_audit(
    model_dir: str = DEFAULT_MODEL_DIR,
    output_dir: str = "",
) -> dict[str, Any]:
    """Run the full P133 model feature schema audit.

    Parameters
    ----------
    model_dir : str
        Directory containing model artifacts.
    output_dir : str
        Output directory for the audit report.

    Returns
    -------
    dict
        Full audit result.
    """
    if not output_dir:
        output_dir = os.path.join(REPO_ROOT, ".local_artifacts", "p133_feature_schema")
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    result: dict[str, Any] = {
        "phase": "P133",
        "title": "Model Feature Schema Audit",
        "status": "STARTED",
        "model_dir": model_dir,
        "models": {},
        "summary": {
            "total_models": len(MODEL_ARTIFACTS),
            "loaded": 0,
            "failed": 0,
            "missing": 0,
            "trusted": 0,
            "quarantined": 0,
        },
    }

    for model_name, model_def in MODEL_ARTIFACTS.items():
        logger.info(f"Auditing model: {model_name}")
        entry = audit_model(model_name, model_def, model_dir)
        result["models"][model_name] = entry

        if entry["loaded_successfully"]:
            result["summary"]["loaded"] += 1
        elif entry["artifact_exists"]:
            result["summary"]["failed"] += 1
        else:
            result["summary"]["missing"] += 1

        trust = entry.get("trusted_status", {})
        if trust.get("trusted"):
            result["summary"]["trusted"] += 1
        if trust.get("quarantined"):
            result["summary"]["quarantined"] += 1

    result["elapsed_seconds"] = round(time.time() - t_start, 2)

    # Determine overall status
    if result["summary"]["loaded"] == result["summary"]["total_models"]:
        result["status"] = "AUDIT_COMPLETE_ALL_LOADED"
    elif result["summary"]["loaded"] > 0:
        result["status"] = "AUDIT_COMPLETE_PARTIAL"
    else:
        result["status"] = "AUDIT_BLOCKED_NO_MODELS"

    # Save output
    output_path = os.path.join(output_dir, "model_feature_schema.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    result["output_path"] = output_path

    logger.info(f"P133 audit complete: {result['status']}")
    logger.info(f"Output: {output_path}")

    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="P133: Model Feature Schema Audit")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_p133_audit(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n=== P133: Model Feature Schema Audit ===")
        print(f"Status: {result['status']}")
        print(f"Models audited: {result['summary']['total_models']}")
        print(f"  Loaded: {result['summary']['loaded']}")
        print(f"  Failed: {result['summary']['failed']}")
        print(f"  Missing: {result['summary']['missing']}")
        print(f"  Trusted: {result['summary']['trusted']}")
        print(f"  Quarantined: {result['summary']['quarantined']}")
        for name, entry in result["models"].items():
            fc = entry["feature_count"]
            status = "OK" if entry["loaded_successfully"] else "FAIL"
            trust = entry.get("trusted_status", {}).get("trusted", "?")
            print(f"  {name}: {status} features={fc} trusted={trust}")
        print(f"Output: {result.get('output_path', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
