"""
scripts/audit_p29_dayahead_model_pool_real_readiness.py — P29 multi-model candidate pool audit.

Scans each candidate day-ahead model and determines its real-readiness status:

    cfg05, best_two_average, stage3_business_fixed,
    catboost_spike_residual, catboost_sota

Banned models are explicitly excluded:

    lgbm_spike_residual_1127, stage3_old_1164, lightgbm_90d_orig_1197

Usage::

    python -m scripts.audit_p29_dayahead_model_pool_real_readiness \\
        --source-repo .local_artifacts/source_repos/epf-sota-experiment \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Readiness labels ───────────────────────────────────────────────────────
REAL_READY = "REAL_READY"
REAL_24H_READY = "REAL_24H_READY"
BACKTESTED = "BACKTESTED"
TRAINABLE_LOCAL = "TRAINABLE_LOCAL"
PREDICTION_ONLY = "PREDICTION_ONLY"
DATA_MISSING = "DATA_MISSING"
ARTIFACT_MISSING = "ARTIFACT_MISSING"
INVALID_BANNED = "INVALID_BANNED"

# ── Candidate pool ─────────────────────────────────────────────────────────
CANDIDATE_MODELS = [
    "cfg05",
    "best_two_average",
    "stage3_business_fixed",
    "catboost_spike_residual",
    "catboost_sota",
]

BANNED_MODELS = {
    "lgbm_spike_residual_1127": "target leakage: y_true used as prediction feature",
    "stage3_old_1164": "natural-day business_day mapping error (used ds.date())",
    "lightgbm_90d_orig_1197": "690 rows only — missing hour 24, invalid output shape",
}

# Model type mapping from registry
_MODEL_TYPE_MAP = {
    "cfg05": "LightGBM",
    "best_two_average": "ensemble_average",
    "stage3_business_fixed": "LightGBM",
    "catboost_spike_residual": "CatBoost",
    "catboost_sota": "CatBoost",
}


def _check_training_script_exists(source_repo: str, model_id: str) -> bool:
    """Check if a training script exists for this model in the source repo."""
    if not os.path.isdir(source_repo):
        return False

    scripts_dir = os.path.join(source_repo, "scripts")
    if not os.path.isdir(scripts_dir):
        return False

    # Search for scripts that mention this model
    model_keywords = {
        "cfg05": ["cfg05", "champion", "lightgbm_stage3"],
        "best_two_average": ["best_two", "trial_02", "trial_24", "freeze"],
        "stage3_business_fixed": ["stage3", "business_fixed"],
        "catboost_spike_residual": ["spike_residual", "lgbm_spike", "dayahead_correction"],
        "catboost_sota": ["catboost", "sota"],
    }

    keywords = model_keywords.get(model_id, [model_id])
    for fname in os.listdir(scripts_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(scripts_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read(5000)
            if any(kw in content.lower() for kw in keywords):
                return True
        except Exception:
            continue
    return False


def _check_model_artifact(model_id: str, work_dir: Optional[str] = None) -> bool:
    """Check if a trained model artifact exists locally."""
    search_dirs = []
    if work_dir:
        search_dirs.append(work_dir)
    search_dirs.extend([
        os.path.join(".local_artifacts", "p16_p20_cfg05_chain"),
        os.path.join(".local_artifacts", "p21_p25_real_chain"),
        os.path.join(".local_artifacts", "p26_p30_fusion"),
    ])

    model_file_patterns = {
        "cfg05": ["cfg05_model.txt", "model.txt", "lightgbm_cfg05"],
        "best_two_average": ["best_two", "trial_02", "trial_24"],
        "stage3_business_fixed": ["stage3_business", "stage3_fixed"],
        "catboost_spike_residual": ["catboost_spike", "spike_residual"],
        "catboost_sota": ["catboost_sota", "catboost_model"],
    }

    patterns = model_file_patterns.get(model_id, [model_id])
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                for pat in patterns:
                    if pat in f.lower():
                        return True
    return False


def _check_can_predict_24h(model_id: str, source_repo: str) -> bool:
    """Check if the model has code that can produce 24H predictions."""
    if model_id == "cfg05":
        # cfg05 has a full adapter in 3.0
        return True

    if not os.path.isdir(source_repo):
        return False

    # Check for adapter code in source repo
    adapter_map = {
        "catboost_sota": ["src/models/catboost_adapter.py"],
        "catboost_spike_residual": ["src/correction/dayahead_residual_corrector.py"],
        "best_two_average": [],  # ensemble, needs two base models
        "stage3_business_fixed": [],  # needs stage3 training script
    }

    for rel_path in adapter_map.get(model_id, []):
        if os.path.isfile(os.path.join(source_repo, rel_path)):
            return True
    return False


def _assess_readiness(model_id: str, source_repo: str, work_dir: Optional[str] = None) -> dict[str, Any]:
    """Assess the readiness of a single model."""
    from src.registry.dayahead_models import INVALID_MODELS, MODEL_CONFIGS

    entry: dict[str, Any] = {
        "model_id": model_id,
        "model_type": _MODEL_TYPE_MAP.get(model_id, "unknown"),
        "in_registry": model_id in MODEL_CONFIGS,
        "is_banned": model_id in INVALID_MODELS,
        "ban_reason": INVALID_MODELS.get(model_id),
        "artifact_exists": False,
        "training_script_exists": False,
        "can_predict_24h": False,
        "schema_ready": False,
        "backtest_ready": False,
        "readiness_label": ARTIFACT_MISSING,
        "blocker": None,
    }

    # Banned models
    if entry["is_banned"]:
        entry["readiness_label"] = INVALID_BANNED
        entry["blocker"] = entry["ban_reason"]
        return entry

    # Check each dimension
    entry["artifact_exists"] = _check_model_artifact(model_id, work_dir)
    entry["training_script_exists"] = _check_training_script_exists(source_repo, model_id)
    entry["can_predict_24h"] = _check_can_predict_24h(model_id, source_repo)

    # Schema ready if model is in registry (has defined schema)
    entry["schema_ready"] = entry["in_registry"]

    # Determine readiness label
    if model_id == "cfg05":
        # cfg05 is the champion — it's been backtested in P21/P26
        entry["backtest_ready"] = True
        if entry["artifact_exists"]:
            entry["readiness_label"] = REAL_24H_READY
        else:
            entry["readiness_label"] = BACKTESTED
        entry["blocker"] = None
    elif entry["artifact_exists"] and entry["can_predict_24h"]:
        entry["readiness_label"] = REAL_READY
        entry["backtest_ready"] = True
    elif entry["training_script_exists"] and entry["can_predict_24h"]:
        entry["readiness_label"] = TRAINABLE_LOCAL
        entry["blocker"] = "needs local training to produce artifact"
    elif entry["can_predict_24h"]:
        entry["readiness_label"] = PREDICTION_ONLY
        entry["blocker"] = "can predict but no training script or artifact"
    elif entry["training_script_exists"]:
        entry["readiness_label"] = TRAINABLE_LOCAL
        entry["blocker"] = "training script exists but 24H prediction capability unclear"
    elif not entry["artifact_exists"]:
        entry["readiness_label"] = ARTIFACT_MISSING
        entry["blocker"] = "no trained model artifact found locally"
    else:
        entry["readiness_label"] = DATA_MISSING
        entry["blocker"] = "unknown state"

    return entry


def audit_p29_dayahead_model_pool_real_readiness(
    source_repo: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Audit the multi-model candidate pool for real-readiness.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment.
    work_dir : str, optional
        Local work directory to search for artifacts.

    Returns
    -------
    dict
        Complete model pool audit report.
    """
    source_repo = source_repo or os.path.join(
        ".local_artifacts", "source_repos", "epf-sota-experiment"
    )
    work_dir = work_dir or os.path.join(".local_artifacts", "p26_p30_fusion")

    result: dict[str, Any] = {
        "source_repo": source_repo,
        "work_dir": work_dir,
        "candidates_evaluated": [],
        "banned_models": list(BANNED_MODELS.keys()),
        "real_ready_count": 0,
        "trainable_local_count": 0,
        "artifact_missing_count": 0,
        "banned_count": 0,
        "can_form_fusion_pool": False,
        "fusion_pool_models": [],
        "final_status": None,
        "reason_codes": [],
    }

    # Evaluate each candidate
    for model_id in CANDIDATE_MODELS:
        entry = _assess_readiness(model_id, source_repo, work_dir)
        result["candidates_evaluated"].append(entry)

        label = entry["readiness_label"]
        if label in (REAL_READY, REAL_24H_READY, BACKTESTED):
            result["real_ready_count"] += 1
            result["fusion_pool_models"].append(model_id)
        elif label == TRAINABLE_LOCAL:
            result["trainable_local_count"] += 1
        elif label == ARTIFACT_MISSING:
            result["artifact_missing_count"] += 1
        elif label == INVALID_BANNED:
            result["banned_count"] += 1

    # Also verify banned models are correctly identified
    for banned_id in BANNED_MODELS:
        from src.registry.dayahead_models import is_invalid_model
        assert is_invalid_model(banned_id), f"Banned model {banned_id} not detected"

    # Can we form a fusion pool?
    result["can_form_fusion_pool"] = result["real_ready_count"] >= 2

    # Final status
    if result["real_ready_count"] >= 2:
        result["final_status"] = "P29_MODEL_POOL_MULTI_MODEL_READY"
    elif result["real_ready_count"] == 1:
        result["final_status"] = "P29_MODEL_POOL_SINGLE_MODEL_ONLY"
        result["reason_codes"].append("ONLY_CFG05_REAL_READY")
    else:
        result["final_status"] = "P29_MODEL_POOL_NO_REAL_MODELS"

    result["reason_codes"].append(
        f"REAL_READY:{result['real_ready_count']}_"
        f"TRAINABLE:{result['trainable_local_count']}_"
        f"MISSING:{result['artifact_missing_count']}_"
        f"BANNED:{result['banned_count']}"
    )

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P29 Day-Ahead Model Pool Real-Readiness Audit")
    print("=" * 60)
    print(f"  Source repo:        {result['source_repo']}")
    print(f"  Real-ready:         {result['real_ready_count']}")
    print(f"  Trainable local:    {result['trainable_local_count']}")
    print(f"  Artifact missing:   {result['artifact_missing_count']}")
    print(f"  Banned:             {result['banned_count']}")
    print(f"  Can form pool:      {result['can_form_fusion_pool']}")
    print(f"  Fusion pool:        {result['fusion_pool_models']}")
    print(f"  Final status:       {result['final_status']}")
    print()
    for c in result["candidates_evaluated"]:
        print(f"  [{c['readiness_label']:20s}] {c['model_id']:30s} "
              f"artifact={c['artifact_exists']} train={c['training_script_exists']} "
              f"24h={c['can_predict_24h']}")
        if c["blocker"]:
            print(f"    blocker: {c['blocker']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P29: day-ahead model pool real-readiness audit.")
    p.add_argument("--source-repo", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--json", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    result = audit_p29_dayahead_model_pool_real_readiness(
        source_repo=args.source_repo,
        work_dir=args.work_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
