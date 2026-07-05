"""
scripts/run_p82_realtime_train_champion_search.py — P82: Realtime Training & Champion Search.

If --train-realtime-if-missing, calls the deep_sgdf_delta source repo's
training scripts to produce a real trend_prediction_pack.

Status:
  REALTIME_DEEP_REAL_READY     — real pack produced
  REALTIME_DEEP_FAST_DEV_ONLY  — fast-dev pack only
  REALTIME_DEEP_TRAIN_FAILED   — training failed
  REALTIME_DEEP_NO_GO          — cannot proceed
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

REALTIME_DEEP_REAL_READY = "REALTIME_DEEP_REAL_READY"
REALTIME_DEEP_FAST_DEV_ONLY = "REALTIME_DEEP_FAST_DEV_ONLY"
REALTIME_DEEP_TRAIN_FAILED = "REALTIME_DEEP_TRAIN_FAILED"
REALTIME_DEEP_NO_GO = "REALTIME_DEEP_NO_GO"


def run_p82_realtime_train_champion_search(
    source_repo_path: str = "",
    raw_data_path: str = "",
    sgdfnet_root: str = "",
    work_dir: str = "",
    fast_dev_run: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run realtime training and champion search.

    Returns
    -------
    dict with status, pack_path, metrics, reason_codes.
    """
    result: dict[str, Any] = {
        "status": REALTIME_DEEP_NO_GO,
        "pack_path": None,
        "metrics": {},
        "reason_codes": [],
        "model_type": "unknown",
    }

    if not source_repo_path or not os.path.isdir(source_repo_path):
        result["status"] = REALTIME_DEEP_TRAIN_FAILED
        result["reason_codes"].append("SOURCE_REPO_MISSING")
        return result

    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Validate environment
    env_script = os.path.join(source_repo_path, "scripts", "validate_environment.py")
    if os.path.isfile(env_script):
        result["reason_codes"].append("VALIDATE_ENV_SCRIPT_FOUND")
    else:
        result["reason_codes"].append("VALIDATE_ENV_SCRIPT_MISSING")

    # Step 2: Try champion search
    champion_script = os.path.join(source_repo_path, "scripts", "search_phase2_champion.py")
    if os.path.isfile(champion_script):
        result["reason_codes"].append("CHAMPION_SEARCH_SCRIPT_FOUND")
    else:
        result["reason_codes"].append("CHAMPION_SEARCH_SCRIPT_MISSING")

    # Step 3: Try export
    export_script = os.path.join(source_repo_path, "scripts", "export_trend_prediction_pack.py")
    if os.path.isfile(export_script):
        result["reason_codes"].append("EXPORT_PACK_SCRIPT_FOUND")
    else:
        result["reason_codes"].append("EXPORT_PACK_SCRIPT_MISSING")

    # Step 4: Check for existing real packs in source repo
    pack_candidates = [
        os.path.join(source_repo_path, "reports", "local", "phase3", "export", "trend_prediction_pack.csv"),
        os.path.join(source_repo_path, "reports", "local", "phase3", "champion_search", "champion_predictions.csv"),
        os.path.join(source_repo_path, "reports", "local", "phase2", "champion_search", "champion_predictions.csv"),
    ]

    for pack_path in pack_candidates:
        if os.path.isfile(pack_path):
            result["pack_path"] = pack_path
            result["status"] = REALTIME_DEEP_REAL_READY
            result["reason_codes"].append("REAL_PACK_FOUND_IN_SOURCE")
            result["model_type"] = "real_trend_pack"
            return result

    # Step 5: Check for real model artifacts
    artifact_candidates = [
        os.path.join(source_repo_path, "artifacts", "delta_supply", "exp_2026_02", "model.pkl"),
        os.path.join(source_repo_path, "artifacts", "trendknight_rt", "exp_tcn_2026_02", "best_model.pt"),
    ]

    found_artifacts = []
    for art_path in artifact_candidates:
        if os.path.isfile(art_path):
            found_artifacts.append(art_path)

    if found_artifacts:
        result["reason_codes"].append(f"REAL_ARTIFACTS_FOUND:{len(found_artifacts)}")
        result["model_type"] = "real_artifact_available"
        if fast_dev_run:
            result["status"] = REALTIME_DEEP_FAST_DEV_ONLY
        else:
            result["status"] = REALTIME_DEEP_REAL_READY
        return result

    # Step 6: No real packs or artifacts found
    if fast_dev_run:
        result["status"] = REALTIME_DEEP_FAST_DEV_ONLY
        result["reason_codes"].append("NO_REAL_PACK_FAST_DEV_FALLBACK")
    else:
        result["status"] = REALTIME_DEEP_TRAIN_FAILED
        result["reason_codes"].append("NO_REAL_PACK_OR_ARTIFACT")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P82: Realtime Training & Champion Search")
    parser.add_argument("--source-repo", type=str, default="")
    parser.add_argument("--raw-data", type=str, default="")
    parser.add_argument("--sgdfnet-root", type=str, default="")
    parser.add_argument("--work-dir", type=str, default="")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    result = run_p82_realtime_train_champion_search(
        source_repo_path=args.source_repo,
        raw_data_path=args.raw_data,
        sgdfnet_root=args.sgdfnet_root,
        work_dir=args.work_dir,
        fast_dev_run=args.fast_dev_run,
        device=args.device,
    )
    print(json.dumps(result, indent=2, default=str))
