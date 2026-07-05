"""
scripts/run_delivery_local_chain.py — P47/P61: Delivery runner with safety supervisor.

Executes the full delivery pipeline: raw data check → source repo check →
trust gate → actual ledger → prediction ledger → safety preflight →
adaptive training days → fusion engine dispatch (P42/P56) →
fallback ladder → write output → postflight → manifest/report →
forbidden file check → claim guard.

Steps that already have valid local artifacts are skipped (reused) unless
--force is passed.

P61 hotfixes:
  - Bug 1: raw data VALID → PASSED (was incorrectly FAILED)
  - Bug 2: adaptive_training_days passes trusted_models + actual_ledger_path
  - Bug 3: safety_preflight runs AFTER ledger generation (not before)
  - Bug 4: leak sentinel returns {"models": [{model_name, status}, ...]}
  - Bug 5: postflight call uses output_path=/target_date=/profile_name=
  - Bug 6: fallback ladder output persisted to final_output.csv
  - Bug 7: --fusion-engine dispatches P56 regime_bgew or P42 period_bgew

Usage::

    python -m scripts.run_delivery_local_chain \\
        --raw-data ../data/shandong_pmos_hourly.csv \\
        --source-repo .local_artifacts/source_repos/epf-sota-experiment \\
        --profile trusted_delivery \\
        --fusion-engine period_bgew \\
        --required-training-days 30 \\
        --allow-degraded \\
        --strict-no-leakage \\
        --start-day 2026-06-01 \\
        --end-day 2026-06-30 \\
        --work-dir .local_artifacts/delivery_run \\
        --json --strict

Options::

    --raw-data PATH               Raw Chinese CSV path.
    --source-repo PATH            Source repo directory (epf-sota-experiment).
    --profile NAME                Fusion profile name (default: trusted_delivery).
    --fusion-engine NAME          Fusion engine: regime_bgew|period_bgew|equal_weight|cfg05 (default: period_bgew).
    --required-training-days N    Required complete training days (default: 30).
    --max-lookback-days N         Max calendar days to scan (default: 180).
    --min-days-for-degraded N     Minimum days for degraded mode (default: 7).
    --allow-degraded              Allow delivery with DEGRADED_MIN_DAYS status.
    --strict-no-leakage           Fail if ANY leakage check triggers.
    --start-day DATE              Start date (YYYY-MM-DD).
    --end-day DATE                End date (YYYY-MM-DD).
    --work-dir PATH               Working directory for outputs (default: .local_artifacts/delivery_run).
    --force                       Force re-run all steps.
    --json                        Output JSON report.
    --strict                      Exit non-zero on any step failure.

Outputs::

    <work-dir>/delivery_summary.json
    <work-dir>/final_output.csv
    <work-dir>/metrics.json
    <work-dir>/run_manifest.json
    <work-dir>/delivery_report.md
    <work-dir>/delivery_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "delivery_run")
_DEFAULT_PROFILE = "trusted_delivery"
_PROFILES_YAML = "config/fusion_profiles.yaml"
_LEDGER_DIR_NAME = "ledger"
_PREDICTION_LEDGER_FILENAME = "prediction_ledger_30d.csv"
_ACTUAL_LEDGER_FILENAME = "actual_ledger_30d.csv"

# ── Profile helpers ──────────────────────────────────────────────────────────


def _load_profile_models(profile_name: str) -> dict[str, Any]:
    """Load profile definition from config/fusion_profiles.yaml."""
    if not os.path.isfile(_PROFILES_YAML):
        return {}
    with open(_PROFILES_YAML, "r") as f:
        data = yaml.safe_load(f)
    profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    return profiles.get(profile_name, {})


def _trusted_models_from_profile(profile: dict[str, Any]) -> list[str]:
    """Extract the allowed model list from a profile definition."""
    return profile.get("allowed_models", [])


# ── Artifact validation ──────────────────────────────────────────────────────


def _file_hash(path: str) -> str:
    """SHA256 of first 8192 bytes + file size (fast fingerprint)."""
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(8192)
        h.update(chunk)
    size = os.path.getsize(path)
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def _csv_row_count(path: str) -> int:
    """Count data rows in a CSV (excludes header)."""
    if not os.path.isfile(path):
        return 0
    with open(path, "r") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def _artifact_valid(path: str, min_rows: int = 0) -> bool:
    """Check artifact exists, has content, and meets minimum row count."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 10:
        return False
    if min_rows > 0 and _csv_row_count(path) < min_rows:
        return False
    return True


def _ledger_path(work_dir: str, filename: str) -> str:
    """Build full path to a ledger file in work_dir/ledger/."""
    return os.path.join(work_dir, _LEDGER_DIR_NAME, filename)


# ── Step functions ───────────────────────────────────────────────────────────


def step_raw_data_check(
    raw_data: str,
    work_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Step 1: Validate raw CSV.

    CFG05_RAW_DATA_VALID → PASSED (data is valid)
    CFG05_RAW_DATA_MISSING → FAILED (data missing)
    Anything else → FAILED
    """
    result: dict[str, Any] = {"step": "raw_data_check", "status": "NOT_STARTED"}
    marker = os.path.join(work_dir, ".step_raw_data_ok")
    if not force and os.path.isfile(marker):
        result["status"] = "SKIPPED"
        return result

    if not os.path.isfile(raw_data):
        result["status"] = "FAILED"
        result["error"] = f"Raw data not found: {raw_data}"
        return result

    try:
        from scripts.check_cfg05_raw_data_contract import run_raw_data_contract
        cr = run_raw_data_contract(raw_data_path=raw_data)
        status = cr["summary"]["status"]
        if status == "CFG05_RAW_DATA_VALID":
            result["rows"] = cr.get("total_rows", 0)
            result["hash"] = _file_hash(raw_data)
            result["status"] = "PASSED"
        elif status == "CFG05_RAW_DATA_MISSING":
            result["status"] = "FAILED"
            result["error"] = f"Raw data check: {status}"
        else:
            result["status"] = "FAILED"
            result["error"] = f"Raw data check: {status}"
            return result
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as f:
            f.write(json.dumps(result))
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_source_repo_check(
    source_repo: str,
    work_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Step 2: Validate source repo exists and has expected structure."""
    result: dict[str, Any] = {"step": "source_repo_check", "status": "NOT_STARTED"}
    marker = os.path.join(work_dir, ".step_source_repo_ok")
    if not force and os.path.isfile(marker):
        result["status"] = "SKIPPED"
        return result

    if not os.path.isdir(os.path.join(source_repo, "models")):
        result["status"] = "FAILED"
        result["error"] = f"Source repo models dir not found: {source_repo}"
        return result

    result["status"] = "PASSED"
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(json.dumps(result))
    return result


def step_load_or_run_trust_gate(
    work_dir: str,
    force: bool,
    trusted_pool_override: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Step 3: Run P41 trust gate or load cached result.

    Returns the *full* profile definition in ``profile_def`` so downstream
    steps (postflight, sentinel) can inspect allowed_models, excluded_models,
    delivery_allowed, etc.
    """
    result: dict[str, Any] = {
        "step": "trust_gate",
        "status": "NOT_STARTED",
    }
    cache = os.path.join(work_dir, "trust_gate_result.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    try:
        if trusted_pool_override:
            result["trusted_models"] = trusted_pool_override
            result["quarantined_models"] = []
            result["status"] = "OVERRIDDEN"
        else:
            from scripts.run_p41_model_trust_gate import run_trust_gate
            gate = run_trust_gate(work_dir=work_dir)
            result["trusted_models"] = gate["summary"].get("trusted_models", [])
            result["quarantined_models"] = gate["summary"].get("suspect_models", [])
            result["status"] = gate["summary"]["p41_status"]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_load_or_run_actual_ledger(
    work_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Step 4: Generate actual ledger or validate existing."""
    result: dict[str, Any] = {"step": "actual_ledger", "status": "NOT_STARTED"}
    actual_path = _ledger_path(work_dir, _ACTUAL_LEDGER_FILENAME)
    cache = os.path.join(work_dir, ".step_actual_ledger_status.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    if _artifact_valid(actual_path, min_rows=24):
        result["rows"] = _csv_row_count(actual_path)
        result["hash"] = _file_hash(actual_path)
        result["status"] = "EXISTING"
    else:
        try:
            from scripts.run_p34_actual_ledger_alignment import run_actual_ledger_alignment
            al = run_actual_ledger_alignment(work_dir=work_dir)
            if al.get("status") == "P34_ACTUAL_LEDGER_READY":
                result["rows"] = _csv_row_count(actual_path)
                result["status"] = "GENERATED"
            else:
                result["status"] = "FAILED"
                result["error"] = f"Actual ledger status: {al.get('status')}"
        except Exception as e:
            result["status"] = "FAILED"
            result["error"] = str(e)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def step_load_or_run_prediction_ledger(
    work_dir: str,
    target_date: str,
    trusted_models: list[str],
    force: bool,
) -> dict[str, Any]:
    """Step 5: Ensure prediction ledger exists.

    Does NOT silently skip — if no prediction ledger is found and generation
    is not possible, the step FAILs explicitly.
    """
    result: dict[str, Any] = {"step": "prediction_ledger", "status": "NOT_STARTED"}
    pred_path = _ledger_path(work_dir, _PREDICTION_LEDGER_FILENAME)
    cache = os.path.join(work_dir, ".step_prediction_ledger_status.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    if _artifact_valid(pred_path, min_rows=24):
        result["rows"] = _csv_row_count(pred_path)
        result["hash"] = _file_hash(pred_path)
        result["status"] = "EXISTING"
    else:
        # Try to generate via P33
        try:
            from scripts.run_p33_prediction_ledger_backfill import run_prediction_ledger_backfill
            pl = run_prediction_ledger_backfill(
                work_dir=work_dir,
                target_date=target_date,
                trusted_models=trusted_models,
            )
            if pl.get("status") == "P33_PREDICTION_LEDGER_READY":
                result["rows"] = _csv_row_count(pred_path)
                result["status"] = "GENERATED"
            else:
                result["status"] = "FAILED"
                result["error"] = f"Prediction ledger status: {pl.get('status')}"
        except ImportError:
            result["status"] = "FAILED"
            result["error"] = "Prediction ledger not found and P33 runner unavailable"
        except Exception as e:
            result["status"] = "FAILED"
            result["error"] = str(e)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def step_safety_preflight(
    work_dir: str,
    trusted_models: list[str],
    force: bool,
    strict_no_leakage: bool = False,
    profile_name: str = "trusted_delivery",
) -> dict[str, Any]:
    """Step 6 (P53): Runtime leakage sentinel preflight check.

    Runs AFTER ledger generation so ledgers are guaranteed to exist.
    Sentinel returns ``{"models": [{model_name, status, ...}, ...]}``.
    """
    result: dict[str, Any] = {"step": "safety_preflight", "status": "NOT_STARTED"}
    cache = os.path.join(work_dir, ".step_safety_preflight.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    pred_path = _ledger_path(work_dir, _PREDICTION_LEDGER_FILENAME)
    actual_path = _ledger_path(work_dir, _ACTUAL_LEDGER_FILENAME)

    if not os.path.isfile(pred_path) or not os.path.isfile(actual_path):
        result["status"] = "FAILED"
        result["reason"] = (
            f"Ledger(s) missing: pred={os.path.isfile(pred_path)}, "
            f"actual={os.path.isfile(actual_path)}"
        )
        return result

    try:
        from safety.leakage_sentinel import run_leakage_sentinel, is_delivery_allowed
        sentinel = run_leakage_sentinel(
            trusted_models=trusted_models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=actual_path,
        )
        # sentinel returns {"models": [{model_name, status, ...}, ...]}
        model_list = sentinel.get("models", [])
        result["model_statuses"] = {
            item["model_name"]: item["status"]
            for item in model_list
        }
        result["blocked_models"] = [
            m for m, s in result["model_statuses"].items()
            if s in ("SUSPECT_LEAKAGE", "INVALID_SCHEMA", "INVALID_24H")
        ]
        # CONSERVATIVE_QUARANTINE: blocked only for trusted_delivery
        result["quarantined_models"] = [
            m for m, s in result["model_statuses"].items()
            if s == "CONSERVATIVE_QUARANTINE"
        ]
        if profile_name == "trusted_delivery":
            result["blocked_models"].extend(result["quarantined_models"])

        if strict_no_leakage and result["blocked_models"]:
            result["status"] = "FAILED"
            result["error"] = (
                f"Strict: blocked models={result['blocked_models']}"
            )
        elif result["blocked_models"]:
            result["status"] = "WARNING"
        else:
            result["status"] = "PASSED"

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_adaptive_training_days(
    work_dir: str,
    target_date: str,
    trusted_models: list[str],
    force: bool,
    required_days: int = 30,
    max_lookback_days: int = 180,
    min_days_for_degraded: int = 7,
    allow_degraded: bool = False,
) -> dict[str, Any]:
    """Step 7 (P52): Adaptive complete training day selector.

    Now correctly passes trusted_models and actual_ledger_path.
    """
    result: dict[str, Any] = {
        "step": "adaptive_training_days",
        "status": "NOT_STARTED",
    }
    cache = os.path.join(work_dir, ".step_adaptive_training_days.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    pred_path = _ledger_path(work_dir, _PREDICTION_LEDGER_FILENAME)
    actual_path = _ledger_path(work_dir, _ACTUAL_LEDGER_FILENAME)

    if not os.path.isfile(pred_path):
        result["status"] = "FAILED"
        result["reason"] = "Prediction ledger not found"
        return result

    try:
        from fusion.adaptive_training_days import select_complete_training_days
        td = select_complete_training_days(
            target_date=target_date,
            trusted_models=trusted_models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=actual_path,
            required_days=required_days,
            max_lookback_days=max_lookback_days,
            min_days_for_degraded=min_days_for_degraded,
        )
        result["training_days"] = td.get("selected_count", 0)
        result["days_status"] = td.get("status", "UNKNOWN")

        days_status = td.get("status", "")
        if days_status == "COMPLETE_30D":
            result["status"] = "PASSED"
        elif days_status == "DEGRADED_MIN_DAYS" and allow_degraded:
            result["status"] = "PASSED"
        elif days_status == "DEGRADED_MIN_DAYS":
            result["status"] = "WARNING"
        else:
            result["status"] = "FAILED"

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_trusted_fusion(
    work_dir: str,
    trusted_models: list[str],
    force: bool,
) -> dict[str, Any]:
    """Step 8a (P42): Run trusted fusion backtest (period_bgew)."""
    result: dict[str, Any] = {"step": "trusted_fusion", "status": "NOT_STARTED"}
    cache = os.path.join(work_dir, "trusted_fusion_result.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    try:
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        p42 = run_trusted_fusion_backtest(
            work_dir=work_dir,
            trusted_models=trusted_models,
        )
        result["metrics"] = p42.get("fusion_metrics", {})
        result["cfg05_metrics"] = p42.get("cfg05_metrics", {})
        result["best_single"] = {
            "model": p42.get("best_single_model"),
            "metrics": p42.get("best_single_metrics", {}),
        }
        result["equal_weight"] = p42.get("equal_weight_metrics", {})
        result["weights"] = p42.get("fusion_weights", {})
        result["fusion_vs_cfg05_delta"] = (
            p42.get("summary", {}).get("fusion_vs_cfg05_delta")
        )
        result["status"] = p42.get("summary", {}).get("p42_status", "FAILED")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_regime_bgew_fusion(
    work_dir: str,
    target_date: str,
    trusted_models: list[str],
    profile_name: str,
    force: bool,
) -> dict[str, Any]:
    """Step 8b (P56): Run regime-aware BGEW fusion.

    Dispatched when ``--fusion-engine regime_bgew``.
    """
    result: dict[str, Any] = {
        "step": "regime_bgew_fusion",
        "status": "NOT_STARTED",
    }
    cache = os.path.join(work_dir, "regime_bgew_fusion_result.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    pred_path = _ledger_path(work_dir, _PREDICTION_LEDGER_FILENAME)
    actual_path = _ledger_path(work_dir, _ACTUAL_LEDGER_FILENAME)

    if not os.path.isfile(pred_path) or not os.path.isfile(actual_path):
        result["status"] = "FAILED"
        result["reason"] = "Ledger(s) missing for regime_bgew fusion"
        return result

    try:
        from fusion.trust_gated_regime_bgew import run_trust_gated_regime_bgew
        rg = run_trust_gated_regime_bgew(
            target_date=target_date,
            trusted_models=trusted_models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=actual_path,
            profile_name=profile_name,
        )
        result["fusion_method"] = rg.get("fusion_method", "unknown")
        result["regime"] = rg.get("regime", "unknown")
        result["weights"] = rg.get("weights", {})
        result["training_days"] = rg.get("training_days", 0)
        result["output_rows"] = len(rg.get("output", pd.DataFrame()))

        # Persist fused output
        output_df = rg.get("output")
        if isinstance(output_df, pd.DataFrame) and len(output_df) > 0:
            out_path = os.path.join(work_dir, "final_output.csv")
            output_df.to_csv(out_path, index=False)
            result["output_file"] = out_path

        if rg.get("success"):
            result["status"] = "PASSED"
        else:
            result["status"] = "FAILED"
            result["error"] = rg.get("error", "regime_bgew returned no success")

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_rolling_validation(
    work_dir: str,
    trusted_models: list[str],
    force: bool,
) -> dict[str, Any]:
    """Step 8c (P43): Rolling weight validation (only for period_bgew)."""
    result: dict[str, Any] = {
        "step": "rolling_validation",
        "status": "NOT_STARTED",
    }
    cache = os.path.join(work_dir, "rolling_validation_result.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    try:
        from scripts.run_p43_rolling_weight_fusion_validation import (
            run_rolling_validation,
        )
        p43 = run_rolling_validation(
            work_dir=work_dir,
            trusted_models=trusted_models,
        )
        result["split"] = p43.get("split", {})
        result["rolling"] = p43.get("rolling", {}).get("metrics", {})
        result["full_period"] = p43.get("full_period", {})
        result["status"] = p43.get("summary", {}).get("p43_status", "FAILED")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_fallback_ladder(
    work_dir: str,
    target_date: str,
    trusted_models: list[str],
    raw_data_path: str,
    force: bool,
) -> dict[str, Any]:
    """Step 9 (P54): Run 6-level fallback ladder.

    Persists the output DataFrame to final_output.csv on success.
    """
    result: dict[str, Any] = {"step": "fallback_ladder", "status": "NOT_STARTED"}
    cache = os.path.join(work_dir, ".step_fallback_ladder.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    pred_path = _ledger_path(work_dir, _PREDICTION_LEDGER_FILENAME)
    actual_path = _ledger_path(work_dir, _ACTUAL_LEDGER_FILENAME)

    try:
        from delivery.fallback_ladder import run_fallback_ladder
        ladder = run_fallback_ladder(
            target_date=target_date,
            trusted_models=trusted_models,
            prediction_ledger_path=pred_path,
            actual_ledger_path=actual_path,
            raw_data_path=raw_data_path,
        )
        result["level_used"] = ladder.get("level_used", "unknown")
        result["fusion_method"] = ladder.get("fusion_method", "unknown")

        # Persist output DataFrame to CSV
        output_df = ladder.get("output")
        if ladder.get("success") and isinstance(output_df, pd.DataFrame):
            out_path = os.path.join(work_dir, "final_output.csv")
            output_df.to_csv(out_path, index=False)
            result["output_rows"] = len(output_df)
            result["output_columns"] = list(output_df.columns)
            result["output_file"] = out_path
        elif ladder.get("success"):
            result["output_rows"] = 0
            result["output_file"] = None

        result["delivery_status"] = ladder.get("delivery_status", "unknown")
        if ladder.get("success"):
            result["status"] = "PASSED"
        else:
            result["status"] = "FAILED"
            result["error"] = ladder.get(
                "error", "Fallback ladder returned no success"
            )

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_postflight_validation(
    work_dir: str,
    target_date: str,
    profile: str,
    profile_def: dict[str, Any] | None,
    force: bool,
) -> dict[str, Any]:
    """Step 10 (P55): Postflight checks on final_output.csv.

    Uses correct API: ``run_postflight(output_path=, target_date=, ...)``.
    """
    result: dict[str, Any] = {
        "step": "postflight_validation",
        "status": "NOT_STARTED",
    }
    cache = os.path.join(work_dir, ".step_postflight_validation.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    final_output = os.path.join(work_dir, "final_output.csv")
    if not os.path.isfile(final_output):
        result["status"] = "SKIPPED"
        result["reason"] = "final_output.csv not found"
        return result

    try:
        from delivery.postflight import run_postflight
        pf = run_postflight(
            output_path=final_output,
            target_date=target_date,
            profile_name=profile,
            profile_def=profile_def,
            work_dir=work_dir,
        )
        result["postflight_status"] = pf.get("status", "UNKNOWN")
        result["postflight_checks"] = pf.get("checks", {})
        result["postflight_passed"] = pf.get("status") == "PASS"

        # Write manifest and report
        manifest = _create_manifest(
            profile_name=profile,
            target_date=target_date,
            output_path=final_output,
            delivery_status=result.get("delivery_status", "unknown"),
            postflight_status=result["postflight_status"],
            profile_def=profile_def,
        )
        manifest_path = os.path.join(work_dir, "run_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        result["manifest_path"] = manifest_path

        try:
            from delivery.report import generate_delivery_report
            generate_delivery_report(
                manifest=manifest,
                output_dir=work_dir,
            )
        except Exception as report_err:
            result["report_warning"] = str(report_err)

        result["status"] = (
            "PASSED" if result["postflight_passed"] else "WARNING"
        )

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def _create_manifest(
    profile_name: str,
    target_date: str,
    output_path: str,
    delivery_status: str,
    postflight_status: str,
    profile_def: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal delivery manifest dict.

    Falls back to manual construction if ``delivery.manifest.create_manifest``
    is unavailable.
    """
    try:
        from delivery.manifest import create_manifest
        return create_manifest(
            profile_name=profile_name,
            output_path=output_path,
        )
    except Exception:
        pass
    return {
        "profile": profile_name,
        "target_date": target_date,
        "delivery_status": delivery_status,
        "postflight_status": postflight_status,
        "output_file": output_path,
    }


def step_delivery_summary(
    work_dir: str,
) -> dict[str, Any]:
    """Step 11 (P44): Generate delivery summary from previous step results."""
    result: dict[str, Any] = {
        "step": "delivery_summary",
        "status": "NOT_STARTED",
    }

    try:
        from scripts.run_p44_delivery_readiness_packager import (
            run_delivery_packager,
        )
        p44 = run_delivery_packager(work_dir=work_dir)
        result["delivery_summary"] = p44
        result["status"] = "PASSED"

        summary_path = os.path.join(work_dir, "delivery_summary.json")
        with open(summary_path, "w") as f:
            json.dump(p44, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_forbidden_file_check(work_dir: str) -> dict[str, Any]:
    """Step 12: Check no forbidden files in outputs."""
    result: dict[str, Any] = {
        "step": "forbidden_file_check",
        "status": "NOT_STARTED",
        "forbidden_found": [],
    }

    forbidden_patterns = ["y_true", ".pkl", ".joblib"]
    for fname in os.listdir(work_dir):
        fpath = os.path.join(work_dir, fname)
        if os.path.isfile(fpath) and any(
            p in fname.lower() for p in forbidden_patterns
        ):
            result["forbidden_found"].append(fname)

    result["status"] = (
        "PASSED" if not result["forbidden_found"] else "WARNING"
    )
    return result


def step_claim_guard_check() -> dict[str, Any]:
    """Step 13: Run claim guard on docs."""
    result: dict[str, Any] = {"step": "claim_guard", "status": "NOT_STARTED"}
    try:
        from scripts.validate_delivery_claims import run_claim_guard
        cg = run_claim_guard()
        result["violations"] = cg.get("violations", [])
        result["warnings"] = cg.get("warnings", [])
        result["status"] = "PASSED" if not cg["violations"] else "FAILED"
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


# ── Orchestrator ─────────────────────────────────────────────────────────────


def run_delivery_chain(
    raw_data: str = "",
    source_repo: str = "",
    profile: str = _DEFAULT_PROFILE,
    start_day: str = "2026-06-01",
    end_day: str = "2026-06-30",
    work_dir: Optional[str] = None,
    force: bool = False,
    fusion_engine: str = "period_bgew",
    required_training_days: int = 30,
    max_lookback_days: int = 180,
    min_days_for_degraded: int = 7,
    allow_degraded: bool = False,
    strict_no_leakage: bool = False,
) -> dict[str, Any]:
    """Run full delivery chain (P47/P61).

    Step order:
      1. raw_data_check
      2. source_repo_check
      3. trust_gate (loads profile, trusted_models, profile_def)
      4. actual_ledger
      5. prediction_ledger
      6. safety_preflight (ledgers now exist — Bug 3 fix)
      7. adaptive_training_days
      8. fusion dispatch (regime_bgew / period_bgew / equal_weight / cfg05)
      9. fallback_ladder (if fusion did not produce output)
     10. postflight_validation
     11. delivery_summary
     12. forbidden_file_check
     13. claim_guard
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    # Load profile
    profile_def = (
        _load_profile_models(profile)
        or _load_profile_models(_DEFAULT_PROFILE)
    )
    trusted_models = _trusted_models_from_profile(profile_def)
    delivery_allowed = profile_def.get("delivery_allowed", False)

    result: dict[str, Any] = {
        "phase": "P47/P61",
        "profile": profile,
        "profile_def": profile_def,
        "fusion_engine": fusion_engine,
        "profile_delivery_allowed": delivery_allowed,
        "trusted_models": trusted_models,
        "start_day": start_day,
        "end_day": end_day,
        "work_dir": work_dir,
        "steps": {},
        "step_order": [],
        "overall_status": "P47_NOT_STARTED",
        "metrics": {},
        "output_files": {},
        "errors": [],
        "p61_config": {
            "fusion_engine": fusion_engine,
            "required_training_days": required_training_days,
            "max_lookback_days": max_lookback_days,
            "min_days_for_degraded": min_days_for_degraded,
            "allow_degraded": allow_degraded,
            "strict_no_leakage": strict_no_leakage,
        },
    }

    # ── Build step list ────────────────────────────────────────────────
    steps: list[tuple[str, Any]] = [
        ("raw_data_check", lambda: step_raw_data_check(raw_data, work_dir, force)),
        ("source_repo_check", lambda: step_source_repo_check(source_repo, work_dir, force)),
        ("trust_gate", lambda: step_load_or_run_trust_gate(work_dir, force, trusted_models)),
        ("actual_ledger", lambda: step_load_or_run_actual_ledger(work_dir, force)),
    ]

    # Prediction ledger needs target_date from trust_gate result
    def _pred_step() -> dict[str, Any]:
        tg = result.get("steps", {}).get("trust_gate", {})
        tm = tg.get("trusted_models", trusted_models)
        return step_load_or_run_prediction_ledger(work_dir, start_day, tm, force)

    steps.append(("prediction_ledger", _pred_step))

    def _safety_step() -> dict[str, Any]:
        return step_safety_preflight(
            work_dir, trusted_models, force,
            strict_no_leakage=strict_no_leakage,
            profile_name=profile,
        )

    steps.append(("safety_preflight", _safety_step))

    steps.append((
        "adaptive_training_days",
        lambda: step_adaptive_training_days(
            work_dir, start_day, trusted_models, force,
            required_days=required_training_days,
            max_lookback_days=max_lookback_days,
            min_days_for_degraded=min_days_for_degraded,
            allow_degraded=allow_degraded,
        ),
    ))

    # ── Fusion dispatch (Bug 7 fix) ────────────────────────────────────
    has_fusion = len(trusted_models) >= 2

    if fusion_engine == "regime_bgew":
        steps.append((
            "regime_bgew_fusion",
            lambda: step_regime_bgew_fusion(
                work_dir, start_day, trusted_models, profile, force,
            ),
        ))
    elif fusion_engine == "period_bgew" and has_fusion:
        steps.append((
            "trusted_fusion",
            lambda: step_trusted_fusion(work_dir, trusted_models, force),
        ))
        steps.append((
            "rolling_validation",
            lambda: step_rolling_validation(work_dir, trusted_models, force),
        ))
    # equal_weight / cfg05 → skip fusion, fallback ladder handles it

    steps.append((
        "fallback_ladder",
        lambda: step_fallback_ladder(
            work_dir, start_day, trusted_models, raw_data, force,
        ),
    ))
    steps.append((
        "postflight_validation",
        lambda: step_postflight_validation(
            work_dir, start_day, profile, profile_def, force,
        ),
    ))
    steps.append((
        "delivery_summary",
        lambda: step_delivery_summary(work_dir),
    ))
    steps.append((
        "forbidden_file_check",
        lambda: step_forbidden_file_check(work_dir),
    ))
    steps.append((
        "claim_guard",
        lambda: step_claim_guard_check(),
    ))

    # ── Execute ────────────────────────────────────────────────────────
    all_passed = True
    for step_name, step_fn in steps:
        if not all_passed and step_name not in (
            "claim_guard", "forbidden_file_check",
        ):
            result["steps"][step_name] = {
                "status": "SKIPPED", "step": step_name,
            }
            result["step_order"].append(step_name)
            continue
        step_result = step_fn()
        result["steps"][step_name] = step_result
        result["step_order"].append(step_name)
        if step_result.get("status") in ("FAILED",):
            all_passed = False
            result["errors"].append(
                f"{step_name}: {step_result.get('error', 'unknown')}"
            )

    # ── Extract metrics ────────────────────────────────────────────────
    fusion_step = result["steps"].get("trusted_fusion", {})
    if fusion_step.get("metrics"):
        result["metrics"]["fusion_sMAPE"] = (
            fusion_step["metrics"].get("sMAPE_floor50")
        )
        result["metrics"]["cfg05_sMAPE"] = (
            fusion_step.get("cfg05_metrics", {}).get("sMAPE_floor50")
        )
        result["metrics"]["improvement_vs_cfg05"] = (
            fusion_step.get("fusion_vs_cfg05_delta")
        )

    rg_step = result["steps"].get("regime_bgew_fusion", {})
    if rg_step.get("fusion_method"):
        result["metrics"]["regime_fusion_method"] = rg_step["fusion_method"]
        result["metrics"]["regime"] = rg_step.get("regime")

    rol_step = result["steps"].get("rolling_validation", {})
    if rol_step.get("rolling"):
        result["metrics"]["rolling_fusion"] = (
            rol_step["rolling"].get("fusion_sMAPE")
        )
        result["metrics"]["rolling_cfg05"] = (
            rol_step["rolling"].get("cfg05_sMAPE")
        )

    ladder_step = result["steps"].get("fallback_ladder", {})
    if ladder_step.get("level_used"):
        result["metrics"]["fallback_level"] = ladder_step["level_used"]
        result["metrics"]["delivery_status"] = ladder_step.get(
            "delivery_status", "unknown"
        )

    # ── Collect output files ───────────────────────────────────────────
    for fname in (
        "delivery_summary.json",
        "final_output.csv",
        "metrics.json",
        "run_manifest.json",
        "delivery_report.md",
        "delivery_report.json",
    ):
        fpath = os.path.join(work_dir, fname)
        if os.path.isfile(fpath):
            result["output_files"][fname] = fpath

    metrics_path = os.path.join(work_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2, default=str)
    result["output_files"]["metrics.json"] = metrics_path

    # ── Overall status ─────────────────────────────────────────────────
    if all_passed:
        result["overall_status"] = "P47_DELIVERY_CHAIN_PASS"
    else:
        result["overall_status"] = "P47_DELIVERY_CHAIN_FAILED"

    return result


# ── Report ────────────────────────────────────────────────────────────────────


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P47/P61 — Delivery Local Chain + Safety Supervisor")
    print("=" * 60)
    print(f"  Profile:          {result['profile']}")
    print(f"  Fusion engine:    {result.get('fusion_engine', 'period_bgew')}")
    print(f"  Delivery allowed: {result['profile_delivery_allowed']}")
    print(f"  Trusted models:   {result['trusted_models']}")
    print(f"  Work dir:         {result['work_dir']}")
    p61 = result.get("p61_config", {})
    if p61:
        print(f"  Training days:    {p61.get('required_training_days')}")
        print(f"  Allow degraded:   {p61.get('allow_degraded')}")
        print(f"  Strict leakage:   {p61.get('strict_no_leakage')}")
    print()

    for step_name in result.get("step_order", []):
        step = result["steps"].get(step_name, {})
        status = step.get("status", "UNKNOWN")
        if status in ("PASSED", "CACHED", "EXISTING", "OVERRIDDEN", "SKIPPED"):
            symbol = "\u2705"
        elif status == "WARNING":
            symbol = "\u26a0\ufe0f"
        else:
            symbol = "\u274c"
        print(f"  {symbol} {step_name}: {status}")
        if step.get("error"):
            print(f"     error: {step['error']}")
        if step.get("level_used"):
            print(f"     level: {step['level_used']}")
        if step.get("training_days"):
            print(f"     days:  {step['training_days']}")
        if step.get("blocked_models"):
            print(f"     blocked: {step['blocked_models']}")
        if step.get("manifest_path"):
            print(f"     manifest: {step['manifest_path']}")
        if step.get("fusion_method"):
            print(f"     method: {step['fusion_method']}")
        if step.get("regime"):
            print(f"     regime: {step['regime']}")

    m = result.get("metrics", {})
    if m:
        print()
        print("── Metrics ──")
        for k, v in m.items():
            if v is not None:
                print(f"  {k}: {v}")

    of = result.get("output_files", {})
    if of:
        print()
        print("── Output Files ──")
        for name, path in of.items():
            print(f"  {name}: {path}")

    print()
    print(f"  Status: {result['overall_status']}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P47/P61: Delivery local chain + safety supervisor.",
    )
    parser.add_argument(
        "--raw-data", type=str, default="",
        help="Raw Chinese CSV path",
    )
    parser.add_argument(
        "--source-repo", type=str, default="",
        help="Source repo directory",
    )
    parser.add_argument(
        "--profile", type=str, default=_DEFAULT_PROFILE,
        help="Fusion profile name",
    )
    parser.add_argument(
        "--fusion-engine", type=str, default="period_bgew",
        choices=["regime_bgew", "period_bgew", "equal_weight", "cfg05"],
        help="Fusion engine (default: period_bgew)",
    )
    parser.add_argument(
        "--required-training-days", type=int, default=30,
        help="Required complete training days (default: 30)",
    )
    parser.add_argument(
        "--max-lookback-days", type=int, default=180,
        help="Max calendar days to scan (default: 180)",
    )
    parser.add_argument(
        "--min-days-for-degraded", type=int, default=7,
        help="Minimum days for degraded mode (default: 7)",
    )
    parser.add_argument(
        "--allow-degraded", action="store_true", default=False,
        help="Allow delivery with DEGRADED_MIN_DAYS status",
    )
    parser.add_argument(
        "--strict-no-leakage", action="store_true", default=False,
        help="Fail if ANY leakage check triggers",
    )
    parser.add_argument("--start-day", type=str, default="2026-06-01")
    parser.add_argument("--end-day", type=str, default="2026-06-30")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_delivery_chain(
        raw_data=args.raw_data,
        source_repo=args.source_repo,
        profile=args.profile,
        start_day=args.start_day,
        end_day=args.end_day,
        work_dir=args.work_dir,
        force=args.force,
        fusion_engine=args.fusion_engine,
        required_training_days=args.required_training_days,
        max_lookback_days=args.max_lookback_days,
        min_days_for_degraded=args.min_days_for_degraded,
        allow_degraded=args.allow_degraded,
        strict_no_leakage=args.strict_no_leakage,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and "PASS" not in result["overall_status"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
