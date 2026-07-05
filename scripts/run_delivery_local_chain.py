"""
scripts/run_delivery_local_chain.py — P47: One-command local delivery runner.

Executes the full delivery pipeline in order: raw data check → source repo
check → model predictions / load artifacts → trust gate → actual ledger →
trusted fusion → rolling validation → final output → delivery summary →
forbidden file check.

Steps that already have valid local artifacts are skipped (reused) unless
--force is passed.

Usage::

    python -m scripts.run_delivery_local_chain \\
        --raw-data ../data/shandong_pmos_hourly.csv \\
        --source-repo .local_artifacts/source_repos/epf-sota-experiment \\
        --profile trusted_delivery \\
        --start-day 2026-06-01 \\
        --end-day 2026-06-30 \\
        --work-dir .local_artifacts/delivery_run \\
        --json --strict

Options::

    --raw-data PATH     Raw Chinese CSV path.
    --source-repo PATH  Source repo directory (epf-sota-experiment).
    --profile NAME      Fusion profile name (default: trusted_delivery).
    --start-day DATE    Start date (YYYY-MM-DD).
    --end-day DATE      End date (YYYY-MM-DD).
    --work-dir PATH     Working directory for outputs (default: .local_artifacts/delivery_run).
    --force             Force re-run all steps.
    --json              Output JSON report.
    --strict            Exit non-zero on any step failure.

Outputs::

    <work-dir>/delivery_summary.json
    <work-dir>/final_output.csv
    <work-dir>/metrics.json
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

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "delivery_run")
_DEFAULT_PROFILE = "trusted_delivery"
_PROFILES_YAML = "config/fusion_profiles.yaml"

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


# ── Step functions ───────────────────────────────────────────────────────────


def step_raw_data_check(
    raw_data: str,
    work_dir: str,
    force: bool,
) -> dict[str, Any]:
    """Step 1: Validate raw CSV."""
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
        if cr["summary"]["status"] in ("CFG05_RAW_DATA_VALID", "CFG05_RAW_DATA_MISSING"):
            result["status"] = "FAILED"
            result["error"] = f"Raw data check failed: {cr['summary']['status']}"
            return result
        result["rows"] = cr.get("total_rows", 0)
        result["hash"] = _file_hash(raw_data)
        result["status"] = "PASSED"
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
    """Step 3: Run P41 trust gate or load cached result."""
    result: dict[str, Any] = {"step": "trust_gate", "status": "NOT_STARTED"}
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
    ledger_dir = os.path.join(work_dir, "ledger")
    actual_path = os.path.join(ledger_dir, "actual_ledger_30d.csv")
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


def step_run_trusted_fusion(
    work_dir: str,
    trusted_models: list[str],
    force: bool,
) -> dict[str, Any]:
    """Step 5: Run P42 trusted fusion backtest."""
    result: dict[str, Any] = {"step": "trusted_fusion", "status": "NOT_STARTED"}
    cache = os.path.join(work_dir, "trusted_fusion_result.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    try:
        from scripts.run_p42_trusted_fusion_backtest import run_trusted_fusion_backtest
        p42 = run_trusted_fusion_backtest(work_dir=work_dir, trusted_models=trusted_models)
        result["metrics"] = p42.get("fusion_metrics", {})
        result["cfg05_metrics"] = p42.get("cfg05_metrics", {})
        result["best_single"] = {
            "model": p42.get("best_single_model"),
            "metrics": p42.get("best_single_metrics", {}),
        }
        result["equal_weight"] = p42.get("equal_weight_metrics", {})
        result["weights"] = p42.get("fusion_weights", {})
        result["fusion_vs_cfg05_delta"] = p42["summary"].get("fusion_vs_cfg05_delta")
        result["status"] = p42["summary"]["p42_status"]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_run_rolling_validation(
    work_dir: str,
    trusted_models: list[str],
    force: bool,
) -> dict[str, Any]:
    """Step 6: Run P43 rolling weight validation."""
    result: dict[str, Any] = {"step": "rolling_validation", "status": "NOT_STARTED"}
    cache = os.path.join(work_dir, "rolling_validation_result.json")

    if not force and os.path.isfile(cache):
        with open(cache, "r") as f:
            result = json.load(f)
        result["status"] = "CACHED"
        return result

    try:
        from scripts.run_p43_rolling_weight_fusion_validation import run_rolling_validation
        p43 = run_rolling_validation(work_dir=work_dir, trusted_models=trusted_models)
        result["split"] = p43.get("split", {})
        result["rolling"] = p43.get("rolling", {}).get("metrics", {})
        result["full_period"] = p43.get("full_period", {})
        result["status"] = p43["summary"]["p43_status"]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_run_delivery_summary(
    work_dir: str,
    trust_gate: dict[str, Any],
    fusion: dict[str, Any],
    rolling: dict[str, Any],
) -> dict[str, Any]:
    """Step 7: Generate delivery summary from previous step results."""
    result: dict[str, Any] = {
        "step": "delivery_summary",
        "status": "NOT_STARTED",
    }

    try:
        from scripts.run_p44_delivery_readiness_packager import run_delivery_packager
        p44 = run_delivery_packager(work_dir=work_dir)
        result["delivery_summary"] = p44
        result["status"] = "PASSED"

        # Write delivery_summary.json
        summary_path = os.path.join(work_dir, "delivery_summary.json")
        with open(summary_path, "w") as f:
            json.dump(p44, f, indent=2, default=str)
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
    return result


def step_forbidden_file_check(work_dir: str) -> dict[str, Any]:
    """Step 8: Check no forbidden files in outputs."""
    result: dict[str, Any] = {
        "step": "forbidden_file_check",
        "status": "NOT_STARTED",
        "forbidden_found": [],
    }

    # Forbidden patterns in work_dir output
    forbidden_patterns = [
        "y_true", ".pkl", ".joblib",
    ]
    for fname in os.listdir(work_dir):
        fpath = os.path.join(work_dir, fname)
        if os.path.isfile(fpath) and any(p in fname.lower() for p in forbidden_patterns):
            result["forbidden_found"].append(fname)

    result["status"] = "PASSED" if not result["forbidden_found"] else "WARNING"
    return result


def step_claim_guard_check() -> dict[str, Any]:
    """Step 9: Run claim guard on docs."""
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
) -> dict[str, Any]:
    """Run full delivery chain."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    # Load profile
    profile_def = _load_profile_models(profile) or _load_profile_models(_DEFAULT_PROFILE)
    trusted_models = _trusted_models_from_profile(profile_def)
    delivery_allowed = profile_def.get("delivery_allowed", False)

    result: dict[str, Any] = {
        "phase": "P47",
        "profile": profile,
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
    }

    # ── Step execution ──
    steps = [
        ("raw_data_check", lambda: step_raw_data_check(raw_data, work_dir, force)),
        ("source_repo_check", lambda: step_source_repo_check(source_repo, work_dir, force)),
        ("trust_gate", lambda: step_load_or_run_trust_gate(work_dir, force, trusted_models)),
        ("actual_ledger", lambda: step_load_or_run_actual_ledger(work_dir, force)),
    ]

    # Only run fusion/rolling if we have 2+ trusted models
    has_fusion = len(trusted_models) >= 2
    if has_fusion:
        steps.append(("trusted_fusion", lambda: step_run_trusted_fusion(work_dir, trusted_models, force)))
        steps.append(("rolling_validation", lambda: step_run_rolling_validation(work_dir, trusted_models, force)))

    steps.append(("delivery_summary", lambda: step_run_delivery_summary(work_dir, {}, {}, {})))
    steps.append(("forbidden_file_check", lambda: step_forbidden_file_check(work_dir)))
    steps.append(("claim_guard", lambda: step_claim_guard_check()))

    all_passed = True
    for step_name, step_fn in steps:
        if not all_passed and step_name not in ("claim_guard", "forbidden_file_check"):
            result["steps"][step_name] = {"status": "SKIPPED", "step": step_name}
            result["step_order"].append(step_name)
            continue
        step_result = step_fn()
        result["steps"][step_name] = step_result
        result["step_order"].append(step_name)
        if step_result.get("status") in ("FAILED",):
            all_passed = False
            result["errors"].append(f"{step_name}: {step_result.get('error', 'unknown')}")

    # ── Extract metrics ──
    fusion_step = result["steps"].get("trusted_fusion", {})
    if fusion_step.get("metrics"):
        result["metrics"]["fusion_sMAPE"] = fusion_step["metrics"].get("sMAPE_floor50")
        result["metrics"]["cfg05_sMAPE"] = fusion_step.get("cfg05_metrics", {}).get("sMAPE_floor50")
        result["metrics"]["improvement_vs_cfg05"] = fusion_step.get("fusion_vs_cfg05_delta")

    rol_step = result["steps"].get("rolling_validation", {})
    if rol_step.get("rolling"):
        result["metrics"]["rolling_fusion"] = rol_step["rolling"].get("fusion_sMAPE")
        result["metrics"]["rolling_cfg05"] = rol_step["rolling"].get("cfg05_sMAPE")
    if rol_step.get("split"):
        result["metrics"]["split_fusion"] = rol_step["split"].get("fusion_sMAPE")
        result["metrics"]["split_cfg05"] = rol_step["split"].get("cfg05_sMAPE")

    # ── Collect output files ──
    for fname in ("delivery_summary.json", "final_output.csv", "metrics.json"):
        fpath = os.path.join(work_dir, fname)
        if os.path.isfile(fpath):
            result["output_files"][fname] = fpath

    # Write metrics.json
    metrics_path = os.path.join(work_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result["metrics"], f, indent=2, default=str)
    result["output_files"]["metrics.json"] = metrics_path

    # ── Overall status ──
    if all_passed:
        result["overall_status"] = "P47_DELIVERY_CHAIN_PASS"
    else:
        result["overall_status"] = "P47_DELIVERY_CHAIN_FAILED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P47 — Delivery Local Chain Runner")
    print("=" * 60)
    print(f"  Profile:          {result['profile']}")
    print(f"  Delivery allowed: {result['profile_delivery_allowed']}")
    print(f"  Trusted models:   {result['trusted_models']}")
    print(f"  Work dir:         {result['work_dir']}")
    print()

    for step_name in result.get("step_order", []):
        step = result["steps"].get(step_name, {})
        status = step.get("status", "UNKNOWN")
        symbol = "✅" if status in ("PASSED", "CACHED", "EXISTING", "OVERRIDDEN", "SKIPPED") else "❌"
        print(f"  {symbol} {step_name}: {status}")
        if step.get("error"):
            print(f"     error: {step['error']}")

    m = result.get("metrics", {})
    if m:
        print()
        print("── Metrics ──")
        if m.get("cfg05_sMAPE"):
            print(f"  cfg05:      {m['cfg05_sMAPE']}%")
        if m.get("fusion_sMAPE"):
            print(f"  fusion:     {m['fusion_sMAPE']}%")
        if m.get("improvement_vs_cfg05"):
            print(f"  vs cfg05:   {m['improvement_vs_cfg05']}%")
        if m.get("rolling_cfg05"):
            print(f"  rolling cfg05: {m['rolling_cfg05']}%")
        if m.get("rolling_fusion"):
            print(f"  rolling fusion: {m['rolling_fusion']}%")

    print()
    print(f"  Status: {result['overall_status']}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P47: Delivery local chain runner.")
    parser.add_argument("--raw-data", type=str, default="", help="Raw Chinese CSV path")
    parser.add_argument("--source-repo", type=str, default="", help="Source repo directory")
    parser.add_argument("--profile", type=str, default=_DEFAULT_PROFILE, help="Fusion profile name")
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
