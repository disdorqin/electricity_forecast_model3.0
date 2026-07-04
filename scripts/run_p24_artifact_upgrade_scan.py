"""
scripts/run_p24_artifact_upgrade_scan.py — P24 upgradeable artifact scan.

Scans the local filesystem for trained model artifacts that could upgrade
the current 3.0 pipeline from placeholder/fallback to REAL implementations.

Scans for:
  - P5M: residual / risk / canonical correction packs (.pkl, .joblib)
  - ExtremPriceClf: negative-price classifier model files
  - BGEW: actual_ledger.csv for weight-learner training

IMPORTANT: Only *trained* model artifacts (.pkl, .joblib, .pt, .txt model
files) count as REAL.  Source code, README, or placeholder files are NOT
counted as real artifacts.

Usage::

    python -m scripts.run_p24_artifact_upgrade_scan --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p21_p25_real_chain")

_P5M_SCAN_DIRS = [
    os.path.join("..", "electricity_forecast_model2.0_exp"),
]

_EXTREMPRICE_SCAN_DIRS = [
    os.path.join("..", "electricity_forecast_model2.1", "ExtremPriceClf"),
]

# Extensions that indicate a *trained* model artifact
_TRAINED_MODEL_EXTENSIONS = (".pkl", ".joblib", ".pt", ".pth")
# .txt files that are model files (e.g. LightGBM saved models)
_MODEL_TXT_PATTERNS = ("model", "lgbm", "lightgbm", "booster")

_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "reports/local/")
_ALLOWED_WORK_DIRS = (".local_artifacts",)

# ── Final statuses ─────────────────────────────────────────────────────────
P24_UPGRADE_ARTIFACTS_FOUND = "P24_UPGRADE_ARTIFACTS_FOUND"
P24_UPGRADE_PARTIAL = "P24_UPGRADE_PARTIAL"
P24_NO_REAL_ARTIFACTS_FOUND = "P24_NO_REAL_ARTIFACTS_FOUND"

# ── Component statuses ─────────────────────────────────────────────────────
ARTIFACT_REAL = "REAL_TRAINED_MODEL_FOUND"
ARTIFACT_MISSING = "NO_REAL_ARTIFACT_FOUND"
ARTIFACT_CODE_ONLY = "CODE_ONLY_NO_TRAINED_MODEL"


# ── Helpers ────────────────────────────────────────────────────────────────

def _path_is_safe(path: str) -> bool:
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def _is_trained_model_file(filepath: str) -> bool:
    """Return True if the file is a trained model artifact (not source code).

    Only .pkl, .joblib, .pt, .pth, or model .txt files count.
    Code files (.py, .md, .yaml, .json, .csv, etc.) do NOT count.
    """
    if not os.path.isfile(filepath):
        return False
    basename = os.path.basename(filepath).lower()
    _, ext = os.path.splitext(basename)

    # Check trained model extensions
    if ext in _TRAINED_MODEL_EXTENSIONS:
        return True

    # Check .txt files that look like model files (e.g. cfg05_model.txt)
    if ext == ".txt":
        return any(pat in basename for pat in _MODEL_TXT_PATTERNS)

    return False


def _scan_directory_for_models(scan_dir: str, extra_extensions: Optional[set[str]] = None) -> list[str]:
    """Recursively scan a directory for trained model files.

    Returns a list of absolute paths to trained model artifacts found.
    """
    found: list[str] = []
    if not os.path.isdir(scan_dir):
        return found

    for root, _dirs, files in os.walk(scan_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if _is_trained_model_file(fpath):
                found.append(os.path.abspath(fpath))

    return found


def scan_p5m_artifacts(
    extra_scan_dirs: Optional[list[str]] = None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Scan for P5M residual/risk/canonical model artifacts.

    Parameters
    ----------
    extra_scan_dirs : list[str], optional
        Additional directories to scan.
    include_defaults : bool
        If True (default), also scan built-in default directories.
        Set to False in tests to isolate from real filesystem state.
    """
    scan_dirs: list[str] = list(extra_scan_dirs) if extra_scan_dirs else []
    if include_defaults:
        scan_dirs.extend(_P5M_SCAN_DIRS)

    all_found: list[str] = []
    dirs_scanned: list[str] = []

    for scan_dir in scan_dirs:
        if os.path.isdir(scan_dir):
            dirs_scanned.append(os.path.abspath(scan_dir))
            found = _scan_directory_for_models(scan_dir)
            all_found.extend(found)

    # Deduplicate
    all_found = sorted(set(all_found))

    if all_found:
        status = ARTIFACT_REAL
    else:
        status = ARTIFACT_MISSING

    return {
        "status": status,
        "dirs_scanned": dirs_scanned,
        "artifacts_found": all_found,
        "n_artifacts": len(all_found),
    }


def scan_extrempriceclf_artifacts(
    extra_scan_dirs: Optional[list[str]] = None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Scan for ExtremPriceClf negative-price classifier model artifacts.

    Parameters
    ----------
    extra_scan_dirs : list[str], optional
        Additional directories to scan.
    include_defaults : bool
        If True (default), also scan built-in default directories.
        Set to False in tests to isolate from real filesystem state.
    """
    scan_dirs: list[str] = list(extra_scan_dirs) if extra_scan_dirs else []
    if include_defaults:
        scan_dirs.extend(_EXTREMPRICE_SCAN_DIRS)

    all_found: list[str] = []
    dirs_scanned: list[str] = []

    for scan_dir in scan_dirs:
        if os.path.isdir(scan_dir):
            dirs_scanned.append(os.path.abspath(scan_dir))
            found = _scan_directory_for_models(scan_dir)
            all_found.extend(found)

    all_found = sorted(set(all_found))

    if all_found:
        status = ARTIFACT_REAL
    elif dirs_scanned:
        # Directory exists but no trained models — could be code-only
        status = ARTIFACT_CODE_ONLY
    else:
        status = ARTIFACT_MISSING

    return {
        "status": status,
        "dirs_scanned": dirs_scanned,
        "artifacts_found": all_found,
        "n_artifacts": len(all_found),
    }


def scan_actual_ledger(work_dir: str) -> dict[str, Any]:
    """Check if actual_ledger.csv exists in work_dir/ledgers/."""
    ledger_dir = os.path.join(work_dir, "ledgers")
    ledger_path = os.path.join(ledger_dir, "actual_ledger.csv")

    if os.path.isfile(ledger_path):
        return {
            "status": ARTIFACT_REAL,
            "ledger_path": os.path.abspath(ledger_path),
            "exists": True,
        }
    return {
        "status": ARTIFACT_MISSING,
        "ledger_path": ledger_path,
        "exists": False,
    }


def run_p24_artifact_upgrade_scan(
    work_dir: Optional[str] = None,
    extra_p5m_dirs: Optional[list[str]] = None,
    extra_extremprice_dirs: Optional[list[str]] = None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Run P24 artifact upgrade scan.

    Parameters
    ----------
    work_dir : str, optional
        Working directory (default: .local_artifacts/p21_p25_real_chain).
    extra_p5m_dirs : list[str], optional
        Additional directories to scan for P5M artifacts.
    extra_extremprice_dirs : list[str], optional
        Additional directories to scan for ExtremPriceClf artifacts.
    include_defaults : bool
        If True (default), also scan built-in default directories.
        Set to False in tests to isolate from real filesystem state.

    Returns a summary dict with ``final_status``.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    # ── Scan all components ──
    p5m = scan_p5m_artifacts(extra_p5m_dirs, include_defaults=include_defaults)
    extremprice = scan_extrempriceclf_artifacts(extra_extremprice_dirs, include_defaults=include_defaults)
    actual_ledger = scan_actual_ledger(work_dir)

    # ── BGEW training status ──
    # BGEW requires actual_ledger to exist; if it does, BGEW can be trained
    if actual_ledger["exists"]:
        bgew_status = "BGEW_TRAINING_FEASIBLE"
    else:
        bgew_status = "BGEW_TRAINING_BLOCKED_NO_ACTUAL_LEDGER"

    # ── Upgrade recommendations ──
    recommendations: list[str] = []
    real_count = 0

    if p5m["status"] == ARTIFACT_REAL:
        real_count += 1
        recommendations.append(
            f"UPGRADE_P5M: {p5m['n_artifacts']} trained model(s) found — "
            f"can upgrade residual correction from NO_OP to REAL"
        )
    else:
        recommendations.append("P5M_NO_UPGRADE: no trained residual models found")

    if extremprice["status"] == ARTIFACT_REAL:
        real_count += 1
        recommendations.append(
            f"UPGRADE_EXTREMPRICECLF: {extremprice['n_artifacts']} model(s) found — "
            f"can upgrade negative classifier from RULE_FALLBACK to REAL"
        )
    elif extremprice["status"] == ARTIFACT_CODE_ONLY:
        recommendations.append(
            "EXTREMPRICECLF_CODE_ONLY: source directory exists but no trained model — "
            "training required before upgrade"
        )
    else:
        recommendations.append("EXTREMPRICECLF_NO_UPGRADE: source directory not found")

    if actual_ledger["exists"]:
        real_count += 1
        recommendations.append("UPGRADE_BGEW: actual_ledger.csv found — weight learner training feasible")
    else:
        recommendations.append("BGEW_NO_UPGRADE: actual_ledger.csv not found")

    # ── Final status ──
    if real_count >= 2:
        final_status = P24_UPGRADE_ARTIFACTS_FOUND
    elif real_count == 1:
        final_status = P24_UPGRADE_PARTIAL
    else:
        final_status = P24_NO_REAL_ARTIFACTS_FOUND

    result: dict[str, Any] = {
        "p5m_pack_status": p5m["status"],
        "p5m_artifacts_found": p5m["artifacts_found"],
        "p5m_n_artifacts": p5m["n_artifacts"],
        "extrempriceclf_status": extremprice["status"],
        "extrempriceclf_artifacts_found": extremprice["artifacts_found"],
        "extrempriceclf_n_artifacts": extremprice["n_artifacts"],
        "actual_ledger_status": actual_ledger["status"],
        "actual_ledger_path": actual_ledger.get("ledger_path"),
        "actual_ledger_exists": actual_ledger["exists"],
        "bgew_training_status": bgew_status,
        "upgrade_recommendations": recommendations,
        "real_artifact_count": real_count,
        "final_status": final_status,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # Reason codes
    if p5m["status"] == ARTIFACT_REAL:
        result["reason_codes"].append(f"P5M_REAL:{p5m['n_artifacts']}_artifacts")
    if extremprice["status"] == ARTIFACT_REAL:
        result["reason_codes"].append(f"EXTREMPRICE_REAL:{extremprice['n_artifacts']}_artifacts")
    if actual_ledger["exists"]:
        result["reason_codes"].append("ACTUAL_LEDGER_EXISTS")

    # Forbidden files check
    work_dir_norm = os.path.abspath(work_dir).replace("\\", "/")
    work_dir_is_safe = (
        any(a.lstrip(".") in work_dir_norm for a in _ALLOWED_WORK_DIRS)
        or os.path.isabs(work_dir)
    )
    if not work_dir_is_safe:
        result["forbidden_files_check"] = "FAIL"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P24 Artifact Upgrade Scan Report")
    print("=" * 60)
    print(f"  P5M status:           {result['p5m_pack_status']}")
    print(f"  P5M artifacts:        {result['p5m_n_artifacts']}")
    print(f"  ExtremPriceClf status:{result['extrempriceclf_status']}")
    print(f"  ExtremPriceClf arts:  {result['extrempriceclf_n_artifacts']}")
    print(f"  Actual ledger:        {result['actual_ledger_status']}")
    print(f"  BGEW training:        {result['bgew_training_status']}")
    print(f"  Real artifact count:  {result['real_artifact_count']}")
    print(f"  Final status:         {result['final_status']}")
    print(f"  Forbidden chk:        {result['forbidden_files_check']}")
    print()
    for rec in result.get("upgrade_recommendations", []):
        print(f"    * {rec}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P24: artifact upgrade scan.")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--json", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    work_dir = args.work_dir or _DEFAULT_WORK_DIR

    result = run_p24_artifact_upgrade_scan(work_dir=work_dir)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
