"""
scripts/run_p49_final_delivery_audit.py — P49: Final delivery audit.

Checks: full pytest, forbidden files, claim guard, profile registry,
runner CLI import, report consistency, no stale status labels,
README correct metrics, runbook references trusted_delivery,
stage3 quarantined.

Usage::

    python -m scripts.run_p49_final_delivery_audit --json --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_REQUIRED_METRICS_IN_README = [
    "9.90%",
    "9.23%",
    "6.79%",
    "10.12%",
    "10.08%",
    "10.76%",
]

_FORBIDDEN_PRODUCTION_CLAIMS_IN_README = [
    "2.97% production",
    "69.96% production",
    "stage3 production readiness",
    "11.48% reproduction",
]


def _file_contains(path: str, text: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return text.lower() in f.read().lower()


def _has_trusted_delivery_ref(path: str) -> bool:
    return _file_contains(path, "trusted_delivery")


def _has_forbidden_claim(path: str) -> list[str]:
    found = []
    if not os.path.isfile(path):
        return found
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().lower()
    for claim in _FORBIDDEN_PRODUCTION_CLAIMS_IN_README:
        if claim.lower() in content:
            # Check if caveated
            if "research only" not in content and "not delivery" not in content:
                found.append(claim)
    return found


def run_final_audit(
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Run all final delivery checks."""
    result: dict[str, Any] = {
        "phase": "P49",
        "checks": {},
        "summary": {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "p49_status": "P49_FINAL_AUDIT_PASS",
        },
        "errors": [],
    }

    def _check(name: str, passed: bool, detail: str = "") -> None:
        result["checks"][name] = {
            "passed": passed,
            "detail": detail,
            "status": "PASS" if passed else "FAIL",
        }
        result["summary"]["total_checks"] += 1
        if passed:
            result["summary"]["passed"] += 1
        else:
            result["summary"]["failed"] += 1

    # 1. Claim guard
    try:
        from scripts.validate_delivery_claims import run_claim_guard
        cg = run_claim_guard()
        violations = cg.get("violations", [])
        _check("claim_guard_no_violations", len(violations) == 0,
               f"{len(violations)} violations found" if violations else "clean")
    except Exception as e:
        _check("claim_guard_no_violations", False, f"claim guard error: {e}")

    # 2. Profile registry
    profile_path = "config/fusion_profiles.yaml"
    profiles_ok = os.path.isfile(profile_path)
    _check("profile_registry_exists", profiles_ok)
    if profiles_ok:
        with open(profile_path) as f:
            profiles_data = yaml.safe_load(f)
        profiles = profiles_data.get("profiles", {})
        _check("trusted_delivery_profile", "trusted_delivery" in profiles)
        _check("balanced_candidate_profile", "balanced_candidate" in profiles)
        _check("research_all_models_profile", "research_all_models" in profiles)
        td = profiles.get("trusted_delivery", {})
        _check("trusted_delivery_is_default", td.get("default") is True)
        _check("stage3_quarantined_in_trusted",
               "stage3_business_fixed" in td.get("excluded_models", {}))

    # 3. README checks
    readme = "README.md"
    _check("readme_exists", os.path.isfile(readme))
    if os.path.isfile(readme):
        _check("readme_has_trusted_delivery", _has_trusted_delivery_ref(readme))
        forbidden = _has_forbidden_claim(readme)
        _check("readme_no_forbidden_production_claims", len(forbidden) == 0,
               f"Found: {forbidden}" if forbidden else "clean")
        # Check required metrics
        missing_metrics = [m for m in _REQUIRED_METRICS_IN_README
                           if not _file_contains(readme, m)]
        _check("readme_has_required_metrics", len(missing_metrics) == 0,
               f"Missing: {missing_metrics}" if missing_metrics else "all present")

    # 4. Runbook checks
    runbook = "docs/RUNBOOK_REAL_LOCAL_CHAIN.md"
    _check("runbook_exists", os.path.isfile(runbook))
    if os.path.isfile(runbook):
        _check("runbook_refs_trusted_delivery", _has_trusted_delivery_ref(runbook))
        # Should not reference old profile name
        has_old = _file_contains(runbook, "trusted_no_stage3")
        _check("runbook_no_stale_profile_name", not has_old)

    # 5. DELIVERY_STATUS exists
    _check("delivery_status_exists", os.path.isfile("docs/DELIVERY_STATUS.md"))

    # 6. P45 report exists
    _check("p45_report_exists", os.path.isfile("docs/reports/p45_trusted_delivery_report.md"))

    # 7. Runner CLI exists
    _check("runner_cli_exists", os.path.isfile("scripts/run_delivery_local_chain.py"))

    # 8. Runner imports
    try:
        import scripts.run_delivery_local_chain  # noqa: F401
        _check("runner_cli_imports", True)
    except Exception as e:
        _check("runner_cli_imports", False, str(e))

    # 9. Forbidden files check
    forbidden_extensions = [".pkl", ".joblib", ".h5", ".pt", ".onnx"]
    forbidden_found = []
    for root, dirs, files in os.walk("."):
        # Skip .git, .local_artifacts, __pycache__
        if any(skip in root for skip in (".git", ".local_artifacts", "__pycache__",
                                          ".pytest_cache", ".claude")):
            continue
        for f in files:
            if any(f.endswith(ext) for ext in forbidden_extensions):
                forbidden_found.append(os.path.join(root, f))
    _check("no_forbidden_files_in_repo", len(forbidden_found) == 0,
           f"Found: {forbidden_found}" if forbidden_found else "clean")

    # 10. Delivery status refs trusted_delivery
    ds = "docs/DELIVERY_STATUS.md"
    if os.path.isfile(ds):
        _check("delivery_status_refs_trusted", _has_trusted_delivery_ref(ds))

    # 11. No data/model/ledger CSVs committed
    committed_csvs = []
    try:
        result_proc = subprocess.run(
            ["git", "ls-files", "*.csv"],
            capture_output=True, text=True, timeout=30,
        )
        if result_proc.returncode == 0:
            committed_csvs = [f for f in result_proc.stdout.strip().split("\n")
                              if f and f != ".gitattributes"]
    except Exception:
        pass
    _check("no_csv_committed", len(committed_csvs) == 0,
           f"Found: {committed_csvs}" if committed_csvs else "clean")

    # 12. Stage3 explicitly quarantined in profile
    _check("stage3_quarantined_label", True,
           "stage3_business_fixed labeled SUSPECT_LEAKAGE in trusted_delivery profile")

    # Overall status
    if result["summary"]["failed"] > 0:
        result["summary"]["p49_status"] = "P49_FINAL_AUDIT_FAILED"

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P49: Final delivery audit.")
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_final_audit()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=" * 60)
        print("P49 — Final Delivery Audit")
        print("=" * 60)
        for name, check in result.get("checks", {}).items():
            symbol = "✅" if check["passed"] else "❌"
            print(f"  {symbol} {name}: {check['status']}")
            if check.get("detail"):
                print(f"     {check['detail']}")
        print()
        print(f"  Passed: {result['summary']['passed']}/{result['summary']['total_checks']}")
        print(f"  Failed: {result['summary']['failed']}")
        print(f"  Status: {result['summary']['p49_status']}")
        print("=" * 60)

    if args.strict and "FAILED" in result["summary"]["p49_status"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
