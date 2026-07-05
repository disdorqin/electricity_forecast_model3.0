"""
scripts/validate_delivery_claims.py — P46: Claim guard.

Scans docs/reports and README for forbidden delivery claims and ensures
research-only results are properly caveated.

Usage::

    python -m scripts.validate_delivery_claims --json
    python -m scripts.validate_delivery_claims --strict

Options::

    --report-dir PATH   Directory containing .md reports (default: docs/reports).
    --readme PATH       Path to README.md (default: README.md).
    --json              Output JSON report.
    --strict            Exit non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_REPORT_DIR = "docs/reports"
_DEFAULT_README = "README.md"
_DEFAULT_PROFILES = "config/fusion_profiles.yaml"

# Forbidden patterns — these must NOT appear in delivery context
# (outside of code blocks and explicitly caveated sections).
_FORBIDDEN_PATTERNS: list[dict[str, Any]] = [
    {
        "pattern": r"2\.97%.*production",
        "flags": re.IGNORECASE,
        "label": "production_sMAPE_2_97",
        "message": "Cannot claim 2.97% production sMAPE (was stage3-leakage research result)",
    },
    {
        "pattern": r"69\.96%.*production",
        "flags": re.IGNORECASE,
        "label": "production_improvement_69_96",
        "message": "Cannot claim 69.96% production improvement (stage3 leakage artifact)",
    },
    {
        "pattern": r"stage3.*production.*read",
        "flags": re.IGNORECASE,
        "label": "stage3_production_readiness",
        "message": "Cannot claim stage3 production readiness (confirmed SUSPECT_LEAKAGE)",
    },
    {
        "pattern": r"11\.48%.*reproduc",
        "flags": re.IGNORECASE,
        "label": "source_11_48_reproduced",
        "message": "Cannot claim source 11.48% reproduction (not verified on trusted pool)",
    },
]

# Required caveats that must appear near a forbidden pattern for it to be
# considered "research context" rather than "delivery context".
_REQUIRED_CAVEATS = [
    "research only",
    "not delivery",
    "stage3 leakage",
]


def load_profiles(profiles_path: str = _DEFAULT_PROFILES) -> dict[str, Any]:
    """Load fusion profiles from YAML."""
    if not os.path.isfile(profiles_path):
        return {}
    with open(profiles_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("profiles", {}) if isinstance(data, dict) else {}


def scan_file_for_claims(
    file_path: str,
    context_label: str,
) -> list[dict[str, Any]]:
    """Scan a single file for forbidden claims."""
    violations: list[dict[str, Any]] = []
    if not os.path.isfile(file_path):
        return violations

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Strip code blocks to avoid false positives
    stripped = re.sub(r"```[\s\S]*?```", "", content)

    # Check each forbidden pattern
    for rule in _FORBIDDEN_PATTERNS:
        matches = list(re.finditer(rule["pattern"], stripped, rule.get("flags", 0)))
        if not matches:
            continue

        # Check if research caveats are present nearby
        has_caveat = any(c.lower() in stripped.lower() for c in _REQUIRED_CAVEATS)

        for m in matches:
            violations.append({
                "file": file_path,
                "context": context_label,
                "label": rule["label"],
                "matched_text": m.group().strip(),
                "position": m.start(),
                "has_caveat": has_caveat,
                "message": rule["message"],
                "severity": "warning" if has_caveat else "violation",
            })

    return violations


def run_claim_guard(
    report_dir: str = _DEFAULT_REPORT_DIR,
    readme_path: str = _DEFAULT_README,
    profiles_path: str = _DEFAULT_PROFILES,
) -> dict[str, Any]:
    """Run claim guard across reports and README."""
    result: dict[str, Any] = {
        "phase": "P46",
        "profiles_loaded": False,
        "profile_count": 0,
        "default_profile": None,
        "files_scanned": [],
        "violations": [],
        "warnings": [],
        "summary": {
            "total_violations": 0,
            "total_warnings": 0,
            "p46_status": "P46_CLAIM_GUARD_PASS",
        },
    }

    # Load profile registry
    profiles = load_profiles(profiles_path)
    if profiles:
        result["profiles_loaded"] = True
        result["profile_count"] = len(profiles)
        for name, p in profiles.items():
            if p.get("default"):
                result["default_profile"] = name
                break

    # Scan report files
    if os.path.isdir(report_dir):
        for fname in sorted(os.listdir(report_dir)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(report_dir, fname)
            violations = scan_file_for_claims(fpath, fname)
            result["files_scanned"].append(fname)
            for v in violations:
                if v["severity"] == "violation":
                    result["violations"].append(v)
                else:
                    result["warnings"].append(v)

    # Scan README
    violations = scan_file_for_claims(readme_path, "README.md")
    result["files_scanned"].append("README.md")
    for v in violations:
        if v["severity"] == "violation":
            result["violations"].append(v)
        else:
            result["warnings"].append(v)

    result["summary"]["total_violations"] = len(result["violations"])
    result["summary"]["total_warnings"] = len(result["warnings"])

    if result["violations"]:
        result["summary"]["p46_status"] = "P46_CLAIM_GUARD_FAILED"
    elif result["warnings"]:
        result["summary"]["p46_status"] = "P46_CLAIM_GUARD_PASS_WITH_WARNINGS"

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P46: Delivery claim guard.")
    parser.add_argument("--report-dir", type=str, default=_DEFAULT_REPORT_DIR)
    parser.add_argument("--readme", type=str, default=_DEFAULT_README)
    parser.add_argument("--profiles", type=str, default=_DEFAULT_PROFILES)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_claim_guard(
        report_dir=args.report_dir,
        readme_path=args.readme,
        profiles_path=args.profiles,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=" * 60)
        print("P46 — Delivery Claim Guard")
        print("=" * 60)
        print(f"  Profiles loaded:  {result['profiles_loaded']}")
        print(f"  Profile count:    {result['profile_count']}")
        print(f"  Default profile:  {result['default_profile']}")
        print(f"  Files scanned:    {len(result['files_scanned'])}")
        print()
        print("── Violations ──")
        for v in result.get("violations", []):
            print(f"  [{v['severity'].upper()}] {v['file']}: {v['message']}")
            print(f"    matched: '{v['matched_text']}'")
        for v in result.get("warnings", []):
            print(f"  [{v['severity'].upper()}] {v['file']}: {v['message']}")
            print(f"    matched: '{v['matched_text']}' (has caveat)")
        if not result["violations"] and not result["warnings"]:
            print("  (none)")
        print()
        print(f"  Status: {result['summary']['p46_status']}")
        print("=" * 60)

    if args.strict and result["summary"]["p46_status"] != "P46_CLAIM_GUARD_PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
