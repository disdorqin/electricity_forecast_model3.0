"""
delivery/report.py — P55: Generate delivery reports in markdown and JSON formats.

Produces a human-readable markdown report and a machine-readable JSON report
from a delivery manifest, plus a colored terminal summary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

_REPORT_MD_FILENAME = "delivery_report.md"
_REPORT_JSON_FILENAME = "delivery_report.json"


def _build_report_data(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build structured report data from a manifest.

    Parameters
    ----------
    manifest : dict
        Delivery manifest dict.

    Returns
    -------
    dict
        Structured report data with computed summary fields.
    """
    postflight = manifest.get("postflight", {})
    postflight_summary = postflight.get("summary", {})
    metrics = manifest.get("metrics", {})
    fallback = manifest.get("fallback", {})

    # Postflight check pass/fail breakdown
    checks = postflight.get("checks", {})
    passed_checks = sum(1 for c in checks.values() if c.get("passed"))
    failed_checks = len(checks) - passed_checks

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": manifest.get("run_id", "N/A"),
        "target_day": manifest.get("target_day", "N/A"),
        "profile": manifest.get("profile", "N/A"),
        "status": manifest.get("status", "UNKNOWN"),
        "delivery_status": manifest.get("delivery_status", "UNKNOWN"),
        "selected_training_days": manifest.get("selected_training_days", 0),
        "trusted_models": manifest.get("trusted_models", []),
        "quarantined_models": manifest.get("quarantined_models", []),
        "fusion_method": manifest.get("fusion_method", "N/A"),
        "fallback_used": fallback.get("fallback_used", False),
        "fallback_method": fallback.get("fallback_method", ""),
        "postflight_status": postflight.get("status", "N/A"),
        "postflight_checks_total": postflight_summary.get("total", 0),
        "postflight_checks_passed": postflight_summary.get("passed", 0),
        "postflight_checks_failed": postflight_summary.get("failed", 0),
        "postflight_errors": postflight.get("errors", []),
        "postflight_checks": checks,
        "metrics": metrics,
        "warnings": manifest.get("warnings", []),
        "errors": manifest.get("errors", []),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
    }


def _report_markdown(report_data: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Generate markdown report string from report data.

    Parameters
    ----------
    report_data : dict
        Structured report data from ``_build_report_data``.
    manifest : dict
        Original manifest dict (for output paths).

    Returns
    -------
    str
        Markdown-formatted report.
    """
    p = report_data
    lines: list[str] = []

    # Header
    lines.append("# P55 Delivery Report")
    lines.append("")
    lines.append(f"> **Generated**: {p['generated_at']}")
    lines.append(f"> **Run ID**: {p['run_id']}")
    lines.append(f"> **Target Day**: {p['target_day']}")
    lines.append("")

    # Delivery Status
    lines.append("## Delivery Status")
    lines.append("")
    status_icon = "PASS" if p["status"] == "PASS" else "FAIL"
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Status** | {status_icon} |")
    lines.append(f"| **Delivery Status** | {p['delivery_status']} |")
    lines.append(f"| **Profile** | {p['profile']} |")
    lines.append(f"| **Fusion Method** | {p['fusion_method']} |")
    lines.append("")

    # Fallback info
    fallback_icon = "Yes" if p["fallback_used"] else "No"
    lines.append(f"| **Fallback Used** | {fallback_icon} |")
    if p["fallback_used"] and p["fallback_method"]:
        lines.append(f"| **Fallback Method** | {p['fallback_method']} |")
    lines.append("")

    # Training Days
    lines.append("## Training Summary")
    lines.append("")
    lines.append(f"- **Selected Training Days**: {p['selected_training_days']}")
    lines.append("")

    # Models
    lines.append("## Model Pool")
    lines.append("")
    trusted = p["trusted_models"]
    quarantined = p["quarantined_models"]
    if trusted:
        lines.append("### Trusted Models")
        for m in trusted:
            lines.append(f"- `{m}`")
    lines.append("")
    if quarantined:
        lines.append("### Quarantined Models")
        for m in quarantined:
            lines.append(f"- `{m}`")
        lines.append("")

    # Postflight Results
    lines.append("## Postflight Results")
    lines.append("")
    lines.append(f"**Overall**: {p['postflight_status']} "
                 f"({p['postflight_checks_passed']}/{p['postflight_checks_total']} checks passed)")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|-------|--------|--------|")
    for check_name, check_result in p["postflight_checks"].items():
        icon = "PASS" if check_result.get("passed") else "FAIL"
        detail = check_result.get("detail", "")
        lines.append(f"| `{check_name}` | {icon} | {detail} |")
    lines.append("")

    if p["postflight_errors"]:
        lines.append("### Postflight Errors")
        for err in p["postflight_errors"]:
            lines.append(f"- {err}")
        lines.append("")

    # Metrics
    lines.append("## Metrics")
    lines.append("")
    if p["metrics"]:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in p["metrics"].items():
            lines.append(f"| `{key}` | {value} |")
    else:
        lines.append("No metrics recorded.")
    lines.append("")

    # Warnings and Errors
    if p["warnings"]:
        lines.append("## Warnings")
        for w in p["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    if p["errors"]:
        lines.append("## Errors")
        for e in p["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    # Output Files
    lines.append("## Output Files")
    lines.append("")
    output_path = manifest.get("postflight", {}).get("output_path", "")
    submission_path = manifest.get("postflight", {}).get("submission_ready_path", "")
    if output_path:
        lines.append(f"- **Final Output**: `{output_path}`")
    if submission_path:
        lines.append(f"- **Submission Ready**: `{submission_path}`")
    if output_path or submission_path:
        lines.append("")

    return "\n".join(lines)


def generate_delivery_report(
    manifest: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Write ``delivery_report.md`` and ``delivery_report.json`` to output_dir.

    Parameters
    ----------
    manifest : dict
        Delivery manifest dict.
    output_dir : str
        Directory to write report files into.

    Returns
    -------
    dict
        Dict with keys ``"markdown_path"`` and ``"json_path"`` pointing to
        the written files.
    """
    os.makedirs(output_dir, exist_ok=True)

    report_data = _build_report_data(manifest)

    # Write markdown report
    md_content = _report_markdown(report_data, manifest)
    md_path = os.path.join(output_dir, _REPORT_MD_FILENAME)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Write JSON report
    json_path = os.path.join(output_dir, _REPORT_JSON_FILENAME)
    json_report = {
        "generated_at": report_data["generated_at"],
        "manifest": manifest,
        "report_data": report_data,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)

    return {
        "markdown_path": os.path.abspath(md_path),
        "json_path": os.path.abspath(json_path),
    }


def print_terminal_report(manifest: dict[str, Any]) -> None:
    """Print a colored terminal delivery report summary.

    Parameters
    ----------
    manifest : dict
        Delivery manifest dict.
    """
    p = _build_report_data(manifest)

    sep = "=" * 60
    print(sep)
    print("P55 — Delivery Report")
    print(sep)
    print(f"  Run ID:             {p['run_id']}")
    print(f"  Target Day:         {p['target_day']}")
    print(f"  Profile:            {p['profile']}")
    print(f"  Status:             {p['status']}")
    print(f"  Delivery Status:    {p['delivery_status']}")
    print()
    print(f"── Model Pool ──")
    print(f"  Trusted models:     {len(p['trusted_models'])}")
    for m in p["trusted_models"]:
        print(f"    - {m}")
    print(f"  Quarantined models: {len(p['quarantined_models'])}")
    for m in p["quarantined_models"]:
        print(f"    - {m}")
    print()
    print(f"── Fusion ──")
    print(f"  Method:             {p['fusion_method']}")
    print(f"  Fallback used:      {'Yes' if p['fallback_used'] else 'No'}")
    if p["fallback_used"] and p["fallback_method"]:
        print(f"  Fallback method:    {p['fallback_method']}")
    print(f"  Training days:      {p['selected_training_days']}")
    print()
    print(f"── Postflight ──")
    print(f"  Status:             {p['postflight_status']}")
    print(f"  Passed:             {p['postflight_checks_passed']}/{p['postflight_checks_total']}")
    print(f"  Failed:             {p['postflight_checks_failed']}")
    if p["postflight_errors"]:
        for err in p["postflight_errors"]:
            print(f"  Error:              {err}")
    print()
    print(f"── Metrics ──")
    if p["metrics"]:
        for key, value in p["metrics"].items():
            print(f"  {key}: {value}")
    else:
        print("  (no metrics)")
    print()
    print(f"── Issues ──")
    if p["warnings"]:
        for w in p["warnings"]:
            print(f"  Warning: {w}")
    if p["errors"]:
        for e in p["errors"]:
            print(f"  Error:   {e}")
    if not p["warnings"] and not p["errors"]:
        print("  (none)")
    print(sep)
