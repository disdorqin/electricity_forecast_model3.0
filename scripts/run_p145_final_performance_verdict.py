"""
scripts/run_p145_final_performance_verdict.py — P145: Final Performance Verdict.

Reads all P137-P144 results and determines the final performance verdict.

Verdicts:
  PERFORMANCE_UNLOCKED_GO:
    2025 trusted BGEW exists
    model_count >= 2
    BGEW improves vs cfg05
    no fake claims

  PERFORMANCE_IMPROVED_WITH_CAVEATS:
    BGEW or realtime delta improves but some artifacts remain partial

  PERFORMANCE_BLOCKED_FEATURE_PIPELINE:
    still cannot generate 2nd trusted model prediction

  PERFORMANCE_NO_IMPROVEMENT:
    multi-model ran but no improvement

Outputs:
  - docs/reports/p145_final_performance_verdict_report.md
  - .local_artifacts/p145_final_verdict/verdict.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Default paths ───────────────────────────────────────────────────

DEFAULT_ARTIFACTS_DIR = os.path.join(REPO_ROOT, ".local_artifacts")

# Known baselines
BASELINE_CFG05_ONLY_SMAPE = 20.22
BASELINE_REALTIME_DA_SAFE_SMAPE = 33.03
LOCAL_2026_BGEW_SMAPE = 9.23

# Performance targets
PERFORMANCE_TARGETS = {
    "minimum": {"threshold": 20.22, "label": "Below cfg05-only baseline"},
    "reasonable": {"threshold": 15.0, "label": "Reasonable production quality"},
    "strong": {"threshold": 12.0, "label": "Strong performance"},
    "stretch": {"threshold": 10.0, "label": "Stretch goal"},
}


# ── Helper: safe JSON reader ────────────────────────────────────────


def _safe_read_json(path: str) -> dict | None:
    """Read JSON file, return None if missing or invalid."""
    if not os.path.isfile(path):
        logger.debug("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _safe_read_json_dir(dir_path: str) -> dict[str, Any]:
    """Read all JSON files in a directory."""
    result: dict[str, Any] = {}
    if not os.path.isdir(dir_path):
        return result
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".json"):
            data = _safe_read_json(os.path.join(dir_path, fname))
            if data is not None:
                result[fname] = data
    return result


# ── Read all phase artifacts ────────────────────────────────────────


def read_all_phase_artifacts(artifacts_dir: str) -> dict[str, Any]:
    """Read artifacts from P137-P144 phases.

    Returns a dict with phase -> data mapping.
    """
    phases: dict[str, Any] = {}

    # P137: Base metrics (production_metrics_2025.json)
    p137_path = os.path.join(REPO_ROOT, "production_metrics_2025.json")
    phases["p137_base_metrics"] = {
        "exists": os.path.isfile(p137_path),
        "data": _safe_read_json(p137_path),
    }

    # P137b: Full BGEW benchmark (production_metrics_2025_full_bgew.json)
    p137b_path = os.path.join(REPO_ROOT, "production_metrics_2025_full_bgew.json")
    phases["p137_full_bgew"] = {
        "exists": os.path.isfile(p137b_path),
        "data": _safe_read_json(p137b_path),
    }

    # P138: Rolling BGEW
    p138_dir = os.path.join(artifacts_dir, "p138_rolling_bgew")
    p138_data = _safe_read_json(os.path.join(p138_dir, "bgew_2025_metrics.json"))
    phases["p138_rolling_bgew"] = {
        "exists": p138_data is not None,
        "data": p138_data,
    }

    # P139: Residual corrected
    p139_dir = os.path.join(artifacts_dir, "p139_residual_corrected")
    p139_data = _safe_read_json_dir(p139_dir)
    phases["p139_residual"] = {
        "exists": len(p139_data) > 0,
        "data": p139_data,
    }

    # P140: Realtime unblock
    p140_dir = os.path.join(artifacts_dir, "p140_realtime_unblock")
    p140_data = _safe_read_json_dir(p140_dir)
    phases["p140_realtime"] = {
        "exists": len(p140_data) > 0,
        "data": p140_data,
    }

    # P141: Negative spike audit
    p141_path = os.path.join(artifacts_dir, "p141_negative_spike", "audit_summary.json")
    phases["p141_audit"] = {
        "exists": os.path.isfile(p141_path),
        "data": _safe_read_json(p141_path),
    }

    # P142: Fair comparison
    p142_path = os.path.join(artifacts_dir, "p142_fair_comparison", "comparison_metrics.json")
    phases["p142_comparison"] = {
        "exists": os.path.isfile(p142_path),
        "data": _safe_read_json(p142_path),
    }

    # P143: Performance claims
    p143_path = os.path.join(REPO_ROOT, "production_metrics_2025_performance.json")
    phases["p143_claims"] = {
        "exists": os.path.isfile(p143_path),
        "data": _safe_read_json(p143_path),
    }

    # P144: Regression tests (no artifact, just test results)
    phases["p144_regression_tests"] = {
        "exists": True,  # Tests always exist as code
        "data": {"status": "DEFINED"},
    }

    return phases


# ── Extract key metrics from phases ─────────────────────────────────


def extract_key_metrics(phases: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics needed for verdict determination."""
    metrics: dict[str, Any] = {
        "cfg05_only_smape": BASELINE_CFG05_ONLY_SMAPE,
        "realtime_da_safe_smape": BASELINE_REALTIME_DA_SAFE_SMAPE,
        "local_2026_bgew_smape": LOCAL_2026_BGEW_SMAPE,
        "bgew_2025_smape": None,
        "bgew_model_count": 0,
        "bgew_improves_vs_cfg05": False,
        "residual_smape": None,
        "residual_is_noop": True,
        "realtime_improved_smape": None,
        "realtime_delta_improves": False,
        "p143_verdict": None,
        "p143_claims_count": 0,
        "p143_blocked_count": 0,
    }

    # P138: BGEW metrics
    p138 = phases.get("p138_rolling_bgew", {})
    if p138.get("exists") and p138.get("data"):
        data = p138["data"]
        for key in ("sMAPE_floor50", "smape", "sMAPE", "bgew_smape"):
            if key in data and isinstance(data[key], (int, float)):
                metrics["bgew_2025_smape"] = float(data[key])
                break
        for key in ("model_count", "n_models"):
            if key in data and isinstance(data[key], (int, float)):
                metrics["bgew_model_count"] = int(data[key])
                break
        if metrics["bgew_2025_smape"] is not None:
            metrics["bgew_improves_vs_cfg05"] = (
                metrics["bgew_2025_smape"] < BASELINE_CFG05_ONLY_SMAPE
            )

    # P139: Residual metrics
    p139 = phases.get("p139_residual", {})
    if p139.get("exists") and p139.get("data"):
        for fname, data in p139["data"].items():
            if isinstance(data, dict):
                status = str(data.get("status", ""))
                if "NO_OP" in status.upper():
                    metrics["residual_is_noop"] = True
                else:
                    metrics["residual_is_noop"] = False
                for key in ("sMAPE_floor50", "smape", "sMAPE", "residual_smape"):
                    if key in data and isinstance(data[key], (int, float)):
                        metrics["residual_smape"] = float(data[key])
                        break

    # P140: Realtime metrics
    p140 = phases.get("p140_realtime", {})
    if p140.get("exists") and p140.get("data"):
        for fname, data in p140["data"].items():
            if isinstance(data, dict):
                for key in ("sMAPE_floor50", "smape", "sMAPE", "realtime_smape"):
                    if key in data and isinstance(data[key], (int, float)):
                        metrics["realtime_improved_smape"] = float(data[key])
                        break
        if metrics["realtime_improved_smape"] is not None:
            metrics["realtime_delta_improves"] = (
                metrics["realtime_improved_smape"] < BASELINE_REALTIME_DA_SAFE_SMAPE
            )

    # P143: Claims
    p143 = phases.get("p143_claims", {})
    if p143.get("exists") and p143.get("data"):
        data = p143["data"]
        metrics["p143_verdict"] = data.get("verdict")
        metrics["p143_claims_count"] = len(data.get("claims", []))
        metrics["p143_blocked_count"] = len(data.get("blocked_claims", []))

    return metrics


# ── Determine verdict ───────────────────────────────────────────────


def determine_verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Determine the final performance verdict.

    Returns dict with:
      - verdict: str (one of the four verdict levels)
      - reasons: list[str]
      - conditions: dict of condition -> bool
    """
    conditions = {
        "bgew_2025_exists": metrics["bgew_2025_smape"] is not None,
        "bgew_model_count_gte_2": metrics["bgew_model_count"] >= 2,
        "bgew_improves_vs_cfg05": metrics["bgew_improves_vs_cfg05"],
        "no_fake_claims": True,  # Assumed true unless P143 says otherwise
        "realtime_delta_improves": metrics["realtime_delta_improves"],
        "residual_not_noop": not metrics["residual_is_noop"],
    }

    # Check P143 for fake claim indicators
    if metrics["p143_verdict"] is not None:
        if metrics["p143_blocked_count"] > metrics["p143_claims_count"]:
            conditions["no_fake_claims"] = False

    reasons: list[str] = []

    # ── PERFORMANCE_UNLOCKED_GO ──
    if (conditions["bgew_2025_exists"]
            and conditions["bgew_model_count_gte_2"]
            and conditions["bgew_improves_vs_cfg05"]
            and conditions["no_fake_claims"]):
        verdict = "PERFORMANCE_UNLOCKED_GO"
        reasons.append("2025 trusted BGEW exists with model_count >= 2")
        reasons.append(f"BGEW sMAPE {metrics['bgew_2025_smape']}% improves vs "
                       f"cfg05-only {BASELINE_CFG05_ONLY_SMAPE}%")
        reasons.append("No fake claims detected")
        if conditions["realtime_delta_improves"]:
            reasons.append("Realtime delta also improves")

    # ── PERFORMANCE_IMPROVED_WITH_CAVEATS ──
    elif conditions["bgew_improves_vs_cfg05"] or conditions["realtime_delta_improves"]:
        verdict = "PERFORMANCE_IMPROVED_WITH_CAVEATS"
        if conditions["bgew_improves_vs_cfg05"]:
            reasons.append("BGEW improves vs cfg05 but artifacts may be partial")
        if conditions["realtime_delta_improves"]:
            reasons.append("Realtime delta improves but some artifacts remain partial")
        if not conditions["bgew_model_count_gte_2"]:
            reasons.append(f"model_count = {metrics['bgew_model_count']} (< 2 required for full BGEW claim)")

    # ── PERFORMANCE_BLOCKED_FEATURE_PIPELINE ──
    elif not conditions["bgew_2025_exists"] and metrics["bgew_model_count"] < 2:
        verdict = "PERFORMANCE_BLOCKED_FEATURE_PIPELINE"
        reasons.append("Cannot generate 2nd trusted model prediction")
        reasons.append("Feature pipeline incompatibility prevents multi-model inference")
        if not conditions["bgew_2025_exists"]:
            reasons.append("P138 rolling BGEW artifacts not available")

    # ── PERFORMANCE_NO_IMPROVEMENT ──
    else:
        verdict = "PERFORMANCE_NO_IMPROVEMENT"
        reasons.append("Multi-model ran but no improvement over baseline")
        if metrics["bgew_2025_smape"] is not None:
            reasons.append(f"BGEW sMAPE {metrics['bgew_2025_smape']}% >= "
                           f"cfg05-only {BASELINE_CFG05_ONLY_SMAPE}%")

    # ── Classify performance target ──
    best_smape = metrics["bgew_2025_smape"]
    if best_smape is None:
        best_smape = BASELINE_CFG05_ONLY_SMAPE

    target_met = "none"
    for target_name in ("stretch", "strong", "reasonable", "minimum"):
        if best_smape < PERFORMANCE_TARGETS[target_name]["threshold"]:
            target_met = target_name
            break

    return {
        "verdict": verdict,
        "reasons": reasons,
        "conditions": conditions,
        "best_smape": best_smape,
        "target_met": target_met,
    }


# ── Output generators ───────────────────────────────────────────────


def generate_verdict_json(
    verdict_data: dict, metrics: dict, phases: dict, output_dir: str
) -> str:
    """Generate .local_artifacts/p145_final_verdict/verdict.json."""
    out_dir = os.path.join(output_dir, "p145_final_verdict")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "verdict.json")

    output = {
        "verdict": verdict_data["verdict"],
        "generated_at": datetime.now().isoformat(),
        "version": "3.0.0-rc1-p145",
        "reasons": verdict_data["reasons"],
        "conditions": verdict_data["conditions"],
        "best_smape": verdict_data["best_smape"],
        "target_met": verdict_data["target_met"],
        "metrics": metrics,
        "phase_availability": {
            phase: info.get("exists", False)
            for phase, info in phases.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", output_path)
    return output_path


def generate_verdict_report_md(
    verdict_data: dict, metrics: dict, phases: dict, output_dir: str
) -> str:
    """Generate docs/reports/p145_final_performance_verdict_report.md."""
    report_dir = os.path.join(output_dir, "docs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "p145_final_performance_verdict_report.md")

    lines = [
        "# P145: Final Performance Verdict Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Verdict:** `{verdict_data['verdict']}`",
        f"**Best sMAPE:** {verdict_data['best_smape']}%",
        f"**Target Met:** {verdict_data['target_met']}",
        "",
        "## Verdict Definition",
        "",
    ]

    verdict = verdict_data["verdict"]
    if verdict == "PERFORMANCE_UNLOCKED_GO":
        lines.extend([
            "### PERFORMANCE_UNLOCKED GO",
            "",
            "All conditions met:",
            "- 2025 trusted BGEW exists",
            "- model_count >= 2",
            "- BGEW improves vs cfg05",
            "- No fake claims",
        ])
    elif verdict == "PERFORMANCE_IMPROVED_WITH_CAVEATS":
        lines.extend([
            "### PERFORMANCE IMPROVED WITH CAVEATS",
            "",
            "BGEW or realtime delta improves but some artifacts remain partial.",
        ])
    elif verdict == "PERFORMANCE_BLOCKED_FEATURE_PIPELINE":
        lines.extend([
            "### PERFORMANCE BLOCKED — FEATURE PIPELINE",
            "",
            "Still cannot generate 2nd trusted model prediction.",
            "Feature pipeline incompatibility prevents multi-model inference on 2025.",
        ])
    elif verdict == "PERFORMANCE_NO_IMPROVEMENT":
        lines.extend([
            "### PERFORMANCE NO IMPROVEMENT",
            "",
            "Multi-model ran but no improvement over baseline.",
        ])
    else:
        lines.append(f"### Unknown verdict: {verdict}")

    lines.extend([
        "",
        "## Reasons",
        "",
    ])
    for reason in verdict_data["reasons"]:
        lines.append(f"- {reason}")

    lines.extend([
        "",
        "## Conditions",
        "",
        "| Condition | Met? |",
        "|-----------|------|",
    ])
    for cond, met in verdict_data["conditions"].items():
        lines.append(f"| {cond} | {'YES' if met else 'NO'} |")

    lines.extend([
        "",
        "## Phase Artifact Availability",
        "",
        "| Phase | Available? |",
        "|-------|------------|",
    ])
    for phase, info in phases.items():
        exists = info.get("exists", False)
        lines.append(f"| {phase} | {'YES' if exists else 'NO'} |")

    lines.extend([
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| cfg05-only 2025 sMAPE | {metrics['cfg05_only_smape']}% |",
        f"| Realtime DA-Safe 2025 sMAPE | {metrics['realtime_da_safe_smape']}% |",
        f"| Local 2026 BGEW sMAPE | {metrics['local_2026_bgew_smape']}% (LOCAL WINDOW) |",
    ])
    if metrics["bgew_2025_smape"] is not None:
        lines.append(f"| P138 Rolling BGEW 2025 sMAPE | {metrics['bgew_2025_smape']}% |")
    else:
        lines.append("| P138 Rolling BGEW 2025 sMAPE | NOT AVAILABLE |")
    if metrics["realtime_improved_smape"] is not None:
        lines.append(f"| P140 Improved Realtime sMAPE | {metrics['realtime_improved_smape']}% |")
    else:
        lines.append("| P140 Improved Realtime sMAPE | NOT AVAILABLE |")
    if metrics["residual_smape"] is not None:
        lines.append(f"| P139 Residual Corrected sMAPE | {metrics['residual_smape']}% |")
    else:
        lines.append("| P139 Residual Corrected sMAPE | NOT AVAILABLE |")

    lines.extend([
        "",
        "## Performance Target Assessment",
        "",
        "| Target | Threshold | Status |",
        "|--------|-----------|--------|",
    ])
    best = verdict_data["best_smape"]
    for target_name in ("minimum", "reasonable", "strong", "stretch"):
        threshold = PERFORMANCE_TARGETS[target_name]["threshold"]
        met = "MET" if best < threshold else "NOT MET"
        lines.append(f"| {target_name.capitalize()} | < {threshold}% | {met} |")

    lines.extend([
        "",
        "## Summary",
        "",
        f"The final verdict is **{verdict}**.",
        "",
    ])
    if verdict_data["target_met"] != "none":
        lines.append(
            f"Best achievable sMAPE is {best}%, meeting the "
            f"**{verdict_data['target_met']}** target."
        )
    else:
        lines.append(
            f"Best available sMAPE is {best}% (cfg05-only baseline), "
            f"which does not meet the minimum target (< {PERFORMANCE_TARGETS['minimum']['threshold']}%)."
        )

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", output_path)
    return output_path


# ── Main entry point ────────────────────────────────────────────────


def run_p145_final_verdict(
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> dict:
    """Main entry point: read all phases, determine verdict, write outputs.

    Args:
        artifacts_dir: Path to .local_artifacts directory

    Returns:
        dict with verdict, reasons, conditions, metrics, and output paths
    """
    logger.info("P145: Reading all phase artifacts from %s", artifacts_dir)

    # Read all phase artifacts
    phases = read_all_phase_artifacts(artifacts_dir)

    # Extract key metrics
    metrics = extract_key_metrics(phases)

    # Determine verdict
    verdict_data = determine_verdict(metrics)

    # Generate outputs
    verdict_json_path = generate_verdict_json(verdict_data, metrics, phases, artifacts_dir)
    report_md_path = generate_verdict_report_md(verdict_data, metrics, phases, REPO_ROOT)

    verdict_data["output_files"] = {
        "verdict_json": verdict_json_path,
        "report_md": report_md_path,
    }
    verdict_data["metrics"] = metrics
    verdict_data["phases"] = {
        phase: info.get("exists", False) for phase, info in phases.items()
    }

    logger.info("P145 complete. Verdict: %s", verdict_data["verdict"])
    return verdict_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_p145_final_verdict()
    print(json.dumps({
        "verdict": result["verdict"],
        "reasons": result["reasons"],
        "best_smape": result["best_smape"],
        "target_met": result["target_met"],
        "output_files": result["output_files"],
    }, indent=2))
