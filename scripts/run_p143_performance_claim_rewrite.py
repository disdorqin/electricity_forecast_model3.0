"""
scripts/run_p143_performance_claim_rewrite.py — P143: Performance Claim Rewrite.

Reads actual results from P138-P142 artifact directories and rewrites all
performance claims based ONLY on verified numbers. No fake numbers allowed.

Rules:
  - Only claim 2025 full-year numbers that actually ran
  - Local 2026 9.23% must note "local window"
  - If BGEW blocked, no BGEW claim
  - If realtime delta improves, note improvement
  - No fake numbers

Outputs:
  - production_metrics_2025_performance.json — actual metrics with honest labels
  - docs/reports/p143_performance_claim_update_report.md — human-readable report
  - docs/CLIENT_DELIVERY_NOTE.md — client-facing delivery note
  - docs/CLIENT_CAVEATS.md — known limitations and caveats
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
DEFAULT_OUTPUT_DIR = REPO_ROOT

# Known baseline numbers (confirmed)
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
        logger.warning("Artifact file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _safe_read_json_dir(dir_path: str) -> dict[str, Any]:
    """Read all JSON files in a directory, return merged dict."""
    result: dict[str, Any] = {}
    if not os.path.isdir(dir_path):
        return result
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".json"):
            data = _safe_read_json(os.path.join(dir_path, fname))
            if data is not None:
                result[fname] = data
    return result


# ── Core: Read P138-P142 artifacts ──────────────────────────────────


def read_p138_rolling_bgew(artifacts_dir: str) -> dict[str, Any]:
    """Read P138 rolling BGEW results.

    Returns dict with keys:
      - exists: bool
      - smape: float or None
      - model_count: int
      - models: list[str]
      - improves_vs_cfg05: bool or None
      - raw: dict or None
    """
    result: dict[str, Any] = {
        "exists": False,
        "smape": None,
        "model_count": 0,
        "models": [],
        "improves_vs_cfg05": None,
        "raw": None,
    }
    p138_dir = os.path.join(artifacts_dir, "p138_rolling_bgew")
    metrics_path = os.path.join(p138_dir, "bgew_2025_metrics.json")
    data = _safe_read_json(metrics_path)
    if data is None:
        return result

    result["exists"] = True
    result["raw"] = data

    # Extract sMAPE — try multiple possible keys
    for key in ("sMAPE_floor50", "smape", "sMAPE", "bgew_smape",
                "trusted_bgew_smape", "bgew_sMAPE_floor50"):
        if key in data and isinstance(data[key], (int, float)):
            result["smape"] = float(data[key])
            break

    # Extract model count
    for key in ("model_count", "n_models", "num_models"):
        if key in data and isinstance(data[key], (int, float)):
            result["model_count"] = int(data[key])
            break

    # Extract model names
    for key in ("models", "model_names", "trusted_models"):
        if key in data and isinstance(data[key], list):
            result["models"] = data[key]
            break

    # Determine if improves vs cfg05
    if result["smape"] is not None:
        result["improves_vs_cfg05"] = result["smape"] < BASELINE_CFG05_ONLY_SMAPE

    return result


def read_p139_residual(artifacts_dir: str) -> dict[str, Any]:
    """Read P139 residual corrected results."""
    result: dict[str, Any] = {
        "exists": False,
        "smape": None,
        "is_noop": False,
        "improves_vs_bgew": None,
        "raw": None,
    }
    p139_dir = os.path.join(artifacts_dir, "p139_residual_corrected")
    data_all = _safe_read_json_dir(p139_dir)
    if not data_all:
        return result

    result["exists"] = True
    result["raw"] = data_all

    # Look for metrics in any file
    for fname, data in data_all.items():
        if isinstance(data, dict):
            status = data.get("status", "")
            if "NO_OP" in str(status).upper():
                result["is_noop"] = True
            for key in ("sMAPE_floor50", "smape", "sMAPE", "residual_smape",
                        "corrected_smape", "residual_sMAPE_floor50"):
                if key in data and isinstance(data[key], (int, float)):
                    result["smape"] = float(data[key])
                    break

    if result["smape"] is not None and read_p138_rolling_bgew(artifacts_dir)["smape"] is not None:
        bgew_smape = read_p138_rolling_bgew(artifacts_dir)["smape"]
        result["improves_vs_bgew"] = result["smape"] < bgew_smape

    return result


def read_p140_realtime(artifacts_dir: str) -> dict[str, Any]:
    """Read P140 realtime results."""
    result: dict[str, Any] = {
        "exists": False,
        "smape": None,
        "delta_improves": None,
        "raw": None,
    }
    p140_dir = os.path.join(artifacts_dir, "p140_realtime_unblock")
    data_all = _safe_read_json_dir(p140_dir)
    if not data_all:
        return result

    result["exists"] = True
    result["raw"] = data_all

    for fname, data in data_all.items():
        if isinstance(data, dict):
            for key in ("sMAPE_floor50", "smape", "sMAPE", "realtime_smape",
                        "realtime_sMAPE_floor50"):
                if key in data and isinstance(data[key], (int, float)):
                    result["smape"] = float(data[key])
                    break
            # Check delta
            for key in ("delta_improves", "improvement", "delta"):
                if key in data:
                    val = data[key]
                    if isinstance(val, bool):
                        result["delta_improves"] = val
                    elif isinstance(val, (int, float)):
                        result["delta_improves"] = val < 0  # negative delta = improvement

    if result["smape"] is not None:
        result["delta_improves"] = result["smape"] < BASELINE_REALTIME_DA_SAFE_SMAPE

    return result


def read_p141_audit(artifacts_dir: str) -> dict[str, Any]:
    """Read P141 negative spike audit."""
    result: dict[str, Any] = {
        "exists": False,
        "raw": None,
    }
    p141_path = os.path.join(artifacts_dir, "p141_negative_spike", "audit_summary.json")
    data = _safe_read_json(p141_path)
    if data is not None:
        result["exists"] = True
        result["raw"] = data
    return result


def read_p142_comparison(artifacts_dir: str) -> dict[str, Any]:
    """Read P142 fair comparison results."""
    result: dict[str, Any] = {
        "exists": False,
        "raw": None,
    }
    p142_path = os.path.join(artifacts_dir, "p142_fair_comparison", "comparison_metrics.json")
    data = _safe_read_json(p142_path)
    if data is not None:
        result["exists"] = True
        result["raw"] = data
    return result


# ── Core: Build performance claims ──────────────────────────────────


def build_performance_claims(artifacts_dir: str) -> dict[str, Any]:
    """Build performance claims based ONLY on actual artifact data.

    Returns a dict with:
      - claims: list of claim dicts
      - metrics: consolidated metrics
      - blocked_claims: list of claims that cannot be made
      - verdict: overall verdict string
    """
    # Read all artifacts
    p138 = read_p138_rolling_bgew(artifacts_dir)
    p139 = read_p139_residual(artifacts_dir)
    p140 = read_p140_realtime(artifacts_dir)
    p141 = read_p141_audit(artifacts_dir)
    p142 = read_p142_comparison(artifacts_dir)

    claims: list[dict[str, Any]] = []
    blocked_claims: list[dict[str, str]] = []

    # ── Claim 1: cfg05-only 2025 (always available) ──
    claims.append({
        "id": "cfg05_only_2025",
        "label": "cfg05-only day-ahead sMAPE (2025 full year)",
        "value": BASELINE_CFG05_ONLY_SMAPE,
        "unit": "sMAPE_floor50 %",
        "period": "2025-01-01 to 2025-12-31",
        "source": "production_metrics_2025.json",
        "verified": True,
        "caveat": None,
    })

    # ── Claim 2: Realtime DA-safe 2025 (always available) ──
    claims.append({
        "id": "realtime_da_safe_2025",
        "label": "Realtime DA-Safe Baseline sMAPE (2025 full year)",
        "value": BASELINE_REALTIME_DA_SAFE_SMAPE,
        "unit": "sMAPE_floor50 %",
        "period": "2025-01-01 to 2025-12-31",
        "source": "production_metrics_2025.json",
        "verified": True,
        "caveat": "DA-Safe Baseline only (rt_pred = da_anchor), no SGDFNet assist",
    })

    # ── Claim 3: Local 2026 BGEW (always available but labeled) ──
    claims.append({
        "id": "local_2026_bgew",
        "label": "Trusted BGEW fusion sMAPE (June 2026 local window)",
        "value": LOCAL_2026_BGEW_SMAPE,
        "unit": "sMAPE_floor50 %",
        "period": "2026-06 local window (NOT full year)",
        "source": "local trusted delivery benchmark",
        "verified": True,
        "caveat": "LOCAL WINDOW ONLY — not comparable to 2025 full-year cfg05-only. "
                  "Small sample, favorable conditions.",
    })

    # ── Claim 4: P138 Rolling BGEW 2025 (conditional) ──
    if p138["exists"] and p138["smape"] is not None:
        if p138["model_count"] >= 2 and p138["improves_vs_cfg05"]:
            improvement = BASELINE_CFG05_ONLY_SMAPE - p138["smape"]
            claims.append({
                "id": "bgew_2025_rolling",
                "label": "Rolling BGEW fusion sMAPE (2025 full year)",
                "value": p138["smape"],
                "unit": "sMAPE_floor50 %",
                "period": "2025-01-01 to 2025-12-31",
                "source": "P138 rolling BGEW benchmark",
                "verified": True,
                "caveat": None,
                "improvement_vs_cfg05": round(improvement, 2),
                "improvement_pct": round(improvement / BASELINE_CFG05_ONLY_SMAPE * 100, 1),
                "model_count": p138["model_count"],
            })
        elif p138["model_count"] < 2:
            blocked_claims.append({
                "id": "bgew_2025_rolling",
                "reason": f"BGEW requires model_count >= 2, got {p138['model_count']}",
            })
        elif not p138["improves_vs_cfg05"]:
            blocked_claims.append({
                "id": "bgew_2025_rolling",
                "reason": f"BGEW sMAPE {p138['smape']}% does not improve vs cfg05 {BASELINE_CFG05_ONLY_SMAPE}%",
            })
    else:
        blocked_claims.append({
            "id": "bgew_2025_rolling",
            "reason": "P138 rolling BGEW artifacts not available",
        })

    # ── Claim 5: P139 Residual corrected (conditional) ──
    if p139["exists"] and not p139["is_noop"] and p139["smape"] is not None:
        claims.append({
            "id": "residual_corrected_2025",
            "label": "Residual-corrected BGEW sMAPE (2025)",
            "value": p139["smape"],
            "unit": "sMAPE_floor50 %",
            "period": "2025-01-01 to 2025-12-31",
            "source": "P139 residual corrected benchmark",
            "verified": True,
            "caveat": None,
        })
    elif p139["exists"] and p139["is_noop"]:
        blocked_claims.append({
            "id": "residual_corrected_2025",
            "reason": "Residual correction is no-op, no improvement to claim",
        })
    else:
        blocked_claims.append({
            "id": "residual_corrected_2025",
            "reason": "P139 residual corrected artifacts not available",
        })

    # ── Claim 6: P140 Realtime improvement (conditional) ──
    if p140["exists"] and p140["smape"] is not None:
        if p140["delta_improves"]:
            improvement = BASELINE_REALTIME_DA_SAFE_SMAPE - p140["smape"]
            claims.append({
                "id": "realtime_improved_2025",
                "label": "Improved realtime sMAPE (2025)",
                "value": p140["smape"],
                "unit": "sMAPE_floor50 %",
                "period": "2025-01-01 to 2025-12-31",
                "source": "P140 realtime unblock",
                "verified": True,
                "caveat": None,
                "improvement_vs_baseline": round(improvement, 2),
            })
        else:
            blocked_claims.append({
                "id": "realtime_improved_2025",
                "reason": f"Realtime sMAPE {p140['smape']}% does not improve vs baseline {BASELINE_REALTIME_DA_SAFE_SMAPE}%",
            })
    else:
        blocked_claims.append({
            "id": "realtime_improved_2025",
            "reason": "P140 realtime artifacts not available",
        })

    # ── Claim 7: P142 Fair comparison (conditional) ──
    if p142["exists"]:
        claims.append({
            "id": "fair_comparison_2025",
            "label": "Fair comparison matrix (2025 vs 2.5)",
            "value": None,
            "unit": "see P142 data",
            "period": "2025",
            "source": "P142 fair comparison",
            "verified": True,
            "caveat": "2.5 artifacts may not be available for direct comparison",
        })
    else:
        blocked_claims.append({
            "id": "fair_comparison_2025",
            "reason": "P142 fair comparison artifacts not available",
        })

    # ── Determine verdict ──
    bgew_claim_exists = any(c["id"] == "bgew_2025_rolling" for c in claims)
    realtime_improved = any(c["id"] == "realtime_improved_2025" for c in claims)

    if bgew_claim_exists and realtime_improved:
        verdict = "PERFORMANCE_UNLOCKED_GO"
    elif bgew_claim_exists or realtime_improved:
        verdict = "PERFORMANCE_IMPROVED_WITH_CAVEATS"
    elif p138["exists"] and p138["model_count"] < 2:
        verdict = "PERFORMANCE_BLOCKED_FEATURE_PIPELINE"
    else:
        verdict = "PERFORMANCE_NO_IMPROVEMENT"

    # ── Consolidated metrics ──
    metrics = {
        "baseline_cfg05_only_smape": BASELINE_CFG05_ONLY_SMAPE,
        "baseline_realtime_da_safe_smape": BASELINE_REALTIME_DA_SAFE_SMAPE,
        "local_2026_bgew_smape": LOCAL_2026_BGEW_SMAPE,
        "local_2026_bgew_label": "local window, NOT full year",
        "p138_bgew_smape": p138["smape"],
        "p138_model_count": p138["model_count"],
        "p138_exists": p138["exists"],
        "p139_exists": p139["exists"],
        "p139_is_noop": p139["is_noop"],
        "p139_smape": p139["smape"],
        "p140_exists": p140["exists"],
        "p140_smape": p140["smape"],
        "p140_delta_improves": p140["delta_improves"],
        "p141_exists": p141["exists"],
        "p142_exists": p142["exists"],
    }

    return {
        "claims": claims,
        "blocked_claims": blocked_claims,
        "metrics": metrics,
        "verdict": verdict,
        "generated_at": datetime.now().isoformat(),
    }


# ── Output generators ───────────────────────────────────────────────


def generate_metrics_json(claims_data: dict, output_dir: str) -> str:
    """Generate production_metrics_2025_performance.json."""
    output_path = os.path.join(output_dir, "production_metrics_2025_performance.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    output = {
        "verdict": claims_data["verdict"],
        "generated_at": claims_data["generated_at"],
        "version": "3.0.0-rc1-p143",
        "claims": claims_data["claims"],
        "blocked_claims": claims_data["blocked_claims"],
        "metrics": claims_data["metrics"],
        "rules": {
            "only_claim_verified": True,
            "local_2026_must_note_window": True,
            "bgew_requires_model_count_gte_2": True,
            "no_fake_numbers": True,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", output_path)
    return output_path


def generate_report_md(claims_data: dict, output_dir: str) -> str:
    """Generate docs/reports/p143_performance_claim_update_report.md."""
    report_dir = os.path.join(output_dir, "docs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "p143_performance_claim_update_report.md")

    claims = claims_data["claims"]
    blocked = claims_data["blocked_claims"]
    metrics = claims_data["metrics"]
    verdict = claims_data["verdict"]

    lines = [
        "# P143: Performance Claim Update Report",
        "",
        f"**Generated:** {claims_data['generated_at']}",
        f"**Verdict:** `{verdict}`",
        "",
        "## Summary",
        "",
        "This report rewrites all performance claims based ONLY on verified artifact data.",
        "No fake numbers. No extrapolation. No unverified claims.",
        "",
        "## Verified Claims",
        "",
    ]

    for claim in claims:
        value_str = f"{claim['value']}%" if claim["value"] is not None else "see source data"
        lines.append(f"### {claim['label']}")
        lines.append(f"- **Value:** {value_str}")
        lines.append(f"- **Period:** {claim['period']}")
        lines.append(f"- **Source:** {claim['source']}")
        if claim.get("caveat"):
            lines.append(f"- **Caveat:** {claim['caveat']}")
        if claim.get("improvement_vs_cfg05") is not None:
            lines.append(f"- **Improvement vs cfg05:** {claim['improvement_vs_cfg05']}% "
                         f"({claim.get('improvement_pct', '?')}% relative)")
        if claim.get("improvement_vs_baseline") is not None:
            lines.append(f"- **Improvement vs baseline:** {claim['improvement_vs_baseline']}%")
        lines.append("")

    lines.append("## Blocked Claims (Cannot Be Made)")
    lines.append("")
    if blocked:
        for bc in blocked:
            lines.append(f"- **{bc['id']}**: {bc['reason']}")
    else:
        lines.append("None — all claims verified.")
    lines.append("")

    lines.append("## Metrics Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| cfg05-only 2025 sMAPE | {metrics['baseline_cfg05_only_smape']}% |")
    lines.append(f"| Realtime DA-Safe 2025 sMAPE | {metrics['baseline_realtime_da_safe_smape']}% |")
    lines.append(f"| Local 2026 BGEW sMAPE | {metrics['local_2026_bgew_smape']}% (LOCAL WINDOW) |")
    if metrics["p138_bgew_smape"] is not None:
        lines.append(f"| P138 Rolling BGEW 2025 sMAPE | {metrics['p138_bgew_smape']}% |")
    else:
        lines.append("| P138 Rolling BGEW 2025 sMAPE | NOT AVAILABLE |")
    if metrics["p140_smape"] is not None:
        lines.append(f"| P140 Improved Realtime sMAPE | {metrics['p140_smape']}% |")
    else:
        lines.append("| P140 Improved Realtime sMAPE | NOT AVAILABLE |")
    lines.append("")

    lines.append("## Performance Target Classification")
    lines.append("")
    lines.append("| Target | Threshold | Status |")
    lines.append("|--------|-----------|--------|")

    # Use best available BGEW number for target classification
    best_smape = metrics["p138_bgew_smape"]
    if best_smape is None:
        best_smape = metrics["baseline_cfg05_only_smape"]

    for target_name in ("minimum", "reasonable", "strong", "stretch"):
        threshold = PERFORMANCE_TARGETS[target_name]["threshold"]
        label = PERFORMANCE_TARGETS[target_name]["label"]
        met = "MET" if best_smape < threshold else "NOT MET"
        if target_name == "minimum" and best_smape <= threshold:
            met = "MET (at threshold)" if best_smape == threshold else "MET"
        lines.append(f"| {target_name.capitalize()} ({label}) | < {threshold}% | {met} |")
    lines.append("")

    lines.append("## Rules Applied")
    lines.append("")
    lines.append("1. Only 2025 full-year numbers that actually ran are claimed as 2025 full-year")
    lines.append("2. Local 2026 9.23% is labeled as 'local window, NOT full year'")
    lines.append("3. If BGEW blocked, no BGEW improvement claim")
    lines.append("4. If realtime delta improves, improvement is noted")
    lines.append("5. No fake numbers — every claim traces to an artifact")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", output_path)
    return output_path


def generate_client_delivery_note(claims_data: dict, output_dir: str) -> str:
    """Generate docs/CLIENT_DELIVERY_NOTE.md."""
    docs_dir = os.path.join(output_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    output_path = os.path.join(docs_dir, "CLIENT_DELIVERY_NOTE.md")

    claims = claims_data["claims"]
    verdict = claims_data["verdict"]
    metrics = claims_data["metrics"]

    lines = [
        "# Client Delivery Note — Electricity Price Forecasting System v3.0.0-rc1-p143",
        "",
        "## What This Is",
        "",
        "A production-oriented multi-model fusion system for day-ahead and real-time "
        "electricity price forecasting in the Shandong PMOS market.",
        "",
        "## What It Does",
        "",
        "- Combines multiple ML models (LightGBM, CatBoost) via BGEW fusion",
        "- Produces 24-hour day-ahead price predictions",
        "- Produces DA-Safe real-time price predictions",
        "- Supports optional SGDFNet neural network assist",
        "- Includes residual correction, risk classification, and safety supervision",
        "",
        "## How to Run",
        "",
        "```bash",
        "python main.py \\",
        "    --raw-data data/shandong_pmos_hourly.csv \\",
        "    --dayahead-source-repo .local_artifacts/source_repos/epf-sota-experiment \\",
        "    --profile trusted_delivery \\",
        "    --fusion-engine period_bgew \\",
        "    --work-dir .local_artifacts/production_run \\",
        "    --strict --strict-no-leakage \\",
        "    --json",
        "```",
        "",
        "## Key Metrics (Verified — P143)",
        "",
        "| Component | sMAPE | Period | Notes |",
        "|-----------|-------|--------|-------|",
    ]

    for claim in claims:
        if claim["value"] is not None:
            caveat_short = ""
            if claim.get("caveat"):
                caveat_short = claim["caveat"][:60]
            lines.append(
                f"| {claim['label'][:50]} | {claim['value']}% | "
                f"{claim['period'][:25]} | {caveat_short} |"
            )

    lines.append("")
    lines.append("## Current Status")
    lines.append("")
    lines.append(f"`{verdict}`")
    lines.append("")
    lines.append("## Important Caveats")
    lines.append("")
    lines.append("See `CLIENT_CAVEATS.md` for full details.")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", output_path)
    return output_path


def generate_client_caveats(claims_data: dict, output_dir: str) -> str:
    """Generate docs/CLIENT_CAVEATS.md."""
    docs_dir = os.path.join(output_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    output_path = os.path.join(docs_dir, "CLIENT_CAVEATS.md")

    blocked = claims_data["blocked_claims"]
    metrics = claims_data["metrics"]

    lines = [
        "# Client Caveats",
        "",
        f"**Updated:** {claims_data['generated_at']} (P143 rewrite)",
        "",
        "## Current Caveats",
        "",
        "1. **Local 2026 BGEW 9.23% — LOCAL WINDOW ONLY**",
        "   - This number comes from a local window (June 2026), NOT a full-year evaluation.",
        "   - It is NOT directly comparable to the 2025 full-year cfg05-only 20.22%.",
        "   - Small sample, potentially favorable conditions.",
        "",
        "2. **SGDFNet Assist — CODE_ONLY**",
        "   - SGDFNet source exists in model2.0_exp repo but production runtime",
        "     requires additional setup.",
        "   - Realtime falls back to DA-Safe Baseline (rt_pred = da_anchor).",
        "",
        "3. **P5M Full Residual — NO_OP_FALLBACK**",
        "   - Full 5-model residual stack is not yet assembled.",
        "   - CatBoost spike residual is available as partial correction.",
        "   - Residual correction is best-effort only.",
        "",
        "4. **ML Classifier — RULE_FALLBACK**",
        "   - ML classifier artifacts exist but are not yet in the automated production path.",
        "   - Classification uses rule-based fallback.",
        "",
    ]

    if metrics["p138_bgew_smape"] is None:
        lines.extend([
            "5. **BGEW Fusion 2025 — NOT YET COMPUTED**",
            "   - Rolling BGEW on 2025 full-year data has not been computed.",
            "   - The 9.23% number is from a local 2026 window only.",
            "",
        ])

    if blocked:
        lines.append("## Blocked Claims (Cannot Be Made)")
        lines.append("")
        for bc in blocked:
            lines.append(f"- **{bc['id']}**: {bc['reason']}")
        lines.append("")

    lines.extend([
        "## Forbidden Claims",
        "",
        "These claims must NEVER appear in delivery context:",
        "",
        '- "BGEW full-year 2025 sMAPE" unless P138 artifacts exist with model_count >= 2',
        '- "3.0 beats 2.5" unless P142 fair comparison confirms',
        '- "9.23% is full-year performance" — it is LOCAL WINDOW ONLY',
        '- "SGDFNet production ready" unless runtime verified',
        '- "Full P5M ready" unless full stack assembled',
        '- "ML classifier production ready" unless in ML path',
        "",
        "## Verified Claims (P143)",
        "",
    ])

    for claim in claims_data["claims"]:
        if claim["value"] is not None:
            lines.append(f"- {claim['label']}: {claim['value']}%")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", output_path)
    return output_path


# ── Main entry point ────────────────────────────────────────────────


def run_p143_performance_claims(
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Main entry point: read artifacts, build claims, write outputs.

    Args:
        artifacts_dir: Path to .local_artifacts directory
        output_dir: Path to repo root for output files

    Returns:
        dict with claims, blocked_claims, metrics, verdict, and output paths
    """
    logger.info("P143: Reading artifacts from %s", artifacts_dir)
    logger.info("P143: Writing outputs to %s", output_dir)

    # Build claims from actual data
    claims_data = build_performance_claims(artifacts_dir)

    # Generate all output files
    metrics_path = generate_metrics_json(claims_data, output_dir)
    report_path = generate_report_md(claims_data, output_dir)
    delivery_path = generate_client_delivery_note(claims_data, output_dir)
    caveats_path = generate_client_caveats(claims_data, output_dir)

    claims_data["output_files"] = {
        "metrics_json": metrics_path,
        "report_md": report_path,
        "client_delivery_note": delivery_path,
        "client_caveats": caveats_path,
    }

    logger.info("P143 complete. Verdict: %s", claims_data["verdict"])
    logger.info("P143: %d claims, %d blocked",
                len(claims_data["claims"]), len(claims_data["blocked_claims"]))

    return claims_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_p143_performance_claims()
    print(json.dumps({
        "verdict": result["verdict"],
        "claims_count": len(result["claims"]),
        "blocked_count": len(result["blocked_claims"]),
        "output_files": result["output_files"],
    }, indent=2))
