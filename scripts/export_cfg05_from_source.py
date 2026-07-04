"""
scripts/export_cfg05_from_source.py — Attempt cfg05 artifact export from source.

Locates the cfg05 LightGBM champion artifact in the epf-sota-experiment
source repository and attempts export. Reports exact blockers if export
cannot be completed.

Usage::

    # Check source repo for cfg05 artifact
    python -m scripts.export_cfg05_from_source --source-repo /path/to/epf-sota-experiment

    # Copy found artifact to local work dir
    python -m scripts.export_cfg05_from_source \\
        --source-repo /path/to/epf-sota-experiment \\
        --work-dir .local_artifacts/p11_cfg05 \\
        --copy-if-found

    # Fallback: search for training/export scripts
    python -m scripts.export_cfg05_from_source \\
        --source-repo /path/to/epf-sota-experiment --verbose

Options::

    --source-repo PATH          Path to epf-sota-experiment repository.
    --target-day YYYY-MM-DD     Target day (informational).
    --work-dir PATH             Ignored local directory for artifact copies.
    --copy-if-found             Copy artifact to work-dir (safe, ignored dir only).
    --run-export                Allow safe export/regeneration commands.
    --json                      Output JSON report.
    --strict                    Exit non-zero if export cannot complete.
    --verbose, -v               Increase log verbosity.

Output statuses::

    "CFG05_EXPORT_BLOCKED"      Source repo not found or no artifact candidates.
    "CFG05_ARTIFACT_FOUND"      Artifact located but not yet REAL_READY.
    "CFG05_COPIED_TO_WORKDIR"  Artifact copied to local work dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from typing import Any, Optional

from artifacts.readiness import (
    check_cfg05_artifact, LOADABLE, SCHEMA_READY, REAL_READY, MISSING,
    PRESENT, INVALID, status_to_dict,
)

logger = logging.getLogger(__name__)

# ── Blacklist ───────────────────────────────────────────────────────────
_BLACKLISTED_NAMES: frozenset[str] = frozenset({
    "lgbm_spike_residual_1127",
    "stage3_old_1164",
    "lightgbm_90d_orig_1197",
})


# --- Candidate search (reuses locate logic) ------------------------------

def _is_blacklisted(path: str) -> bool:
    lower = path.lower()
    return any(bl in lower for bl in _BLACKLISTED_NAMES)


def _find_cfg05_candidates(source_repo: str) -> list[str]:
    """Walk source_repo for cfg05 model file candidates."""
    candidates: list[str] = []
    if not os.path.isdir(source_repo):
        return candidates

    for root, _dirs, files in os.walk(source_repo):
        rel = os.path.relpath(root, source_repo)
        parts = [p for p in rel.split(os.sep) if p and p != "."]
        if any(part.startswith(".") or part in ("__pycache__", "venv", ".venv")
               for part in parts):
            continue

        for fname in files:
            if fname in ("cfg05_model.txt", "model.txt", "lightgbm_cfg05_dayahead.txt"):
                candidates.append(os.path.join(root, fname))
                continue
            if fname.endswith(".txt"):
                full = os.path.join(root, fname)
                lower = full.lower()
                if any(kw in lower for kw in ("cfg05", "lgbm", "lightgbm", "champion")):
                    candidates.append(full)

    return candidates


# --- Training/export script search --------------------------------------

_TRAINING_SCRIPT_PATTERNS = [
    "cfg05", "micro-search", "micro_search", "lgbm", "lightgbm",
    "save_model", "Booster", "model.txt", "train",
]


def _find_training_scripts(source_repo: str) -> list[str]:
    """Find scripts that might contain cfg05 training or export logic."""
    found: list[str] = []
    for root, _dirs, files in os.walk(source_repo):
        for fname in files:
            if not fname.endswith((".py", ".ipynb", ".sh", ".md", ".txt")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if any(p in content.lower() for p in _TRAINING_SCRIPT_PATTERNS):
                    found.append(fpath)
            except Exception:
                continue
    return found


# --- Core export logic --------------------------------------------------

def export_cfg05_from_source(
    source_repo: Optional[str] = None,
    target_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    copy_if_found: bool = False,
    run_export: bool = False,
) -> dict[str, Any]:
    """Attempt to export cfg05 artifact from source repository.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment repository.
    target_day : str, optional
        Target day for prediction (informational).
    work_dir : str, optional
        Ignored local directory for artifact copies.
    copy_if_found : bool
        Whether to copy found artifact to work-dir.
    run_export : bool
        Whether to run safe export commands.

    Returns
    -------
    dict with export attempt summary.
    """
    result: dict[str, Any] = {
        "source_repo": source_repo,
        "target_day": target_day,
        "export_status": "CFG05_EXPORT_BLOCKED",
        "artifact_path": None,
        "artifact_status": MISSING,
        "candidates": [],
        "blacklisted_skipped": [],
        "training_scripts_found": [],
        "work_dir": work_dir,
        "copy_performed": False,
        "reason_codes": [],
        "next_commands": [],
    }

    # --- Check source repo ---
    if not source_repo:
        result["reason_codes"].append("SOURCE_REPO_NOT_PROVIDED")
        result["next_commands"].append(
            "Provide --source-repo pointing to disdorqin/epf-sota-experiment"
        )
        return result

    if not os.path.isdir(source_repo):
        result["reason_codes"].append(f"SOURCE_REPO_NOT_FOUND: {source_repo}")
        result["next_commands"].extend([
            f"Clone epf-sota-experiment: git clone https://github.com/disdorqin/epf-sota-experiment <path>",
            f"Then re-run with: --source-repo <path>",
        ])
        return result

    result["reason_codes"].append(f"SOURCE_REPO_FOUND: {source_repo}")

    # --- Search for artifact candidates ---
    candidates = _find_cfg05_candidates(source_repo)
    non_blacklisted: list[str] = []
    for path in candidates:
        if _is_blacklisted(path):
            result["blacklisted_skipped"].append(path)
            logger.info("Skipped blacklisted: %s", path)
            continue
        non_blacklisted.append(path)

        status = check_cfg05_artifact(path)
        entry = {
            "path": path,
            "status": status.status,
            "reason_codes": status.reason_codes,
            "file_size_bytes": status.details.get("file_size_bytes"),
        }
        result["candidates"].append(entry)

        if status.status in (LOADABLE, SCHEMA_READY, REAL_READY):
            result["artifact_path"] = path
            result["artifact_status"] = status.status
            result["artifact_report"] = status_to_dict(status)

    # --- Determine export status ---
    if result["artifact_path"] and result["artifact_status"] in (LOADABLE, SCHEMA_READY, REAL_READY):
        result["export_status"] = "CFG05_ARTIFACT_FOUND"
        result["reason_codes"].append(
            f"CFG05_ARTIFACT_LOADABLE: {result['artifact_path']}"
        )

        # Copy if requested
        if copy_if_found and work_dir:
            os.makedirs(work_dir, exist_ok=True)
            dst = os.path.join(work_dir, os.path.basename(result["artifact_path"]))
            shutil.copy2(result["artifact_path"], dst)
            result["copy_performed"] = True
            result["export_status"] = "CFG05_COPIED_TO_WORKDIR"
            result["reason_codes"].append(f"CFG05_COPIED_TO_WORKDIR: {dst}")

        result["next_commands"].extend([
            f"Artifact ready at: {result['artifact_path']}",
            "Run cfg05 REAL smoke: "
            f"python -m scripts.run_cfg05_real_smoke_pipeline "
            f"--cfg05-model {result['artifact_path']} "
            f"--cfg05-input <input_csv> --strict",
        ])
    else:
        # No loadable artifact found — search for training scripts
        training_scripts = _find_training_scripts(source_repo)
        result["training_scripts_found"] = training_scripts

        if training_scripts:
            result["reason_codes"].append(
                f"NO_LOADABLE_ARTIFACT_BUT_FOUND_{len(training_scripts)}_TRAINING_SCRIPTS"
            )
            result["next_commands"].extend([
                "Inspect training scripts listed above to locate model save logic.",
                "Common patterns: lgb.Booster.save_model(), model.txt, cfg05_model.txt",
                "If training script found, run it with --run-export (if safe) or manually.",
            ])
        else:
            result["reason_codes"].append(
                "NO_CFG05_ARTIFACT_AND_NO_TRAINING_SCRIPTS_FOUND"
            )
            result["next_commands"].extend([
                "1. Clone: git clone https://github.com/disdorqin/epf-sota-experiment.git",
                "2. Check docs/reports/ for champion training commands",
                "3. Run cfg05 micro-search or training script",
                "4. Re-run this script with --source-repo pointing to the clone",
            ])

    # --- Handle --run-export (safe export commands) ---
    if run_export and not copy_if_found:
        result["reason_codes"].append("RUN_EXPORT_FLAG_SET_BUT_NO_COPY_REQUESTED")
        result["next_commands"].append(
            "Use --copy-if-found to copy artifact to work-dir"
        )

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable export report."""
    print("=" * 60)
    print("cfg05 Export Attempt Report")
    print("=" * 60)
    print(f"  Source repo:  {result['source_repo'] or 'N/A'}")
    print(f"  Export status: {result['export_status']}")
    print(f"  Artifact:     {result['artifact_path'] or 'NOT FOUND'}")
    if result["artifact_path"]:
        print(f"  Status:       {result['artifact_status']}")
    print(f"  Candidates:   {len(result['candidates'])}")
    print(f"  Blacklisted:  {len(result['blacklisted_skipped'])}")
    print(f"  Copy done:    {result['copy_performed']}")
    print()
    if result["candidates"]:
        print("  Candidates:")
        for c in result["candidates"]:
            sz = f" ({c['file_size_bytes']}B)" if c.get("file_size_bytes") else ""
            print(f"    [{c['status']:20s}] {c['path']}{sz}")
    if result["training_scripts_found"]:
        print(f"  Training scripts ({len(result['training_scripts_found'])}):")
        for s in result["training_scripts_found"]:
            print(f"    {s}")
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("  Next commands:")
    for cmd in result.get("next_commands", []):
        print(f"    -> {cmd}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = export_cfg05_from_source(
        source_repo=args.source_repo,
        target_day=args.target_day,
        work_dir=args.work_dir,
        copy_if_found=args.copy_if_found,
        run_export=args.run_export,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["export_status"] in ("CFG05_ARTIFACT_FOUND", "CFG05_COPIED_TO_WORKDIR"):
            return 0
        else:
            logger.error("Export strict mode FAILED: %s", result["export_status"])
            return 1

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cfg05 LightGBM champion from source repo.",
    )
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment repository.")
    parser.add_argument("--target-day", type=str, default=None,
                        help="Target day (informational).")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Ignored local directory for artifact copies.")
    parser.add_argument("--copy-if-found", action="store_true", default=False,
                        help="Copy artifact to work-dir.")
    parser.add_argument("--run-export", action="store_true", default=False,
                        help="Allow safe export commands.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero if export fails.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
