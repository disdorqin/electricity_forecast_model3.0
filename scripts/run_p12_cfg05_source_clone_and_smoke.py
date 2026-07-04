"""
scripts/run_p12_cfg05_source_clone_and_smoke.py — P12 orchestration.

Clones or locates the epf-sota-experiment source repo, runs artifact export,
feature input build, and optionally the REAL smoke pipeline.

Usage::

    # Clone and run all gates (default)
    python -m scripts.run_p12_cfg05_source_clone_and_smoke \\
        --clone-url https://github.com/disdorqin/epf-sota-experiment.git \\
        --target-day 2026-07-01

    # With existing source repo
    python -m scripts.run_p12_cfg05_source_clone_and_smoke \\
        --source-repo /path/to/epf-sota-experiment --target-day 2026-07-01

    # With input CSV and REAL smoke attempt
    python -m scripts.run_p12_cfg05_source_clone_and_smoke \\
        --source-repo /path/to/epf-sota-experiment --target-day 2026-07-01 \\
        --input-csv /path/to/features.csv --run-real-smoke --strict

    # Full pipe with JSON output
    python -m scripts.run_p12_cfg05_source_clone_and_smoke \\
        --clone-url https://github.com/disdorqin/epf-sota-experiment.git \\
        --target-day 2026-07-01 --work-dir .local_artifacts/p12_cfg05 \\
        --copy-if-found --run-real-smoke --json

Options::

    --source-repo PATH              Existing path to epf-sota-experiment.
    --clone-url URL                 Git URL to clone (default: GitHub).
    --clone-dir PATH                Where to clone (default: .local_artifacts/source_repos/...).
    --work-dir PATH                 Local work dir for artifact copies.
    --target-day YYYY-MM-DD         Target day for prediction.
    --input-csv PATH                Pre-existing input CSV to validate.
    --copy-if-found                 Copy artifact to work-dir.
    --run-real-smoke                Attempt REAL smoke if artifact+input ready.
    --json                          Output JSON report.
    --strict                        Exit non-zero on any blocker.
    --verbose, -v                   Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Safe paths (ignored by .gitignore) ─────────────────────────────────
_ALLOWED_WORK_DIRS = (".local_artifacts",)
_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "ledgers/", "reports/local/")


def _path_is_safe(path: str) -> bool:
    """Check path is under an ignored, allowed directory."""
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        # Relative path: must start with an allowed prefix
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True  # absolute paths outside repo are OK


def _run_git_clone(clone_url: str, clone_dir: str) -> bool:
    """Clone source repo. Returns True on success."""
    try:
        logger.info("Cloning %s -> %s", clone_url, clone_dir)
        os.makedirs(os.path.dirname(clone_dir), exist_ok=True)
        result = subprocess.run(
            ["git", "clone", clone_url, clone_dir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("Clone succeeded")
            return True
        else:
            logger.error("Clone failed: %s", result.stderr.strip())
            return False
    except Exception as e:
        logger.error("Clone exception: %s", e)
        return False


def run_p12_cfg05_source_clone_and_smoke(
    source_repo: Optional[str] = None,
    clone_url: Optional[str] = None,
    clone_dir: Optional[str] = None,
    work_dir: Optional[str] = None,
    target_day: Optional[str] = None,
    input_csv: Optional[str] = None,
    copy_if_found: bool = False,
    run_real_smoke: bool = False,
) -> dict[str, Any]:
    """P12 orchestration: clone → export → build input → REAL smoke.

    Parameters
    ----------
    source_repo : str, optional
        Existing path to epf-sota-experiment.
    clone_url : str, optional
        Git URL for cloning.
    clone_dir : str, optional
        Destination for clone (default: .local_artifacts/source_repos/...).
    work_dir : str, optional
        Local work dir for copies.
    target_day : str, optional
        Target day for prediction.
    input_csv : str, optional
        Pre-existing input CSV to validate.
    copy_if_found : bool
        Copy artifact to work-dir.
    run_real_smoke : bool
        Run REAL smoke if artifact + input ready.

    Returns
    -------
    dict with complete P12 summary.
    """
    default_clone_dir = os.path.join(
        ".local_artifacts", "source_repos", "epf-sota-experiment",
    )
    clone_dir = clone_dir or default_clone_dir
    work_dir = work_dir or os.path.join(".local_artifacts", "p12_cfg05")

    result: dict[str, Any] = {
        "source_repo_status": "NOT_CHECKED",
        "source_repo_path": None,
        "artifact_status": "NOT_CHECKED",
        "artifact_candidates": [],
        "copied_artifact_path": None,
        "input_status": "NOT_CHECKED",
        "input_candidates": [],
        "prepared_input_path": None,
        "real_smoke_attempted": False,
        "real_smoke_status": None,
        "prediction_rows": 0,
        "validator_passed": False,
        "readiness_label": None,
        "final_status": "CFG05_EXPORT_BLOCKED",
        "reason_codes": [],
        "forbidden_files_check": "NOT_RUN",
    }

    # ── Path safety check ──
    if not _path_is_safe(clone_dir):
        result["reason_codes"].append(
            f"UNSAFE_CLONE_DIR: {clone_dir} — must be under .local_artifacts/"
        )
        result["final_status"] = "CFG05_EXPORT_BLOCKED"
        return result
    if not _path_is_safe(work_dir):
        result["reason_codes"].append(
            f"UNSAFE_WORK_DIR: {work_dir} — must be under .local_artifacts/"
        )
        result["final_status"] = "CFG05_EXPORT_BLOCKED"
        return result

    # ── Step 1: Locate or clone source repo ──
    effective_repo = source_repo
    if effective_repo and os.path.isdir(effective_repo):
        result["source_repo_status"] = "EXISTING_PATH"
        result["source_repo_path"] = effective_repo
        result["reason_codes"].append(f"USING_EXISTING_SOURCE_REPO: {effective_repo}")
    elif os.path.isdir(clone_dir):
        effective_repo = clone_dir
        result["source_repo_status"] = "ALREADY_CLONED"
        result["source_repo_path"] = clone_dir
        result["reason_codes"].append(f"USING_EXISTING_CLONE: {clone_dir}")
    elif clone_url:
        result["reason_codes"].append(f"CLONING: {clone_url} -> {clone_dir}")
        if _run_git_clone(clone_url, clone_dir):
            effective_repo = clone_dir
            result["source_repo_status"] = "CLONED_OK"
            result["source_repo_path"] = clone_dir
            result["reason_codes"].append(f"CLONE_SUCCESS: {clone_dir}")
        else:
            result["source_repo_status"] = "CLONE_FAILED"
            result["final_status"] = "CFG05_EXPORT_BLOCKED"
            result["reason_codes"].append("CLONE_FAILED")
            return result
    else:
        result["source_repo_status"] = "NO_SOURCE_PROVIDED"
        result["final_status"] = "CFG05_EXPORT_BLOCKED"
        result["reason_codes"].append("NO_SOURCE_REPO_OR_CLONE_URL_PROVIDED")
        return result

    # ── Step 2: Run artifact export ──
    from scripts.export_cfg05_from_source import export_cfg05_from_source

    export_result = export_cfg05_from_source(
        source_repo=effective_repo,
        target_day=target_day,
        work_dir=work_dir,
        copy_if_found=copy_if_found,
    )

    result["artifact_status"] = export_result.get("export_status", "UNKNOWN")
    result["artifact_candidates"] = export_result.get("candidates", [])
    result["copied_artifact_path"] = str(
        os.path.join(work_dir, os.path.basename(export_result["candidates"][0]["path"]))
    ) if (export_result.get("candidates") and export_result.get("copy_performed")) else None

    result["reason_codes"].extend(
        [f"ARTIFACT:{rc}" for rc in export_result.get("reason_codes", [])]
    )

    artifact_loadable = any(
        c.get("status") in ("LOADABLE", "SCHEMA_READY", "REAL_READY")
        for c in export_result.get("candidates", [])
    )

    # ── Step 3: Build/validate feature input ──
    from scripts.build_cfg05_feature_input_from_source import (
        build_cfg05_feature_input_from_source,
    )

    # If user provided --input-csv, use that directly
    if input_csv:
        input_result = build_cfg05_feature_input_from_source(
            input_csv=input_csv,
            target_day=target_day,
            out_path=os.path.join(work_dir, "prepared_input.csv") if copy_if_found else None,
        )
    else:
        input_result = build_cfg05_feature_input_from_source(
            source_repo=effective_repo,
            target_day=target_day,
        )

    result["input_status"] = input_result.get("input_status", "UNKNOWN")
    result["input_candidates"] = input_result.get("candidate_csvs_found", [])
    if input_result.get("out_written"):
        result["prepared_input_path"] = os.path.join(work_dir, "prepared_input.csv")

    result["reason_codes"].extend(
        [f"INPUT:{rc}" for rc in input_result.get("reason_codes", [])]
    )

    from artifacts.readiness import SCHEMA_READY
    input_schema_ready = input_result.get("input_status") == SCHEMA_READY

    # ── Step 4: REAL smoke (only if both gates pass) ──
    if run_real_smoke and artifact_loadable and input_schema_ready:
        result["real_smoke_attempted"] = True

        # Find the best artifact path
        best_artifact = None
        for c in export_result.get("candidates", []):
            if c.get("status") in ("LOADABLE", "SCHEMA_READY", "REAL_READY"):
                best_artifact = c["path"]
                break

        input_path = input_csv or (
            os.path.join(work_dir, "prepared_input.csv")
            if input_result.get("out_written") else None
        )

        if best_artifact and input_path and os.path.isfile(input_path):
            from scripts.run_cfg05_real_smoke_pipeline import (
                run_cfg05_real_smoke_pipeline,
            )

            smoke_result = run_cfg05_real_smoke_pipeline(
                cfg05_model=best_artifact,
                cfg05_input=input_path,
                target_day=target_day,
                production=True,
            )

            result["real_smoke_status"] = smoke_result.get("readiness_label")
            result["prediction_rows"] = smoke_result.get("prediction_rows", 0)
            result["validator_passed"] = smoke_result.get("validator_passed", False)
            result["readiness_label"] = smoke_result.get("readiness_label")
            result["reason_codes"].extend(
                [f"SMOKE:{rc}" for rc in smoke_result.get("reason_codes", [])]
            )
        else:
            result["real_smoke_status"] = "SKIPPED_MISSING_ARTIFACT_OR_INPUT"
            result["reason_codes"].append("REAL_SMOKE_SKIPPED_NO_ARTIFACT_OR_INPUT_PATH")
    elif run_real_smoke and not artifact_loadable:
        result["reason_codes"].append(
            "REAL_SMOKE_NOT_ATTEMPTED_ARTIFACT_NOT_LOADABLE"
        )
    elif run_real_smoke and not input_schema_ready:
        result["reason_codes"].append(
            "REAL_SMOKE_NOT_ATTEMPTED_INPUT_NOT_SCHEMA_READY"
        )

    # ── Step 5: Determine final status ──
    if result.get("readiness_label") == "REAL_READY":
        result["final_status"] = "CFG05_REAL_READY_LOCAL"
    elif artifact_loadable and input_schema_ready:
        result["final_status"] = "CFG05_REAL_SMOKE_FAILED"
    elif artifact_loadable and not input_schema_ready:
        result["final_status"] = "CFG05_ARTIFACT_FOUND_INPUT_BLOCKED"
    elif input_schema_ready and not artifact_loadable:
        result["final_status"] = "CFG05_ARTIFACT_BLOCKED_INPUT_FOUND"
    elif export_result.get("export_status") == "CFG05_EXPORT_BLOCKED":
        result["final_status"] = "CFG05_EXPORT_BLOCKED"
    elif input_result.get("input_status") == "CFG05_INPUT_BLOCKED":
        result["final_status"] = "CFG05_INPUT_BLOCKED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable P12 report."""
    print("=" * 60)
    print("P12 Source Clone + cfg05 REAL Smoke Report")
    print("=" * 60)
    print(f"  Source repo:      {result['source_repo_path'] or 'N/A'} ({result['source_repo_status']})")
    print(f"  Artifact status:  {result['artifact_status']}")
    print(f"  Input status:     {result['input_status']}")
    print(f"  REAL smoke:       {'YES' if result['real_smoke_attempted'] else 'NO'}")
    if result["real_smoke_attempted"]:
        print(f"  Smoke status:     {result['real_smoke_status']}")
        print(f"  Prediction rows:  {result['prediction_rows']}")
        print(f"  Validator passed: {result['validator_passed']}")
        print(f"  Readiness label:  {result['readiness_label']}")
    print(f"  Final status:     {result['final_status']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = run_p12_cfg05_source_clone_and_smoke(
        source_repo=args.source_repo,
        clone_url=args.clone_url,
        clone_dir=args.clone_dir,
        work_dir=args.work_dir,
        target_day=args.target_day,
        input_csv=args.input_csv,
        copy_if_found=args.copy_if_found,
        run_real_smoke=args.run_real_smoke,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["final_status"] == "CFG05_REAL_READY_LOCAL":
            logger.info("P12 strict mode PASS: REAL_READY_LOCAL")
            return 0
        else:
            logger.error("P12 strict mode FAIL: %s", result["final_status"])
            return 1

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P12: clone source repo + run cfg05 REAL export smoke.",
    )
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Existing path to epf-sota-experiment.")
    parser.add_argument("--clone-url", type=str,
                        default="https://github.com/disdorqin/epf-sota-experiment.git",
                        help="Git URL to clone.")
    parser.add_argument("--clone-dir", type=str, default=None,
                        help="Destination for clone.")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Local work dir for artifact copies.")
    parser.add_argument("--target-day", type=str, default="2026-07-01",
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Pre-existing input CSV to validate.")
    parser.add_argument("--copy-if-found", action="store_true", default=False,
                        help="Copy artifact to work-dir.")
    parser.add_argument("--run-real-smoke", action="store_true", default=False,
                        help="Run REAL smoke if artifact+input ready.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero on blocker.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
