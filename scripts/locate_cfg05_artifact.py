"""
scripts/locate_cfg05_artifact.py — Locate cfg05 LightGBM champion artifact.

Searches a source repository for cfg05-compatible LightGBM model files,
ignoring known invalid blacklisted models.

Usage::

    # Search in epf-sota-experiment repo
    python -m scripts.locate_cfg05_artifact \\
        --source-repo /path/to/epf-sota-experiment

    # Search a specific model directory
    python -m scripts.locate_cfg05_artifact --model-dir /path/to/weights

    # JSON output
    python -m scripts.locate_cfg05_artifact \\
        --source-repo /path/to/repo --json --verbose

Options::

    --source-repo PATH      Path to epf-sota-experiment repository.
    --model-dir PATH        Specific model directory to check.
    --json                  Output JSON report.
    --verbose, -v           Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from artifacts.readiness import check_cfg05_artifact, MISSING, LOADABLE, status_to_dict
from artifacts.readiness import ArtifactStatus

logger = logging.getLogger(__name__)

# ── Blacklist: invalid model identifiers ──────────────────────────────
# These are known stale/broken models that must NOT be used.
_BLACKLISTED_NAMES: frozenset[str] = frozenset({
    "lgbm_spike_residual_1127",
    "stage3_old_1164",
    "lightgbm_90d_orig_1197",
})


def _is_blacklisted(path: str) -> bool:
    """Check if a path contains any blacklisted model name."""
    lower = path.lower()
    return any(bl in lower for bl in _BLACKLISTED_NAMES)


def _find_cfg05_candidates(source_repo: str) -> list[str]:
    """Search source_repo for plausible cfg05 model files.

    Scans for files matching::

        cfg05_model.txt
        model.txt
        lightgbm_cfg05_dayahead.txt

    Also scans paths containing::

        cfg05
        lgbm
        lightgbm
        champion

    with ``.txt`` extension for LightGBM text models.
    """
    candidates: list[str] = []

    if not os.path.isdir(source_repo):
        return candidates

    for root, _dirs, files in os.walk(source_repo):
        # Skip hidden dirs, venv, __pycache__
        rel = os.path.relpath(root, source_repo)
        parts = [p for p in rel.split(os.sep) if p and p != "."]
        if any(part.startswith(".") or part in ("__pycache__", "venv", ".venv")
               for part in parts):
            continue

        for fname in files:
            # Check exact known names first
            if fname in ("cfg05_model.txt", "model.txt", "lightgbm_cfg05_dayahead.txt"):
                candidates.append(os.path.join(root, fname))
                logger.debug("Found candidate (exact): %s", os.path.join(root, fname))
                continue

            # Broader: .txt files with cfg05/lgbm/lightgbm/champion in path
            if fname.endswith(".txt"):
                full = os.path.join(root, fname)
                lower_full = full.lower()
                if any(kw in lower_full for kw in ("cfg05", "lgbm", "lightgbm", "champion")):
                    candidates.append(full)
                    logger.debug("Found candidate (keyword): %s", full)

    return candidates


def locate_cfg05_artifact(
    source_repo: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Locate cfg05 artifact candidates and run readiness checks.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment repository for broad search.
    model_dir : str, optional
        Specific model directory to check.

    Returns
    -------
    dict with keys:
        source_repo_checked, candidates, blacklisted_skipped,
        total_candidates, best_status, best_path, artifact_report
    """
    result: dict[str, Any] = {
        "source_repo_checked": source_repo if source_repo else None,
        "candidates": [],
        "blacklisted_skipped": [],
        "total_candidates": 0,
        "best_status": MISSING,
        "best_path": None,
        "artifact_report": None,
    }

    # Check specific model directory first
    if model_dir:
        logger.info("Checking specific model-dir: %s", model_dir)
        status = check_cfg05_artifact(model_dir)
        cand_entry = {
            "path": model_dir,
            "status": status.status,
            "reason_codes": status.reason_codes,
            "file_size_bytes": status.details.get("file_size_bytes"),
        }
        result["candidates"].append(cand_entry)
        if status.status in (LOADABLE, "SCHEMA_READY", "REAL_READY"):
            result["best_status"] = status.status
            result["best_path"] = model_dir
            result["artifact_report"] = status_to_dict(status)
        result["total_candidates"] = len(result["candidates"])
        return result

    # Search source repo
    if source_repo and not os.path.isdir(source_repo):
        logger.warning("Source repo not found: %s", source_repo)
        result["source_repo_checked"] = source_repo
        result["total_candidates"] = len(result["candidates"])
        return result

    if source_repo:
        candidates = _find_cfg05_candidates(source_repo)

        for path in candidates:
            if _is_blacklisted(path):
                result["blacklisted_skipped"].append(path)
                logger.info("Skipped blacklisted: %s", path)
                continue

            status = check_cfg05_artifact(path)
            cand_entry = {
                "path": path,
                "status": status.status,
                "reason_codes": status.reason_codes,
                "file_size_bytes": status.details.get("file_size_bytes"),
            }
            result["candidates"].append(cand_entry)

            # Track best status (LOADABLE > PRESENT > MISSING)
            _rank = {MISSING: 0, "PRESENT": 1, LOADABLE: 2, "SCHEMA_READY": 3, "REAL_READY": 4}
            best_rank = _rank.get(result["best_status"], 0)
            this_rank = _rank.get(status.status, 0)
            if this_rank > best_rank:
                result["best_status"] = status.status
                result["best_path"] = path
                result["artifact_report"] = status_to_dict(status)

    result["total_candidates"] = len(result["candidates"])
    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable locate report."""
    print("=" * 60)
    print("cfg05 Artifact Locate Report")
    print("=" * 60)
    print(f"  Source repo:    {result['source_repo_checked'] or 'N/A'}")
    print(f"  Candidates:     {result['total_candidates']}")
    print(f"  Blacklisted:    {len(result['blacklisted_skipped'])}")
    print(f"  Best status:    {result['best_status']}")
    print(f"  Best path:      {result['best_path'] or 'N/A'}")
    print()
    if result["candidates"]:
        print("  Candidate details:")
        for c in result["candidates"]:
            size = f"({c['file_size_bytes']} bytes)" if c.get("file_size_bytes") else ""
            print(f"    [{c['status']:20s}] {c['path']} {size}")
            for rc in c.get("reason_codes", []):
                print(f"      -> {rc}")
    if result["blacklisted_skipped"]:
        print("  Blacklisted (skipped):")
        for p in result["blacklisted_skipped"]:
            print(f"    [BLOCKED] {p}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = locate_cfg05_artifact(
        source_repo=args.source_repo,
        model_dir=args.model_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate cfg05 LightGBM champion artifact.",
    )
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment repository.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Specific model directory to check.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON report.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
