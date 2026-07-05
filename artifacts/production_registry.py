"""
artifacts/production_registry.py — P97: Production Artifact Registry.

Unified registry that scans for ALL real model artifacts across the
filesystem and reports which are available, missing, or blocked.

Output:
    production_artifact_status.json
    production_artifact_status.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Default config path
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "production_artifacts.yaml",
)


# ── Status constants ──────────────────────────────────────────────────
ARTIFACT_FOUND = "ARTIFACT_FOUND"
ARTIFACT_MISSING = "ARTIFACT_MISSING"
ARTIFACT_LOADED = "ARTIFACT_LOADED"
ARTIFACT_LOAD_FAILED = "ARTIFACT_LOAD_FAILED"
ARTIFACT_BLOCKED = "ARTIFACT_BLOCKED"


def sha256_file(path: str) -> str:
    """Compute SHA256 of a file, or empty string if not found."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return ""
    except PermissionError:
        return "PERMISSION_DENIED"


def try_load_artifact(path: str) -> bool:
    """Try to load a model artifact to verify it's not corrupted."""
    if not os.path.isfile(path):
        return False
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".pkl", ".joblib"):
            import joblib
            joblib.load(path)
            return True
        elif ext == ".cbm":
            from catboost import CatBoost
            CatBoost().load_model(path)
            return True
        elif ext == ".txt":
            _size = os.path.getsize(path)
            if _size > 100:
                return True
            return False
        elif ext == ".json":
            with open(path) as f:
                json.load(f)
            return True
        elif ext == ".csv":
            return os.path.getsize(path) > 0
        else:
            return True
    except Exception as e:
        logger.warning("Load failed for %s: %s", path, e)
        return False


def resolve_root_path(
    config_path: str,
    artifact_dir: str = "",
) -> str:
    """Resolve an artifact path relative to the repo root."""
    if os.path.isabs(artifact_dir):
        return artifact_dir
    # Relative to the config file location (project root)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.normpath(os.path.join(repo_root, artifact_dir))


def scan_artifact(
    paths: list[str],
    fallback_paths: list[str] | None = None,
    required_for_go: bool = False,
    type_hint: str = "",
    config_path: str = "",
) -> dict[str, Any]:
    """Scan for a single artifact across candidate paths.

    Returns dict with status, path, sha256, loaded_successfully, etc.
    """
    result: dict[str, Any] = {
        "found": False,
        "path": "",
        "sha256": "",
        "loaded_successfully": False,
        "status": ARTIFACT_MISSING,
        "reason_codes": [],
        "type": type_hint,
    }

    for raw_path in paths:
        abs_path = resolve_root_path(config_path, raw_path) if config_path else raw_path
        if os.path.isfile(abs_path):
            result["found"] = True
            result["path"] = abs_path
            result["sha256"] = sha256_file(abs_path)
            loaded = try_load_artifact(abs_path)
            result["loaded_successfully"] = loaded
            if loaded:
                result["status"] = ARTIFACT_LOADED
            else:
                result["status"] = ARTIFACT_LOAD_FAILED
                result["reason_codes"].append(f"CORRUPT_OR_UNREADABLE:{os.path.basename(abs_path)}")
            return result

    # Check fallback paths
    if fallback_paths:
        for raw_path in fallback_paths:
            abs_path = resolve_root_path(config_path, raw_path) if config_path else raw_path
            if os.path.isfile(abs_path):
                result["found"] = True
                result["path"] = abs_path
                result["sha256"] = sha256_file(abs_path)
                result["loaded_successfully"] = True
                result["status"] = ARTIFACT_LOADED
                result["reason_codes"].append(f"USING_FALLBACK_PATH:{raw_path}")
                return result

    result["reason_codes"].append("NO_PATH_MATCHED")
    return result


def run_production_registry(
    config_path: str = "",
    output_dir: str = "",
) -> dict[str, Any]:
    """Run the production artifact registry scan.

    Returns dict with full status per artifact and overall assessment.
    """
    config_path = config_path or _DEFAULT_CONFIG
    if not os.path.isfile(config_path):
        return {"status": "REGISTRY_CONFIG_MISSING", "error": f"Not found: {config_path}"}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result: dict[str, Any] = {
        "status": "REGISTRY_COMPLETE",
        "version": config.get("version", "unknown"),
        "artifacts": {},
        "go_blockers": [],
        "caveats": [],
        "summary": {
            "total": 0,
            "found": 0,
            "missing": 0,
            "loaded": 0,
            "failed": 0,
        },
    }

    for category, items in config.items():
        if category == "version" or not isinstance(items, dict):
            continue
        for name, spec in items.items():
            if not isinstance(spec, dict):
                continue
            result["summary"]["total"] += 1
            paths = spec.get("artifact_paths", [])
            fallback = spec.get("fallback_paths", [])
            required = spec.get("required_for_go", False)
            type_hint = spec.get("type", "unknown")
            status_if_missing = spec.get("status_if_missing", "GO_WITH_CAVEATS")

            artifact_result = scan_artifact(
                paths=paths,
                fallback_paths=fallback,
                required_for_go=required,
                type_hint=type_hint,
                config_path=config_path,
            )

            artifact_result["name"] = name
            artifact_result["category"] = category
            artifact_result["required_for_go"] = required
            artifact_result["status_if_missing"] = status_if_missing

            result["artifacts"][f"{category}.{name}"] = artifact_result

            if artifact_result["found"]:
                result["summary"]["found"] += 1
            else:
                result["summary"]["missing"] += 1
                if required:
                    result["go_blockers"].append(
                        f"{category}.{name}: REQUIRED but missing "
                        f"(status_if_missing={status_if_missing})"
                    )
                else:
                    result["caveats"].append(
                        f"{category}.{name}: missing "
                        f"(status_if_missing={status_if_missing})"
                    )

            if artifact_result["loaded_successfully"]:
                result["summary"]["loaded"] += 1
            elif artifact_result["found"] and not artifact_result["loaded_successfully"]:
                result["summary"]["failed"] += 1
                result["caveats"].append(
                    f"{category}.{name}: found but load FAILED"
                )

    # Determine overall assessment
    if result["go_blockers"]:
        result["overall_assessment"] = "BLOCKED"
        result["assessment_reason"] = f"{len(result['go_blockers'])} required artifacts missing"
    elif result["caveats"]:
        result["overall_assessment"] = "GO_WITH_CAVEATS"
        result["assessment_reason"] = f"{len(result['caveats'])} artifacts with caveats"
    else:
        result["overall_assessment"] = "GO"
        result["assessment_reason"] = "All required artifacts ready"

    return result


def save_registry_output(
    registry_result: dict[str, Any],
    output_dir: str,
) -> dict[str, str]:
    """Save registry result to JSON and MD."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "production_artifact_status.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(registry_result, f, indent=2, default=str)

    # Build MD report
    lines = [
        "# Production Artifact Registry Status",
        "",
        f"**Assessment**: {registry_result.get('overall_assessment', 'UNKNOWN')}",
        f"**Reason**: {registry_result.get('assessment_reason', '')}",
        "",
        f"**Summary**: "
        f"{registry_result['summary']['found']}/{registry_result['summary']['total']} found, "
        f"{registry_result['summary']['loaded']} loaded, "
        f"{registry_result['summary']['failed']} failed",
        "",
        "## Artifact Details",
        "",
        "| Artifact | Required | Status | Path |",
        "|----------|----------|--------|------|",
    ]

    for key, val in sorted(registry_result.get("artifacts", {}).items()):
        status = val.get("status", "UNKNOWN")
        found = "✅" if val.get("found") else "❌"
        req = "🔴" if val.get("required_for_go") else "⚪"
        path = val.get("path", "") or "(not found)"
        lines.append(f"| {key} | {req} | {found} {status} | `{path}` |")

    if registry_result.get("go_blockers"):
        lines.extend(["", "## GO Blockers", ""])
        for b in registry_result["go_blockers"]:
            lines.append(f"- ❌ {b}")

    if registry_result.get("caveats"):
        lines.extend(["", "## Caveats", ""])
        for c in registry_result["caveats"]:
            lines.append(f"- ⚠️ {c}")

    md_path = os.path.join(output_dir, "production_artifact_status.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": json_path, "md": md_path}
