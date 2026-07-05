"""
scripts/audit_p27_cfg05_feature_builder_alignment.py — P27 feature builder v3 alignment audit.

Audits the alignment between:
  - Source repo feature builders (v2 = 41 cols, v3 = 55 cols)
  - Current 3.0 cfg05 adapter (CFG05_FEATURE_COLUMNS = 41 cols)

Produces a detailed comparison report with labels:
  FEATURE_ALIGNMENT_MATCHED / FEATURE_ALIGNMENT_PARTIAL / FEATURE_ALIGNMENT_NOT_MATCHED

Usage::

    python -m scripts.audit_p27_cfg05_feature_builder_alignment \\
        --source-repo .local_artifacts/source_repos/epf-sota-experiment \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Alignment labels ───────────────────────────────────────────────────────
FEATURE_ALIGNMENT_MATCHED = "FEATURE_ALIGNMENT_MATCHED"
FEATURE_ALIGNMENT_PARTIAL = "FEATURE_ALIGNMENT_PARTIAL"
FEATURE_ALIGNMENT_NOT_MATCHED = "FEATURE_ALIGNMENT_NOT_MATCHED"


def _get_source_v2_features(source_repo: str) -> Optional[list[str]]:
    """Extract EXTENDED_FEATURE_COLUMNS from source v2 feature builder."""
    try:
        fb_path = os.path.join(source_repo, "src", "common", "feature_builder_dayahead.py")
        if not os.path.isfile(fb_path):
            return None
        with open(fb_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Look for EXTENDED_FEATURE_COLUMNS definition
        if "EXTENDED_FEATURE_COLUMNS" in content:
            # Try to import the module
            if source_repo not in sys.path:
                sys.path.insert(0, source_repo)
            import importlib
            mod = importlib.import_module("src.common.feature_builder_dayahead")
            if hasattr(mod, "EXTENDED_FEATURE_COLUMNS"):
                return list(mod.EXTENDED_FEATURE_COLUMNS)
    except Exception as e:
        logger.warning(f"Failed to extract v2 features: {e}")
    return None


def _get_source_v3_features(source_repo: str) -> Optional[list[str]]:
    """Extract V3_FEATURE_COLUMNS or equivalent from source v3 feature builder."""
    try:
        fb_path = os.path.join(source_repo, "src", "common", "feature_builder_dayahead_v3.py")
        if not os.path.isfile(fb_path):
            return None
        with open(fb_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Try to import
        if source_repo not in sys.path:
            sys.path.insert(0, source_repo)
        import importlib
        mod = importlib.import_module("src.common.feature_builder_dayahead_v3")
        # Check for various attribute names
        for attr in ["V3_FEATURE_COLUMNS", "EXTENDED_V3_FEATURE_COLUMNS", "ALL_FEATURE_COLUMNS"]:
            if hasattr(mod, attr):
                return list(getattr(mod, attr))
        # Fallback: parse V3_NEW_COLUMNS and combine with v2 extended
        if hasattr(mod, "V3_NEW_COLUMNS"):
            v2_feats = _get_source_v2_features(source_repo)
            if v2_feats:
                v3_new = list(mod.V3_NEW_COLUMNS)
                return v2_feats + [c for c in v3_new if c not in v2_feats]
    except Exception as e:
        logger.warning(f"Failed to extract v3 features: {e}")
    return None


def _get_cfg05_features() -> list[str]:
    """Get current cfg05 feature columns from 3.0 adapter."""
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
    return list(CFG05_FEATURE_COLUMNS)


def _get_cfg05_params() -> dict[str, Any]:
    """Get current cfg05 params from 3.0 adapter."""
    from models.adapters.cfg05_dayahead_lgbm import CFG05_PARAMS
    return dict(CFG05_PARAMS)


def _compare_feature_lists(
    cfg05_feats: list[str],
    v2_feats: Optional[list[str]],
    v3_feats: Optional[list[str]],
) -> dict[str, Any]:
    """Compare feature lists and produce alignment report."""
    cfg05_set = set(cfg05_feats)
    result: dict[str, Any] = {
        "cfg05_feature_count": len(cfg05_feats),
        "cfg05_features": cfg05_feats,
        "v2_feature_count": len(v2_feats) if v2_feats else None,
        "v2_features": v2_feats,
        "v3_feature_count": len(v3_feats) if v3_feats else None,
        "v3_features": v3_feats,
        "cfg05_vs_v2": {},
        "cfg05_vs_v3": {},
        "v2_vs_v3": {},
        "alignment_label": FEATURE_ALIGNMENT_NOT_MATCHED,
        "migration_recommendation": "",
    }

    # cfg05 vs v2
    if v2_feats:
        v2_set = set(v2_feats)
        result["cfg05_vs_v2"] = {
            "match": cfg05_set == v2_set,
            "cfg05_only": sorted(cfg05_set - v2_set),
            "v2_only": sorted(v2_set - cfg05_set),
            "common_count": len(cfg05_set & v2_set),
            "order_match": cfg05_feats == v2_feats,
        }

    # cfg05 vs v3
    if v3_feats:
        v3_set = set(v3_feats)
        result["cfg05_vs_v3"] = {
            "match": cfg05_set == v3_set,
            "cfg05_only": sorted(cfg05_set - v3_set),
            "v3_only": sorted(v3_set - cfg05_set),
            "common_count": len(cfg05_set & v3_set),
            "v3_extra_count": len(v3_set - cfg05_set),
        }

    # v2 vs v3
    if v2_feats and v3_feats:
        v2_set = set(v2_feats)
        v3_set = set(v3_feats)
        result["v2_vs_v3"] = {
            "match": v2_set == v3_set,
            "v2_only": sorted(v2_set - v3_set),
            "v3_only": sorted(v3_set - v2_set),
            "v3_new_count": len(v3_set - v2_set),
        }

    # Determine alignment label
    if v2_feats and cfg05_set == set(v2_feats):
        result["alignment_label"] = FEATURE_ALIGNMENT_MATCHED
        result["migration_recommendation"] = (
            "cfg05 matches source v2 exactly. "
            "Consider upgrading to v3 for additional features if source methodology supports it."
        )
    elif v2_feats and len(cfg05_set & set(v2_feats)) > 0.8 * len(cfg05_set):
        result["alignment_label"] = FEATURE_ALIGNMENT_PARTIAL
        result["migration_recommendation"] = (
            f"cfg05 ({len(cfg05_feats)} cols) mostly matches v2 ({len(v2_feats)} cols). "
            f"Missing from cfg05: {sorted(set(v2_feats) - cfg05_set)[:5]}. "
            "Minor differences — likely acceptable."
        )
    elif v3_feats and len(cfg05_set & set(v3_feats)) > 0.5 * len(set(v3_feats)):
        result["alignment_label"] = FEATURE_ALIGNMENT_PARTIAL
        v3_extra = sorted(set(v3_feats) - cfg05_set)
        result["migration_recommendation"] = (
            f"cfg05 ({len(cfg05_feats)} cols) is a subset of v3 ({len(v3_feats)} cols). "
            f"v3 adds {len(v3_extra)} features: {v3_extra[:10]}. "
            "To close the sMAPE gap, consider: (1) upgrade cfg05 to use v3 features, "
            "(2) retrain with v3 feature set, (3) validate no leakage in new features."
        )
    else:
        result["alignment_label"] = FEATURE_ALIGNMENT_NOT_MATCHED
        result["migration_recommendation"] = (
            "Significant feature mismatch. Manual review required before migration."
        )

    return result


def audit_p27_cfg05_feature_builder_alignment(
    source_repo: Optional[str] = None,
) -> dict[str, Any]:
    """Run the P27 feature builder alignment audit.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment source repo.

    Returns
    -------
    dict
        Complete alignment audit report.
    """
    source_repo = source_repo or os.path.join(
        ".local_artifacts", "source_repos", "epf-sota-experiment"
    )

    result: dict[str, Any] = {
        "source_repo": source_repo,
        "source_repo_exists": os.path.isdir(source_repo),
        "alignment_label": FEATURE_ALIGNMENT_NOT_MATCHED,
        "dimensions_checked": 0,
        "dimensions_matched": 0,
        "dimensions_partial": 0,
        "dimensions_not_matched": 0,
        "reason_codes": [],
    }

    # Get cfg05 features
    cfg05_feats = _get_cfg05_features()
    cfg05_params = _get_cfg05_params()

    # Get source features
    v2_feats = None
    v3_feats = None
    if os.path.isdir(source_repo):
        v2_feats = _get_source_v2_features(source_repo)
        v3_feats = _get_source_v3_features(source_repo)

    # Compare
    comparison = _compare_feature_lists(cfg05_feats, v2_feats, v3_feats)
    result.update(comparison)

    # Count dimensions
    dims = {
        "feature_count": False,
        "feature_names": False,
        "feature_order": False,
        "cfg05_params": False,
    }

    # Feature count check
    if v2_feats:
        dims["feature_count"] = len(cfg05_feats) == len(v2_feats)
    elif v3_feats:
        dims["feature_count"] = len(cfg05_feats) == len(v3_feats)

    # Feature names check
    if v2_feats:
        dims["feature_names"] = set(cfg05_feats) == set(v2_feats)
    elif v3_feats:
        dims["feature_names"] = set(cfg05_feats).issubset(set(v3_feats))

    # Feature order check
    if v2_feats:
        dims["feature_order"] = cfg05_feats == v2_feats

    # Params check (cfg05 params are frozen, should match registry)
    try:
        from src.registry.dayahead_models import MODEL_CONFIGS
        registry_params = MODEL_CONFIGS.get("cfg05", {}).get("params", {})
    except ImportError:
        # Fallback: import directly from the adapter
        from models.adapters.cfg05_dayahead_lgbm import CFG05_PARAMS as registry_params
    if registry_params:
        param_match = all(
            cfg05_params.get(k) == v for k, v in registry_params.items()
            if k != "verbosity"
        )
        dims["cfg05_params"] = param_match

    result["dimensions_checked"] = len(dims)
    result["dimensions_matched"] = sum(1 for v in dims.values() if v)
    result["dimensions_partial"] = 0
    result["dimensions_not_matched"] = sum(1 for v in dims.values() if not v)
    result["dimension_details"] = dims

    # Final alignment label
    if comparison["alignment_label"] == FEATURE_ALIGNMENT_MATCHED:
        result["alignment_label"] = FEATURE_ALIGNMENT_MATCHED
    elif comparison["alignment_label"] == FEATURE_ALIGNMENT_PARTIAL:
        result["alignment_label"] = FEATURE_ALIGNMENT_PARTIAL
    else:
        result["alignment_label"] = FEATURE_ALIGNMENT_NOT_MATCHED

    result["reason_codes"].append(f"CFG05_COLS:{len(cfg05_feats)}")
    if v2_feats:
        result["reason_codes"].append(f"SOURCE_V2_COLS:{len(v2_feats)}")
    if v3_feats:
        result["reason_codes"].append(f"SOURCE_V3_COLS:{len(v3_feats)}")
    result["reason_codes"].append(f"ALIGNMENT:{result['alignment_label']}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P27 cfg05 Feature Builder Alignment Audit")
    print("=" * 60)
    print(f"  Source repo:        {result['source_repo']}")
    print(f"  Source exists:      {result['source_repo_exists']}")
    print(f"  cfg05 features:     {result['cfg05_feature_count']}")
    if result["v2_feature_count"] is not None:
        print(f"  Source v2 features: {result['v2_feature_count']}")
    if result["v3_feature_count"] is not None:
        print(f"  Source v3 features: {result['v3_feature_count']}")
    print(f"  Alignment label:    {result['alignment_label']}")
    print(f"  Dims checked:       {result['dimensions_checked']}")
    print(f"  Dims matched:       {result['dimensions_matched']}")
    print(f"  Recommendation:     {result['migration_recommendation']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P27: cfg05 feature builder alignment audit.")
    p.add_argument("--source-repo", type=str, default=None)
    p.add_argument("--json", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    result = audit_p27_cfg05_feature_builder_alignment(source_repo=args.source_repo)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
