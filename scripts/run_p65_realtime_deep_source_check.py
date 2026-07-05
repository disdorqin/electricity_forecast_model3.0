"""
scripts/run_p65_realtime_deep_source_check.py — P65: Realtime Deep Source Check.

Validates the deep realtime source repo and adapter availability.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

P65_SOURCE_READY = "P65_SOURCE_READY"
P65_SOURCE_PARTIAL = "P65_SOURCE_PARTIAL"
P65_SOURCE_MISSING = "P65_SOURCE_MISSING"


def run_p65_realtime_deep_source_check(
    realtime_source_repo: str = "",
    sgdfnet_root: str = "",
    raw_data: str = "",
) -> dict[str, Any]:
    """Check realtime deep source availability."""
    result: dict[str, Any] = {
        "p65_status": P65_SOURCE_MISSING,
        "source_repo": {"exists": False, "path": realtime_source_repo},
        "sgdfnet_root": {"exists": False, "path": sgdfnet_root},
        "raw_data": {"exists": False, "path": raw_data},
        "adapter_available": False,
        "reason_codes": [],
    }

    if realtime_source_repo and os.path.isdir(realtime_source_repo):
        result["source_repo"]["exists"] = True
        result["reason_codes"].append("SOURCE_REPO_FOUND")

    if sgdfnet_root and os.path.isdir(sgdfnet_root):
        result["sgdfnet_root"]["exists"] = True
        result["reason_codes"].append("SGDFNET_ROOT_FOUND")

    if raw_data and os.path.isfile(raw_data):
        result["raw_data"]["exists"] = True
        result["reason_codes"].append("RAW_DATA_FOUND")

    # Check adapter
    try:
        from models.adapters.realtime_deep_adapter import RealtimeDeepAdapter
        adapter = RealtimeDeepAdapter(
            source_repo_path=realtime_source_repo,
            raw_data_path=raw_data,
            sgdfnet_root=sgdfnet_root,
        )
        env = adapter.check_environment()
        result["adapter_available"] = True
        result["environment"] = env
    except Exception as e:
        result["reason_codes"].append(f"ADAPTER_ERROR:{e}")

    # Determine status
    if result["source_repo"]["exists"] and result["adapter_available"]:
        result["p65_status"] = P65_SOURCE_READY
    elif result["source_repo"]["exists"] or result["adapter_available"]:
        result["p65_status"] = P65_SOURCE_PARTIAL
    else:
        result["p65_status"] = P65_SOURCE_MISSING

    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="P65: Realtime deep source check")
    p.add_argument("--realtime-source-repo", type=str, default="")
    p.add_argument("--sgdfnet-root", type=str, default="")
    p.add_argument("--raw-data", type=str, default="")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = run_p65_realtime_deep_source_check(
        realtime_source_repo=args.realtime_source_repo,
        sgdfnet_root=args.sgdfnet_root,
        raw_data=args.raw_data,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"P65 Status: {result['p65_status']}")
        for rc in result.get("reason_codes", []):
            print(f"  -> {rc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
