"""
scripts/run_p97_production_artifact_registry.py — Run production artifact registry.

Usage::

    python -m scripts.run_p97_production_artifact_registry \\
        --output-dir .local_artifacts/p97_registry \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="P97: Run production artifact registry")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--output-dir", type=str, default=".local_artifacts/p97_registry")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from artifacts.production_registry import run_production_registry, save_registry_output

    result = run_production_registry(
        config_path=args.config,
        output_dir=args.output_dir,
    )

    paths = save_registry_output(result, args.output_dir)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"P97 Production Artifact Registry: {result.get('overall_assessment', 'UNKNOWN')}")
        print(f"{'='*60}")
        print(f"  Total:  {result['summary']['total']}")
        print(f"  Found:  {result['summary']['found']}")
        print(f"  Loaded: {result['summary']['loaded']}")
        print(f"  Failed: {result['summary']['failed']}")
        print(f"  JSON: {paths.get('json', 'N/A')}")
        print(f"  MD:   {paths.get('md', 'N/A')}")
        if result.get("go_blockers"):
            print(f"\n  GO Blockers ({len(result['go_blockers'])}):")
            for b in result["go_blockers"]:
                print(f"    ❌ {b}")
        if result.get("caveats"):
            print(f"\n  Caveats ({len(result['caveats'])}):")
            for c in result["caveats"]:
                print(f"    ⚠️ {c}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
