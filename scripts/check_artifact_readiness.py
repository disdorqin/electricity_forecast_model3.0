"""
scripts/check_artifact_readiness.py — CLI for artifact readiness checking.

Usage::

    # Quick structural check (all gates MISSING → exit 0)
    python -m scripts.check_artifact_readiness

    # Check with specific paths
    python -m scripts.check_artifact_readiness \\
        --cfg05-model /path/to/cfg05_model.txt \\
        --cfg05-input /path/to/features.csv

    # Strict mode (exit non-zero if any gate not REAL_READY)
    python -m scripts.check_artifact_readiness \\
        --cfg05-model /real/path --strict

    # JSON output
    python -m scripts.check_artifact_readiness --json

Options::

    --cfg05-model PATH              Path to cfg05 model file or directory.
    --cfg05-input PATH              Path to cfg05 input CSV.
    --rt-assist-pack PATH           Path to RT assist pack directory.
    --p5m-pack PATH                 Path to P5M pack directory.
    --actual-ledger PATH            Path to actual ledger CSV.
    --extrempriceclf-dir PATH       Path to ExtremPriceClf model directory.
    --json                          Output raw JSON report.
    --strict                        Exit non-zero if any gate is not REAL_READY.
    --verbose, -v                   Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from artifacts.readiness import run_all_artifact_readiness

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check artifact readiness for all gates.",
    )
    parser.add_argument("--cfg05-model", type=str, default=None,
                        help="Path to cfg05 model file or directory.")
    parser.add_argument("--cfg05-input", type=str, default=None,
                        help="Path to cfg05 input CSV.")
    parser.add_argument("--rt-assist-pack", type=str, default=None,
                        help="Path to RT assist pack directory.")
    parser.add_argument("--p5m-pack", type=str, default=None,
                        help="Path to P5M pack directory.")
    parser.add_argument("--actual-ledger", type=str, default=None,
                        help="Path to actual ledger CSV.")
    parser.add_argument("--extrempriceclf-dir", type=str, default=None,
                        help="Path to ExtremPriceClf model directory.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output raw JSON report.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero if any gate is not REAL_READY.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    report = run_all_artifact_readiness(
        cfg05_model=args.cfg05_model,
        cfg05_input=args.cfg05_input,
        rt_assist_pack=args.rt_assist_pack,
        p5m_pack=args.p5m_pack,
        actual_ledger=args.actual_ledger,
        extrempriceclf_dir=args.extrempriceclf_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report)

    if args.strict:
        if report["summary"]["all_real_ready"]:
            logger.info("All gates REAL_READY")
            return 0
        else:
            logger.error("Not all gates are REAL_READY (strict mode)")
            return 1

    return 0


def _print_report(report: dict) -> None:
    """Print a human-readable readiness report."""
    summary = report["summary"]
    print("=" * 60)
    print("Artifact Readiness Report")
    print("=" * 60)
    print(f"  Total gates:        {summary['total_gates']}")
    print(f"  Status counts:      {summary['status_counts']}")
    print(f"  REAL_READY gates:   {summary['real_ready_gates']}")
    print(f"  All REAL_READY:     {summary['all_real_ready']}")
    print(f"  Any missing:        {summary['any_missing']}")
    print()
    for name, gate in report["gates"].items():
        print(f"  [{gate['status']:20s}] {name}: {gate.get('path', 'N/A')}")
        for rc in gate.get("reason_codes", []):
            print(f"    -> {rc}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
