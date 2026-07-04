"""
scripts/run_full_chain_smoke.py — CLI for single-day full-chain structural smoke.

Usage::

    # Default dry-run smoke
    python -m scripts.run_full_chain_smoke --target-day 2026-07-04

    # With ledger output to a specific directory
    python -m scripts.run_full_chain_smoke --target-day 2026-07-04 \\
        --ledger-dir /tmp/smoke_ledgers

    # With artifact paths (if available)
    python -m scripts.run_full_chain_smoke --target-day 2026-07-04 \\
        --cfg05-artifact-path /path/to/cfg05/model.pkl

Options:

    --target-day YYYY-MM-DD     Target day for the smoke (required).
    --ledger-dir PATH           Directory for ledger CSV output.
    --allow-dry-run             Allow dry-run models in fusion (default: on).
    --use-realtime              Include realtime path (default: off).
    --classifier-rule-fallback  Apply rule-based negative flag (default: on).
    --no-classifier-rule-fallback
    --cfg05-artifact-path PATH  Path to cfg05 model artifact.
    --rt-assist-pack-path PATH  Path to realtime assist pack.
    --residual-pack-path PATH   Path to residual canonical pack.
    --classifier-model-dir PATH Directory for classifier artifacts.
    --production                Production mode (default: True).
    --verbose, -v               Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pipelines.full_chain_smoke import run_full_chain_smoke

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-day full-chain structural smoke (P7).",
    )
    parser.add_argument("--target-day", type=str, required=True,
                        help="Target day (YYYY-MM-DD) for the smoke.")
    parser.add_argument("--ledger-dir", type=str, default=None,
                        help="Directory for ledger CSV output.")
    parser.add_argument("--allow-dry-run", action="store_true", default=True,
                        help="Allow dry-run models in fusion (default: on).")
    parser.add_argument("--no-allow-dry-run", dest="allow_dry_run",
                        action="store_false",
                        help="Disable dry-run models in fusion.")
    parser.add_argument("--use-realtime", action="store_true", default=False,
                        help="Include realtime path.")
    parser.add_argument("--classifier-rule-fallback", action="store_true",
                        default=True,
                        help="Apply rule-based negative flag (default: on).")
    parser.add_argument("--no-classifier-rule-fallback",
                        dest="classifier_rule_fallback",
                        action="store_false",
                        help="Disable rule-based negative flag.")
    parser.add_argument("--cfg05-artifact-path", type=str, default=None,
                        help="Path to cfg05 model artifact.")
    parser.add_argument("--rt-assist-pack-path", type=str, default=None,
                        help="Path to realtime assist pack.")
    parser.add_argument("--residual-pack-path", type=str, default=None,
                        help="Path to residual canonical pack.")
    parser.add_argument("--classifier-model-dir", type=str, default=None,
                        help="Directory for classifier artifacts.")
    parser.add_argument("--no-production", dest="production",
                        action="store_false", default=True,
                        help="Disable production mode.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        default=False, help="Increase verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    logger.info(
        "Starting full-chain structural smoke for target_day=%s",
        args.target_day,
    )

    summary = run_full_chain_smoke(
        target_day=args.target_day,
        ledger_dir=args.ledger_dir,
        allow_dry_run=args.allow_dry_run,
        use_realtime=args.use_realtime,
        classifier_rule_fallback=args.classifier_rule_fallback,
        cfg05_artifact_path=args.cfg05_artifact_path,
        rt_assist_pack_path=args.rt_assist_pack_path,
        residual_pack_path=args.residual_pack_path,
        classifier_model_dir=args.classifier_model_dir,
        production=args.production,
    )

    # Output summary as JSON to stdout
    print(json.dumps(summary, indent=2, default=str))

    if summary["overall_status"] == "PASS":
        logger.info("Full-chain smoke PASSED")
        return 0
    else:
        logger.error("Full-chain smoke FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
