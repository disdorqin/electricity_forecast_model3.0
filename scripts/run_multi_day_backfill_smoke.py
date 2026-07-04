"""
scripts/run_multi_day_backfill_smoke.py — CLI for multi-day backfill smoke.

Usage::

    # 30-day default smoke
    python -m scripts.run_multi_day_backfill_smoke --start-day 2026-06-01

    # With ledger output
    python -m scripts.run_multi_day_backfill_smoke --start-day 2026-06-01 \\
        --n-days 30 --ledger-dir /tmp/smoke_ledgers

    # 3-day quick smoke
    python -m scripts.run_multi_day_backfill_smoke --start-day 2026-07-01 \\
        --n-days 3

Options:

    --start-day YYYY-MM-DD             First target day (required).
    --n-days N                         Number of days (default: 30).
    --ledger-dir PATH                  Directory for ledger CSV output.
    --allow-dry-run                    Allow dry-run models in fusion (default: on).
    --classifier-rule-fallback         Apply rule-based negative flag (default: on).
    --no-classifier-rule-fallback
    --generate-synthetic-actuals       Generate synthetic actuals (default: on).
    --no-generate-synthetic-actuals
    --fusion-method METHOD             equal_weight | prior_weight | bgew_skeleton
    --production                       Production mode (default: True).
    --verbose, -v                      Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pipelines.multi_day_backfill_smoke import run_multi_day_backfill_smoke

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-day ledger backfill structural smoke (P8).",
    )
    parser.add_argument("--start-day", type=str, required=True,
                        help="First target day (YYYY-MM-DD).")
    parser.add_argument("--n-days", type=int, default=30,
                        help="Number of consecutive days (default: 30).")
    parser.add_argument("--ledger-dir", type=str, default=None,
                        help="Directory for ledger CSV output.")
    parser.add_argument("--allow-dry-run", action="store_true", default=True,
                        help="Allow dry-run models in fusion (default: on).")
    parser.add_argument("--no-allow-dry-run", dest="allow_dry_run",
                        action="store_false",
                        help="Disable dry-run models in fusion.")
    parser.add_argument("--classifier-rule-fallback", action="store_true",
                        default=True,
                        help="Apply rule-based negative flag (default: on).")
    parser.add_argument("--no-classifier-rule-fallback",
                        dest="classifier_rule_fallback",
                        action="store_false",
                        help="Disable rule-based negative flag.")
    parser.add_argument("--generate-synthetic-actuals", action="store_true",
                        default=True,
                        help="Generate synthetic actuals (default: on).")
    parser.add_argument("--no-generate-synthetic-actuals",
                        dest="generate_synthetic_actuals",
                        action="store_false",
                        help="Skip synthetic actual generation.")
    parser.add_argument("--fusion-method", type=str, default="equal_weight",
                        choices=["equal_weight", "prior_weight", "bgew_skeleton"],
                        help="Fusion method (default: equal_weight).")
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
        "Starting multi-day backfill smoke: start=%s, n_days=%d, method=%s",
        args.start_day, args.n_days, args.fusion_method,
    )

    summary = run_multi_day_backfill_smoke(
        start_day=args.start_day,
        n_days=args.n_days,
        ledger_dir=args.ledger_dir,
        allow_dry_run=args.allow_dry_run,
        classifier_rule_fallback=args.classifier_rule_fallback,
        generate_synthetic_actuals=args.generate_synthetic_actuals,
        fusion_method=args.fusion_method,
        production=args.production,
    )

    print(json.dumps(summary, indent=2, default=str))

    if summary["overall_status"] == "PASS":
        logger.info("Multi-day backfill smoke PASSED")
        return 0
    else:
        logger.error("Multi-day backfill smoke FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
