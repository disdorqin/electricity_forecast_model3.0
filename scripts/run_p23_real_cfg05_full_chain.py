"""
scripts/run_p23_real_cfg05_full_chain.py — P23 real cfg05 full chain orchestration.

Reads the P22 prediction ledger and delegates to P18 for the full chain
(residual correction -> fusion -> negative classifier -> final output).

Usage::

    python -m scripts.run_p23_real_cfg05_full_chain --json --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p21_p25_real_chain")
_LEDGER_SUBDIR = "ledgers"
_LEDGER_FILE = "prediction_ledger.csv"

_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "reports/local/")
_ALLOWED_WORK_DIRS = (".local_artifacts",)

# ── Final statuses ─────────────────────────────────────────────────────────
P23_REAL_CFG05_FULL_CHAIN_READY_WITH_FALLBACKS = "P23_REAL_CFG05_FULL_CHAIN_READY_WITH_FALLBACKS"
P23_REAL_CFG05_FULL_CHAIN_PARTIAL = "P23_REAL_CFG05_FULL_CHAIN_PARTIAL"
P23_REAL_CFG05_FULL_CHAIN_FAILED = "P23_REAL_CFG05_FULL_CHAIN_FAILED"


# ── Helpers ────────────────────────────────────────────────────────────────

def _path_is_safe(path: str) -> bool:
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def run_p23_real_cfg05_full_chain(
    work_dir: Optional[str] = None,
    ledger_path_override: Optional[str] = None,
    production: bool = True,
) -> dict[str, Any]:
    """Run P23 real cfg05 full chain orchestration.

    Returns a summary dict with ``final_status``.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    result: dict[str, Any] = {
        "p23_stage": "initialization",
        "ledger_path": None,
        "p18_summary": None,
        "final_output_path_local": None,
        "validators_passed": [],
        "final_status": None,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # ── Step 1: locate prediction ledger ──
    ledger_path = ledger_path_override or os.path.join(work_dir, _LEDGER_SUBDIR, _LEDGER_FILE)
    if not os.path.isfile(ledger_path):
        result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_FAILED
        result["reason_codes"].append(f"PREDICTION_LEDGER_NOT_FOUND:{ledger_path}")
        return result
    result["ledger_path"] = ledger_path
    result["reason_codes"].append(f"PREDICTION_LEDGER_FOUND:{ledger_path}")

    # ── Step 2: validate ledger is non-empty ──
    try:
        ledger_df = pd.read_csv(ledger_path)
        if len(ledger_df) == 0:
            result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_FAILED
            result["reason_codes"].append("PREDICTION_LEDGER_EMPTY")
            return result
        result["reason_codes"].append(f"LEDGER_ROWS:{len(ledger_df)}")
    except Exception as e:
        result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_FAILED
        result["reason_codes"].append(f"LEDGER_READ_FAILED:{e}")
        return result

    # ── Step 3: call P18 ──
    result["p23_stage"] = "full_chain_execution"
    try:
        from scripts.run_p18_cfg05_real_full_chain_local import (
            run_p18_cfg05_real_full_chain_local,
            CHAIN_READY,
            CHAIN_READY_FALLBACKS,
        )
        p18_result = run_p18_cfg05_real_full_chain_local(
            prediction_ledger_path=ledger_path,
            work_dir=work_dir,
            production=production,
        )
        result["p18_summary"] = p18_result
        result["final_output_path_local"] = p18_result.get("final_output_path_local")
        result["validators_passed"] = p18_result.get("validators_passed", [])
        result["reason_codes"].extend(p18_result.get("reason_codes", []))

        # Map P18 status to P23 status
        p18_status = p18_result.get("final_status")
        if p18_status in (CHAIN_READY, CHAIN_READY_FALLBACKS):
            if p18_result.get("final_rows", 0) > 0:
                result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_READY_WITH_FALLBACKS
            else:
                result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_FAILED
        elif p18_result.get("final_rows", 0) > 0:
            result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_PARTIAL
        else:
            result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_FAILED

    except Exception as e:
        result["final_status"] = P23_REAL_CFG05_FULL_CHAIN_FAILED
        result["reason_codes"].append(f"P18_CALL_FAILED:{e}")

    # ── Forbidden files check ──
    work_dir_norm = os.path.abspath(work_dir).replace("\\", "/")
    work_dir_is_safe = (
        any(a.lstrip(".") in work_dir_norm for a in _ALLOWED_WORK_DIRS)
        or os.path.isabs(work_dir)
    )
    if not work_dir_is_safe:
        result["forbidden_files_check"] = "FAIL"
    else:
        result["forbidden_files_check"] = "PASS"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P23 Real cfg05 Full Chain Report")
    print("=" * 60)
    print(f"  Ledger path:       {result['ledger_path']}")
    print(f"  Final output:      {result['final_output_path_local']}")
    print(f"  Validators passed: {result['validators_passed']}")
    print(f"  Final status:      {result['final_status']}")
    print(f"  Forbidden chk:     {result['forbidden_files_check']}")
    if result.get("p18_summary"):
        p18 = result["p18_summary"]
        print(f"  P18 status:        {p18.get('final_status')}")
        print(f"  P18 final rows:    {p18.get('final_rows')}")
        print(f"  P18 readiness:     {p18.get('readiness_label')}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P23: real cfg05 full chain orchestration.")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--ledger", type=str, default=None, help="Override prediction ledger path.")
    p.add_argument("--json", action="store_true", default=False)
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    work_dir = args.work_dir or _DEFAULT_WORK_DIR
    if not _path_is_safe(work_dir):
        logger.error("Unsafe work-dir: %s", work_dir)
        return 1

    result = run_p23_real_cfg05_full_chain(
        work_dir=work_dir,
        ledger_path_override=args.ledger,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and result["final_status"] != P23_REAL_CFG05_FULL_CHAIN_READY_WITH_FALLBACKS:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
