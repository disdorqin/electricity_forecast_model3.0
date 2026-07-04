"""
scripts/run_p22_real_cfg05_ledger_materialization.py — P22 real prediction ledger materialization.

Reads P21 predictions (all_predictions.csv from the P21 work directory) and
delegates to P17 for canonical ledger conversion.

Usage::

    python -m scripts.run_p22_real_cfg05_ledger_materialization --json --strict
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
_PREDICTIONS_FILE = "all_predictions.csv"

_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "reports/local/")
_ALLOWED_WORK_DIRS = (".local_artifacts",)

# ── Final statuses ─────────────────────────────────────────────────────────
P22_REAL_PREDICTION_LEDGER_READY = "P22_REAL_PREDICTION_LEDGER_READY"
P22_REAL_PREDICTION_LEDGER_PARTIAL = "P22_REAL_PREDICTION_LEDGER_PARTIAL"
P22_REAL_PREDICTION_LEDGER_FAILED = "P22_REAL_PREDICTION_LEDGER_FAILED"


# ── Helpers ────────────────────────────────────────────────────────────────

def _path_is_safe(path: str) -> bool:
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def run_p22_real_cfg05_ledger_materialization(
    work_dir: Optional[str] = None,
    predictions_path_override: Optional[str] = None,
    mode: str = "eval",
    run_id: str = "p22_real_backtest",
) -> dict[str, Any]:
    """Run P22 real prediction ledger materialization.

    Returns a summary dict with ``final_status``.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    result: dict[str, Any] = {
        "p22_stage": "initialization",
        "predictions_path": None,
        "p17_summary": None,
        "ledger_path_local": None,
        "final_status": None,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # ── Step 1: locate predictions ──
    predictions_path = predictions_path_override or os.path.join(work_dir, _PREDICTIONS_FILE)
    if not os.path.isfile(predictions_path):
        result["final_status"] = P22_REAL_PREDICTION_LEDGER_FAILED
        result["reason_codes"].append(f"PREDICTIONS_FILE_NOT_FOUND:{predictions_path}")
        return result
    result["predictions_path"] = predictions_path
    result["reason_codes"].append(f"PREDICTIONS_FOUND:{predictions_path}")

    # ── Step 2: validate predictions are non-empty ──
    try:
        pred_df = pd.read_csv(predictions_path)
        if len(pred_df) == 0:
            result["final_status"] = P22_REAL_PREDICTION_LEDGER_FAILED
            result["reason_codes"].append("PREDICTIONS_EMPTY")
            return result
        result["reason_codes"].append(f"PREDICTIONS_ROWS:{len(pred_df)}")
    except Exception as e:
        result["final_status"] = P22_REAL_PREDICTION_LEDGER_FAILED
        result["reason_codes"].append(f"PREDICTIONS_READ_FAILED:{e}")
        return result

    # ── Step 3: call P17 ──
    result["p22_stage"] = "ledger_materialization"
    try:
        from scripts.run_p17_cfg05_predictions_to_ledger import (
            run_p17_cfg05_predictions_to_ledger,
            LEDGER_READY,
        )
        p17_result = run_p17_cfg05_predictions_to_ledger(
            predictions_path=predictions_path,
            work_dir=work_dir,
            mode=mode,
            run_id=run_id,
        )
        result["p17_summary"] = p17_result
        result["ledger_path_local"] = p17_result.get("ledger_path_local")
        result["reason_codes"].extend(p17_result.get("reason_codes", []))

        # Map P17 status to P22 status
        if p17_result.get("final_status") == LEDGER_READY:
            result["final_status"] = P22_REAL_PREDICTION_LEDGER_READY
        elif p17_result.get("ledger_rows", 0) > 0:
            result["final_status"] = P22_REAL_PREDICTION_LEDGER_PARTIAL
        else:
            result["final_status"] = P22_REAL_PREDICTION_LEDGER_FAILED

    except Exception as e:
        result["final_status"] = P22_REAL_PREDICTION_LEDGER_FAILED
        result["reason_codes"].append(f"P17_CALL_FAILED:{e}")

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
    print("P22 Real Prediction Ledger Materialization Report")
    print("=" * 60)
    print(f"  Predictions path:  {result['predictions_path']}")
    print(f"  Ledger path:       {result['ledger_path_local']}")
    print(f"  Final status:      {result['final_status']}")
    print(f"  Forbidden chk:     {result['forbidden_files_check']}")
    if result.get("p17_summary"):
        p17 = result["p17_summary"]
        print(f"  P17 status:        {p17.get('final_status')}")
        print(f"  P17 ledger rows:   {p17.get('ledger_rows')}")
        print(f"  P17 complete days: {p17.get('complete_days')}/{p17.get('target_days')}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P22: real prediction ledger materialization.")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--predictions", type=str, default=None, help="Override predictions CSV path.")
    p.add_argument("--mode", type=str, default="eval", choices=["eval", "production"])
    p.add_argument("--run-id", type=str, default="p22_real_backtest")
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

    result = run_p22_real_cfg05_ledger_materialization(
        work_dir=work_dir,
        predictions_path_override=args.predictions,
        mode=args.mode,
        run_id=args.run_id,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and result["final_status"] != P22_REAL_PREDICTION_LEDGER_READY:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
