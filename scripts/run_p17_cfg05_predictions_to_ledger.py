"""
scripts/run_p17_cfg05_predictions_to_ledger.py — P17 predictions → prediction ledger.

Converts P16 local predictions into the 3.0 canonical prediction ledger schema.

Usage::

    python -m scripts.run_p17_cfg05_predictions_to_ledger \\
        --predictions .local_artifacts/p16_p20_cfg05_chain/all_predictions.csv \\
        --work-dir .local_artifacts/p16_p20_cfg05_chain \\
        --mode eval --json --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from data.schema import (
    PREDICTION_LEDGER_COLUMNS,
    PREDICTION_LEDGER_KEY,
    EVAL_ONLY_COLUMNS,
)
from data.business_day import add_business_time_columns
from scripts.check_cfg05_hour24_completeness import COMPLETE_24H

logger = logging.getLogger(__name__)

# ── Statuses ───────────────────────────────────────────────────────────────
LEDGER_READY = "CFG05_PREDICTION_LEDGER_READY_LOCAL"
LEDGER_INCOMPLETE = "CFG05_PREDICTION_LEDGER_INCOMPLETE"
LEDGER_INVALID = "CFG05_PREDICTION_LEDGER_INVALID"

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p16_p20_cfg05_chain")
_LEDGER_SUBDIR = "ledgers"


def _ensure_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure business_day, hour_business, period columns exist."""
    df = df.copy()
    if "ds" in df.columns and "business_day" not in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
        df = add_business_time_columns(df, timestamp_col="ds")
    return df


def run_p17_cfg05_predictions_to_ledger(
    predictions_path: Optional[str] = None,
    predictions_df: Optional[pd.DataFrame] = None,
    work_dir: Optional[str] = None,
    mode: str = "eval",
    run_id: str = "p17_backtest",
) -> dict[str, Any]:
    """Convert predictions to canonical prediction ledger.

    Parameters
    ----------
    predictions_path : str
        Path to P16 all_predictions.csv.
    predictions_df : DataFrame
        Alternative: pass DataFrame directly.
    work_dir : str
        Local work directory.
    mode : str
        'eval' (allow y_true) or 'production' (strip y_true).
    run_id : str
        Run identifier.

    Returns
    -------
    dict with summary.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, _LEDGER_SUBDIR)
    os.makedirs(ledger_dir, exist_ok=True)

    result: dict[str, Any] = {
        "input_rows": 0,
        "ledger_rows": 0,
        "target_days": 0,
        "complete_days": 0,
        "duplicate_keys": 0,
        "schema_valid": False,
        "completeness_status": "NOT_CHECKED",
        "ledger_path_local": None,
        "final_status": None,
        "reason_codes": [],
    }

    # ── Load predictions ──
    if predictions_df is not None:
        df = predictions_df.copy()
    elif predictions_path and os.path.isfile(predictions_path):
        df = pd.read_csv(predictions_path)
    else:
        result["final_status"] = LEDGER_INCOMPLETE
        result["reason_codes"].append("NO_PREDICTIONS_INPUT")
        return result

    result["input_rows"] = len(df)
    if len(df) == 0:
        result["final_status"] = LEDGER_INCOMPLETE
        result["reason_codes"].append("EMPTY_PREDICTIONS")
        return result

    # ── Ensure business columns ──
    df = _ensure_business_columns(df)

    # ── Production mode: strip y_true ──
    if mode == "production":
        for col in EVAL_ONLY_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
                result["reason_codes"].append(f"STRIPPED_EVAL_COLUMN:{col}")

    # ── Add ledger metadata columns ──
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()
    df["run_id"] = run_id
    df["created_at"] = now_str
    df["updated_at"] = now_str

    # ── Ensure all PREDICTION_LEDGER_COLUMNS exist ──
    for col in PREDICTION_LEDGER_COLUMNS:
        if col not in df.columns:
            if col == "source_confidence":
                df[col] = None
            elif col == "model_version":
                df[col] = "1.0.0"
            else:
                df[col] = None

    # Select only ledger columns (+ y_true if eval mode)
    ledger_cols = list(PREDICTION_LEDGER_COLUMNS)
    if mode == "eval" and "y_true" in df.columns:
        ledger_cols.append("y_true")

    ledger = df[ledger_cols].copy()

    # ── Check duplicate keys ──
    dup_mask = ledger.duplicated(subset=PREDICTION_LEDGER_KEY, keep=False)
    n_dups = int(dup_mask.sum())
    result["duplicate_keys"] = n_dups
    if n_dups > 0:
        ledger = ledger.drop_duplicates(subset=PREDICTION_LEDGER_KEY, keep="last")
        result["reason_codes"].append(f"DEDUPED:{n_dups}_DUPLICATE_ROWS")

    result["ledger_rows"] = len(ledger)

    # ── Check completeness per target_day ──
    if "target_day" in ledger.columns:
        target_days = ledger["target_day"].unique()
        result["target_days"] = len(target_days)

        complete = 0
        for td in target_days:
            day_df = ledger[ledger["target_day"] == td]
            if "hour_business" in day_df.columns:
                hours = sorted(day_df["hour_business"].unique())
                if len(hours) == 24 and hours == list(range(1, 25)):
                    complete += 1
        result["complete_days"] = complete
    else:
        result["target_days"] = 0
        result["complete_days"] = 0

    # ── Schema validation ──
    missing_cols = [c for c in PREDICTION_LEDGER_COLUMNS if c not in ledger.columns]
    if not missing_cols:
        result["schema_valid"] = True
    else:
        result["reason_codes"].append(f"MISSING_SCHEMA_COLS:{missing_cols}")

    # ── hour_business range check ──
    if "hour_business" in ledger.columns:
        invalid_hours = ledger[~ledger["hour_business"].between(1, 24)]
        if len(invalid_hours) > 0:
            result["reason_codes"].append(f"INVALID_HOURS:{len(invalid_hours)}")
            result["schema_valid"] = False

    # ── Production mode: no y_true ──
    if mode == "production" and "y_true" in ledger.columns:
        result["reason_codes"].append("PRODUCTION_MODE_BUT_YTRUE_PRESENT")
        result["schema_valid"] = False

    result["completeness_status"] = COMPLETE_24H if result["complete_days"] == result["target_days"] else "INCOMPLETE"

    # ── Save ledger ──
    ledger_path = os.path.join(ledger_dir, "prediction_ledger.csv")
    ledger.to_csv(ledger_path, index=False)
    result["ledger_path_local"] = ledger_path

    # ── Final status ──
    if result["schema_valid"] and result["complete_days"] == result["target_days"] and result["ledger_rows"] > 0:
        result["final_status"] = LEDGER_READY
    elif result["ledger_rows"] > 0:
        result["final_status"] = LEDGER_INCOMPLETE
        result["reason_codes"].append(f"COMPLETENESS:{result['complete_days']}/{result['target_days']}")
    else:
        result["final_status"] = LEDGER_INVALID

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P17 cfg05 Prediction Ledger Report")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="P17: predictions → prediction ledger.")
    p.add_argument("--predictions", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--mode", type=str, default="eval", choices=["eval", "production"])
    p.add_argument("--run-id", type=str, default="p17_backtest")
    p.add_argument("--json", action="store_true", default=False)
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    result = run_p17_cfg05_predictions_to_ledger(
        predictions_path=args.predictions,
        work_dir=args.work_dir,
        mode=args.mode,
        run_id=args.run_id,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and result["final_status"] != LEDGER_READY:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
