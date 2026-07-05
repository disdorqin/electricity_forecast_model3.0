"""
scripts/run_p33_multimodel_prediction_ledger.py — P33: Multi-model prediction ledger.

Consolidates individual model 30-day prediction CSVs into a unified prediction
ledger in canonical PREDICTION_LEDGER_COLUMNS format.

Usage::

    python -m scripts.run_p33_multimodel_prediction_ledger

Options::

    --work-dir PATH    Model artifacts dir.
    --output-dir PATH  Output dir for ledger CSVs.
    --force            Overwrite existing.
    --json             Output JSON report.
    --strict           Exit non-zero on failures.
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

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")

_MODEL_NAMES = [
    "cfg05_dayahead_lgbm",
    "best_two_average",
    "stage3_business_fixed",
    "catboost_sota",
    "catboost_spike_residual",
]

PREDICTION_LEDGER_COLUMNS = [
    "task", "model_name", "target_day", "business_day", "ds",
    "hour_business", "period", "y_pred", "source_confidence",
    "model_version", "run_id", "created_at", "updated_at",
]


def build_prediction_ledger(
    work_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build unified prediction ledger from per-model 30d CSVs."""
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = output_dir or os.path.join(work_dir, "ledger")
    os.makedirs(ledger_dir, exist_ok=True)

    result: dict[str, Any] = {
        "phase": "P33",
        "models_found": 0,
        "models_missing": 0,
        "total_rows": 0,
        "ledger_path": os.path.join(ledger_dir, "prediction_ledger_30d.csv"),
        "p33_status": "P33_NOT_STARTED",
        "reason_codes": [],
    }

    all_dfs = []
    for model_name in _MODEL_NAMES:
        csv_path = os.path.join(ledger_dir, f"predictions_{model_name}_30d.csv")
        if not os.path.isfile(csv_path):
            result["models_missing"] += 1
            result["reason_codes"].append(f"MISSING:{model_name}")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Ensure columns match prediction ledger format
            for col in ["run_id", "created_at", "updated_at"]:
                if col not in df.columns:
                    df[col] = "p33_ledger" if col == "run_id" else pd.Timestamp.now().isoformat()
            # Reorder to canonical format
            available = [c for c in PREDICTION_LEDGER_COLUMNS if c in df.columns]
            df = df[available]
            all_dfs.append(df)
            result["models_found"] += 1
            result["reason_codes"].append(f"LOADED:{model_name}({len(df)}rows)")
        except Exception as e:
            result["reason_codes"].append(f"LOAD_FAILED:{model_name}:{e}")

    if len(all_dfs) == 0:
        result["p33_status"] = "P33_NO_DATA"
        return result

    ledger = pd.concat(all_dfs, ignore_index=True)
    ledger = ledger.sort_values(["model_name", "business_day", "hour_business"]).reset_index(drop=True)
    result["total_rows"] = len(ledger)

    try:
        ledger.to_csv(result["ledger_path"], index=False)
        result["reason_codes"].append(f"SAVED:{result['ledger_path']}({len(ledger)}rows)")
        result["p33_status"] = "P33_LEDGER_BUILT"
    except Exception as e:
        result["reason_codes"].append(f"SAVE_FAILED:{e}")
        result["p33_status"] = "P33_SAVE_FAILED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P33 — Multi-Model Prediction Ledger")
    print("=" * 60)
    print(f"  Models found:      {result['models_found']}")
    print(f"  Models missing:    {result['models_missing']}")
    print(f"  Total rows:        {result['total_rows']}")
    print(f"  Ledger path:       {result['ledger_path']}")
    print(f"  Status:            {result['p33_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P33: Build prediction ledger.")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    result = build_prediction_ledger(
        work_dir=args.work_dir, output_dir=args.output_dir, force=args.force,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict and result["p33_status"] != "P33_LEDGER_BUILT":
        logger.error("P33: FAIL")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
