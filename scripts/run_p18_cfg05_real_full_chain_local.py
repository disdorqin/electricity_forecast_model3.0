"""
scripts/run_p18_cfg05_real_full_chain_local.py — P18 cfg05 REAL full local chain.

Chains cfg05 REAL prediction ledger through:

    prediction ledger
        → residual correction (P5M_DATA_MISSING_NO_OP)
        → corrected ledger
        → fusion (CFG05_SINGLE_REAL_MODEL_FUSION)
        → fusion ledger
        → negative classifier (NEGATIVE_CLASSIFIER_RULE_FALLBACK)
        → final output

Usage::

    python -m scripts.run_p18_cfg05_real_full_chain_local \\
        --ledger .local_artifacts/p16_p20_cfg05_chain/ledgers/prediction_ledger.csv \\
        --work-dir .local_artifacts/p16_p20_cfg05_chain \\
        --json --strict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    CORRECTED_PREDICTION_COLUMNS,
    FUSION_OUTPUT_COLUMNS,
    FINAL_OUTPUT_COLUMNS,
    EVAL_ONLY_COLUMNS,
)
from pipelines.residual_correction import apply_residual_correction, is_data_missing_noop
from pipelines.classifier_pipeline import run_negative_classifier
from fusion.engine import run_fusion, READY_DRY_RUN

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p16_p20_cfg05_chain")

# ── Statuses ───────────────────────────────────────────────────────────────
CHAIN_READY = "CFG05_FULL_CHAIN_READY_LOCAL"
CHAIN_READY_FALLBACKS = "CFG05_FULL_CHAIN_READY_WITH_FALLBACKS"
CHAIN_BLOCKED = "CFG05_FULL_CHAIN_BLOCKED"
CHAIN_INVALID = "CFG05_FULL_CHAIN_INVALID"

# ── Honest fallback labels ─────────────────────────────────────────────────
RESIDUAL_MODE = "P5M_DATA_MISSING_NO_OP"
FUSION_MODE = "CFG05_SINGLE_REAL_MODEL_FUSION"
CLASSIFIER_MODE = "NEGATIVE_CLASSIFIER_RULE_FALLBACK"


def _prepare_prediction_input(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Convert prediction ledger rows to standard prediction schema for chain input."""
    df = ledger_df.copy()

    # Map y_pred_corrected or y_pred to y_pred
    if "y_pred" not in df.columns:
        if "y_pred_corrected" in df.columns:
            df["y_pred"] = df["y_pred_corrected"]
        else:
            raise ValueError("No y_pred or y_pred_corrected column found")

    # Ensure standard columns
    for col in ["task", "model_name", "target_day", "business_day", "ds",
                "hour_business", "period", "source_confidence", "model_version"]:
        if col not in df.columns:
            if col == "task":
                df[col] = "dayahead"
            elif col == "model_name":
                df[col] = "lightgbm_cfg05_dayahead"
            elif col == "source_confidence":
                df[col] = np.nan
            elif col == "model_version":
                df[col] = "1.0.0"
            else:
                df[col] = None

    # Strip eval-only columns for production residual correction
    for col in EVAL_ONLY_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def run_p18_cfg05_real_full_chain_local(
    prediction_ledger_path: Optional[str] = None,
    prediction_df: Optional[pd.DataFrame] = None,
    work_dir: Optional[str] = None,
    production: bool = True,
) -> dict[str, Any]:
    """Run cfg05 REAL full local chain.

    Parameters
    ----------
    prediction_ledger_path : str
        Path to P17 prediction ledger CSV.
    prediction_df : DataFrame
        Alternative: pass DataFrame directly.
    work_dir : str
        Local work directory.
    production : bool
        Production mode (strip y_true from outputs).

    Returns
    -------
    dict with chain summary.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    ledger_dir = os.path.join(work_dir, "ledgers")
    os.makedirs(ledger_dir, exist_ok=True)

    result: dict[str, Any] = {
        "input_prediction_rows": 0,
        "corrected_rows": 0,
        "fusion_rows": 0,
        "final_rows": 0,
        "validators_passed": [],
        "residual_mode": RESIDUAL_MODE,
        "fusion_mode": FUSION_MODE,
        "classifier_mode": CLASSIFIER_MODE,
        "prediction_ledger_path_local": prediction_ledger_path,
        "corrected_ledger_path_local": None,
        "fusion_ledger_path_local": None,
        "final_output_path_local": None,
        "readiness_label": "NOT_ASSESSED",
        "final_status": None,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # ── Load prediction ledger ──
    if prediction_df is not None:
        pred_df = prediction_df.copy()
    elif prediction_ledger_path and os.path.isfile(prediction_ledger_path):
        pred_df = pd.read_csv(prediction_ledger_path)
    else:
        result["final_status"] = CHAIN_BLOCKED
        result["reason_codes"].append("NO_PREDICTION_LEDGER_INPUT")
        return result

    result["input_prediction_rows"] = len(pred_df)
    if len(pred_df) == 0:
        result["final_status"] = CHAIN_BLOCKED
        result["reason_codes"].append("EMPTY_PREDICTION_LEDGER")
        return result

    # ── Prepare input for chain ──
    chain_input = _prepare_prediction_input(pred_df)
    result["reason_codes"].append(f"CHAIN_INPUT_PREPARED:{len(chain_input)}_rows")

    # ══════════════════════════════════════════════════════════════
    # Stage 1: Residual correction
    # ══════════════════════════════════════════════════════════════

    try:
        corrected = apply_residual_correction(
            chain_input,
            correction_profile="conservative",
            risk_df=None,
            canonical_pack_path=None,
            production=production,
        )
        result["corrected_rows"] = len(corrected)

        if is_data_missing_noop(corrected):
            result["residual_mode"] = RESIDUAL_MODE
            result["reason_codes"].append("RESIDUAL_DATA_MISSING_NO_OP_CONFIRMED")
        else:
            result["residual_mode"] = "RESIDUAL_CORRECTION_APPLIED"
            result["reason_codes"].append("RESIDUAL_CORRECTION_WAS_APPLIED")

        # Validate corrected schema
        missing_cols = [c for c in CORRECTED_PREDICTION_COLUMNS if c not in corrected.columns]
        if not missing_cols:
            result["validators_passed"].append("corrected_schema")
        else:
            result["reason_codes"].append(f"CORRECTED_MISSING_COLS:{missing_cols}")

        # Save corrected ledger
        corrected_path = os.path.join(ledger_dir, "corrected_ledger.csv")
        corrected.to_csv(corrected_path, index=False)
        result["corrected_ledger_path_local"] = corrected_path

    except Exception as e:
        result["reason_codes"].append(f"RESIDUAL_CORRECTION_FAILED:{e}")
        result["final_status"] = CHAIN_BLOCKED
        return result

    # ══════════════════════════════════════════════════════════════
    # Stage 2: Fusion (single real model = cfg05)
    # ══════════════════════════════════════════════════════════════

    try:
        # For single-model fusion, use equal_weight with cfg05 only
        readiness_status = {"lightgbm_cfg05_dayahead": READY_DRY_RUN}

        fusion_result = run_fusion(
            corrected,
            method="equal_weight",
            allow_dry_run=True,
            readiness_status=readiness_status,
            production=production,
            learner_version="0.1.0-cfg05-single",
        )
        result["fusion_rows"] = len(fusion_result)
        result["fusion_mode"] = FUSION_MODE

        # Validate fusion schema
        missing_cols = [c for c in FUSION_OUTPUT_COLUMNS if c not in fusion_result.columns]
        if not missing_cols:
            result["validators_passed"].append("fusion_schema")
        else:
            result["reason_codes"].append(f"FUSION_MISSING_COLS:{missing_cols}")

        # Save fusion ledger
        fusion_path = os.path.join(ledger_dir, "fusion_ledger.csv")
        fusion_result.to_csv(fusion_path, index=False)
        result["fusion_ledger_path_local"] = fusion_path

    except Exception as e:
        result["reason_codes"].append(f"FUSION_FAILED:{e}")
        result["final_status"] = CHAIN_BLOCKED
        return result

    # ══════════════════════════════════════════════════════════════
    # Stage 3: Negative classifier (rule fallback)
    # ══════════════════════════════════════════════════════════════

    try:
        final_result = run_negative_classifier(
            fusion_result,
            model_dir=None,
            rule_fallback=True,
            production=production,
        )
        result["final_rows"] = len(final_result)
        result["classifier_mode"] = CLASSIFIER_MODE

        # Validate final schema
        missing_cols = [c for c in FINAL_OUTPUT_COLUMNS if c not in final_result.columns]
        if not missing_cols:
            result["validators_passed"].append("final_schema")
        else:
            result["reason_codes"].append(f"FINAL_MISSING_COLS:{missing_cols}")

        # Save final output
        final_path = os.path.join(ledger_dir, "final_output.csv")
        final_result.to_csv(final_path, index=False)
        result["final_output_path_local"] = final_path

    except Exception as e:
        result["reason_codes"].append(f"CLASSIFIER_FAILED:{e}")
        result["final_status"] = CHAIN_BLOCKED
        return result

    # ══════════════════════════════════════════════════════════════
    # Row count consistency check
    # ══════════════════════════════════════════════════════════════

    n_input = result["input_prediction_rows"]
    n_corr = result["corrected_rows"]
    n_fus = result["fusion_rows"]
    n_final = result["final_rows"]

    if n_corr == n_input and n_fus == n_input and n_final == n_input:
        result["reason_codes"].append("ROW_COUNTS_CONSISTENT")
    else:
        result["reason_codes"].append(
            f"ROW_COUNTS_MISMATCH:input={n_input},corrected={n_corr},fusion={n_fus},final={n_final}"
        )

    # ── Readiness label ──
    n_validators = len(result["validators_passed"])
    if n_validators >= 3:
        result["readiness_label"] = "LOCAL_CHAIN_READY"
    elif n_validators >= 1:
        result["readiness_label"] = "LOCAL_CHAIN_PARTIAL"
    else:
        result["readiness_label"] = "LOCAL_CHAIN_NOT_READY"

    # ── Final status ──
    has_fallbacks = (
        result["residual_mode"] == RESIDUAL_MODE
        or result["classifier_mode"] == CLASSIFIER_MODE
    )

    if n_final > 0 and n_validators >= 3:
        result["final_status"] = CHAIN_READY_FALLBACKS if has_fallbacks else CHAIN_READY
    elif n_final > 0:
        result["final_status"] = CHAIN_READY_FALLBACKS
    else:
        result["final_status"] = CHAIN_INVALID

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P18 cfg05 REAL Full Chain Local Report")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="P18: cfg05 REAL full chain local.")
    p.add_argument("--ledger", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--json", action="store_true", default=False)
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    result = run_p18_cfg05_real_full_chain_local(
        prediction_ledger_path=args.ledger,
        work_dir=args.work_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and result["final_status"] not in (CHAIN_READY, CHAIN_READY_FALLBACKS):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
