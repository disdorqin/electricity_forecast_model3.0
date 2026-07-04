"""
pipelines/full_chain_smoke.py — Single-day full-chain structural smoke.

Chains P1–P6 into a single orchestrated dry-run flow:

    synthetic predictions
        → residual correction (DATA_MISSING no-op)
        → corrected output validator
        → corrected ledger append
        → fusion (STRUCTURAL_ONLY, equal_weight)
        → fusion output validator
        → fusion ledger append
        → weight extraction → weight ledger append
        → negative classifier (RULE_FALLBACK / no-artifact)
        → final output validator
        → summary dict

This is **structural / dry-run smoke, not production real inference**.
No REAL labels appear unless artifacts are path-verified.

Usage::

    from pipelines.full_chain_smoke import run_full_chain_smoke

    summary = run_full_chain_smoke(target_day="2026-07-04")
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    CORRECTED_LEDGER_COLUMNS,
    CORRECTED_PREDICTION_COLUMNS,
    FUSION_OUTPUT_COLUMNS,
    FINAL_OUTPUT_COLUMNS,
)
from pipelines.residual_correction import (
    apply_residual_correction,
    is_data_missing_noop,
)
from pipelines.classifier_pipeline import (
    run_negative_classifier,
)
from fusion.engine import run_fusion, READY_DRY_RUN
from ledgers.prediction_ledger import (
    append_corrected_predictions_to_ledger,
    validate_corrected_ledger,
)
from ledgers.fusion_ledger import (
    append_fusion_to_ledger,
    validate_fusion_ledger,
)
from ledgers.weight_ledger import (
    extract_weight_rows,
    append_weights_to_ledger,
    validate_weight_ledger,
)
from scripts.validate_prediction_output import validate_prediction_dataframe
from scripts.validate_residual_output import validate_residual_dataframe
from scripts.validate_fusion_output import validate_fusion_dataframe
from scripts.validate_final_output import validate_final_dataframe

logger = logging.getLogger(__name__)

# ── Stage label constants ──────────────────────────────────────────────

LABEL_DRY_RUN = "DRY_RUN"
LABEL_STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
LABEL_DATA_MISSING = "DATA_MISSING"
LABEL_RULE_FALLBACK = "RULE_FALLBACK"
LABEL_REAL = "REAL"


def _build_synthetic_predictions(
    target_day: str,
    n_models: int = 2,
    include_negative: bool = False,
) -> pd.DataFrame:
    """Build synthetic prediction output for dry-run smoke.

    Parameters
    ----------
    target_day : str
        Target day (YYYY-MM-DD).
    n_models : int
        Number of models to simulate (default 2).
    include_negative : bool
        If True, some hours get negative prices.

    Returns
    -------
    pd.DataFrame
        Synthetic PREDICTION_OUTPUT_COLUMNS DataFrame (24 rows per model).
    """
    from data.business_day import add_business_time_columns

    model_names = ["cfg05", "best_two_average"][:n_models]
    rng = np.random.default_rng(42)

    base_price = -20.0 if include_negative else 120.0
    timestamps = pd.date_range(f"{target_day} 01:00", periods=24, freq="h")

    rows: list[dict[str, Any]] = []
    for model in model_names:
        prices = base_price + rng.uniform(-10, 30, 24)
        for i in range(24):
            rows.append({
                "task": "dayahead",
                "model_name": model,
                "target_day": target_day,
                "ds": timestamps[i],
                "y_pred": float(prices[i]),
                "source_confidence": 0.9,
                "model_version": "1.0.0",
            })

    df = pd.DataFrame(rows)
    df = add_business_time_columns(df, timestamp_col="ds")

    # Ensure all PREDICTION_OUTPUT_COLUMNS exist
    from data.schema import PREDICTION_OUTPUT_COLUMNS
    for c in PREDICTION_OUTPUT_COLUMNS:
        if c not in df.columns:
            if c == "model_name":
                df[c] = "cfg05"
            else:
                df[c] = None

    return df[PREDICTION_OUTPUT_COLUMNS]


def run_full_chain_smoke(
    target_day: str,
    ledger_dir: Optional[str] = None,
    allow_dry_run: bool = True,
    use_realtime: bool = False,
    classifier_rule_fallback: bool = True,
    cfg05_artifact_path: Optional[str] = None,
    rt_assist_pack_path: Optional[str] = None,
    residual_pack_path: Optional[str] = None,
    classifier_model_dir: Optional[str] = None,
    production: bool = True,
) -> dict[str, Any]:
    """Run a single-day full-chain structural smoke.

    Parameters
    ----------
    target_day : str
        Target day in YYYY-MM-DD format.
    ledger_dir : str, optional
        If provided, ledgers are persisted to this directory.
        Use pytest ``tmp_path`` in tests; do not write to repo paths.
    allow_dry_run : bool
        If True, dry-run models are eligible for fusion (default True).
    use_realtime : bool
        If True, include realtime prediction path (default False; not
        structurally wired, falls back to DA-only).
    classifier_rule_fallback : bool
        Apply rule-based negative price detection (default True).
    cfg05_artifact_path : str, optional
        Path to cfg05 model artifact.  If not None and file exists,
        the day-ahead stage label becomes REAL.
    rt_assist_pack_path : str, optional
        Path to realtime assist pack (unused in default structural smoke).
    residual_pack_path : str, optional
        Path to residual canonical pack (unused in default structural smoke).
    classifier_model_dir : str, optional
        Path to classifier artifacts (unused in default structural smoke).
    production : bool
        Production mode flag (default True).

    Returns
    -------
    dict
        Summary dict with keys:
        - target_day
        - overall_status (PASS / FAIL)
        - mode_label
        - prediction_rows
        - corrected_rows
        - fusion_rows
        - weight_rows
        - final_rows
        - ledger_dir_used
        - stage_labels
        - validators_passed
        - reason_codes
        - forbidden_files_check
    """
    summary: dict[str, Any] = {
        "target_day": target_day,
        "overall_status": "PASS",
        "mode_label": [],
        "prediction_rows": 0,
        "corrected_rows": 0,
        "fusion_rows": 0,
        "weight_rows": 0,
        "final_rows": 0,
        "ledger_dir_used": ledger_dir,
        "stage_labels": {},
        "validators_passed": [],
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    stage_labels: dict[str, str] = {}
    all_reason_codes: list[str] = []
    run_id = f"smoke_{target_day}"

    # ════════════════════════════════════════════
    # 1. Day-ahead prediction
    # ════════════════════════════════════════════

    logger.info("Stage 1/6: Day-ahead prediction (DRY_RUN)")

    # Determine day-ahead label
    da_label = LABEL_REAL if (
        cfg05_artifact_path is not None and os.path.isfile(cfg05_artifact_path)
    ) else LABEL_DRY_RUN

    predictions = _build_synthetic_predictions(
        target_day,
        n_models=2,
        include_negative=True,
    )
    summary["prediction_rows"] = len(predictions)

    # Validate prediction output
    pred_valid, pred_errors = validate_prediction_dataframe(
        predictions, production=production
    )
    if pred_valid:
        summary["validators_passed"].append("prediction_validator")
        all_reason_codes.append("PREDICTION_SCHEMA_VALID")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(f"PREDICTION_VALIDATION_FAILED: {'; '.join(pred_errors)}")

    stage_labels["dayahead"] = da_label
    all_reason_codes.append(f"STAGE_DAYAHEAD_{da_label}")

    # ════════════════════════════════════════════
    # 2. Residual correction
    # ════════════════════════════════════════════

    logger.info("Stage 2/6: Residual correction (DATA_MISSING)")

    residual_label = LABEL_DATA_MISSING
    if residual_pack_path is not None and os.path.isfile(residual_pack_path):
        # Could be upgraded to REAL if artifact verified
        residual_label = "CANONICAL_PACK_AVAILABLE"
        # Still structural-only until real correction applied
        residual_label = f"{LABEL_STRUCTURAL_ONLY}_CANONICAL_PACK"

    corrected = apply_residual_correction(
        predictions,
        correction_profile="conservative",
        risk_df=None,
        canonical_pack_path=residual_pack_path if (
            residual_pack_path and os.path.isfile(residual_pack_path)
        ) else None,
        production=production,
    )
    summary["corrected_rows"] = len(corrected)

    # Verify DATA-MISSING no-op holds
    if is_data_missing_noop(corrected):
        all_reason_codes.append("RESIDUAL_DATA_MISSING_NO_OP")
    else:
        all_reason_codes.append("RESIDUAL_CORRECTION_APPLIED")

    # Validate corrected output
    corr_valid, corr_errors = validate_residual_dataframe(
        corrected, production=production
    )
    if corr_valid:
        summary["validators_passed"].append("residual_validator")
        all_reason_codes.append("CORRECTED_SCHEMA_VALID")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(
            f"CORRECTED_VALIDATION_FAILED: {'; '.join(corr_errors)}"
        )

    stage_labels["residual"] = residual_label
    all_reason_codes.append(f"STAGE_RESIDUAL_{residual_label}")

    # ════════════════════════════════════════════
    # 3. Corrected ledger append
    # ════════════════════════════════════════════

    logger.info("Stage 3/6: Corrected ledger append (STRUCTURAL_ONLY)")

    if ledger_dir is not None:
        ledger_path = os.path.join(ledger_dir, "corrected_ledger.csv")
        from ledgers.store import load_ledger
        existing_corrected = load_ledger(
            ledger_path, columns=CORRECTED_LEDGER_COLUMNS  # type: ignore[arg-type]
        )
    else:
        existing_corrected = None

    corrected_ledger = append_corrected_predictions_to_ledger(
        corrected, ledger_df=existing_corrected, run_id=run_id
    )

    # Validate corrected ledger
    cl_valid, cl_issues = validate_corrected_ledger(corrected_ledger)
    if cl_valid:
        summary["validators_passed"].append("corrected_ledger_validator")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(f"CORRECTED_LEDGER_INVALID: {'; '.join(cl_issues)}")

    if ledger_dir is not None:
        from ledgers.store import save_ledger
        save_ledger(corrected_ledger, ledger_path)

    stage_labels["corrected_ledger"] = LABEL_STRUCTURAL_ONLY
    all_reason_codes.append("STAGE_CORRECTED_LEDGER_STRUCTURAL_ONLY")

    # ════════════════════════════════════════════
    # 4. Fusion
    # ════════════════════════════════════════════

    logger.info("Stage 4/6: Fusion (STRUCTURAL_ONLY)")

    fusion_label = LABEL_STRUCTURAL_ONLY
    readiness_status = {
        "cfg05": READY_DRY_RUN,
        "best_two_average": READY_DRY_RUN,
    }

    fusion_result = run_fusion(
        corrected,
        method="equal_weight",
        allow_dry_run=allow_dry_run,
        readiness_status=readiness_status,
        production=production,
        learner_version="0.1.0-skeleton",
    )
    summary["fusion_rows"] = len(fusion_result)

    # Validate fusion output
    fusion_valid, fusion_errors = validate_fusion_dataframe(
        fusion_result, allow_empty=True, production=production
    )
    if fusion_valid:
        summary["validators_passed"].append("fusion_validator")
        all_reason_codes.append("FUSION_SCHEMA_VALID")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(
            f"FUSION_VALIDATION_FAILED: {'; '.join(fusion_errors)}"
        )

    stage_labels["fusion"] = fusion_label
    all_reason_codes.append(f"STAGE_FUSION_{fusion_label}")

    # ════════════════════════════════════════════
    # 5. Fusion ledger + weight ledger
    # ════════════════════════════════════════════

    logger.info("Stage 5/6: Ledger append (STRUCTURAL_ONLY)")

    # Fusion ledger
    if ledger_dir is not None:
        fl_path = os.path.join(ledger_dir, "fusion_ledger.csv")
        from ledgers.store import load_ledger
        existing_fusion = load_ledger(
            fl_path, columns=FUSION_OUTPUT_COLUMNS  # type: ignore[arg-type]
        )
    else:
        existing_fusion = None

    fusion_ledger = append_fusion_to_ledger(
        fusion_result, ledger_df=existing_fusion, run_id=run_id
    )

    fl_valid, fl_issues = validate_fusion_ledger(fusion_ledger)
    if fl_valid:
        summary["validators_passed"].append("fusion_ledger_validator")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(f"FUSION_LEDGER_INVALID: {'; '.join(fl_issues)}")

    if ledger_dir is not None:
        from ledgers.store import save_ledger
        save_ledger(fusion_ledger, fl_path)

    # Weight extraction
    weight_rows = extract_weight_rows(fusion_result)
    summary["weight_rows"] = len(weight_rows)

    # Weight ledger
    if ledger_dir is not None:
        wl_path = os.path.join(ledger_dir, "weight_ledger.csv")
        from ledgers.store import load_ledger
        existing_weights = load_ledger(wl_path, columns=weight_rows.columns.tolist() if len(weight_rows) > 0 else None)
    else:
        existing_weights = None

    weight_ledger = append_weights_to_ledger(
        weight_rows, ledger_df=existing_weights, run_id=run_id
    )

    wl_valid, wl_issues = validate_weight_ledger(weight_ledger)
    if wl_valid:
        summary["validators_passed"].append("weight_ledger_validator")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(f"WEIGHT_LEDGER_INVALID: {'; '.join(wl_issues)}")

    if ledger_dir is not None:
        from ledgers.store import save_ledger
        save_ledger(weight_ledger, wl_path)

    stage_labels["fusion_ledger"] = LABEL_STRUCTURAL_ONLY
    stage_labels["weight_ledger"] = LABEL_STRUCTURAL_ONLY
    all_reason_codes.append("STAGE_FUSION_LEDGER_STRUCTURAL_ONLY")
    all_reason_codes.append("STAGE_WEIGHT_LEDGER_STRUCTURAL_ONLY")

    # ════════════════════════════════════════════
    # 6. Negative classifier + final output
    # ════════════════════════════════════════════

    logger.info("Stage 6/6: Negative classifier (RULE_FALLBACK)")

    classifier_label = LABEL_RULE_FALLBACK if classifier_rule_fallback else "CLASSIFIER_ARTIFACT_MISSING"

    # Check if real classifier is available
    if classifier_model_dir is not None and os.path.isdir(classifier_model_dir):
        from extreme.negative_classifier import NegativeClassifierAdapter
        adapter = NegativeClassifierAdapter()
        adapter.load(model_dir=classifier_model_dir)
        if adapter._artifact_found:
            classifier_label = LABEL_REAL

    final_result = run_negative_classifier(
        fusion_result,
        model_dir=classifier_model_dir,
        rule_fallback=classifier_rule_fallback,
        production=production,
    )
    summary["final_rows"] = len(final_result)

    # Validate final output
    final_valid, final_errors = validate_final_dataframe(
        final_result, allow_empty=True, production=production
    )
    if final_valid:
        summary["validators_passed"].append("final_output_validator")
        all_reason_codes.append("FINAL_SCHEMA_VALID")
    else:
        summary["overall_status"] = "FAIL"
        all_reason_codes.append(
            f"FINAL_VALIDATION_FAILED: {'; '.join(final_errors)}"
        )

    stage_labels["negative_classifier"] = classifier_label
    all_reason_codes.append(f"STAGE_CLASSIFIER_{classifier_label}")

    # Final stage: final output
    stage_labels["final_output"] = LABEL_STRUCTURAL_ONLY
    all_reason_codes.append("STAGE_FINAL_OUTPUT_STRUCTURAL_ONLY")

    # ════════════════════════════════════════════
    # Summary construction
    # ════════════════════════════════════════════

    # Build mode_label as a list of unique labels used
    mode_labels = sorted(set(stage_labels.values()))
    summary["mode_label"] = mode_labels

    # Ensure REAL is not present without verification
    if LABEL_REAL in mode_labels:
        if not (
            (cfg05_artifact_path and os.path.isfile(cfg05_artifact_path))
            or (classifier_model_dir and os.path.isdir(classifier_model_dir))
        ):
            summary["overall_status"] = "FAIL"
            all_reason_codes.append("ILLEGAL_REAL_LABEL_WITHOUT_ARTIFACT")

    summary["stage_labels"] = stage_labels
    summary["reason_codes"] = all_reason_codes

    # Forbidden files check
    forbidden_paths = ["data/", "outputs/", "reports/local/", "ledgers/*.csv"]
    if ledger_dir is not None:
        # Only allow if ledger_dir is NOT inside repo data paths
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        abs_ledger = os.path.abspath(ledger_dir)
        for forbidden in forbidden_paths:
            forbidden_abs = os.path.abspath(os.path.join(repo_root, forbidden.replace("/*.csv", "").replace("/", os.sep)))
            if abs_ledger.startswith(forbidden_abs):
                summary["forbidden_files_check"] = "FAIL"
                all_reason_codes.append(f"FORBIDDEN_LEDGER_DIR: {ledger_dir}")
                break

    logger.info(
        "Full-chain smoke complete: status=%s, mode=%s, "
        "rows(pred=%d, corr=%d, fusion=%d, weights=%d, final=%d)",
        summary["overall_status"],
        mode_labels,
        summary["prediction_rows"],
        summary["corrected_rows"],
        summary["fusion_rows"],
        summary["weight_rows"],
        summary["final_rows"],
    )

    return summary
