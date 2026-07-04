"""
pipelines/multi_day_backfill_smoke.py — Multi-day ledger backfill structural smoke.

Extends the P7 single-day full-chain smoke to N days (default 30),
validating ledger continuity, idempotency, key uniqueness, and
actual-ledger no-leakage.

This is **structural / dry-run multi-day smoke, not production real inference**.

Usage::

    from pipelines.multi_day_backfill_smoke import run_multi_day_backfill_smoke

    summary = run_multi_day_backfill_smoke(
        start_day="2026-06-01", n_days=30,
        ledger_dir="/tmp/smoke_ledgers",
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.schema import (
    CORRECTED_LEDGER_COLUMNS,
    CORRECTED_LEDGER_KEY,
    FUSION_OUTPUT_COLUMNS,
    FUSION_LEDGER_COLUMNS,
    FUSION_LEDGER_KEY,
    WEIGHT_LEDGER_COLUMNS,
    WEIGHT_LEDGER_KEY,
    ACTUAL_LEDGER_COLUMNS,
    ACTUAL_LEDGER_KEY,
    FINAL_OUTPUT_COLUMNS,
)
from pipelines.residual_correction import apply_residual_correction
from pipelines.classifier_pipeline import run_negative_classifier
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
from ledgers.actual_ledger import (
    append_actuals_to_ledger,
    validate_actual_ledger,
    filter_actuals_for_training,
)
from ledgers.store import load_ledger, save_ledger, validate_ledger_keys
from scripts.validate_prediction_output import validate_prediction_dataframe
from scripts.validate_residual_output import validate_residual_dataframe
from scripts.validate_fusion_output import validate_fusion_dataframe
from scripts.validate_final_output import validate_final_dataframe

logger = logging.getLogger(__name__)

# ── Label constants ───────────────────────────────────────────────────

LABEL_DRY_RUN = "DRY_RUN"
LABEL_STRUCTURAL_ONLY = "STRUCTURAL_ONLY"
LABEL_DATA_MISSING = "DATA_MISSING"
LABEL_RULE_FALLBACK = "RULE_FALLBACK"
LABEL_REAL = "REAL"


def _build_synthetic_predictions_for_day(
    target_day: str,
    n_models: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Build synthetic prediction output for a single day.

    Parameters
    ----------
    target_day : str
        Target day (YYYY-MM-DD).
    n_models : int
        Number of models (default 2).
    rng : np.random.Generator, optional
        Deterministic RNG for reproducibility.

    Returns
    -------
    pd.DataFrame
        ``PREDICTION_OUTPUT_COLUMNS`` with 24 rows per model.
    """
    from data.business_day import add_business_time_columns
    from data.schema import PREDICTION_OUTPUT_COLUMNS

    if rng is None:
        rng = np.random.default_rng(42)

    model_names = ["cfg05", "best_two_average"][:n_models]
    base_price = 120.0
    timestamps = pd.date_range(f"{target_day} 01:00", periods=24, freq="h")

    rows: list[dict[str, Any]] = []
    for model in model_names:
        prices = base_price + rng.uniform(-15, 25, 24)
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

    for c in PREDICTION_OUTPUT_COLUMNS:
        if c not in df.columns:
            if c == "model_name":
                df[c] = "cfg05"
            else:
                df[c] = None

    return df[PREDICTION_OUTPUT_COLUMNS]


def _build_synthetic_actuals_for_day(
    target_day: str,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Build synthetic actuals for a single day.

    Parameters
    ----------
    target_day : str
        Target day (YYYY-MM-DD).
    rng : np.random.Generator, optional
        Deterministic RNG.

    Returns
    -------
    pd.DataFrame
        Actuals with task, target_day, ds, y_true columns.
    """
    from data.business_day import add_business_time_columns

    if rng is None:
        rng = np.random.default_rng(42)

    timestamps = pd.date_range(f"{target_day} 01:00", periods=24, freq="h")

    rows: list[dict[str, Any]] = []
    for i in range(24):
        rows.append({
            "task": "dayahead",
            "target_day": target_day,
            "ds": timestamps[i],
            "y_true": float(rng.uniform(80, 200)),
            "actual_source": "synthetic_smoke",
        })

    df = pd.DataFrame(rows)
    df = add_business_time_columns(df, timestamp_col="ds")
    return df


def _check_key_uniqueness(
    ledger_df: pd.DataFrame,
    key_cols: list[str],
    label: str,
) -> list[str]:
    """Check that a ledger has no duplicate keys.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []
    valid, issues = validate_ledger_keys(ledger_df, key_cols)
    if not valid:
        errors.append(f"{label}: {'; '.join(issues)}")
    return errors


def run_multi_day_backfill_smoke(
    start_day: str,
    n_days: int = 30,
    ledger_dir: Optional[str] = None,
    allow_dry_run: bool = True,
    classifier_rule_fallback: bool = True,
    generate_synthetic_actuals: bool = True,
    fusion_method: str = "equal_weight",
    production: bool = True,
) -> dict[str, Any]:
    """Run a multi-day ledger backfill structural smoke.

    Parameters
    ----------
    start_day : str
        First target day (YYYY-MM-DD).
    n_days : int
        Number of consecutive days (default 30).
    ledger_dir : str, optional
        Directory for ledger CSVs.  Use ``tmp_path`` in tests.
    allow_dry_run : bool
        Include dry-run models in fusion (default True).
    classifier_rule_fallback : bool
        Apply rule-based negative flag (default True).
    generate_synthetic_actuals : bool
        If True, generate synthetic actuals and test no-leakage (default True).
    fusion_method : str
        Fusion method: ``"equal_weight"`` (default), ``"prior_weight"``,
        or ``"bgew_skeleton"``.
    production : bool
        Production mode (default True).

    Returns
    -------
    dict
        Summary dict with per-day status, row counts, and all checks.
    """
    days = _date_range(start_day, n_days)
    end_day = days[-1]
    rng = np.random.default_rng(42)

    summary: dict[str, Any] = {
        "start_day": start_day,
        "end_day": end_day,
        "n_days": n_days,
        "overall_status": "PASS",
        "mode_label": [],
        "per_day_status": {},
        "prediction_rows_total": 0,
        "corrected_rows_total": 0,
        "fusion_rows_total": 0,
        "weight_rows_total": 0,
        "final_rows_total": 0,
        "corrected_ledger_rows": 0,
        "fusion_ledger_rows": 0,
        "weight_ledger_rows": 0,
        "actual_ledger_rows": 0,
        "idempotency_check": "SKIPPED",
        "key_uniqueness_check": {},
        "validators_passed": [],
        "no_leakage_check": "SKIPPED",
        "forbidden_files_check": "PASS",
        "reason_codes": [],
    }

    all_reason_codes: list[str] = []
    validators_passed: list[str] = []
    stage_labels: dict[str, str] = {}
    run_id = f"backfill_{start_day}_to_{end_day}"

    # ── Initialise accumulators ─────────────────────────────────────

    # Ledger accumulators (in-memory)
    corrected_ledger = pd.DataFrame(columns=CORRECTED_LEDGER_COLUMNS)
    fusion_ledger = pd.DataFrame(columns=FUSION_LEDGER_COLUMNS)
    weight_ledger = pd.DataFrame(columns=WEIGHT_LEDGER_COLUMNS)
    actual_ledger = pd.DataFrame(columns=ACTUAL_LEDGER_COLUMNS) if generate_synthetic_actuals else pd.DataFrame()

    # Readiness status for all days
    readiness_status = {
        "cfg05": READY_DRY_RUN,
        "best_two_average": READY_DRY_RUN,
    }

    # Fusion config
    from data.schema import VALID_FUSION_METHODS
    if fusion_method not in VALID_FUSION_METHODS:
        fusion_method = "equal_weight"
        all_reason_codes.append(f"FUSION_METHOD_FALLBACK_TO_EQUAL_WEIGHT")

    # ════════════════════════════════════════════
    # Per-day loop
    # ════════════════════════════════════════════

    for day_idx, day in enumerate(days):
        day_status: dict[str, Any] = {
            "day": day,
            "status": "PASS",
            "prediction_rows": 0,
            "corrected_rows": 0,
            "fusion_rows": 0,
            "final_rows": 0,
        }

        # ── 1. Predictions ─────────────────────────────────
        predictions = _build_synthetic_predictions_for_day(day, rng=rng)
        day_status["prediction_rows"] = len(predictions)

        # Validate predictions (once for the whole run)
        if day_idx == 0:
            pred_valid, pred_errors = validate_prediction_dataframe(
                predictions, production=production
            )
            if pred_valid:
                validators_passed.append("prediction_validator")
                all_reason_codes.append("PREDICTION_SCHEMA_VALID")
            else:
                summary["overall_status"] = "FAIL"
                all_reason_codes.append(
                    f"PREDICTION_VALIDATION_FAILED: {'; '.join(pred_errors)}"
                )

        # ── 2. Residual correction ───────────────────────────
        corrected = apply_residual_correction(
            predictions,
            correction_profile="conservative",
            risk_df=None,
            production=production,
        )
        day_status["corrected_rows"] = len(corrected)

        if day_idx == 0:
            corr_valid, corr_errors = validate_residual_dataframe(
                corrected, production=production
            )
            if corr_valid:
                validators_passed.append("residual_validator")
            else:
                summary["overall_status"] = "FAIL"
                all_reason_codes.append(
                    f"CORRECTED_VALIDATION_FAILED: {'; '.join(corr_errors)}"
                )

        # ── 3. Corrected ledger ──────────────────────────────
        corrected_ledger = append_corrected_predictions_to_ledger(
            corrected, ledger_df=corrected_ledger, run_id=run_id,
        )

        # ── 4. Fusion ────────────────────────────────────────

        # For bgew_skeleton, try to use actuals if available
        actuals_for_fusion = None
        if fusion_method == "bgew_skeleton" and len(actual_ledger) > 0:
            filtered = filter_actuals_for_training(
                actual_ledger, target_day=day, window=30,
            )
            if len(filtered) > 0:
                actuals_for_fusion = filtered
                all_reason_codes.append(f"BGEW_ACTUALS_AVAILABLE_DAY_{day}")

        fusion_result = run_fusion(
            corrected,
            method=fusion_method,
            actuals_df=actuals_for_fusion,
            allow_dry_run=allow_dry_run,
            readiness_status=readiness_status,
            production=production,
            learner_version="0.1.0-skeleton",
        )
        day_status["fusion_rows"] = len(fusion_result)

        if day_idx == 0 and len(fusion_result) > 0:
            fusion_valid, fusion_errors = validate_fusion_dataframe(
                fusion_result, allow_empty=True, production=production,
            )
            if fusion_valid:
                validators_passed.append("fusion_validator")
            else:
                summary["overall_status"] = "FAIL"
                all_reason_codes.append(
                    f"FUSION_VALIDATION_FAILED: {'; '.join(fusion_errors)}"
                )

        # ── 5. Fusion ledger ─────────────────────────────────
        fusion_ledger = append_fusion_to_ledger(
            fusion_result, ledger_df=fusion_ledger, run_id=run_id,
        )

        # ── 6. Weight ledger ─────────────────────────────────
        weight_rows = extract_weight_rows(fusion_result)
        weight_ledger = append_weights_to_ledger(
            weight_rows, ledger_df=weight_ledger, run_id=run_id,
        )

        # ── 7. Negative classifier / final output ────────────
        final = run_negative_classifier(
            fusion_result,
            model_dir=None,
            rule_fallback=classifier_rule_fallback,
            production=production,
        )
        day_status["final_rows"] = len(final)

        if day_idx == 0 and len(final) > 0:
            final_valid, final_errors = validate_final_dataframe(
                final, allow_empty=True, production=production,
            )
            if final_valid:
                validators_passed.append("final_output_validator")
            else:
                summary["overall_status"] = "FAIL"
                all_reason_codes.append(
                    f"FINAL_VALIDATION_FAILED: {'; '.join(final_errors)}"
                )

        # ── Accumulate totals ───────────────────────────
        summary["prediction_rows_total"] += day_status["prediction_rows"]
        summary["corrected_rows_total"] += day_status["corrected_rows"]
        summary["fusion_rows_total"] += day_status["fusion_rows"]
        summary["final_rows_total"] += day_status["final_rows"]
        summary["per_day_status"][day] = day_status

    # ── Ledger totals (after all days accumulated) ──────────────
    summary["corrected_ledger_rows"] = len(corrected_ledger)
    summary["fusion_ledger_rows"] = len(fusion_ledger)
    summary["weight_ledger_rows"] = len(weight_ledger)

    # ════════════════════════════════════════════
    # Actual ledger (synthetic)
    # ════════════════════════════════════════════

    if generate_synthetic_actuals:
        logger.info("Generating synthetic actuals for %d days", n_days)
        actual_rng = np.random.default_rng(123)
        for day in days:
            actuals = _build_synthetic_actuals_for_day(day, rng=actual_rng)
            actual_ledger = append_actuals_to_ledger(
                actuals, ledger_df=actual_ledger, run_id=run_id,
            )

        summary["actual_ledger_rows"] = len(actual_ledger)

        # Validate actual ledger
        act_valid, act_issues = validate_actual_ledger(actual_ledger)
        if act_valid:
            validators_passed.append("actual_ledger_validator")
        else:
            summary["overall_status"] = "FAIL"
            all_reason_codes.append(
                f"ACTUAL_LEDGER_INVALID: {'; '.join(act_issues)}"
            )

        # ── No-leakage check ──────────────────────────────
        leakage_found = False
        for day in days:
            trained = filter_actuals_for_training(
                actual_ledger, target_day=day, window=30,
            )
            if len(trained) > 0:
                leaked = trained[trained["business_day"] >= pd.Timestamp(day)]
                if len(leaked) > 0:
                    leakage_found = True
                    all_reason_codes.append(f"LEAKAGE_DETECTED_ON_DAY_{day}")

        if not leakage_found:
            summary["no_leakage_check"] = "PASS"
            all_reason_codes.append("NO_LEAKAGE_ALL_DAYS_PASS")
        else:
            summary["no_leakage_check"] = "FAIL"
            all_reason_codes.append("LEAKAGE_DETECTED")

    # ════════════════════════════════════════════
    # Key uniqueness checks
    # ════════════════════════════════════════════

    key_checks: dict[str, str] = {}

    corr_key_errors = _check_key_uniqueness(
        corrected_ledger, CORRECTED_LEDGER_KEY, "corrected_ledger",
    )
    key_checks["corrected_ledger"] = "PASS" if not corr_key_errors else "FAIL"
    all_reason_codes.extend(corr_key_errors)

    fusion_key_errors = _check_key_uniqueness(
        fusion_ledger, FUSION_LEDGER_KEY, "fusion_ledger",
    )
    key_checks["fusion_ledger"] = "PASS" if not fusion_key_errors else "FAIL"
    all_reason_codes.extend(fusion_key_errors)

    weight_key_errors = _check_key_uniqueness(
        weight_ledger, WEIGHT_LEDGER_KEY, "weight_ledger",
    )
    key_checks["weight_ledger"] = "PASS" if not weight_key_errors else "FAIL"
    all_reason_codes.extend(weight_key_errors)

    if generate_synthetic_actuals:
        actual_key_errors = _check_key_uniqueness(
            actual_ledger, ACTUAL_LEDGER_KEY, "actual_ledger",
        )
        key_checks["actual_ledger"] = "PASS" if not actual_key_errors else "FAIL"
        all_reason_codes.extend(actual_key_errors)

    summary["key_uniqueness_check"] = key_checks

    # ════════════════════════════════════════════
    # Idempotency check — run same backfill again,
    # ledger sizes should not double
    # ════════════════════════════════════════════

    corrected_size_1 = len(corrected_ledger)

    # Re-run one day (simulates idempotent re-append)
    rerun_day = days[0]
    rerun_preds = _build_synthetic_predictions_for_day(rerun_day, rng=np.random.default_rng(42))
    rerun_corrected = apply_residual_correction(rerun_preds, production=production)
    corrected_ledger_2 = append_corrected_predictions_to_ledger(
        rerun_corrected, ledger_df=corrected_ledger, run_id=f"rerun_{rerun_day}",
    )

    if len(corrected_ledger_2) == corrected_size_1:
        summary["idempotency_check"] = "PASS"
        all_reason_codes.append("IDEMPOTENCY_CHECK_PASSED")
    else:
        # First run may have been empty (size 0 + 48 = 48), second run dedupes to 48
        if corrected_size_1 == len(rerun_corrected) and len(corrected_ledger_2) == corrected_size_1:
            summary["idempotency_check"] = "PASS"
            all_reason_codes.append("IDEMPOTENCY_CHECK_PASSED")
        else:
            summary["idempotency_check"] = "FAIL"
            all_reason_codes.append(
                f"IDEMPOTENCY_FAILED: size_before={corrected_size_1}, "
                f"size_after={len(corrected_ledger_2)}"
            )

    # ════════════════════════════════════════════
    # Write ledgers to disk (if ledger_dir set)
    # ════════════════════════════════════════════

    if ledger_dir is not None:
        # Forbidden files check
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        abs_ledger = os.path.abspath(ledger_dir)
        for forbidden in ["data", "outputs", os.path.join("reports", "local")]:
            forbidden_abs = os.path.abspath(
                os.path.join(repo_root, forbidden.replace("/", os.sep))
            )
            if abs_ledger.startswith(forbidden_abs):
                summary["forbidden_files_check"] = "FAIL"
                all_reason_codes.append(f"FORBIDDEN_LEDGER_DIR: {ledger_dir}")
                break

        if summary["forbidden_files_check"] == "PASS":
            save_ledger(corrected_ledger, os.path.join(ledger_dir, "corrected_ledger.csv"))
            save_ledger(fusion_ledger, os.path.join(ledger_dir, "fusion_ledger.csv"))
            save_ledger(weight_ledger, os.path.join(ledger_dir, "weight_ledger.csv"))
            if generate_synthetic_actuals:
                save_ledger(actual_ledger, os.path.join(ledger_dir, "actual_ledger.csv"))

    # ════════════════════════════════════════════
    # Stage labels
    # ════════════════════════════════════════════

    stage_labels["dayahead"] = LABEL_DRY_RUN
    stage_labels["residual"] = LABEL_DATA_MISSING
    stage_labels["fusion"] = LABEL_STRUCTURAL_ONLY
    stage_labels["corrected_ledger"] = LABEL_STRUCTURAL_ONLY
    stage_labels["fusion_ledger"] = LABEL_STRUCTURAL_ONLY
    stage_labels["weight_ledger"] = LABEL_STRUCTURAL_ONLY
    stage_labels["negative_classifier"] = (
        LABEL_RULE_FALLBACK if classifier_rule_fallback else "CLASSIFIER_ARTIFACT_MISSING"
    )
    stage_labels["final_output"] = LABEL_STRUCTURAL_ONLY

    mode_labels = sorted(set(stage_labels.values()))
    summary["mode_label"] = mode_labels
    summary["validators_passed"] = validators_passed
    summary["reason_codes"] = all_reason_codes
    summary["stage_labels"] = stage_labels

    # Weight rows total
    summary["weight_rows_total"] = len(weight_ledger)

    # Update overall status for any failures
    if any(v != "PASS" for v in key_checks.values()):
        summary["overall_status"] = "FAIL"
    if summary["idempotency_check"] == "FAIL":
        summary["overall_status"] = "FAIL"
    if summary["no_leakage_check"] == "FAIL":
        summary["overall_status"] = "FAIL"
    if summary["forbidden_files_check"] == "FAIL":
        summary["overall_status"] = "FAIL"

    logger.info(
        "Multi-day backfill smoke complete: days=%d, status=%s, "
        "corrected_ledger=%d, fusion_ledger=%d, weight_ledger=%d, "
        "actual_ledger=%d, keys=%s, idempotency=%s",
        n_days, summary["overall_status"],
        summary["corrected_ledger_rows"],
        summary["fusion_ledger_rows"],
        summary["weight_ledger_rows"],
        summary["actual_ledger_rows"],
        key_checks,
        summary["idempotency_check"],
    )

    return summary


def _date_range(start_day: str, n_days: int) -> list[str]:
    """Generate a list of consecutive day strings."""
    start = pd.Timestamp(start_day)
    return [(start + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]
