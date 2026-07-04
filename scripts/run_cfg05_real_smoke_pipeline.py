"""
scripts/run_cfg05_real_smoke_pipeline.py — Full cfg05 REAL smoke pipeline.

Chains artifact check → input check → adapter predict → validate.

Usage::

    # Structural smoke (no real artifacts)
    python -m scripts.run_cfg05_real_smoke_pipeline --no-production

    # With real artifact paths
    python -m scripts.run_cfg05_real_smoke_pipeline \\
        --cfg05-model /path/to/weights --cfg05-input /path/to/data.csv \\
        --target-day 2026-07-01

    # Strict mode
    python -m scripts.run_cfg05_real_smoke_pipeline --strict \\
        --cfg05-model /path/to/model.txt --cfg05-input /path/to/data.csv

    # With output file
    python -m scripts.run_cfg05_real_smoke_pipeline \\
        --cfg05-model /path --cfg05-input /path --target-day 2026-07-01 \\
        --out /tmp/cfg05_smoke.json

Options::

    --cfg05-model PATH          Path to cfg05 model file or directory.
    --cfg05-input PATH          Path to cfg05 input CSV.
    --target-day YYYY-MM-DD     Target day for prediction.
    --out PATH                  Output JSON path (default: don't write).
    --strict                    Exit non-zero if artifacts missing or smoke fails.
    --production                Production mode (default: True).
    --no-production             Disable production mode.
    --json                      Output JSON to stdout.
    --verbose, -v               Increase log verbosity.

Summary JSON contains::

    cfg05_artifact_status   — Artifact readiness status code
    cfg05_input_status      — Input readiness status code
    cfg05_adapter_loaded    — Whether adapter was loaded successfully
    prediction_rows         — Number of prediction rows produced
    validator_passed        — Whether validate_output passed
    readiness_label         — "REAL_READY" | "NOT_READY" | "DATA_MISSING" | "INVALID"
    overall_status          — "PASS" | "PASS_STRUCTURAL" | "FAIL"
    reason_codes            — Audit trail
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from artifacts.readiness import (
    check_cfg05_artifact, check_cfg05_input,
    LOADABLE, SCHEMA_READY, REAL_READY, MISSING, PRESENT, INVALID,
)

logger = logging.getLogger(__name__)


def run_cfg05_real_smoke_pipeline(
    cfg05_model: Optional[str] = None,
    cfg05_input: Optional[str] = None,
    target_day: Optional[str] = None,
    production: bool = True,
) -> dict[str, Any]:
    """Run the full cfg05 REAL smoke pipeline.

    Parameters
    ----------
    cfg05_model : str, optional
        Path to cfg05 model file or directory.
    cfg05_input : str, optional
        Path to cfg05 input CSV.
    target_day : str, optional
        Target day in YYYY-MM-DD.
    production : bool
        Production mode flag.

    Returns
    -------
    dict with smoke summary keys.
    """
    # Step 1: Run readiness gates
    artifact_status = check_cfg05_artifact(cfg05_model)
    input_status = check_cfg05_input(cfg05_input)

    cfg05_adapter_loaded = False
    prediction_rows = 0
    validator_passed = False
    reason_codes: list[str] = []
    readiness_label = "DATA_MISSING"
    overall_status = "PASS_STRUCTURAL"

    # Step 2: Try real inference if both gates are ready
    if (artifact_status.status in (LOADABLE, SCHEMA_READY, REAL_READY)
            and input_status.status == SCHEMA_READY):
        try:
            from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter

            # Determine effective model path for predict()
            effective_model_dir = cfg05_model
            if effective_model_dir and not os.path.isdir(effective_model_dir):
                effective_model_dir = os.path.dirname(effective_model_dir)

            adapter = CFG05DayaheadAdapter()
            adapter.load()
            cfg05_adapter_loaded = True
            reason_codes.append("CFG05_ADAPTER_LOADED")

            result = adapter.predict(
                data_path=cfg05_input,
                target_date=target_day,
                model_dir=effective_model_dir,
            )

            prediction_rows = len(result)
            reason_codes.append(f"CFG05_PREDICTION_ROWS: {prediction_rows}")

            if prediction_rows > 0:
                # validate_output is called inside predict(), but we also
                # check the output schema explicitly
                from data.schema import PREDICTION_OUTPUT_COLUMNS
                actual_cols = list(result.columns)
                missing_cols = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in actual_cols]

                if not missing_cols:
                    validator_passed = True
                    reason_codes.append("CFG05_VALIDATOR_PASSED")
                    readiness_label = "REAL_READY"
                    overall_status = "PASS"
                    logger.info(
                        "cfg05 REAL smoke: REAL_READY — %d rows, validator passed",
                        prediction_rows,
                    )
                else:
                    reason_codes.append(
                        f"CFG05_VALIDATOR_FAILED_MISSING_COLS: {missing_cols}"
                    )
                    readiness_label = "INVALID"
                    overall_status = "FAIL"
            else:
                reason_codes.append("CFG05_PREDICTION_EMPTY")
                readiness_label = "INVALID"
                overall_status = "FAIL"

        except Exception as e:
            reason_codes.append(f"CFG05_REAL_SMOKE_FAILED: {e}")
            logger.error("cfg05 REAL smoke failed: %s", e)
            if cfg05_adapter_loaded:
                readiness_label = "INVALID"
                overall_status = "FAIL"
            else:
                readiness_label = "NOT_READY"
    else:
        # Structural-only smoke
        if artifact_status.status == MISSING:
            reason_codes.append("CFG05_ARTIFACT_MISSING_SKIPPED")
        elif artifact_status.status == PRESENT:
            reason_codes.append("CFG05_ARTIFACT_PRESENT_NOT_LOADABLE")
        elif artifact_status.status == INVALID:
            reason_codes.append(f"CFG05_ARTIFACT_INVALID: {artifact_status.reason_codes}")

        if input_status.status == MISSING:
            reason_codes.append("CFG05_INPUT_MISSING_SKIPPED")
        elif input_status.status == INVALID:
            reason_codes.append(f"CFG05_INPUT_INVALID: {input_status.reason_codes}")

        if artifact_status.status in (PRESENT, INVALID) or input_status.status in (PRESENT, INVALID):
            readiness_label = "NOT_READY"
        else:
            readiness_label = "DATA_MISSING"

        reason_codes.append("CFG05_REAL_SMOKE_SKIPPED_STRUCTURAL_ONLY")

    summary: dict[str, Any] = {
        "cfg05_artifact_status": artifact_status.status,
        "cfg05_input_status": input_status.status,
        "cfg05_adapter_loaded": cfg05_adapter_loaded,
        "prediction_rows": prediction_rows,
        "validator_passed": validator_passed,
        "readiness_label": readiness_label,
        "overall_status": overall_status,
        "reason_codes": reason_codes,
    }

    logger.info(
        "cfg05 REAL smoke pipeline: label=%s status=%s rows=%d",
        readiness_label, overall_status, prediction_rows,
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cfg05 REAL smoke pipeline (P10).",
    )
    parser.add_argument("--cfg05-model", type=str, default=None,
                        help="Path to cfg05 model file or directory.")
    parser.add_argument("--cfg05-input", type=str, default=None,
                        help="Path to cfg05 input CSV.")
    parser.add_argument("--target-day", type=str, default=None,
                        help="Target day for prediction (YYYY-MM-DD).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: don't write).")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero if artifacts missing or smoke fails.")
    parser.add_argument("--no-production", dest="production",
                        action="store_false", default=True,
                        help="Disable production mode.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON to stdout.")
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

    logger.info(
        "Starting cfg05 REAL smoke pipeline: model=%s input=%s target=%s",
        args.cfg05_model, args.cfg05_input, args.target_day,
    )

    summary = run_cfg05_real_smoke_pipeline(
        cfg05_model=args.cfg05_model,
        cfg05_input=args.cfg05_input,
        target_day=args.target_day,
        production=args.production,
    )

    # Write output only when --out is explicitly provided
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Wrote smoke summary to %s", args.out)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        _print_summary(summary)

    if args.strict:
        if summary["readiness_label"] == "REAL_READY":
            logger.info("cfg05 REAL smoke: strict mode PASS (REAL_READY)")
            return 0
        else:
            logger.error(
                "cfg05 REAL smoke: strict mode FAIL (label=%s)",
                summary["readiness_label"],
            )
            return 1

    return 0 if summary["overall_status"] in ("PASS", "PASS_STRUCTURAL") else 1


def _print_summary(summary: dict[str, Any]) -> None:
    """Print human-readable smoke summary."""
    print("=" * 60)
    print("cfg05 REAL Smoke Pipeline Summary")
    print("=" * 60)
    print(f"  Artifact status:    {summary['cfg05_artifact_status']}")
    print(f"  Input status:       {summary['cfg05_input_status']}")
    print(f"  Adapter loaded:     {summary['cfg05_adapter_loaded']}")
    print(f"  Prediction rows:    {summary['prediction_rows']}")
    print(f"  Validator passed:   {summary['validator_passed']}")
    print(f"  Readiness label:    {summary['readiness_label']}")
    print(f"  Overall status:     {summary['overall_status']}")
    print("  Reason codes:")
    for rc in summary.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
