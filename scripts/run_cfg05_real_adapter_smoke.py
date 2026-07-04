"""
scripts/run_cfg05_real_adapter_smoke.py — cfg05 REAL adapter smoke.

Usage::

    # Structural smoke (no real artifacts, non-strict → exit 0)
    python -m scripts.run_cfg05_real_adapter_smoke --no-production

    # With real artifact paths
    python -m scripts.run_cfg05_real_adapter_smoke \\
        --model-dir /path/to/weights --input /path/to/data.csv \\
        --target-day 2026-07-01

    # With output file
    python -m scripts.run_cfg05_real_adapter_smoke \\
        --model-dir /path/to/weights --input /path/to/data.csv \\
        --target-day 2026-07-01 --out /tmp/cfg05_smoke.json

    # Strict mode (exit non-zero if artifacts missing)
    python -m scripts.run_cfg05_real_adapter_smoke --strict \\
        --model-dir /path/to/weights --input /path/to/data.csv

Options::

    --model-dir PATH                Directory containing model weight file(s).
    --model-file PATH               Direct path to model weight file.
    --input PATH                    Path to input CSV with cfg05 feature columns.
    --target-day YYYY-MM-DD         Target day for prediction (required with --model-dir/--model-file).
    --out PATH                      Output JSON path (default: don't write).
    --production                    Production mode (default: True).
    --no-production                 Disable production mode.
    --strict                        Exit non-zero if real artifacts are missing.
    --verbose, -v                   Increase log verbosity.

Summary JSON contains::

    cfg05_artifact_status   — MISSING | PRESENT | LOADABLE | etc
    cfg05_input_status      — MISSING | PRESENT | SCHEMA_READY | etc
    cfg05_adapter_loaded    — bool
    prediction_rows         — int
    validator_passed        — bool
    readiness_label         — "REAL" | "DRY_RUN" | "DATA_MISSING"
    reason_codes            — list[str]
    overall_status          — "PASS" | "FAIL"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from artifacts.readiness import check_cfg05_artifact, check_cfg05_input, LOADABLE, SCHEMA_READY, REAL_READY, MISSING, PRESENT
from data.schema import PREDICTION_OUTPUT_COLUMNS

logger = logging.getLogger(__name__)


def run_cfg05_real_adapter_smoke(
    model_dir: Optional[str] = None,
    model_file: Optional[str] = None,
    input_path: Optional[str] = None,
    target_day: Optional[str] = None,
    production: bool = True,
) -> dict[str, Any]:
    """Run cfg05 REAL adapter smoke test.

    Parameters
    ----------
    model_dir : str, optional
        Directory containing model weight file(s).
    model_file : str, optional
        Direct path to model weight file.
    input_path : str, optional
        Path to input CSV with cfg05 feature columns.
    target_day : str, optional
        Target day in YYYY-MM-DD format.
    production : bool
        Production mode (default True).

    Returns
    -------
    dict
        Summary with keys: cfg05_artifact_status, cfg05_input_status,
        cfg05_adapter_loaded, prediction_rows, validator_passed,
        readiness_label, reason_codes, overall_status.
    """
    # Determine model path from dir or file
    model_path = model_file or model_dir

    # Run readiness checks
    artifact_status = check_cfg05_artifact(model_path)
    input_status = check_cfg05_input(input_path)

    cfg05_adapter_loaded = False
    prediction_rows = 0
    validator_passed = False
    reason_codes: list[str] = []

    # Try prediction if artifact is loadable and input has schema
    if (artifact_status.status in (LOADABLE, SCHEMA_READY, REAL_READY)
            and input_status.status in (SCHEMA_READY, REAL_READY)):
        try:
            from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter

            # Determine effective model directory for predict()
            effective_model_dir = model_dir
            if effective_model_dir is None and model_file is not None:
                effective_model_dir = os.path.dirname(model_file)

            adapter = CFG05DayaheadAdapter()
            adapter.load()

            result = adapter.predict(
                data_path=input_path,
                target_date=target_day,
                model_dir=effective_model_dir,
            )

            cfg05_adapter_loaded = True
            prediction_rows = len(result)

            # Validate output schema
            expected_cols = list(PREDICTION_OUTPUT_COLUMNS)
            actual_cols = list(result.columns)
            missing = [c for c in expected_cols if c not in actual_cols]

            if not missing and prediction_rows > 0:
                validator_passed = True
                reason_codes.append("CFG05_REAL_SMOKE_PASSED")
                logger.info(
                    "cfg05 REAL smoke: %d rows, all columns valid",
                    prediction_rows,
                )
            else:
                reason_codes.append(
                    f"CFG05_REAL_SMOKE_VALIDATION_FAILED: "
                    f"missing_cols={missing}, rows={prediction_rows}"
                )
                logger.error("cfg05 REAL smoke validation failed: missing=%s", missing)

        except Exception as e:
            reason_codes.append(f"CFG05_REAL_SMOKE_PREDICTION_FAILED: {e}")
            logger.error("cfg05 REAL smoke prediction failed: %s", e)
    else:
        if artifact_status.status == MISSING:
            reason_codes.append("CFG05_ARTIFACT_MISSING_SKIPPED")
        if input_status.status == MISSING:
            reason_codes.append("CFG05_INPUT_MISSING_SKIPPED")
        if artifact_status.status not in (LOADABLE, SCHEMA_READY, REAL_READY):
            reason_codes.append(f"CFG05_ARTIFACT_NOT_READY: {artifact_status.status}")
        if input_status.status not in (SCHEMA_READY, REAL_READY):
            reason_codes.append(f"CFG05_INPUT_NOT_READY: {input_status.status}")
        reason_codes.append("CFG05_REAL_SMOKE_SKIPPED")

    # Determine readiness_label
    if cfg05_adapter_loaded and validator_passed:
        readiness_label = "REAL"
    elif cfg05_adapter_loaded:
        readiness_label = "DRY_RUN"
    elif artifact_status.status not in (MISSING, PRESENT) or input_status.status not in (MISSING, PRESENT):
        readiness_label = "DRY_RUN"
    else:
        readiness_label = "DATA_MISSING"

    # Determine overall status
    # Prediction failure with loaded adapter → FAIL
    if cfg05_adapter_loaded and prediction_rows > 0 and not validator_passed:
        overall_status = "FAIL"
    elif cfg05_adapter_loaded and prediction_rows == 0:
        overall_status = "FAIL"
    else:
        overall_status = "PASS"

    summary = {
        "cfg05_artifact_status": artifact_status.status,
        "cfg05_input_status": input_status.status,
        "cfg05_adapter_loaded": cfg05_adapter_loaded,
        "prediction_rows": prediction_rows,
        "validator_passed": validator_passed,
        "readiness_label": readiness_label,
        "reason_codes": reason_codes,
        "overall_status": overall_status,
    }

    logger.info(
        "cfg05 REAL smoke: label=%s status=%s rows=%d",
        readiness_label, overall_status, prediction_rows,
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cfg05 REAL adapter smoke test (P9).",
    )
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing model weight file(s).")
    parser.add_argument("--model-file", type=str, default=None,
                        help="Direct path to model weight file.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input CSV with cfg05 feature columns.")
    parser.add_argument("--target-day", type=str, default=None,
                        help="Target day for prediction (YYYY-MM-DD).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: don't write).")
    parser.add_argument("--no-production", dest="production",
                        action="store_false", default=True,
                        help="Disable production mode.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero if real artifacts are missing.")
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
        "cfg05 REAL smoke: model_dir=%s model_file=%s input=%s target_day=%s",
        args.model_dir, args.model_file, args.input, args.target_day,
    )

    summary = run_cfg05_real_adapter_smoke(
        model_dir=args.model_dir,
        model_file=args.model_file,
        input_path=args.input,
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

    print(json.dumps(summary, indent=2, default=str))

    if args.strict:
        if summary["cfg05_artifact_status"] != MISSING and summary["cfg05_input_status"] != MISSING:
            logger.info("cfg05 REAL smoke: strict mode PASS (artifacts present)")
            return 0
        else:
            logger.error("cfg05 REAL smoke: strict mode FAIL (artifacts missing)")
            return 1

    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
