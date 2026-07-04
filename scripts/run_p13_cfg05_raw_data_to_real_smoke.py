"""
scripts/run_p13_cfg05_raw_data_to_real_smoke.py — P13 orchestration.

Chains raw data contract check → local train/export → cfg05 REAL smoke pipeline.

Usage::

    # With raw data
    python -m scripts.run_p13_cfg05_raw_data_to_real_smoke \\
        --raw-data /path/to/raw.csv --target-day 2026-07-01

    # With raw data and REAL smoke attempt
    python -m scripts.run_p13_cfg05_raw_data_to_real_smoke \\
        --raw-data /path/to/raw.csv --target-day 2026-07-01 --run-real-smoke

    # JSON + strict
    python -m scripts.run_p13_cfg05_raw_data_to_real_smoke \\
        --raw-data /path/to/raw.csv --json --strict

Options::

    --source-repo PATH          Path to epf-sota-experiment.
    --raw-data PATH             Path to raw Chinese CSV.
    --target-day YYYY-MM-DD     Target day for prediction.
    --work-dir PATH             Local work dir (default: .local_artifacts/p13_cfg05).
    --train-window-days N       Training window in days (default: 90).
    --run-real-smoke            Attempt REAL smoke if model+features gates pass.
    --json                      Output JSON report.
    --strict                    Exit non-zero on any blocker.
    --verbose, -v               Increase log verbosity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

from artifacts.readiness import SCHEMA_READY, LOADABLE
from scripts.check_cfg05_raw_data_contract import (
    check_cfg05_raw_data_contract,
    RAW_DATA_MISSING,
    RAW_DATA_INVALID,
    RAW_DATA_VALID,
)
from scripts.train_export_cfg05_local import train_export_cfg05_local

logger = logging.getLogger(__name__)

# ── Safe paths ─────────────────────────────────────────────────────────────
_ALLOWED_WORK_DIRS = (".local_artifacts",)
_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "ledgers/", "reports/local/")

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p13_cfg05")
_DEFAULT_SOURCE_REPO = os.path.join(
    ".local_artifacts", "source_repos", "epf-sota-experiment",
)


def _path_is_safe(path: str) -> bool:
    """Check path is under an ignored, allowed directory."""
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def run_p13_cfg05_raw_data_to_real_smoke(
    source_repo: Optional[str] = None,
    raw_data: Optional[str] = None,
    target_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    train_window_days: int = 90,
    run_real_smoke: bool = False,
) -> dict[str, Any]:
    """P13 orchestration: raw data check → train/export → real smoke.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment.
    raw_data : str, optional
        Path to raw Chinese CSV.
    target_day : str, optional
        Target day in YYYY-MM-DD.
    work_dir : str, optional
        Local work dir.
    train_window_days : int
        Training window in days.
    run_real_smoke : bool
        Attempt REAL smoke if model+features gates pass.

    Returns
    -------
    dict with complete P13 summary.
    """
    source_repo = source_repo or _DEFAULT_SOURCE_REPO
    target_day = target_day or "2026-07-01"
    work_dir = work_dir or _DEFAULT_WORK_DIR

    model_out = os.path.join(work_dir, "cfg05_model.txt")
    features_out = os.path.join(work_dir, f"cfg05_features_{target_day}.csv")

    result: dict[str, Any] = {
        "source_repo_status": "NOT_CHECKED",
        "raw_data_status": "NOT_CHECKED",
        "model_export_status": "NOT_ATTEMPTED",
        "feature_export_status": "NOT_ATTEMPTED",
        "cfg05_artifact_status": None,
        "cfg05_input_status": None,
        "real_smoke_attempted": False,
        "readiness_label": None,
        "prediction_rows": 0,
        "validator_passed": False,
        "final_status": None,
        "model_out": model_out,
        "features_out": features_out,
        "reason_codes": [],
        "forbidden_files_check": "NOT_RUN",
    }

    # ── Step 1: Check source repo ──
    if not os.path.isdir(source_repo):
        result["source_repo_status"] = "MISSING"
        result["reason_codes"].append(f"SOURCE_REPO_MISSING: {source_repo}")
        result["final_status"] = "CFG05_RAW_DATA_MISSING"
        return result

    result["source_repo_status"] = "PRESENT"
    result["reason_codes"].append(f"SOURCE_REPO_FOUND: {source_repo}")

    # ── Step 2: Check raw data contract ──
    contract = check_cfg05_raw_data_contract(raw_data=raw_data)
    result["raw_data_status"] = contract["raw_data_status"]
    if contract["reason_codes"]:
        result["reason_codes"].extend(
            [f"RAW_DATA:{rc}" for rc in contract["reason_codes"]]
        )

    if contract["raw_data_status"] in (RAW_DATA_MISSING, RAW_DATA_INVALID):
        result["final_status"] = contract["raw_data_status"]
        result["reason_codes"].append("P13_STOPPED_RAW_DATA_CONTRACT_FAILED")
        return result

    # ── Step 3: Run train/export ──
    export_result = train_export_cfg05_local(
        source_repo=source_repo,
        raw_data=raw_data,
        target_day=target_day,
        train_window_days=train_window_days,
        work_dir=work_dir,
        force=True,
    )

    result["model_export_status"] = (
        "EXPORTED" if export_result.get("model_saved") else "FAILED"
    )
    result["feature_export_status"] = (
        "EXPORTED" if export_result.get("features_saved") else "FAILED"
    )
    result["cfg05_artifact_status"] = export_result.get("cfg05_artifact_status")
    result["cfg05_input_status"] = export_result.get("cfg05_input_status")

    if export_result.get("reason_codes"):
        result["reason_codes"].extend(
            [f"TRAIN_EXPORT:{rc}" for rc in export_result["reason_codes"]]
        )

    # Determine if gates pass
    artifact_gate_pass = result["cfg05_artifact_status"] in (
        LOADABLE, SCHEMA_READY
    )
    input_gate_pass = result["cfg05_input_status"] == SCHEMA_READY

    # ── Step 4: Optionally run REAL smoke ──
    if run_real_smoke and artifact_gate_pass and input_gate_pass:
        result["real_smoke_attempted"] = True
        result["reason_codes"].append("REAL_SMOKE_ATTEMPTED")

        from scripts.run_cfg05_real_smoke_pipeline import (
            run_cfg05_real_smoke_pipeline,
        )

        smoke_result = run_cfg05_real_smoke_pipeline(
            cfg05_model=model_out,
            cfg05_input=features_out,
            target_day=target_day,
            production=True,
        )

        result["readiness_label"] = smoke_result.get("readiness_label")
        result["prediction_rows"] = smoke_result.get("prediction_rows", 0)
        result["validator_passed"] = smoke_result.get("validator_passed", False)

        if smoke_result.get("reason_codes"):
            result["reason_codes"].extend(
                [f"SMOKE:{rc}" for rc in smoke_result["reason_codes"]]
            )

        # REAL_READY check
        from artifacts.readiness import REAL_READY

        if (
            smoke_result.get("readiness_label") == REAL_READY
            and smoke_result.get("prediction_rows", 0) > 0
            and smoke_result.get("validator_passed") is True
        ):
            result["final_status"] = "CFG05_REAL_READY_LOCAL"
            result["reason_codes"].append("CFG05_REAL_READY_LOCAL_ACHIEVED")
        else:
            result["final_status"] = "CFG05_REAL_SMOKE_FAILED"
            result["reason_codes"].append("CFG05_REAL_SMOKE_FAILED")

    elif run_real_smoke and not artifact_gate_pass:
        result["reason_codes"].append("REAL_SMOKE_NOT_ATTEMPTED_ARTIFACT_GATE_FAILED")
    elif run_real_smoke and not input_gate_pass:
        result["reason_codes"].append("REAL_SMOKE_NOT_ATTEMPTED_INPUT_GATE_FAILED")

    # ── Step 5: Determine final status ──
    if result["final_status"] is None:
        if (
            result["model_export_status"] == "EXPORTED"
            and result["feature_export_status"] == "EXPORTED"
        ):
            if result["cfg05_artifact_status"] in (LOADABLE, SCHEMA_READY) \
                    and result["cfg05_input_status"] == SCHEMA_READY:
                result["final_status"] = "CFG05_READY_FOR_SMOKE"
            else:
                result["final_status"] = "CFG05_LOCAL_EXPORT_DONE"
        elif result["model_export_status"] == "FAILED":
            result["final_status"] = "CFG05_LOCAL_TRAIN_FAILED"
        elif result["feature_export_status"] == "FAILED":
            result["final_status"] = "CFG05_INPUT_EXPORT_FAILED"
        else:
            result["final_status"] = "CFG05_LOCAL_EXPORT_FAILED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable P13 report."""
    print("=" * 60)
    print("P13 cfg05 Raw Data → REAL Smoke Report")
    print("=" * 60)
    print(f"  Source repo:        {result['source_repo_status']}")
    print(f"  Raw data:           {result['raw_data_status']}")
    print(f"  Model export:       {result['model_export_status']} -> {result['model_out']}")
    print(f"  Feature export:     {result['feature_export_status']} -> {result['features_out']}")
    print(f"  cfg05 artifact:     {result['cfg05_artifact_status']}")
    print(f"  cfg05 input:        {result['cfg05_input_status']}")
    print(f"  REAL smoke:         {'YES' if result['real_smoke_attempted'] else 'NO'}")
    if result["real_smoke_attempted"]:
        print(f"  Readiness label:    {result['readiness_label']}")
        print(f"  Prediction rows:    {result['prediction_rows']}")
        print(f"  Validator passed:   {result['validator_passed']}")
    print(f"  Final status:       {result['final_status']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P13: raw Chinese CSV → train/export → cfg05 REAL smoke.",
    )
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment.")
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw Chinese CSV.")
    parser.add_argument("--target-day", type=str, default="2026-07-01",
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Local work dir.")
    parser.add_argument("--train-window-days", type=int, default=90,
                        help="Training window in days.")
    parser.add_argument("--run-real-smoke", action="store_true", default=False,
                        help="Attempt REAL smoke if model+features gates pass.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON to stdout.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero on any blocker.")
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

    result = run_p13_cfg05_raw_data_to_real_smoke(
        source_repo=args.source_repo,
        raw_data=args.raw_data,
        target_day=args.target_day,
        work_dir=args.work_dir,
        train_window_days=args.train_window_days,
        run_real_smoke=args.run_real_smoke,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["final_status"] == "CFG05_REAL_READY_LOCAL":
            logger.info("P13 strict mode PASS: REAL_READY_LOCAL")
            return 0
        else:
            logger.error("P13 strict mode FAIL: %s", result["final_status"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
