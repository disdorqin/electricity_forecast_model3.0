"""
scripts/run_p14_raw_csv_intake_cfg05.py — P14 one-command wrapper.

One-command entry point for raw Chinese CSV → schema check → local train/export
→ cfg05 REAL smoke pipeline.

Usage::

    # No raw data (shows help and exits)
    python -m scripts.run_p14_raw_csv_intake_cfg05

    # With raw data (full pipeline)
    python -m scripts.run_p14_raw_csv_intake_cfg05 \\
        --raw-data /path/to/shandong_pmos_hourly.csv \\
        --target-day 2026-07-01

    # Full pipeline with REAL smoke attempt
    python -m scripts.run_p14_raw_csv_intake_cfg05 \\
        --raw-data /path/to/shandong_pmos_hourly.csv \\
        --target-day 2026-07-01 --run-real-smoke --json --strict

Options::

    --raw-data PATH             Path to raw Chinese CSV.
    --source-repo PATH          Path to epf-sota-experiment.
    --target-day YYYY-MM-DD     Target day for prediction (default: 2026-07-01).
    --work-dir PATH             Local work dir (default: .local_artifacts/p14_cfg05).
    --train-window-days N       Training window in days (default: 90).
    --run-real-smoke            Attempt REAL smoke if model+features gates pass.
    --force                     Overwrite existing output files.
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

from scripts.check_cfg05_raw_data_contract import (
    check_cfg05_raw_data_contract,
    RAW_DATA_MISSING,
    RAW_DATA_INVALID,
    RAW_DATA_VALID,
)
from scripts.inspect_cfg05_raw_csv_schema import inspect_cfg05_raw_csv_schema
from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
    run_p13_cfg05_raw_data_to_real_smoke,
)

logger = logging.getLogger(__name__)

# ── Safe paths ─────────────────────────────────────────────────────────────
_ALLOWED_WORK_DIRS = (".local_artifacts",)
_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "ledgers/", "reports/local/")

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p14_cfg05")
_DEFAULT_SOURCE_REPO = os.path.join(
    ".local_artifacts", "source_repos", "epf-sota-experiment",
)

# Command examples shown when raw data is missing
_REQUIRED_COLUMNS = [
    "时刻", "日前电价", "实时电价", "直调负荷预测值",
    "风电总加预测值", "光伏总加预测值", "联络线受电负荷预测值", "竞价空间预测值",
]


def _path_is_safe(path: str) -> bool:
    """Check path is under an ignored, allowed directory."""
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def _print_missing_data_help() -> None:
    """Print helpful message when raw data is not provided."""
    print("=" * 60)
    print("cfg05 Raw Data Intake — Missing Data")
    print("=" * 60)
    print()
    print("No raw Chinese CSV was provided.")
    print()
    print("Required Chinese columns (8):")
    for col in _REQUIRED_COLUMNS:
        print(f"  - {col}")
    print()
    print("To inspect a CSV:")
    print("  python -m scripts.inspect_cfg05_raw_csv_schema \\")
    print("      --raw-data /path/to/shandong_pmos_hourly.csv --json")
    print()
    print("To run the full P14 pipeline:")
    print("  python -m scripts.run_p14_raw_csv_intake_cfg05 \\")
    print("      --raw-data /path/to/shandong_pmos_hourly.csv \\")
    print("      --target-day 2026-07-01 \\")
    print("      --run-real-smoke --json --strict")
    print()
    print("Expected data location:")
    print("  electricity_forecast_model2.1/data/shandong_pmos_hourly.csv")
    print("=" * 60)


def run_p14_raw_csv_intake_cfg05(
    source_repo: Optional[str] = None,
    raw_data: Optional[str] = None,
    target_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    train_window_days: int = 90,
    run_real_smoke: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """P14 one-command wrapper: raw CSV → schema check → train/export → REAL smoke.

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
    force : bool
        Overwrite existing output files.

    Returns
    -------
    dict with complete P14 summary.
    """
    source_repo = source_repo or _DEFAULT_SOURCE_REPO
    target_day = target_day or "2026-07-01"
    work_dir = work_dir or _DEFAULT_WORK_DIR

    model_out = os.path.join(work_dir, "cfg05_model.txt")
    features_out = os.path.join(work_dir, f"cfg05_features_{target_day}.csv")

    result: dict[str, Any] = {
        "raw_data_status": "NOT_CHECKED",
        "source_repo_status": "NOT_CHECKED",
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

    # ── Step 1: Check raw data ──
    if not raw_data:
        result["raw_data_status"] = RAW_DATA_MISSING
        result["final_status"] = "CFG05_RAW_DATA_MISSING"
        result["reason_codes"].append("NO_RAW_DATA_PATH_PROVIDED")
        return result

    if not os.path.isfile(raw_data):
        result["raw_data_status"] = RAW_DATA_MISSING
        result["final_status"] = "CFG05_RAW_DATA_MISSING"
        result["reason_codes"].append(f"RAW_DATA_FILE_NOT_FOUND: {raw_data}")
        return result

    # Run schema inspection
    schema = inspect_cfg05_raw_csv_schema(raw_data=raw_data)
    result["raw_data_status"] = schema["raw_data_status"]
    if schema.get("reason_codes"):
        result["reason_codes"].extend(
            [f"SCHEMA:{rc}" for rc in schema["reason_codes"]]
        )

    if schema["raw_data_status"] != RAW_DATA_VALID:
        result["final_status"] = "CFG05_RAW_DATA_INVALID"
        result["reason_codes"].append("P14_STOPPED_RAW_DATA_INVALID")
        return result

    # ── Step 2: Check source repo ──
    if not os.path.isdir(source_repo):
        result["source_repo_status"] = "MISSING"
        result["reason_codes"].append(f"SOURCE_REPO_MISSING: {source_repo}")
        result["final_status"] = "CFG05_RAW_DATA_INVALID"
        return result

    result["source_repo_status"] = "PRESENT"
    result["reason_codes"].append(f"SOURCE_REPO_FOUND: {source_repo}")

    # ── Step 3: Delegate to P13 orchestration ──
    p13_result = run_p13_cfg05_raw_data_to_real_smoke(
        source_repo=source_repo,
        raw_data=raw_data,
        target_day=target_day,
        work_dir=work_dir,
        train_window_days=train_window_days,
        run_real_smoke=run_real_smoke,
    )

    # Merge P13 results into P14 summary
    for key in [
        "model_export_status", "feature_export_status",
        "cfg05_artifact_status", "cfg05_input_status",
        "real_smoke_attempted", "readiness_label",
        "prediction_rows", "validator_passed", "final_status",
    ]:
        if key in p13_result and p13_result[key] is not None:
            result[key] = p13_result[key]

    if p13_result.get("reason_codes"):
        result["reason_codes"].extend(
            [f"P13:{rc}" for rc in p13_result["reason_codes"]]
        )

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable P14 report."""
    print("=" * 60)
    print("P14 Raw CSV Intake + cfg05 REAL Report")
    print("=" * 60)
    print(f"  Raw data status:     {result['raw_data_status']}")
    print(f"  Source repo:         {result['source_repo_status']}")
    print(f"  Model export:        {result['model_export_status']}")
    print(f"  Feature export:      {result['feature_export_status']}")
    print(f"  cfg05 artifact:      {result['cfg05_artifact_status']}")
    print(f"  cfg05 input:         {result['cfg05_input_status']}")
    print(f"  REAL smoke:          {'YES' if result['real_smoke_attempted'] else 'NO'}")
    if result["real_smoke_attempted"]:
        print(f"  Readiness label:     {result['readiness_label']}")
        print(f"  Prediction rows:     {result['prediction_rows']}")
        print(f"  Validator passed:    {result['validator_passed']}")
    print(f"  Model out:           {result['model_out']}")
    print(f"  Features out:        {result['features_out']}")
    print(f"  Final status:        {result['final_status']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P14: raw Chinese CSV → schema check → train/export → REAL smoke.",
    )
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw Chinese CSV.")
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment.")
    parser.add_argument("--target-day", type=str, default="2026-07-01",
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Local work dir (default: .local_artifacts/p14_cfg05).")
    parser.add_argument("--train-window-days", type=int, default=90,
                        help="Training window in days.")
    parser.add_argument("--run-real-smoke", action="store_true", default=False,
                        help="Run REAL smoke if gates pass.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite existing output files.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON.")
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

    # Validate work-dir safety
    work_dir = args.work_dir or _DEFAULT_WORK_DIR
    if not _path_is_safe(work_dir):
        logger.error("Unsafe work-dir: %s", work_dir)
        return 1

    result = run_p14_raw_csv_intake_cfg05(
        source_repo=args.source_repo,
        raw_data=args.raw_data,
        target_day=args.target_day,
        work_dir=work_dir,
        train_window_days=args.train_window_days,
        run_real_smoke=args.run_real_smoke,
        force=args.force,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif result["final_status"] == "CFG05_RAW_DATA_MISSING" and not args.raw_data:
        _print_report(result)
        print()
        _print_missing_data_help()
    else:
        _print_report(result)

    if args.strict:
        if result["final_status"] == "CFG05_REAL_READY_LOCAL":
            logger.info("P14 strict mode PASS: REAL_READY_LOCAL")
            return 0
        elif result["final_status"] in ("CFG05_RAW_DATA_MISSING", "CFG05_RAW_DATA_INVALID"):
            logger.error("P14 strict mode FAIL: %s", result["final_status"])
            return 1
        elif result.get("real_smoke_attempted") and result["final_status"] != "CFG05_REAL_READY_LOCAL":
            logger.error("P14 strict mode FAIL: %s", result["final_status"])
            return 1
        else:
            # Non-REAL but no blocker (structural pass)
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
