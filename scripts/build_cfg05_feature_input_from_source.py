"""
scripts/build_cfg05_feature_input_from_source.py — Build cfg05 feature input.

Validates candidate input CSV against CFG05_FEATURE_COLUMNS, or searches
source repo for compatible feature CSVs and generation scripts.

Usage::

    # Validate existing CSV
    python -m scripts.build_cfg05_feature_input_from_source \\
        --input-csv /path/to/data.csv

    # Search source repo for feature data
    python -m scripts.build_cfg05_feature_input_from_source \\
        --source-repo /path/to/epf-sota-experiment

    # Validate with target day and optional output
    python -m scripts.build_cfg05_feature_input_from_source \\
        --input-csv /path/to/data.csv --target-day 2026-07-01 \\
        --out .local_artifacts/p11_cfg05/cfg05_input.csv

Options::

    --source-repo PATH          Path to epf-sota-experiment repository.
    --input-csv PATH            Candidate input CSV to validate.
    --target-day YYYY-MM-DD     Target day for row count validation.
    --out PATH                  Local output path for prepared CSV.
    --json                      Output JSON report.
    --strict                    Exit non-zero if input not ready.
    --verbose, -v               Increase log verbosity.

Output statuses::

    CFG05_INPUT_BLOCKED         Source repo not found or no compatible CSV.
    CFG05_INPUT_VALIDATED       Input CSV schema-ready for target day.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from artifacts.readiness import SCHEMA_READY, INVALID, MISSING

logger = logging.getLogger(__name__)

# CSV patterns to search for in source repo
_FEATURE_CSV_PATTERNS = ["features", "data", "input", "cfg05", "prepared", "training"]
_SCRIPT_CSV_PATTERNS = ["csv", "read_csv", "to_csv", "feature", "create_data", "build_data"]


def _find_feature_csvs(source_repo: str) -> list[str]:
    """Find candidate CSV files that might contain cfg05 features."""
    found: list[str] = []
    for root, _dirs, files in os.walk(source_repo):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            lower = fname.lower()
            if any(p in lower for p in _FEATURE_CSV_PATTERNS):
                full = os.path.join(root, fname)
                # Quick check: does it have at least some of our feature columns?
                try:
                    df = pd.read_csv(full, nrows=1)
                    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
                    overlap = len([c for c in CFG05_FEATURE_COLUMNS if c in df.columns])
                    if overlap >= 10:  # threshold: at least 10 feature columns
                        found.append(full)
                        logger.debug("Found candidate CSV: %s (%d/%d features)",
                                     full, overlap, len(CFG05_FEATURE_COLUMNS))
                except Exception:
                    continue
    return found


def _find_feature_scripts(source_repo: str) -> list[str]:
    """Find scripts that might build cfg05 features."""
    found: list[str] = []
    for root, _dirs, files in os.walk(source_repo):
        for fname in files:
            if not fname.endswith((".py", ".ipynb")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                if any(p in content for p in _SCRIPT_CSV_PATTERNS):
                    found.append(fpath)
            except Exception:
                continue
    return found


def build_cfg05_feature_input_from_source(
    source_repo: Optional[str] = None,
    input_csv: Optional[str] = None,
    target_day: Optional[str] = None,
    out_path: Optional[str] = None,
) -> dict[str, Any]:
    """Build or validate cfg05 feature input.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment.
    input_csv : str, optional
        Candidate input CSV to validate.
    target_day : str, optional
        Target day for row count.
    out_path : str, optional
        Output path for prepared CSV (only written if valid).

    Returns
    -------
    dict with build attempt summary.
    """
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

    result: dict[str, Any] = {
        "source_repo": source_repo,
        "input_csv": input_csv,
        "target_day": target_day,
        "feature_count": len(CFG05_FEATURE_COLUMNS),
        "input_status": MISSING,
        "columns_present": 0,
        "columns_missing": [],
        "has_ds": False,
        "ds_parsable": True,
        "target_day_rows": None,
        "candidate_csvs_found": [],
        "feature_scripts_found": [],
        "out_written": False,
        "reason_codes": [],
        "next_commands": [],
    }

    # --- Priority 1: Validate existing input CSV ---
    if input_csv:
        if not os.path.isfile(input_csv):
            result["reason_codes"].append(f"INPUT_CSV_NOT_FOUND: {input_csv}")
            result["input_status"] = MISSING
            result["next_commands"].append("Provide valid --input-csv path")
            return result

        result["reason_codes"].append(f"INPUT_CSV_FOUND: {input_csv}")

        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            result["input_status"] = INVALID
            result["reason_codes"].append(f"INPUT_CSV_LOAD_FAILED: {e}")
            return result

        if len(df) == 0:
            result["input_status"] = INVALID
            result["reason_codes"].append("INPUT_CSV_EMPTY")
            return result

        # Check feature columns
        present = [c for c in CFG05_FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in CFG05_FEATURE_COLUMNS if c not in df.columns]
        result["columns_present"] = len(present)
        result["columns_missing"] = missing

        if missing:
            result["input_status"] = INVALID
            result["reason_codes"].append(
                f"INPUT_MISSING_{len(missing)}_FEATURE_COLUMNS"
            )
        else:
            result["reason_codes"].append(
                f"INPUT_ALL_{len(CFG05_FEATURE_COLUMNS)}_FEATURE_COLUMNS_PRESENT"
            )

        # Check ds
        result["has_ds"] = "ds" in df.columns
        if not result["has_ds"]:
            result["input_status"] = INVALID
            result["reason_codes"].append("INPUT_MISSING_DS")
            return result

        try:
            ds_parsed = pd.to_datetime(df["ds"])
            result["ds_parsable"] = True
        except Exception:
            result["ds_parsable"] = False
            result["input_status"] = INVALID
            result["reason_codes"].append("INPUT_DS_NOT_PARSABLE")
            return result

        # Check target day row count
        if target_day:
            target_dt = pd.Timestamp(target_day)
            start = target_dt + pd.Timedelta(hours=1)
            end = target_dt + pd.Timedelta(days=1)
            mask = (ds_parsed >= start) & (ds_parsed < end)
            result["target_day_rows"] = int(mask.sum())
            result["reason_codes"].append(
                f"TARGET_DAY_ROWS: {result['target_day_rows']}"
            )
            if result["target_day_rows"] != 24:
                result["reason_codes"].append(
                    f"TARGET_DAY_EXPECTED_24_ROWS_GOT_{result['target_day_rows']}"
                )

        if not missing and result["has_ds"] and result["ds_parsable"]:
            result["input_status"] = SCHEMA_READY
            result["reason_codes"].append("CFG05_INPUT_SCHEMA_READY")

            # Write prepared CSV only if --out provided
            if out_path:
                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                out_cols = ["ds"] + list(CFG05_FEATURE_COLUMNS)
                out_df = df[[c for c in out_cols if c in df.columns]]
                out_df.to_csv(out_path, index=False)
                result["out_written"] = True
                logger.info("Wrote prepared input to %s", out_path)
        else:
            result["input_status"] = INVALID

        return result

    # --- Priority 2: Search source repo for feature CSVs ---
    if not source_repo:
        result["reason_codes"].append("NO_SOURCE_REPO_AND_NO_INPUT_CSV")
        result["input_status"] = MISSING
        result["next_commands"].extend([
            "Provide --input-csv with existing feature data, or",
            "Provide --source-repo pointing to epf-sota-experiment",
        ])
        return result

    if not os.path.isdir(source_repo):
        result["reason_codes"].append(f"SOURCE_REPO_NOT_FOUND: {source_repo}")
        result["input_status"] = MISSING
        result["next_commands"].extend([
            f"Clone epf-sota-experiment first",
            f"Then run with --source-repo <path>",
        ])
        return result

    result["reason_codes"].append(f"SOURCE_REPO_FOUND: {source_repo}")

    # Search for candidate CSVs
    csvs = _find_feature_csvs(source_repo)
    result["candidate_csvs_found"] = csvs

    if csvs:
        result["reason_codes"].append(
            f"FOUND_{len(csvs)}_CANDIDATE_CSVS_IN_SOURCE"
        )
        result["next_commands"].extend([
            f"Re-run with --input-csv pointing to one of the candidate CSVs above",
        ])
    else:
        # Search for feature-building scripts
        scripts = _find_feature_scripts(source_repo)
        result["feature_scripts_found"] = scripts
        if scripts:
            result["reason_codes"].append(
                f"NO_CSV_FOUND_BUT_FOUND_{len(scripts)}_FEATURE_SCRIPTS"
            )
            result["next_commands"].extend([
                "Run feature-building scripts to generate input CSV, then",
                "Re-run with --input-csv <generated_csv>",
            ])
        else:
            result["reason_codes"].append("NO_CANDIDATE_CSV_OR_FEATURE_SCRIPTS")
            result["next_commands"].extend([
                "1. Clone epf-sota-experiment",
                "2. Run feature pipeline to generate CSV",
                "3. Re-run with --input-csv",
            ])

    result["input_status"] = "CFG05_INPUT_BLOCKED"
    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable build report."""
    print("=" * 60)
    print("cfg05 Feature Input Build Report")
    print("=" * 60)
    print(f"  Source repo:     {result['source_repo'] or 'N/A'}")
    print(f"  Input CSV:       {result['input_csv'] or 'N/A'}")
    print(f"  Target day:      {result['target_day'] or 'N/A'}")
    print(f"  Feature count:   {result['feature_count']} (from CFG05_FEATURE_COLUMNS)")
    print(f"  Input status:    {result['input_status']}")
    if result["columns_present"]:
        print(f"  Columns present: {result['columns_present']}")
    if result["columns_missing"]:
        print(f"  Columns missing: {len(result['columns_missing'])}")
    if result["has_ds"]:
        print(f"  Has ds:          {result['has_ds']}")
    if result["target_day_rows"] is not None:
        print(f"  Target rows:     {result['target_day_rows']}")
    print(f"  Out written:     {result['out_written']}")
    print()
    if result["candidate_csvs_found"]:
        print(f"  Candidate CSVs ({len(result['candidate_csvs_found'])}):")
        for c in result["candidate_csvs_found"]:
            print(f"    {c}")
    if result["feature_scripts_found"]:
        print(f"  Feature scripts ({len(result['feature_scripts_found'])}):")
        for s in result["feature_scripts_found"]:
            print(f"    {s}")
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("  Next commands:")
    for cmd in result.get("next_commands", []):
        print(f"    -> {cmd}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = build_cfg05_feature_input_from_source(
        source_repo=args.source_repo,
        input_csv=args.input_csv,
        target_day=args.target_day,
        out_path=args.out,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["input_status"] == SCHEMA_READY:
            return 0
        else:
            logger.error("Input strict mode FAILED: %s", result["input_status"])
            return 1

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/validate cfg05 feature input from source.",
    )
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment.")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="Candidate input CSV to validate.")
    parser.add_argument("--target-day", type=str, default=None,
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--out", type=str, default=None,
                        help="Local output path for prepared CSV.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero if input not ready.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
