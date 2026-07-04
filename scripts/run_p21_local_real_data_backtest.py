"""
scripts/run_p21_local_real_data_backtest.py — P21 local real-data backtest orchestration.

Auto-discovers the raw CSV and source repo, then delegates to P16 for the
walk-forward backtest.  Supports quick (7-day), full (30-day), and
three-month (90-day) evaluation windows.

Usage::

    python -m scripts.run_p21_local_real_data_backtest --full-days 30 --json --strict
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

# ── Paths ──────────────────────────────────────────────────────────────────
_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p21_p25_real_chain")

_RAW_CSV_CANDIDATES = [
    os.path.join("..", "electricity_forecast_model2.1", "data", "shandong_pmos_hourly.csv"),
    os.path.join(".local_artifacts", "p14_cfg05", "shandong_pmos_hourly.csv"),
    os.path.join(".local_artifacts", "p15_cfg05", "shandong_pmos_hourly.csv"),
]

_SOURCE_REPO_CANDIDATES = [
    os.path.join(".local_artifacts", "source_repos", "epf-sota-experiment"),
]

_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "ledgers/", "reports/local/")
_ALLOWED_WORK_DIRS = (".local_artifacts",)

# ── Final statuses ─────────────────────────────────────────────────────────
P21_REAL_BACKTEST_COMPLETE = "P21_REAL_BACKTEST_COMPLETE"
P21_REAL_BACKTEST_PARTIAL = "P21_REAL_BACKTEST_PARTIAL"
P21_LOCAL_DATA_MISSING = "P21_LOCAL_DATA_MISSING"
P21_SOURCE_REPO_MISSING = "P21_SOURCE_REPO_MISSING"
P21_BACKTEST_FAILED = "P21_BACKTEST_FAILED"


# ── Helpers ────────────────────────────────────────────────────────────────

def _path_is_safe(path: str) -> bool:
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def discover_raw_csv(
    extra_candidates: Optional[list[str]] = None,
    include_defaults: bool = True,
) -> Optional[str]:
    """Return the first existing raw CSV path, or None.

    Parameters
    ----------
    extra_candidates : list[str], optional
        Additional paths to check (checked first).
    include_defaults : bool
        If True (default), also check built-in default paths.
        Set to False in tests to isolate from real filesystem state.
    """
    candidates: list[str] = list(extra_candidates) if extra_candidates else []
    if include_defaults:
        candidates.extend(_RAW_CSV_CANDIDATES)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def discover_source_repo(
    extra_candidates: Optional[list[str]] = None,
    include_defaults: bool = True,
) -> Optional[str]:
    """Return the first existing source repo directory, or None.

    Parameters
    ----------
    extra_candidates : list[str], optional
        Additional paths to check (checked first).
    include_defaults : bool
        If True (default), also check built-in default paths.
        Set to False in tests to isolate from real filesystem state.
    """
    candidates: list[str] = list(extra_candidates) if extra_candidates else []
    if include_defaults:
        candidates.extend(_SOURCE_REPO_CANDIDATES)
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def run_p21_local_real_data_backtest(
    quick_days: int = 0,
    full_days: int = 0,
    three_month: int = 0,
    work_dir: Optional[str] = None,
    raw_csv_override: Optional[str] = None,
    source_repo_override: Optional[str] = None,
    extra_raw_candidates: Optional[list[str]] = None,
    extra_source_candidates: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run P21 local real-data backtest orchestration.

    Exactly one of *quick_days*, *full_days*, *three_month* should be
    non-zero.  If none is given, defaults to 30.

    Returns a summary dict with ``final_status``.
    """
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    # Determine eval window
    n_days = quick_days or full_days or three_month or 30

    result: dict[str, Any] = {
        "p21_stage": "discovery",
        "raw_csv_path": None,
        "source_repo_path": None,
        "eval_days": n_days,
        "p16_summary": None,
        "final_status": None,
        "reason_codes": [],
        "forbidden_files_check": "PASS",
    }

    # ── Step 1: discover raw CSV ──
    raw_csv = raw_csv_override or discover_raw_csv(extra_raw_candidates)
    if raw_csv is None or not os.path.isfile(raw_csv):
        result["final_status"] = P21_LOCAL_DATA_MISSING
        result["reason_codes"].append("RAW_CSV_NOT_FOUND")
        return result
    result["raw_csv_path"] = raw_csv
    result["reason_codes"].append(f"RAW_CSV_FOUND:{raw_csv}")

    # ── Step 2: discover source repo ──
    source_repo = source_repo_override or discover_source_repo(extra_source_candidates)
    if source_repo is None or not os.path.isdir(source_repo):
        result["final_status"] = P21_SOURCE_REPO_MISSING
        result["reason_codes"].append("SOURCE_REPO_NOT_FOUND")
        return result
    result["source_repo_path"] = source_repo
    result["reason_codes"].append(f"SOURCE_REPO_FOUND:{source_repo}")

    # ── Step 3: determine eval range ──
    try:
        try:
            raw_df = pd.read_csv(raw_csv, encoding="gbk", nrows=5)
        except UnicodeDecodeError:
            raw_df = pd.read_csv(raw_csv, encoding="utf-8", nrows=5)
        raw_df["ds"] = pd.to_datetime(raw_df["时刻"])
        max_ds = raw_df["ds"].max()
        # Read full to get true max
        try:
            raw_full = pd.read_csv(raw_csv, encoding="gbk", usecols=["时刻"])
        except UnicodeDecodeError:
            raw_full = pd.read_csv(raw_csv, encoding="utf-8", usecols=["时刻"])
        raw_full["ds"] = pd.to_datetime(raw_full["时刻"])
        max_ds = raw_full["ds"].max()
        end_day = (max_ds - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_day = (pd.Timestamp(end_day) - pd.Timedelta(days=n_days - 1)).strftime("%Y-%m-%d")
    except Exception as e:
        result["final_status"] = P21_BACKTEST_FAILED
        result["reason_codes"].append(f"EVAL_RANGE_FAILED:{e}")
        return result

    # ── Step 4: call P16 ──
    try:
        from scripts.run_p16_cfg05_30d_walkforward_backtest import (
            run_p16_cfg05_30d_walkforward_backtest,
            BACKTEST_COMPLETE,
        )
        p16_result = run_p16_cfg05_30d_walkforward_backtest(
            raw_data=raw_csv,
            source_repo=source_repo,
            start_day=start_day,
            end_day=end_day,
            work_dir=work_dir,
        )
        result["p16_summary"] = p16_result
        result["reason_codes"].extend(p16_result.get("reason_codes", []))

        # Map P16 status to P21 status
        if p16_result.get("final_status") == BACKTEST_COMPLETE:
            if p16_result.get("complete_days", 0) == p16_result.get("attempted_days", 0):
                result["final_status"] = P21_REAL_BACKTEST_COMPLETE
            else:
                result["final_status"] = P21_REAL_BACKTEST_PARTIAL
        elif p16_result.get("complete_days", 0) > 0:
            result["final_status"] = P21_REAL_BACKTEST_PARTIAL
        else:
            result["final_status"] = P21_BACKTEST_FAILED

    except Exception as e:
        result["final_status"] = P21_BACKTEST_FAILED
        result["reason_codes"].append(f"P16_CALL_FAILED:{e}")

    # ── Step 5: save summary JSON ──
    summary_path = os.path.join(work_dir, "p21_backtest_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        result["summary_path_local"] = summary_path
    except Exception as e:
        result["reason_codes"].append(f"SUMMARY_SAVE_FAILED:{e}")

    # ── Forbidden files check ──
    work_dir_norm = os.path.abspath(work_dir).replace("\\", "/")
    work_dir_is_safe = (
        any(a.lstrip(".") in work_dir_norm for a in _ALLOWED_WORK_DIRS)
        or os.path.isabs(work_dir)
    )
    if not work_dir_is_safe:
        result["forbidden_files_check"] = "FAIL"
    else:
        result["forbidden_files_check"] = "PASS"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P21 Local Real-Data Backtest Report")
    print("=" * 60)
    print(f"  Raw CSV:        {result['raw_csv_path']}")
    print(f"  Source repo:    {result['source_repo_path']}")
    print(f"  Eval days:      {result['eval_days']}")
    print(f"  Final status:   {result['final_status']}")
    print(f"  Forbidden chk:  {result['forbidden_files_check']}")
    if result.get("p16_summary"):
        p16 = result["p16_summary"]
        print(f"  P16 status:     {p16.get('final_status')}")
        print(f"  P16 complete:   {p16.get('complete_days')}/{p16.get('attempted_days')}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P21: local real-data backtest orchestration.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--quick-days", type=int, default=0, help="Quick backtest (e.g. 7 days).")
    group.add_argument("--full-days", type=int, default=0, help="Full backtest (e.g. 30 days).")
    group.add_argument("--three-month", type=int, default=0, help="Three-month backtest (90 days).")
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--raw-csv", type=str, default=None, help="Override raw CSV path.")
    p.add_argument("--source-repo", type=str, default=None, help="Override source repo path.")
    p.add_argument("--json", action="store_true", default=False)
    p.add_argument("--strict", action="store_true", default=False)
    p.add_argument("--verbose", "-v", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stderr)

    work_dir = args.work_dir or _DEFAULT_WORK_DIR
    if not _path_is_safe(work_dir):
        logger.error("Unsafe work-dir: %s", work_dir)
        return 1

    result = run_p21_local_real_data_backtest(
        quick_days=args.quick_days,
        full_days=args.full_days,
        three_month=args.three_month,
        work_dir=work_dir,
        raw_csv_override=args.raw_csv,
        source_repo_override=args.source_repo,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict and result["final_status"] != P21_REAL_BACKTEST_COMPLETE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
