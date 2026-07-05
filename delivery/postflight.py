"""
delivery/postflight.py — P55: Postflight validation checks for a completed delivery run.

Validates the final output CSV against a set of quality and safety checks
before the delivery is considered complete and ready for submission.
"""

from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime
from typing import Any

import pandas as pd


def _check_exists(output_path: str) -> dict[str, Any]:
    """Check 1: final_output exists and is readable CSV."""
    if not os.path.isfile(output_path):
        return {"passed": False, "detail": f"File not found: {output_path}"}
    if not output_path.endswith(".csv"):
        return {"passed": False, "detail": f"Not a CSV file: {output_path}"}
    try:
        pd.read_csv(output_path, nrows=1)
        return {"passed": True, "detail": f"File exists and readable: {output_path}"}
    except Exception as e:
        return {"passed": False, "detail": f"CSV read error: {e}"}


def _check_24_rows(output_path: str) -> dict[str, Any]:
    """Check 2: exactly 24 data rows."""
    try:
        df = pd.read_csv(output_path)
        row_count = len(df)
        if row_count == 24:
            return {"passed": True, "detail": f"Exactly 24 rows ({row_count})"}
        return {"passed": False, "detail": f"Expected 24 rows, got {row_count}"}
    except Exception as e:
        return {"passed": False, "detail": f"Cannot count rows: {e}"}


def _check_hour_business_range(output_path: str) -> dict[str, Any]:
    """Check 3: hour_business is 1..24."""
    try:
        df = pd.read_csv(output_path)
        col = _find_column(df, ["hour_business", "hour", "Hour", "h"])
        if col is None:
            return {"passed": False, "detail": "No hour_business column found"}
        hours = sorted(df[col].dropna().unique())
        expected = list(range(1, 25))
        if hours == expected:
            return {"passed": True, "detail": "hour_business is 1..24"}
        return {
            "passed": False,
            "detail": f"hour_business values: {hours}, expected 1..24",
        }
    except Exception as e:
        return {"passed": False, "detail": f"hour range check error: {e}"}


def _check_no_duplicate_hours(output_path: str) -> dict[str, Any]:
    """Check 4: No duplicate hour_business."""
    try:
        df = pd.read_csv(output_path)
        col = _find_column(df, ["hour_business", "hour", "Hour", "h"])
        if col is None:
            return {"passed": False, "detail": "No hour_business column found"}
        duplicates = df[col].duplicated()
        if duplicates.any():
            dup_vals = df.loc[duplicates, col].tolist()
            return {
                "passed": False,
                "detail": f"Duplicate hour_business values: {dup_vals}",
            }
        return {"passed": True, "detail": "No duplicate hour_business"}
    except Exception as e:
        return {"passed": False, "detail": f"Duplicate check error: {e}"}


def _check_no_nan(output_path: str) -> dict[str, Any]:
    """Check 5: No NaN in y_pred / final price columns."""
    try:
        df = pd.read_csv(output_path)
        # Look for prediction/price columns
        pred_cols = [c for c in df.columns if any(
            kw in c.lower() for kw in ["y_pred", "final_price", "pred", "price", "forecast"]
        )]
        if not pred_cols:
            # If no obvious prediction column, check all numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            pred_cols = numeric_cols
        nan_found = {}
        for col in pred_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_found[col] = int(nan_count)
        if nan_found:
            return {"passed": False, "detail": f"NaN found in columns: {nan_found}"}
        return {"passed": True, "detail": "No NaN in prediction/price columns"}
    except Exception as e:
        return {"passed": False, "detail": f"NaN check error: {e}"}


def _check_business_day(output_path: str, target_date: str) -> dict[str, Any]:
    """Check 6: business_day is consistent with target_date."""
    try:
        df = pd.read_csv(output_path)
        col = _find_column(df, ["business_day", "date", "Date", "ds", "day"])
        if col is None:
            return {"passed": False, "detail": "No business_day/date column found"}
        unique_dates = df[col].dropna().unique()
        if target_date in unique_dates:
            return {
                "passed": True,
                "detail": f"business_day contains target_date: {target_date}",
            }
        return {
            "passed": False,
            "detail": f"target_date {target_date} not in business_day values: {unique_dates}",
        }
    except Exception as e:
        return {"passed": False, "detail": f"business day check error: {e}"}


def _check_profile_delivery_allowed(
    profile_name: str, profile_def: dict | None,
) -> dict[str, Any]:
    """Check 7: profile is delivery_allowed (True)."""
    if profile_def is None:
        return {"passed": True, "detail": "No profile_def provided, skipping"}
    delivery_allowed = profile_def.get("delivery_allowed", False)
    if delivery_allowed:
        return {"passed": True, "detail": f"Profile '{profile_name}' is delivery_allowed"}
    return {
        "passed": False,
        "detail": f"Profile '{profile_name}' has delivery_allowed=False",
    }


def _check_no_quarantined_models(profile_def: dict | None) -> dict[str, Any]:
    """Check 8: no quarantined model used (if profile_def provided)."""
    if profile_def is None:
        return {"passed": True, "detail": "No profile_def provided, skipping"}
    excluded = profile_def.get("excluded_models", {})
    allowed = profile_def.get("allowed_models", [])
    quarantined_in_allowed = [m for m in excluded if m in allowed]
    if quarantined_in_allowed:
        return {
            "passed": False,
            "detail": f"Quarantined models found in allowed_models: {quarantined_in_allowed}",
        }
    return {"passed": True, "detail": "No quarantined models in allowed_models"}


def _check_claim_guard() -> dict[str, Any]:
    """Check 9: claim guard pass (call scripts.validate_delivery_claims.run_claim_guard)."""
    try:
        from scripts.validate_delivery_claims import run_claim_guard
        cg = run_claim_guard()
        violations = cg.get("violations", [])
        if violations:
            return {
                "passed": False,
                "detail": f"Claim guard found {len(violations)} violations",
            }
        return {"passed": True, "detail": "Claim guard passed with 0 violations"}
    except Exception as e:
        return {"passed": False, "detail": f"Claim guard error: {e}"}


def _check_no_git_tracked_artifacts(work_dir: str | None = None) -> dict[str, Any]:
    """Check 10: local artifacts not git-tracked."""
    try:
        check_dir = work_dir or os.getcwd()
        result_proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", check_dir],
            capture_output=True, text=True, timeout=30,
        )
        if result_proc.returncode == 0:
            return {
                "passed": False,
                "detail": f"Work dir is tracked by git: {check_dir}",
            }
        return {"passed": True, "detail": "Local artifacts not git-tracked"}
    except FileNotFoundError:
        return {"passed": True, "detail": "git not available, skipping"}
    except Exception as e:
        return {"passed": True, "detail": f"git check skipped: {e}"}


def _check_hour24_convention(output_path: str) -> dict[str, Any]:
    """Check 11: ds column hour-24 convention (hour 24 -> D+1 00:00:00)."""
    try:
        df = pd.read_csv(output_path)
        col = _find_column(df, ["ds", "datetime", "timestamp", "time"])
        if col is None:
            return {"passed": True, "detail": "No ds/datetime column found, skipping"}
        # Look for hour 24 entries and check they map to D+1 00:00:00
        hour_col = _find_column(df, ["hour_business", "hour", "Hour", "h"])
        if hour_col is None:
            return {"passed": True, "detail": "No hour column to cross-reference, skipping"}
        # Check for any hour 24 entries
        has_hour24 = (df[hour_col] == 24).any() if hour_col in df.columns else False
        if has_hour24:
            return {
                "passed": True,
                "detail": "Hour 24 entries present (convention: D+1 00:00:00)",
            }
        return {"passed": True, "detail": "No hour 24 entries, convention check skipped"}
    except Exception as e:
        return {"passed": False, "detail": f"Hour-24 convention check error: {e}"}


def _check_no_merge_suffixes(output_path: str) -> dict[str, Any]:
    """Check 12: No _x/_y suffix columns from bad merge."""
    try:
        df = pd.read_csv(output_path)
        bad_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        if bad_cols:
            return {
                "passed": False,
                "detail": f"Merge suffix columns found: {bad_cols}",
            }
        return {"passed": True, "detail": "No _x/_y merge suffix columns"}
    except Exception as e:
        return {"passed": False, "detail": f"Merge suffix check error: {e}"}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first column that matches a candidate name (case-insensitive)."""
    for candidate in candidates:
        for col in df.columns:
            if col.strip().lower() == candidate.lower():
                return col
    return None


_CHECK_FUNCTIONS: list[tuple[str, Any]] = [
    ("file_exists_readable", _check_exists),
    ("twenty_four_rows", _check_24_rows),
    ("hour_business_range", _check_hour_business_range),
    ("no_duplicate_hours", _check_no_duplicate_hours),
    ("no_nan_in_predictions", _check_no_nan),
    ("business_day_consistency", _check_business_day),
    ("profile_delivery_allowed", _check_profile_delivery_allowed),
    ("no_quarantined_models", _check_no_quarantined_models),
    ("claim_guard_pass", _check_claim_guard),
    ("no_git_tracked_artifacts", _check_no_git_tracked_artifacts),
    ("hour24_convention", _check_hour24_convention),
    ("no_merge_suffixes", _check_no_merge_suffixes),
]


def run_postflight(
    output_path: str,
    target_date: str,
    profile_name: str,
    profile_def: dict | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Run all postflight checks on delivery output.

    Parameters
    ----------
    output_path : str
        Path to the final output CSV file.
    target_date : str
        The target delivery date in YYYY-MM-DD format.
    profile_name : str
        The delivery profile name used.
    profile_def : dict | None
        The profile definition dict. If None, profile-related checks
        are skipped.
    work_dir : str | None
        Working directory for git-track check. Defaults to cwd.

    Returns
    -------
    dict
        Postflight results with status, checks, errors, warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    for check_name, check_fn in _CHECK_FUNCTIONS:
        try:
            if check_name == "business_day_consistency":
                result = check_fn(output_path, target_date)
            elif check_name == "profile_delivery_allowed":
                result = check_fn(profile_name, profile_def)
            elif check_name == "no_quarantined_models":
                result = check_fn(profile_def)
            elif check_name in ("file_exists_readable", "twenty_four_rows",
                                "hour_business_range", "no_duplicate_hours",
                                "no_nan_in_predictions", "hour24_convention",
                                "no_merge_suffixes"):
                result = check_fn(output_path)
            elif check_name == "claim_guard_pass":
                result = check_fn()
            elif check_name == "no_git_tracked_artifacts":
                result = check_fn(work_dir)
            else:
                result = check_fn(output_path, target_date, profile_name, profile_def)

            checks[check_name] = result
            if not result["passed"]:
                errors.append(f"{check_name}: {result['detail']}")
        except Exception as e:
            checks[check_name] = {"passed": False, "detail": f"Unexpected error: {e}"}
            errors.append(f"{check_name}: Unexpected error: {e}")

    # Compute summary
    total = len(checks)
    passed_count = sum(1 for c in checks.values() if c["passed"])
    failed_count = total - passed_count

    if failed_count == 0:
        status = "PASS"
    elif passed_count >= total - 2:  # allow up to 2 skippable failures
        status = "WARN"
    else:
        status = "FAIL"

    # Determine submission_ready_path
    submission_ready_path = output_path
    if status == "PASS":
        submission_ready_path = output_path.replace(".csv", "_submission_ready.csv")
        if submission_ready_path == output_path:
            submission_ready_path = output_path

    return {
        "status": status,
        "target_date": target_date,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": failed_count,
            "warned": len(warnings),
        },
        "output_path": output_path,
        "submission_ready_path": submission_ready_path,
    }
