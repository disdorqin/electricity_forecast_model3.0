#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P142 - 2.5 Fair Comparison (If Available)
==========================================
Searches for model-2.5 prediction artefacts across known repository paths.
If found, evaluates them on the **same 2025 window** as cfg05 3.0 and BGEW
using the canonical sMAPE_floor50 metric, producing a fair comparison.

If NOT found, reports ``2.5_COMPARISON_UNAVAILABLE`` and logs every path that
was checked -- **no data is fabricated**.

Outputs land in .local_artifacts/p142_fair_comparison/.

Main entry: run_p142_fair_comparison(output_dir, day_start, day_end)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical sMAPE with floor 50
# ---------------------------------------------------------------------------

def compute_smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
) -> float:
    y_true_f = np.maximum(np.asarray(y_true, dtype=float), floor)
    y_pred_f = np.maximum(np.asarray(y_pred, dtype=float), floor)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    if mask.sum() == 0:
        return 0.0
    return float(
        200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask])
    )


def compute_metrics(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> Dict[str, Any]:
    n = len(y_true_arr)
    if n == 0:
        return {"smape_floor50": None, "mae": None, "rmse": None, "count": 0}
    smape = compute_smape_floor50(y_true_arr, y_pred_arr)
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    return {"smape_floor50": round(smape, 4), "mae": round(mae, 4),
            "rmse": round(rmse, 4), "count": n}


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

# Candidate relative paths (relative to repo root) where 2.5 artefacts may live
_CANDIDATE_REL_PATHS: List[str] = [
    "../electricity_forecast_model2.5",
    "../electricity_forecast_model2.0_exp",
    ".local_artifacts/source_repos/electricity_forecast_model2.5",
]

# File-name patterns that would indicate 2.5 predictions
_PREDICTION_FILE_PATTERNS: List[str] = [
    "all_predictions.csv",
    "predictions.csv",
    "dayahead_predictions.csv",
    "full_predictions.csv",
    "output/predictions.csv",
    "artifacts/all_predictions.csv",
]


def _search_paths(repo_root: Path) -> List[Dict[str, Any]]:
    """Check every candidate path and return a log of what was found.

    Returns a list of dicts with keys:
        path        : str   -- the absolute path checked
        exists      : bool  -- whether the directory exists
        predictions : str|None -- path to a predictions CSV if found
        note        : str   -- human-readable note
    """
    log: List[Dict[str, Any]] = []

    # 1. Check the well-known relative paths
    for rel in _CANDIDATE_REL_PATHS:
        abs_path = (repo_root / rel).resolve()
        entry: Dict[str, Any] = {
            "path": str(abs_path),
            "exists": abs_path.is_dir(),
            "predictions": None,
            "note": "",
        }
        if abs_path.is_dir():
            found = _find_predictions_csv(abs_path)
            if found:
                entry["predictions"] = str(found)
                entry["note"] = "predictions CSV found"
            else:
                entry["note"] = "directory exists but no predictions CSV found"
        else:
            entry["note"] = "directory does not exist"
        log.append(entry)

    # 2. Scan .local_artifacts/ for any sub-directory whose name hints at 2.5
    local_art = repo_root / ".local_artifacts"
    if local_art.is_dir():
        for child in sorted(local_art.iterdir()):
            if child.is_dir() and _looks_like_25(child.name):
                abs_path = child.resolve()
                # skip if already covered
                if any(e["path"] == str(abs_path) for e in log):
                    continue
                entry = {
                    "path": str(abs_path),
                    "exists": True,
                    "predictions": None,
                    "note": "",
                }
                found = _find_predictions_csv(abs_path)
                if found:
                    entry["predictions"] = str(found)
                    entry["note"] = "predictions CSV found (discovered via .local_artifacts scan)"
                else:
                    entry["note"] = "name suggests 2.5 but no predictions CSV found"
                log.append(entry)

    return log


def _looks_like_25(dirname: str) -> bool:
    """Heuristic: does this directory name suggest model 2.5?"""
    low = dirname.lower()
    return any(tok in low for tok in ["2.5", "2_5", "model25", "v2.5", "v25"])


def _find_predictions_csv(directory: Path) -> Optional[Path]:
    """Search *directory* (recursively, up to depth 3) for a predictions CSV."""
    for pat in _PREDICTION_FILE_PATTERNS:
        candidate = directory / pat
        if candidate.is_file():
            return candidate
    # Fallback: glob one level deep
    for csv_file in directory.rglob("*.csv"):
        # stop at depth 3
        try:
            rel = csv_file.relative_to(directory)
            if len(rel.parts) <= 3 and csv_file.stat().st_size > 1000:
                # Quick check: does it have y_pred and y_true columns?
                try:
                    header = pd.read_csv(csv_file, nrows=0).columns.tolist()
                    if "y_pred" in header and "y_true" in header:
                        return csv_file
                except Exception:
                    pass
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Evaluation on same window
# ---------------------------------------------------------------------------

def _evaluate_25_predictions(
    predictions_csv: Path,
    day_start: str,
    day_end: str,
) -> Optional[Dict[str, Any]]:
    """Load 2.5 predictions, filter to the same window, compute metrics.

    Returns None if the file cannot be loaded or has no overlapping dates.
    """
    try:
        df = pd.read_csv(predictions_csv)
    except Exception:
        return None

    # Need at minimum: ds (or date), y_pred, y_true
    col_map = _detect_columns(df)
    if col_map is None:
        return None

    ds_col, yp_col, yt_col = col_map
    df["_ds"] = pd.to_datetime(df[ds_col], errors="coerce")
    df = df.dropna(subset=["_ds"])

    # Filter to window
    start_ts = pd.Timestamp(day_start)
    end_ts = pd.Timestamp(day_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = (df["_ds"] >= start_ts) & (df["_ds"] <= end_ts)
    sub = df.loc[mask]

    if len(sub) == 0:
        return None

    y_true = sub[yt_col].values.astype(float)
    y_pred = sub[yp_col].values.astype(float)

    # Remove NaN
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return None

    metrics = compute_metrics(y_true, y_pred)
    metrics["window_start"] = day_start
    metrics["window_end"] = day_end
    metrics["source_file"] = str(predictions_csv)
    return metrics


def _detect_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    """Return (ds_col, y_pred_col, y_true_col) or None."""
    cols_lower = {c.lower(): c for c in df.columns}
    ds_col = None
    for candidate in ["ds", "date", "datetime", "timestamp", "time", "时刻"]:
        if candidate in cols_lower:
            ds_col = cols_lower[candidate]
            break
    yp_col = None
    for candidate in ["y_pred", "pred", "prediction", "forecast"]:
        if candidate in cols_lower:
            yp_col = cols_lower[candidate]
            break
    yt_col = None
    for candidate in ["y_true", "actual", "target", "real"]:
        if candidate in cols_lower:
            yt_col = cols_lower[candidate]
            break
    if ds_col and yp_col and yt_col:
        return ds_col, yp_col, yt_col
    return None


# ---------------------------------------------------------------------------
# Load 3.0 cfg05 metrics for comparison
# ---------------------------------------------------------------------------

def _load_cfg05_metrics(repo_root: Path, day_start: str, day_end: str) -> Optional[Dict[str, Any]]:
    """Load cfg05 all_predictions.csv and compute metrics on the same window."""
    pred_path = repo_root / ".local_artifacts" / "p2025_full" / "dayahead" / "all_predictions.csv"
    if not pred_path.is_file():
        return None
    try:
        df = pd.read_csv(pred_path)
        df["ds"] = pd.to_datetime(df["ds"])
        start_ts = pd.Timestamp(day_start)
        end_ts = pd.Timestamp(day_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = (df["ds"] >= start_ts) & (df["ds"] <= end_ts)
        sub = df.loc[mask]
        if len(sub) == 0:
            return None
        return compute_metrics(sub["y_true"].values.astype(float),
                               sub["y_pred"].values.astype(float))
    except Exception:
        return None


def _load_bgew_metrics(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load BGEW daily_metrics.csv (P138) if available."""
    bgew_path = repo_root / ".local_artifacts" / "p138_rolling_bgew" / "daily_metrics.csv"
    if not bgew_path.is_file():
        return None
    try:
        df = pd.read_csv(bgew_path)
        # Try to compute overall metrics from the daily file
        if "smape" in df.columns:
            return {
                "smape_floor50": round(float(df["smape"].mean()), 4) if "smape" in df.columns else None,
                "mae": round(float(df["mae"].mean()), 4) if "mae" in df.columns else None,
                "rmse": round(float(df["rmse"].mean()), 4) if "rmse" in df.columns else None,
                "count": int(df["count"].sum()) if "count" in df.columns else len(df),
                "source": "p138_rolling_bgew/daily_metrics.csv",
            }
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

STATUS_UNAVAILABLE = "2.5_COMPARISON_UNAVAILABLE"
STATUS_AVAILABLE = "2.5_COMPARISON_AVAILABLE"


def run_p142_fair_comparison(
    output_dir: str,
    day_start: str = "2025-01-01",
    day_end: str = "2025-12-31",
) -> dict:
    """Run the 2.5 fair comparison.

    Parameters
    ----------
    output_dir : str
        Where to write artefacts.
    day_start, day_end : str
        Inclusive date window for fair comparison.

    Returns
    -------
    dict
        Result summary with status and comparison data (or unavailability report).
    """
    repo_root = Path(__file__).resolve().parent.parent
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- 1. Search for 2.5 artefacts ----------------------------------------
    search_log = _search_paths(repo_root)
    _write_json(out / "search_log.json", search_log)

    # -- 2. Determine if we found usable 2.5 predictions --------------------
    found_entry = next((e for e in search_log if e["predictions"] is not None), None)

    # -- 3. Load 3.0 cfg05 baseline -----------------------------------------
    cfg05_metrics = _load_cfg05_metrics(repo_root, day_start, day_end)
    bgew_metrics = _load_bgew_metrics(repo_root)

    if found_entry is None:
        # -- UNAVAILABLE path ------------------------------------------------
        result = {
            "status": STATUS_UNAVAILABLE,
            "day_start": day_start,
            "day_end": day_end,
            "search_summary": [
                {"path": e["path"], "exists": e["exists"], "note": e["note"]}
                for e in search_log
            ],
            "paths_checked": len(search_log),
            "paths_with_predictions": 0,
            "cfg03_metrics": cfg05_metrics,
            "bgew_metrics": bgew_metrics,
            "comparison": None,
            "reason": (
                "No model-2.5 prediction artefacts were found in any of the "
                f"{len(search_log)} searched paths. "
                "Comparison data is NOT fabricated."
            ),
        }
        _write_json(out / "comparison_metrics.json", result)
        return result

    # -- 4. Evaluate 2.5 on same window -------------------------------------
    pred_path = Path(found_entry["predictions"])
    metrics_25 = _evaluate_25_predictions(pred_path, day_start, day_end)

    if metrics_25 is None:
        result = {
            "status": STATUS_UNAVAILABLE,
            "day_start": day_start,
            "day_end": day_end,
            "search_summary": [
                {"path": e["path"], "exists": e["exists"], "note": e["note"]}
                for e in search_log
            ],
            "paths_checked": len(search_log),
            "paths_with_predictions": 1,
            "cfg03_metrics": cfg05_metrics,
            "bgew_metrics": bgew_metrics,
            "comparison": None,
            "reason": (
                f"Found predictions file at {pred_path} but could not "
                "evaluate it on the requested window "
                f"[{day_start}, {day_end}]. "
                "Comparison data is NOT fabricated."
            ),
        }
        _write_json(out / "comparison_metrics.json", result)
        return result

    # -- 5. Build comparison ------------------------------------------------
    comparison = {
        "model_25": metrics_25,
        "model_30_cfg05": cfg05_metrics,
        "bgew": bgew_metrics,
        "window": {"start": day_start, "end": day_end},
    }

    # Delta: lower is better for all metrics
    if cfg05_metrics and metrics_25.get("smape_floor50") is not None and cfg05_metrics.get("smape_floor50") is not None:
        delta_smape = round(metrics_25["smape_floor50"] - cfg05_metrics["smape_floor50"], 4)
        comparison["delta_smape_floor50"] = delta_smape
        comparison["better_model"] = "2.5" if delta_smape < 0 else "3.0_cfg05"

    result = {
        "status": STATUS_AVAILABLE,
        "day_start": day_start,
        "day_end": day_end,
        "search_summary": [
            {"path": e["path"], "exists": e["exists"], "note": e["note"]}
            for e in search_log
        ],
        "paths_checked": len(search_log),
        "paths_with_predictions": sum(1 for e in search_log if e["predictions"] is not None),
        "comparison": comparison,
        "cfg03_metrics": cfg05_metrics,
        "bgew_metrics": bgew_metrics,
    }
    _write_json(out / "comparison_metrics.json", result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_dir() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / ".local_artifacts" / "p142_fair_comparison")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        out_dir = sys.argv[1]
    else:
        out_dir = _default_output_dir()

    day_start = sys.argv[2] if len(sys.argv) >= 3 else "2025-01-01"
    day_end = sys.argv[3] if len(sys.argv) >= 4 else "2025-12-31"

    print(f"[P142] output_dir  : {out_dir}")
    print(f"[P142] day_start   : {day_start}")
    print(f"[P142] day_end     : {day_end}")

    result = run_p142_fair_comparison(out_dir, day_start, day_end)

    print(f"\n[P142] Status: {result['status']}")
    print(f"[P142] Paths checked: {result['paths_checked']}")
    if result["status"] == STATUS_UNAVAILABLE:
        print(f"[P142] Reason: {result['reason']}")
    else:
        comp = result.get("comparison", {})
        print(f"[P142] 2.5 sMAPE_floor50 : {comp.get('model_25', {}).get('smape_floor50')}")
        print(f"[P142] 3.0 sMAPE_floor50 : {comp.get('model_30_cfg05', {}).get('smape_floor50')}")
        if "delta_smape_floor50" in comp:
            print(f"[P142] Delta           : {comp['delta_smape_floor50']}")
            print(f"[P142] Better model    : {comp.get('better_model', 'N/A')}")

    print(f"\n[P142] Outputs written to {out_dir}")
