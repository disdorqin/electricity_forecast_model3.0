"""
scripts/run_p137_trusted_2025_prediction_ledger_v2.py — P137: Trusted 2025 Prediction Ledger V2.

Merges cfg05 and catboost_spike predictions into a single trusted prediction ledger
for the full year 2025.  The ledger is prediction-only (y_true is forbidden).

Inputs:
  - cfg05 predictions:       .local_artifacts/p2025_full/dayahead/all_predictions.csv
  - catboost_spike predictions: .local_artifacts/p136_catboost_spike_2025/catboost_spike_2025_predictions.csv

Outputs:
  - .local_artifacts/p137_trusted_2025/ledger/dayahead_prediction_ledger_2025_trusted.csv
  - .local_artifacts/p137_trusted_2025/ledger/manifest.json

Status:
  TRUSTED_LEDGER_V2_READY              — ledger built successfully with >= 2 models
  TRUSTED_LEDGER_BLOCKED_SINGLE_MODEL  — fewer than 2 models available
  TRUSTED_LEDGER_BLOCKED_NAN_YPRED     — NaN detected in y_pred
  TRUSTED_LEDGER_BLOCKED_MISSING_KEYS  — business_day / hour_business missing
  TRUSTED_LEDGER_BLOCKED_YTRUE_LEAK    — y_true or forbidden column detected
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Constants ─────────────────────────────────────────────────────────
LEDGER_COLUMNS: List[str] = [
    "task", "model_name", "business_day", "ds", "hour_business",
    "period", "y_pred", "source_confidence", "model_version",
]

FORBIDDEN_COLUMNS: List[str] = [
    "y_true", "residual", "error", "abs_error", "future_y",
    "target_actual", "oracle",
]

DEFAULT_CFG05_PATH = os.path.join(
    REPO_ROOT, ".local_artifacts", "p2025_full", "dayahead", "all_predictions.csv"
)
DEFAULT_CATBOOST_SPIKE_PATH = os.path.join(
    REPO_ROOT, ".local_artifacts", "p136_catboost_spike_2025",
    "catboost_spike_2025_predictions.csv",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    REPO_ROOT, ".local_artifacts", "p137_trusted_2025", "ledger"
)


# ── Helpers ───────────────────────────────────────────────────────────


def _derive_target_day(ds_series: pd.Series) -> pd.Series:
    """Derive target_day from ds using the business-day convention.

    For dayahead: the prediction made at wall-clock timestamp *ds* is for
    the *next* calendar day (the day-ahead market clears D-1 for D).

        target_day = date(ds) + 1 day

    This matches the convention that business_day D hour 24 corresponds to
    wall-clock D+1 00:00, and the prediction for business_day D covers
    hours 1..24 of day D.

    Handles the edge case where ds contains hour=24 (e.g. "2025-01-01 24:00:00")
    which is not a standard datetime — it represents the next day's midnight.
    """
    ds_str = ds_series.astype(str)

    # Detect rows with hour=24 (e.g. "2025-01-01 24:00:00")
    is_hour24 = ds_str.str.contains(r" 24:", regex=True, na=False)

    # Replace " 24:" with " 00:" so pd.to_datetime can parse them
    ds_fixed = ds_str.str.replace(r" 24:", " 00:", regex=True)
    ds_dt = pd.to_datetime(ds_fixed, errors="coerce")

    # For hour-24 rows, add 1 day (since 24:00 = next day 00:00)
    ds_dt = ds_dt.where(~is_hour24, ds_dt + pd.Timedelta(days=1))

    return (ds_dt.dt.normalize() + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")


def _read_cfg05(path: str) -> pd.DataFrame:
    """Read cfg05 predictions and strip y_true (forbidden in prediction ledger)."""
    df = pd.read_csv(path)
    logger.info("cfg05 predictions: %d rows, columns=%s", len(df), df.columns.tolist())

    # Strip y_true and any forbidden columns
    for col in FORBIDDEN_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info("Stripped forbidden column: %s", col)

    return df


def _read_catboost_spike(path: str) -> pd.DataFrame:
    """Read catboost_spike predictions.  Derive target_day if missing.

    For dayahead, target_day == business_day (the prediction for business_day D
    covers hours 1..24 of day D).
    """
    df = pd.read_csv(path)
    logger.info(
        "catboost_spike predictions: %d rows, columns=%s", len(df), df.columns.tolist()
    )

    # Derive target_day from business_day (dayahead convention: target_day = business_day)
    if "target_day" not in df.columns and "business_day" in df.columns:
        df["target_day"] = pd.to_datetime(df["business_day"]).dt.strftime("%Y-%m-%d")
        logger.info("Derived target_day from business_day for catboost_spike predictions")

    return df


def _standardize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Keep only the canonical ledger columns that exist in *df*.

    Missing optional columns are filled with safe defaults.
    """
    out = pd.DataFrame()
    for col in LEDGER_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
        elif col == "source_confidence":
            out[col] = 1.0
        elif col == "model_version":
            out[col] = "unknown"
        else:
            logger.warning("[%s] Missing required column: %s", label, col)
            out[col] = np.nan

    # Ensure types
    out["business_day"] = pd.to_datetime(out["business_day"]).dt.strftime("%Y-%m-%d")
    out["hour_business"] = out["hour_business"].astype(int)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    return out


# ── Safety checks ─────────────────────────────────────────────────────


def _safety_checks(ledger: pd.DataFrame) -> tuple[str, list[str]]:
    """Run safety checks on the merged ledger.

    Returns (status, reason_codes).  status is either
    ``TRUSTED_LEDGER_V2_READY`` or a ``TRUSTED_LEDGER_BLOCKED_*`` code.
    """
    reason_codes: List[str] = []

    # 1. model_count >= 2
    models = ledger["model_name"].unique().tolist() if "model_name" in ledger.columns else []
    if len(models) < 2:
        reason_codes.append(f"SINGLE_MODEL_COUNT:{len(models)}")
        return "TRUSTED_LEDGER_BLOCKED_SINGLE_MODEL", reason_codes

    # 2. No y_true or forbidden columns
    for col in FORBIDDEN_COLUMNS:
        if col in ledger.columns:
            reason_codes.append(f"FORBIDDEN_COLUMN_PRESENT:{col}")
            return "TRUSTED_LEDGER_BLOCKED_YTRUE_LEAK", reason_codes

    # 3. No NaN in y_pred
    nan_count = int(ledger["y_pred"].isna().sum())
    if nan_count > 0:
        reason_codes.append(f"NAN_IN_YPRED:{nan_count}")
        return "TRUSTED_LEDGER_BLOCKED_NAN_YPRED", reason_codes

    # 4. business_day / hour_business present and valid
    for key in ("business_day", "hour_business"):
        if key not in ledger.columns:
            reason_codes.append(f"MISSING_KEY_COLUMN:{key}")
            return "TRUSTED_LEDGER_BLOCKED_MISSING_KEYS", reason_codes

    hb = ledger["hour_business"]
    if hb.min() < 1 or hb.max() > 24:
        reason_codes.append(f"HOUR_BUSINESS_OUT_OF_RANGE:{hb.min()}-{hb.max()}")
        return "TRUSTED_LEDGER_BLOCKED_MISSING_KEYS", reason_codes

    reason_codes.append("ALL_SAFETY_CHECKS_PASSED")
    return "TRUSTED_LEDGER_V2_READY", reason_codes


# ── Main entry point ──────────────────────────────────────────────────


def run_p137_trusted_ledger(
    cfg05_path: str = DEFAULT_CFG05_PATH,
    catboost_spike_path: str = DEFAULT_CATBOOST_SPIKE_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Build the Trusted 2025 Prediction Ledger V2.

    Parameters
    ----------
    cfg05_path : str
        Path to cfg05 all_predictions.csv.
    catboost_spike_path : str
        Path to catboost_spike_2025_predictions.csv.
    output_dir : str
        Directory for output ledger and manifest.

    Returns
    -------
    dict
        Manifest dictionary with status, model_count, total_rows, etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest: dict[str, Any] = {
        "phase": "P137",
        "title": "Trusted 2025 Prediction Ledger V2",
        "cfg05_path": cfg05_path,
        "catboost_spike_path": catboost_spike_path,
        "status": "STARTED",
        "reason_codes": [],
    }

    # ── Read inputs ───────────────────────────────────────────────────
    if not os.path.isfile(cfg05_path):
        manifest["status"] = "TRUSTED_LEDGER_BLOCKED_CFG05_MISSING"
        manifest["reason_codes"].append("CFG05_FILE_NOT_FOUND")
        _save_manifest(manifest, output_dir)
        return manifest

    if not os.path.isfile(catboost_spike_path):
        manifest["status"] = "TRUSTED_LEDGER_BLOCKED_CATBOOST_MISSING"
        manifest["reason_codes"].append("CATBOOST_SPIKE_FILE_NOT_FOUND")
        _save_manifest(manifest, output_dir)
        return manifest

    cfg05 = _read_cfg05(cfg05_path)
    catboost = _read_catboost_spike(catboost_spike_path)

    # ── Standardize ───────────────────────────────────────────────────
    cfg05_std = _standardize(cfg05, "cfg05")
    catboost_std = _standardize(catboost, "catboost_spike")

    # ── Merge ─────────────────────────────────────────────────────────
    ledger = pd.concat([cfg05_std, catboost_std], ignore_index=True)
    ledger = ledger.sort_values(
        ["business_day", "hour_business", "model_name"]
    ).reset_index(drop=True)

    manifest["total_rows"] = len(ledger)
    manifest["model_count"] = int(ledger["model_name"].nunique())
    manifest["models"] = ledger["model_name"].unique().tolist()
    manifest["rows_per_model"] = {
        m: int(len(ledger[ledger["model_name"] == m]))
        for m in manifest["models"]
    }

    if "business_day" in ledger.columns:
        manifest["date_range"] = {
            "start": str(ledger["business_day"].min()),
            "end": str(ledger["business_day"].max()),
        }

    # ── Safety checks ─────────────────────────────────────────────────
    status, reason_codes = _safety_checks(ledger)
    manifest["status"] = status
    manifest["reason_codes"] = reason_codes

    # ── Save outputs ──────────────────────────────────────────────────
    if status == "TRUSTED_LEDGER_V2_READY":
        ledger_path = os.path.join(output_dir, "dayahead_prediction_ledger_2025_trusted.csv")
        ledger.to_csv(ledger_path, index=False)
        manifest["ledger_path"] = ledger_path
        manifest["ledger_columns"] = ledger.columns.tolist()
        logger.info("Ledger saved: %s (%d rows)", ledger_path, len(ledger))
    else:
        logger.warning("Ledger BLOCKED: %s — %s", status, reason_codes)

    _save_manifest(manifest, output_dir)
    return manifest


def _save_manifest(manifest: dict, output_dir: str) -> None:
    """Write manifest JSON."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    manifest["manifest_path"] = manifest_path


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="P137: Trusted 2025 Prediction Ledger V2")
    parser.add_argument("--cfg05", default=DEFAULT_CFG05_PATH)
    parser.add_argument("--catboost-spike", default=DEFAULT_CATBOOST_SPIKE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_p137_trusted_ledger(
        cfg05_path=args.cfg05,
        catboost_spike_path=args.catboost_spike,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n=== P137: Trusted 2025 Prediction Ledger V2 ===")
        print(f"  Status: {result['status']}")
        print(f"  Models: {result.get('models', [])}")
        print(f"  Total rows: {result.get('total_rows', 0)}")
        print(f"  Date range: {result.get('date_range', {})}")
        print(f"  Reason codes: {result.get('reason_codes', [])}")
        if "ledger_path" in result:
            print(f"  Ledger: {result['ledger_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
