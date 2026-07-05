"""
scripts/run_p139_2025_residual_corrected_bgew.py
================================================
P139: 2025 Residual-Corrected BGEW

Reads P138 rolling-BGEW fused day-ahead predictions (or re-derives them from
the trusted ledger + BGEW weights), attempts residual correction via
ResidualP5MAdapter, and evaluates before/after metrics with the canonical
sMAPE_floor50 formula.

Outputs
-------
.local_artifacts/p139_residual_corrected/
    bgew_raw_metrics.json              -- sMAPE/MAE/RMSE *before* correction
    bgew_residual_corrected_metrics.json -- after correction
    residual_delta_summary.json        -- delta between before/after
    period_metrics.json                -- per-period (1_8, 9_16, 17_24) breakdown

Status codes
------------
RESIDUAL_CORRECTED_IMPROVED  -- real correction applied AND improved sMAPE
RESIDUAL_NO_OP               -- no residual model found; no improvement claim
RESIDUAL_BLOCKED             -- missing inputs prevented evaluation
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from adapters.residual_p5m_adapter import ResidualP5MAdapter  # noqa: E402
from data.business_day import add_business_time_columns  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical metrics
# ---------------------------------------------------------------------------

def compute_smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float = 50.0,
) -> float:
    """Canonical sMAPE with floor=50."""
    y_true_f = np.maximum(np.asarray(y_true, dtype=float), floor)
    y_pred_f = np.maximum(np.asarray(y_pred, dtype=float), floor)
    denom = np.abs(y_true_f) + np.abs(y_pred_f)
    mask = denom > 1e-10
    if mask.sum() == 0:
        return 0.0
    return float(
        200.0 * np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]) / denom[mask])
    )


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return {sMAPE_floor50, MAE, RMSE, n}."""
    return {
        "sMAPE_floor50": round(compute_smape_floor50(y_true, y_pred), 4),
        "MAE": round(compute_mae(y_true, y_pred), 4),
        "RMSE": round(compute_rmse(y_true, y_pred), 4),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# BGEW weight computation (canonical)
# ---------------------------------------------------------------------------

def compute_bgew_weights(
    smape_values: dict[str, float],
    alpha: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.75,
) -> dict[str, float]:
    """Exponential-inverse sMAPE weighting with clip + renormalize."""
    scores = {k: np.exp(-alpha * v) for k, v in smape_values.items()}
    total = sum(scores.values())
    weights = {k: v / total for k, v in scores.items()}
    weights = {k: np.clip(v, min_weight, max_weight) for k, v in weights.items()}
    total2 = sum(weights.values())
    weights = {k: v / total2 for k, v in weights.items()}
    return weights


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_raw_data(raw_data_path: str) -> pd.DataFrame:
    """Load raw Shandong PMOS CSV (GBK) and normalise columns."""
    df = pd.read_csv(raw_data_path, encoding="gbk")
    col_map = {"\u65f6\u523b": "ds", "\u65e5\u524d\u7535\u4ef7": "dayahead_price", "\u5b9e\u65f6\u7535\u4ef7": "realtime_price"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    return df


def _load_bgew_fused(bgew_dir: str) -> Optional[pd.DataFrame]:
    """Try to load P138 rolling BGEW fused predictions.

    Looks for daily_metrics.csv inside *bgew_dir*.
    Returns None when the directory or file is missing.
    """
    if not bgew_dir or not Path(bgew_dir).is_dir():
        return None
    csv_path = Path(bgew_dir) / "daily_metrics.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    return df


def _load_trusted_ledger(ledger_path: str) -> Optional[pd.DataFrame]:
    """Load the P137 trusted 2025 ledger."""
    if not ledger_path or not Path(ledger_path).is_file():
        return None
    df = pd.read_csv(ledger_path)
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    return df


def _load_bgew_weights_json(bgew_dir: str) -> Optional[dict[str, float]]:
    """Load BGEW weights from P138 bgew_2025_metrics.json (overall averages)."""
    p = Path(bgew_dir) / "bgew_2025_metrics.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weights")


def _load_per_day_weights(bgew_dir: str) -> Optional[pd.DataFrame]:
    """Load per-day weights from P138 weights.csv.

    Returns DataFrame with columns: target_day, model_name, weight.
    """
    p = Path(bgew_dir) / "weights.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if "target_day" in df.columns:
        df["target_day"] = pd.to_datetime(df["target_day"], errors="coerce")
    return df


def _rederive_bgew_from_ledger_per_day(
    ledger: pd.DataFrame,
    per_day_weights: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Re-compute BGEW fused y_pred per day using per-day weights.

    For each target_day, pivot model predictions, apply weights, fuse.
    Then merge with actuals.
    Returns DataFrame with columns: ds, hour_business, y_pred, y_true, business_day.
    """
    results = []
    for target_day, grp_w in per_day_weights.groupby("target_day"):
        td_str = str(target_day.date()) if hasattr(target_day, 'date') else str(target_day)
        # Get ledger rows for this day
        day_mask = ledger["business_day"].astype(str) == td_str
        day_ledger = ledger[day_mask].copy()
        if day_ledger.empty:
            continue

        # Capture ds mapping (hour_business -> ds) before pivot
        ds_map = day_ledger.drop_duplicates("hour_business").set_index("hour_business")["ds"].to_dict()

        # Pivot: hour_business -> model_name -> y_pred
        pivot = day_ledger.pivot_table(
            index="hour_business",
            columns="model_name",
            values="y_pred",
            aggfunc="mean",
        )
        models_present = [m for m in grp_w["model_name"] if m in pivot.columns]
        if not models_present:
            continue
        w = np.array([grp_w.loc[grp_w["model_name"] == m, "weight"].values[0]
                       for m in models_present])
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        fused = (pivot[models_present].values * w).sum(axis=1)
        day_result = pivot.reset_index()
        day_result["y_pred"] = fused
        day_result["business_day"] = td_str
        day_result["target_day"] = td_str
        # Restore ds from mapping
        day_result["ds"] = day_result["hour_business"].map(ds_map)
        results.append(day_result)

    if not results:
        return pd.DataFrame()

    fused_df = pd.concat(results, ignore_index=True)
    # Merge with actuals on ds + hour_business
    merge_keys = ["ds", "hour_business"] if "ds" in fused_df.columns and fused_df["ds"].notna().any() else ["hour_business"]
    fused_df = fused_df.merge(
        actuals[["ds", "hour_business", "y_true"]].drop_duplicates(merge_keys),
        on=merge_keys,
        how="inner",
    )
    return fused_df


# ---------------------------------------------------------------------------
# Re-derive BGEW fused predictions from trusted ledger
# ---------------------------------------------------------------------------

def _rederive_bgew_from_ledger(
    ledger: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Re-compute BGEW fused y_pred from a multi-model trusted ledger.

    The ledger must have columns: ds, model_name, y_pred, hour_business.
    Returns a DataFrame with columns: ds, hour_business, y_pred (fused).
    """
    models = sorted(weights.keys())
    available = [m for m in models if m in ledger["model_name"].unique()]
    if len(available) < 1:
        return pd.DataFrame()

    # Pivot: one column per model
    pivot = ledger[ledger["model_name"].isin(available)].pivot_table(
        index=["ds", "hour_business"] if "hour_business" in ledger.columns else ["ds"],
        columns="model_name",
        values="y_pred",
        aggfunc="mean",
    )
    # Only keep models that are present
    present = [m for m in available if m in pivot.columns]
    if not present:
        return pd.DataFrame()

    w = np.array([weights.get(m, 0.0) for m in present])
    w = w / w.sum() if w.sum() > 0 else w
    fused = (pivot[present].values * w).sum(axis=1)

    result = pivot.reset_index()
    result["y_pred"] = fused
    return result


# ---------------------------------------------------------------------------
# Period breakdown
# ---------------------------------------------------------------------------

def _period_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hour_business: np.ndarray,
) -> dict[str, dict]:
    """Compute per-period metrics for 1_8, 9_16, 17_24."""
    out: dict[str, dict] = {}
    for lo, hi, label in [(1, 8, "1_8"), (9, 16, "9_16"), (17, 24, "17_24")]:
        mask = (hour_business >= lo) & (hour_business <= hi)
        if mask.sum() == 0:
            out[label] = {"sMAPE_floor50": None, "MAE": None, "RMSE": None, "n": 0}
        else:
            out[label] = compute_metrics(y_true[mask], y_pred[mask])
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_p139_residual_corrected_bgew(
    bgew_dir: str = ".local_artifacts/p138_rolling_bgew",
    raw_data_path: str = "data/shandong_pmos_hourly.csv",
    output_dir: str = ".local_artifacts/p139_residual_corrected",
    *,
    trusted_ledger_path: str = ".local_artifacts/p137_trusted_2025/ledger/dayahead_prediction_ledger_2025_trusted.csv",
) -> dict:
    """Run P139 residual-corrected BGEW evaluation.

    Parameters
    ----------
    bgew_dir : str
        Directory containing P138 rolling BGEW outputs.
    raw_data_path : str
        Path to raw Shandong PMOS hourly CSV (GBK).
    output_dir : str
        Where to write JSON metrics.
    trusted_ledger_path : str
        Path to P137 trusted 2025 ledger.

    Returns
    -------
    dict
        Result summary with keys: status, bgew_raw_metrics,
        bgew_residual_corrected_metrics, residual_delta_summary,
        period_metrics, reason_codes.
    """
    os.makedirs(output_dir, exist_ok=True)
    reason_codes: list[str] = []

    result: dict[str, Any] = {
        "status": "RESIDUAL_BLOCKED",
        "bgew_raw_metrics": {},
        "bgew_residual_corrected_metrics": {},
        "residual_delta_summary": {},
        "period_metrics": {},
        "reason_codes": reason_codes,
    }

    # ------------------------------------------------------------------
    # 1. Load actuals from raw data
    # ------------------------------------------------------------------
    if not os.path.isfile(raw_data_path):
        reason_codes.append("RAW_DATA_MISSING")
        _save_result(result, output_dir)
        return result

    raw = _load_raw_data(raw_data_path)
    raw = add_business_time_columns(raw, timestamp_col="ds")
    # Actuals for dayahead: dayahead_price
    actuals = raw[["ds", "business_day", "hour_business", "period", "dayahead_price"]].copy()
    actuals = actuals.rename(columns={"dayahead_price": "y_true"})
    actuals = actuals.dropna(subset=["y_true"])

    # ------------------------------------------------------------------
    # 2. Load or re-derive BGEW fused predictions
    # ------------------------------------------------------------------
    fused = _load_bgew_fused(bgew_dir)
    bgew_weights = _load_bgew_weights_json(bgew_dir)
    per_day_weights = _load_per_day_weights(bgew_dir)

    eval_df = None

    if fused is not None and "y_pred" in fused.columns and "y_true" in fused.columns:
        # P138 daily_metrics has both fused prediction and actual
        reason_codes.append("BGEW_FUSED_FROM_P138")
        eval_df = fused.copy()
        if "ds" not in eval_df.columns and "business_day" in eval_df.columns:
            reason_codes.append("NO_DS_COLUMN_IN_FUSED")
            _save_result(result, output_dir)
            return result
        eval_df = add_business_time_columns(eval_df, timestamp_col="ds")
    elif per_day_weights is not None:
        # Re-derive from trusted ledger + per-day weights
        ledger = _load_trusted_ledger(trusted_ledger_path)
        if ledger is not None:
            reason_codes.append("BGEW_REDERIVED_FROM_LEDGER_PER_DAY")
            eval_df = _rederive_bgew_from_ledger_per_day(ledger, per_day_weights, actuals)
            if eval_df.empty:
                reason_codes.append("REDERIVE_BGEW_EMPTY")
                _save_result(result, output_dir)
                return result
        else:
            reason_codes.append("NO_BGEW_DATA_AVAILABLE")
            _save_result(result, output_dir)
            return result
    elif bgew_weights is not None:
        # Fallback: use overall average weights
        ledger = _load_trusted_ledger(trusted_ledger_path)
        if ledger is not None:
            reason_codes.append("BGEW_REDERIVED_FROM_LEDGER")
            fused_derived = _rederive_bgew_from_ledger(ledger, bgew_weights)
            if fused_derived.empty:
                reason_codes.append("REDERIVE_BGEW_EMPTY")
                _save_result(result, output_dir)
                return result
            eval_df = fused_derived.merge(
                actuals[["ds", "hour_business", "y_true"]],
                on=["ds", "hour_business"],
                how="inner",
            )
        else:
            reason_codes.append("NO_BGEW_DATA_AVAILABLE")
            _save_result(result, output_dir)
            return result
    else:
        reason_codes.append("NO_BGEW_DATA_AVAILABLE")
        _save_result(result, output_dir)
        return result

    # ------------------------------------------------------------------
    # 3. Evaluate *before* residual correction
    # ------------------------------------------------------------------
    if "y_pred" not in eval_df.columns or "y_true" not in eval_df.columns:
        reason_codes.append("MISSING_PRED_OR_ACTUAL_COL")
        _save_result(result, output_dir)
        return result

    y_true_raw = eval_df["y_true"].values.astype(float)
    y_pred_raw = eval_df["y_pred"].values.astype(float)
    hour_bus = eval_df["hour_business"].values.astype(int) if "hour_business" in eval_df.columns else np.arange(1, 25)

    raw_metrics = compute_metrics(y_true_raw, y_pred_raw)
    result["bgew_raw_metrics"] = raw_metrics

    # Period breakdown (raw)
    raw_period = _period_metrics(y_true_raw, y_pred_raw, hour_bus)

    # ------------------------------------------------------------------
    # 4. Attempt residual correction
    # ------------------------------------------------------------------
    adapter = ResidualP5MAdapter(
        residual_source_repo=os.path.join(REPO_ROOT, ".local_artifacts", "source_repos", "electricity_forecast_model2.0_exp"),
        work_dir=REPO_ROOT,
        strict=False,
    )
    artifacts = adapter.find_artifacts()
    load_info = adapter.load_correction_model()

    # Build a predictions DataFrame for the adapter
    pred_df = eval_df[["y_pred"]].copy()
    correction_result = adapter.apply_correction(pred_df, task="dayahead")

    correction_applied = correction_result.get("correction_applied", False)
    adapter_status = correction_result.get("status", "RESIDUAL_P5M_NO_OP")

    if correction_applied and "y_pred_corrected" in correction_result.get("output", pd.DataFrame()).columns:
        corrected_output = correction_result["output"]
        y_pred_corrected = corrected_output["y_pred_corrected"].values.astype(float)

        # Check for NaN
        nan_mask = np.isnan(y_pred_corrected)
        if nan_mask.any():
            reason_codes.append(f"CORRECTION_NAN_COUNT_{nan_mask.sum()}")
            y_pred_corrected[nan_mask] = y_pred_raw[nan_mask]

        corrected_metrics = compute_metrics(y_true_raw, y_pred_corrected)
        result["bgew_residual_corrected_metrics"] = corrected_metrics

        # Period breakdown (corrected)
        corrected_period = _period_metrics(y_true_raw, y_pred_corrected, hour_bus)

        # Delta
        delta_smape = corrected_metrics["sMAPE_floor50"] - raw_metrics["sMAPE_floor50"]
        delta_mae = corrected_metrics["MAE"] - raw_metrics["MAE"]
        delta_rmse = corrected_metrics["RMSE"] - raw_metrics["RMSE"]

        result["residual_delta_summary"] = {
            "delta_sMAPE_floor50": round(delta_smape, 4),
            "delta_MAE": round(delta_mae, 4),
            "delta_RMSE": round(delta_rmse, 4),
            "improved": delta_smape < 0,
            "correction_applied": True,
            "adapter_status": adapter_status,
        }

        # Period deltas
        period_delta: dict[str, dict] = {}
        for period_label in ("1_8", "9_16", "17_24"):
            rp = raw_period.get(period_label, {})
            cp = corrected_period.get(period_label, {})
            if rp.get("sMAPE_floor50") is not None and cp.get("sMAPE_floor50") is not None:
                period_delta[period_label] = {
                    "raw_sMAPE": rp["sMAPE_floor50"],
                    "corrected_sMAPE": cp["sMAPE_floor50"],
                    "delta_sMAPE": round(cp["sMAPE_floor50"] - rp["sMAPE_floor50"], 4),
                }
            else:
                period_delta[period_label] = {"raw_sMAPE": None, "corrected_sMAPE": None, "delta_sMAPE": None}

        result["period_metrics"] = {
            "raw": raw_period,
            "corrected": corrected_period,
            "delta": period_delta,
        }

        if delta_smape < 0:
            result["status"] = "RESIDUAL_CORRECTED_IMPROVED"
        else:
            result["status"] = "RESIDUAL_CORRECTED_NOT_IMPROVED"
            reason_codes.append("CORRECTION_DID_NOT_IMPROVE_SMAPE")
    else:
        # No correction applied -- NO_OP
        reason_codes.append(f"ADAPTER_STATUS_{adapter_status}")
        result["bgew_residual_corrected_metrics"] = dict(raw_metrics)
        result["bgew_residual_corrected_metrics"]["note"] = "no correction applied"

        result["residual_delta_summary"] = {
            "delta_sMAPE_floor50": 0.0,
            "delta_MAE": 0.0,
            "delta_RMSE": 0.0,
            "improved": False,
            "correction_applied": False,
            "adapter_status": adapter_status,
        }

        result["period_metrics"] = {
            "raw": raw_period,
            "corrected": raw_period,
            "delta": {p: {"raw_sMAPE": raw_period[p].get("sMAPE_floor50"),
                          "corrected_sMAPE": raw_period[p].get("sMAPE_floor50"),
                          "delta_sMAPE": 0.0} for p in ("1_8", "9_16", "17_24")},
        }

        result["status"] = "RESIDUAL_NO_OP"

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    _save_result(result, output_dir)
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_result(result: dict, output_dir: str) -> None:
    """Write all JSON artefacts to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    _write_json(os.path.join(output_dir, "bgew_raw_metrics.json"),
                result.get("bgew_raw_metrics", {}))
    _write_json(os.path.join(output_dir, "bgew_residual_corrected_metrics.json"),
                result.get("bgew_residual_corrected_metrics", {}))
    _write_json(os.path.join(output_dir, "residual_delta_summary.json"),
                result.get("residual_delta_summary", {}))
    _write_json(os.path.join(output_dir, "period_metrics.json"),
                result.get("period_metrics", {}))


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="P139: 2025 Residual-Corrected BGEW")
    parser.add_argument("--bgew-dir", default=".local_artifacts/p138_rolling_bgew")
    parser.add_argument("--raw-data", default="data/shandong_pmos_hourly.csv")
    parser.add_argument("--output-dir", default=".local_artifacts/p139_residual_corrected")
    parser.add_argument("--trusted-ledger",
                        default=".local_artifacts/p137_trusted_2025/ledger/dayahead_prediction_ledger_2025_trusted.csv")
    parser.add_argument("--json", action="store_true", help="Print JSON result to stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = run_p139_residual_corrected_bgew(
        bgew_dir=args.bgew_dir,
        raw_data_path=args.raw_data,
        output_dir=args.output_dir,
        trusted_ledger_path=args.trusted_ledger,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n=== P139: 2025 Residual-Corrected BGEW ===")
        print(f"  Status: {result['status']}")
        print(f"  Raw sMAPE_floor50:    {result['bgew_raw_metrics'].get('sMAPE_floor50', 'N/A')}")
        print(f"  Corrected sMAPE:      {result['bgew_residual_corrected_metrics'].get('sMAPE_floor50', 'N/A')}")
        delta = result.get("residual_delta_summary", {})
        print(f"  Delta sMAPE:          {delta.get('delta_sMAPE_floor50', 'N/A')}")
        print(f"  Correction applied:   {delta.get('correction_applied', False)}")
        if result.get("reason_codes"):
            print(f"  Reason codes: {result['reason_codes']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
