"""
scripts/run_p121_reproduce_2025_metrics.py — P121: Reproduce 2025 annual evaluation metrics.

Audits the actual prediction ledger to understand exactly which model produced
the 20.22% / 33.03% results and whether BGEW/residual/classifier were used.

Output::
    p121_2025_metrics.json — structured metrics report
    p121_2025_eval_join.csv — merged pred + actual for manual inspection
"""
from __future__ import annotations
import json, logging, os, sys
import pandas as pd, numpy as np

logger = logging.getLogger(__name__)

def reproduce_2025_metrics(
    pred_ledger_path: str = "",
    actual_ledger_path: str = "",
    raw_data_path: str = "",
    target_start: str = "2025-01-01",
    target_end: str = "2025-12-31",
    work_dir: str = ".local_artifacts/p121_2025_reproduce",
) -> dict:
    """Reproduce and audit 2025 evaluation metrics."""
    os.makedirs(work_dir, exist_ok=True)
    result: dict = {
        "audit_status": "STARTED",
        "dayahead": {}, "realtime": {},
        "root_cause": [],
        "discrepancies": [],
    }

    # Load prediction ledger
    if pred_ledger_path and os.path.isfile(pred_ledger_path):
        pred = pd.read_csv(pred_ledger_path)
        if "ds" in pred.columns:
            pred["ds"] = pd.to_datetime(pred["ds"])
        result["prediction_ledger"] = {
            "path": pred_ledger_path, "rows": len(pred),
            "columns": list(pred.columns),
            "models_used": list(pred["model_name"].unique()) if "model_name" in pred.columns else [],
        }
    else:
        result["prediction_ledger"] = {"status": "MISSING"}

    # Load actual ledger
    if actual_ledger_path and os.path.isfile(actual_ledger_path):
        actual = pd.read_csv(actual_ledger_path)
    else:
        actual = None

    # Load raw data for ground truth
    if raw_data_path and os.path.isfile(raw_data_path):
        raw = pd.read_csv(raw_data_path, encoding="gbk")
        raw["ds"] = pd.to_datetime(raw["时刻"])
        raw_2025 = raw[raw["ds"].dt.year == 2025].copy()
    else:
        raw = None
        raw_2025 = None

    # ── Day-ahead audit ──
    if pred_ledger_path and os.path.isfile(pred_ledger_path):
        da_pred = pred[pred["task"] == "dayahead"] if "task" in pred.columns else pred
        result["dayahead"]["pred_rows"] = len(da_pred)
        result["dayahead"]["pred_columns"] = list(da_pred.columns)
        result["dayahead"]["models"] = list(da_pred["model_name"].unique()) if "model_name" in da_pred.columns else []
        result["dayahead"]["dayahead_model_or_fusion"] = list(da_pred["model_name"].unique()) if "model_name" in da_pred.columns else []

        if "y_pred" in da_pred.columns and raw_2025 is not None:
            merged = pd.merge(
                da_pred, raw_2025[["ds", "日前电价", "实时电价"]],
                on="ds", how="inner",
            )
            y_true = merged["日前电价"].values
            y_pred = merged["y_pred"].values
            denom = np.abs(y_true) + np.abs(y_pred)
            denom = np.where(denom < 1e-8, 1e-8, denom)
            smape = np.minimum(2 * np.abs(y_true - y_pred) / denom, 1.0)
            result["dayahead"]["sMAPE_floor50"] = round(float(smape.mean() * 100), 4)
            result["dayahead"]["MAE"] = round(float(np.abs(y_true - y_pred).mean()), 4)
            result["dayahead"]["RMSE"] = round(float(np.sqrt(np.mean((y_true - y_pred)**2))), 4)
            result["dayahead"]["mean_pred"] = round(float(y_pred.mean()), 2)
            result["dayahead"]["mean_actual"] = round(float(y_true.mean()), 2)
            result["dayahead"]["actual_source"] = "日前电价"
            result["dayahead"]["rows_joined"] = len(merged)
        elif "y_pred" in da_pred.columns:
            result["dayahead"]["y_pred_mean"] = float(da_pred["y_pred"].mean())

    # ── Realtime audit ──
    if pred_ledger_path and os.path.isfile(pred_ledger_path):
        rt_pred = pred[pred["task"] == "realtime"] if "task" in pred.columns else pred
        result["realtime"]["pred_rows"] = len(rt_pred)
        result["realtime"]["models"] = list(rt_pred["model_name"].unique()) if "model_name" in rt_pred.columns else []

        if "y_pred" in rt_pred.columns and raw_2025 is not None:
            merged_rt = pd.merge(
                rt_pred, raw_2025[["ds", "日前电价", "实时电价"]],
                on="ds", how="inner",
            )
            # Check if actual source is correct
            result["realtime"]["check_actual_col"] = "实时电价" if "实时电价" in merged_rt.columns else "MISSING"
            y_true_rt = merged_rt["实时电价"].values
            y_pred_rt = merged_rt["y_pred"].values
            denom = np.abs(y_true_rt) + np.abs(y_pred_rt)
            denom = np.where(denom < 1e-8, 1e-8, denom)
            smape_rt = np.minimum(2 * np.abs(y_true_rt - y_pred_rt) / denom, 1.0)
            result["realtime"]["sMAPE_floor50"] = round(float(smape_rt.mean() * 100), 4)
            result["realtime"]["MAE"] = round(float(np.abs(y_true_rt - y_pred_rt).mean()), 4)
            result["realtime"]["RMSE"] = round(float(np.sqrt(np.mean((y_true_rt - y_pred_rt)**2))), 4)
            result["realtime"]["mean_pred"] = round(float(y_pred_rt.mean()), 2)
            result["realtime"]["mean_actual"] = round(float(y_true_rt.mean()), 2)
            result["realtime"]["actual_source"] = "实时电价"
            result["realtime"]["rows_joined"] = len(merged_rt)

            # Check if rt == da
            result["realtime"]["rt_equals_da"] = bool(np.allclose(y_pred_rt, merged_rt["日前电价"].values, rtol=1e-5))

    # ── Root cause analysis ──
    da_models = result["dayahead"].get("models", [])
    if len(da_models) <= 1:
        result["root_cause"].append("MAIN_PATH_USES_CFG05_ONLY: Step 4 calls P16 (cfg05 walkforward), not P31-P40 multi-model pool")
    if "catboost" not in str(da_models).lower():
        result["root_cause"].append("CATBOOST_SPIKE_RESIDUAL_NOT_IN_MAIN_PATH: CatBoost model not loaded in main pipeline")
    if "BGEW" not in str(result.get("dayahead", {}).get("dayahead_model_or_fusion", "")):
        result["root_cause"].append("BGEW_FUSION_NOT_APPLIED: No fusion weights found in prediction ledger (single model)")

    result["audit_status"] = "COMPLETE"
    return result


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pred-ledger", default=".local_artifacts/p2025_full/ledger/dayahead_prediction_ledger.csv")
    p.add_argument("--actual-ledger", default="")
    p.add_argument("--raw-data", default="data/shandong_pmos_hourly.csv")
    p.add_argument("--output-dir", default=".local_artifacts/p121_2025_reproduce")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = reproduce_2025_metrics(
        pred_ledger_path=args.pred_ledger,
        actual_ledger_path=args.actual_ledger,
        raw_data_path=args.raw_data,
        work_dir=args.output_dir,
    )

    out_path = os.path.join(args.output_dir, "p121_2025_metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n=== P121: 2025 Metrics Reproduce ===\n{json.dumps(result, indent=2, default=str)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
