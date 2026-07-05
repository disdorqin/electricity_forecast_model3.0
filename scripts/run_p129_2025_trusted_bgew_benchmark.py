"""
scripts/run_p129_2025_trusted_bgew_benchmark.py — P129: Run 2025 BGEW fusion benchmark.

Reads multi-model prediction ledger and computes cfg05-only, catboost_spike_residual,
and trusted BGEW fusion metrics for the full 2025 year.
"""
from __future__ import annotations
import json, logging, os, sys
import pandas as pd, numpy as np

logger = logging.getLogger(__name__)

def run_2025_bgew_benchmark(
    ledger_path: str = "",
    actual_path: str = "",
    output_dir: str = ".local_artifacts/p129_2025_bgew",
) -> dict:
    """Run full 2025 BGEW fusion benchmark."""
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "status": "STARTED",
        "cfg05_only": {},
        "catboost_spike": {},
        "trusted_bgew": {},
        "daily_comparison": [],
    }

    # Load ledger and actuals
    if not ledger_path or not os.path.isfile(ledger_path):
        result["status"] = "LEDGER_MISSING"
        return result

    ledger = pd.read_csv(ledger_path)
    if "ds" in ledger.columns:
        ledger["ds"] = pd.to_datetime(ledger["ds"])

    # Load actuals (either from path or from raw data)
    actuals = None
    if actual_path and os.path.isfile(actual_path):
        actuals = pd.read_csv(actual_path)
    else:
        raw = pd.read_csv("data/shandong_pmos_hourly.csv", encoding="gbk")
        raw["ds"] = pd.to_datetime(raw["时刻"])
        actuals = raw[["ds", "日前电价"]].rename(columns={"日前电价": "y_true"})

    if "ds" in actuals.columns:
        actuals["ds"] = pd.to_datetime(actuals["ds"])

    # Get unique models
    models = ledger["model_name"].unique().tolist() if "model_name" in ledger.columns else []
    result["model_count"] = len(models)
    result["models"] = models

    if len(models) < 1:
        result["status"] = "NO_MODELS"
        return result

    # Compute per-model metrics
    for model in models:
        m_data = ledger[ledger["model_name"] == model].copy()
        merged = pd.merge(m_data, actuals, on="ds", how="inner")
        if len(merged) == 0:
            result[model] = {"rows": 0}
            continue
        y_true, y_pred = merged["y_true"].values, merged["y_pred"].values
        denom = np.abs(y_true) + np.abs(y_pred)
        denom = np.where(denom < 1e-8, 1e-8, denom)
        smape = np.minimum(2 * np.abs(y_true - y_pred) / denom, 1.0)
        result[model] = {
            "sMAPE_floor50": round(float(np.mean(smape) * 100), 4),
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred)**2))), 4),
            "rows": len(merged),
            "mean_pred": round(float(y_pred.mean()), 2),
            "mean_actual": round(float(y_true.mean()), 2),
        }

    # cfg05-only
    if "lightgbm_cfg05_dayahead" in result:
        result["cfg05_only"] = dict(result["lightgbm_cfg05_dayahead"])
        result["cfg05_only"]["model"] = "lightgbm_cfg05_dayahead"

    # catboost_spike_residual
    if "catboost_spike_residual" in result:
        result["catboost_spike"] = dict(result["catboost_spike_residual"])
        result["catboost_spike"]["model"] = "catboost_spike_residual"

    # Trusted BGEW fusion (cfg05 + catboost_spike_residual)
    trusted = ["lightgbm_cfg05_dayahead", "catboost_spike_residual"]
    available_trusted = [m for m in trusted if m in models and m in result and result[m].get("rows", 0) > 0]
    result["trusted_models_available"] = available_trusted

    if len(available_trusted) >= 2:
        # Simple BGEW: learn weights from the full year
        merged_all = pd.DataFrame()
        for model in available_trusted:
            m_data = ledger[ledger["model_name"] == model].copy()
            m_merged = pd.merge(m_data, actuals, on="ds", how="inner")
            if len(m_merged) > 0:
                merged_all = pd.concat([merged_all, m_merged])

        # Compute model-level sMAPE for weight calculation
        smapes = {}
        for model in available_trusted:
            m = merged_all[merged_all["model_name"] == model]
            yt, yp = m["y_true"].values, m["y_pred"].values
            denom = np.abs(yt) + np.abs(yp)
            denom = np.where(denom < 1e-8, 1e-8, denom)
            s = np.minimum(2 * np.abs(yt - yp) / denom, 1.0)
            smapes[model] = float(np.mean(s) * 100)

        # BGEW weights
        alpha = 0.05
        scores = {m: np.exp(-alpha * s) for m, s in smapes.items()}
        total = sum(scores.values())
        weights = {m: s / total for m, s in scores.items()}

        # Compute fused prediction per day
        daily_results = []
        days = sorted(merged_all["target_day"].unique()) if "target_day" in merged_all.columns else []

        for day in days:
            day_preds = {}
            for model in available_trusted:
                m = merged_all[(merged_all["model_name"] == model) & (merged_all["target_day"] == day)]
                if len(m) > 0:
                    day_preds[model] = m["y_pred"].values
            if len(day_preds) == len(available_trusted):
                fused = sum(weights[m] * day_preds[m] for m in available_trusted)
                day_actual = merged_all[(merged_all["model_name"] == available_trusted[0]) & (merged_all["target_day"] == day)]["y_true"].values
                daily_results.append({"day": day, "fused": fused, "actual": day_actual})

        if daily_results:
            all_fused = np.concatenate([d["fused"] for d in daily_results])
            all_actual = np.concatenate([d["actual"] for d in daily_results])
            denom = np.abs(all_actual) + np.abs(all_fused)
            denom = np.where(denom < 1e-8, 1e-8, denom)
            smape_fused = np.minimum(2 * np.abs(all_actual - all_fused) / denom, 1.0)
            result["trusted_bgew"] = {
                "sMAPE_floor50": round(float(np.mean(smape_fused) * 100), 4),
                "MAE": round(float(np.mean(np.abs(all_actual - all_fused))), 4),
                "RMSE": round(float(np.sqrt(np.mean((all_actual - all_fused)**2))), 4),
                "rows": len(all_fused),
                "weights": {m: round(w, 4) for m, w in weights.items()},
                "models": available_trusted,
                "daily_count": len(daily_results),
            }
        result["status"] = "TRUSTED_BGEW_BENCHMARK_COMPLETE"
    else:
        result["status"] = "TRUSTED_BGEW_BLOCKED_INSUFFICIENT_MODELS"
        result["trusted_bgew"] = {"note": "Need at least 2 trusted models", "available": available_trusted}

    # Save metrics
    metrics = {
        "cfg05_only": result.get("cfg05_only", {}),
        "catboost_spike": result.get("catboost_spike", {}),
        "trusted_bgew": result.get("trusted_bgew", {}),
        "model_count": result["model_count"],
        "models": result["models"],
        "status": result["status"],
    }
    metrics_path = os.path.join(output_dir, "2025_trusted_bgew_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return result

def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", default=".local_artifacts/p128_2025_multimodel/dayahead_prediction_ledger_2025.csv")
    p.add_argument("--actuals", default="")
    p.add_argument("--output-dir", default=".local_artifacts/p129_2025_bgew")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_2025_bgew_benchmark(
        ledger_path=args.ledger,
        actual_path=args.actuals,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n=== P129: 2025 BGEW Benchmark === Status: {result['status']}")
        print(f"  Models: {result.get('models', [])}")
        print(f"  cfg05-only: {result.get('cfg05_only', {}).get('sMAPE_floor50', 'N/A')}")
        print(f"  catboost_spike: {result.get('catboost_spike', {}).get('sMAPE_floor50', 'N/A')}")
        print(f"  trusted_bgew: {result.get('trusted_bgew', {}).get('sMAPE_floor50', 'N/A')}")
        if "weights" in result.get("trusted_bgew", {}):
            print(f"  weights: {result['trusted_bgew']['weights']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
