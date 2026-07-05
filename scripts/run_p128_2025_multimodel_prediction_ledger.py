"""
scripts/run_p128_2025_multimodel_prediction_ledger.py — P128: Build 2025 multi-model prediction ledger.

Loads trained P31-P40 models and runs inference for 2025 full year.
Outputs a unified prediction ledger with multiple models for BGEW fusion.
"""
from __future__ import annotations
import json, logging, os, sys, time
import pandas as pd, numpy as np

logger = logging.getLogger(__name__)

def run_multimodel_inference(
    raw_data_path: str = "data/shandong_pmos_hourly.csv",
    model_dir: str = ".local_artifacts/p31_p40_multimodel_fusion/models",
    output_dir: str = ".local_artifacts/p128_2025_multimodel",
    day_start: str = "2025-01-01",
    day_end: str = "2025-12-31",
) -> dict:
    """Run multi-model inference for all available models across 2025."""
    os.makedirs(output_dir, exist_ok=True)
    result = {"status": "STARTED", "models": {}, "total_rows": 0, "model_count": 0}

    # Load raw data
    raw = pd.read_csv(raw_data_path, encoding="gbk")
    raw["ds"] = pd.to_datetime(raw["时刻"])
    from data.business_day import add_business_time_columns
    raw = add_business_time_columns(raw, timestamp_col="ds")

    # Define available models from P31-P40
    model_defs = {
        "lightgbm_cfg05_dayahead": {
            "path": os.path.join(model_dir, "cfg05_dayahead_lgbm", "cfg05_model.txt"),
            "type": "lightgbm",
            "trusted": True,
            "quarantined": False,
        },
        "catboost_spike_residual": {
            "path": os.path.join(model_dir, "catboost_spike_residual", "catboost_spike_residual.cbm"),
            "type": "catboost",
            "trusted": True,
            "quarantined": False,
        },
        "catboost_sota": {
            "path": os.path.join(model_dir, "catboost_sota", "catboost_sota_model.cbm"),
            "type": "catboost",
            "trusted": False,
            "quarantined": True,
        },
        "best_two_average": {
            "path": os.path.join(model_dir, "best_two_average", "best_two_average_trial_02.txt"),
            "type": "lightgbm",
            "trusted": False,
            "quarantined": True,
        },
        "stage3_business_fixed": {
            "path": os.path.join(model_dir, "stage3_business_fixed", "stage3_model.txt"),
            "type": "lightgbm",
            "trusted": False,
            "quarantined": True,
        },
    }

    # Load models
    loaded_models = {}
    for name, cfg in model_defs.items():
        p = cfg["path"]
        if not os.path.isfile(p):
            result["models"][name] = {"status": "FILE_MISSING", "path": p}
            continue
        try:
            if cfg["type"] == "catboost":
                from catboost import CatBoost
                m = CatBoost()
                m.load_model(p)
            else:
                import lightgbm as lgb
                m = lgb.Booster(model_file=p)
            loaded_models[name] = {"model": m, "cfg": cfg}
            result["models"][name] = {"status": "LOADED", "trusted": cfg["trusted"]}
        except Exception as e:
            result["models"][name] = {"status": f"LOAD_FAILED:{e}"}

    if not loaded_models:
        result["status"] = "NO_MODELS_LOADED"
        return result

    result["model_count"] = len(loaded_models)
    logger.info(f"Loaded {len(loaded_models)} models")

    # Generate predictions day by day for each model
    days = pd.date_range(day_start, day_end)
    all_rows = []
    day_progress = 0
    t_start = time.time()

    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        day_data = raw[raw["business_day"] == day_str].copy()
        if len(day_data) == 0:
            continue

        # Build features once per day
        from data.features.dayahead_features import build_dayahead_features
        features = None
        for try_model in ["lightgbm_cfg05_dayahead", "catboost_spike_residual", next(iter(loaded_models))]:
            try:
                features = build_dayahead_features(day_data, model_id=try_model)
                break
            except Exception:
                continue
        if features is None or len(features) == 0:
            features = day_data.copy()

        for name, lm in loaded_models.items():
            try:
                pred_raw = lm["model"].predict(features)
                if isinstance(pred_raw, np.ndarray):
                    pred_flat = pred_raw.flatten()
                else:
                    pred_flat = np.full(len(day_data), float(pred_raw))
            except Exception as e:
                logger.warning(f"Inference failed for {name} on {day_str}: {e}")
                pred_flat = np.full(len(day_data), np.nan)

            for i, row in day_data.iterrows():
                hb = row.get("hour_business", 0)
                all_rows.append({
                    "task": "dayahead",
                    "model_name": name,
                    "target_day": day_str,
                    "business_day": str(row.get("business_day", day_str)),
                    "ds": row["ds"],
                    "hour_business": int(hb) if not pd.isna(hb) else 0,
                    "period": str(row.get("period", "")),
                    "y_pred": float(pred_flat[i]),
                    "source_confidence": 1.0 if lm["cfg"]["trusted"] else 0.5,
                    "model_version": "p31_v3",
                })

        day_progress += 1
        if day_progress % 30 == 0:
            elapsed = time.time() - t_start
            rate = day_progress / elapsed
            remaining = (len(days) - day_progress) / rate if rate > 0 else 0
            logger.info(f"Day {day_str}: {day_progress}/{len(days)}, {rate:.1f} days/s, ETA {remaining/60:.0f}min")

    ledger = pd.DataFrame(all_rows)
    result["total_rows"] = len(ledger)
    result["day_count"] = ledger["target_day"].nunique() if "target_day" in ledger.columns else 0
    result["elapsed"] = round(time.time() - t_start, 2)

    # Save
    ledger_path = os.path.join(output_dir, "dayahead_prediction_ledger_2025.csv")
    ledger.to_csv(ledger_path, index=False)
    result["ledger_path"] = ledger_path

    # Summary
    for name in loaded_models:
        m = ledger[ledger["model_name"] == name]
        result["models"][name]["rows"] = len(m)
        result["models"][name]["y_pred_mean"] = round(float(m["y_pred"].mean()), 2)

    if len(loaded_models) >= 2:
        result["status"] = "MULTIMODEL_LEDGER_READY"
    else:
        result["status"] = "MULTIMODEL_LEDGER_BLOCKED_CFG05_ONLY"

    return result


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data", default="data/shandong_pmos_hourly.csv")
    p.add_argument("--model-dir", default=".local_artifacts/p31_p40_multimodel_fusion/models")
    p.add_argument("--output-dir", default=".local_artifacts/p128_2025_multimodel")
    p.add_argument("--day-start", default="2025-01-01")
    p.add_argument("--day-end", default="2025-12-31")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_multimodel_inference(
        raw_data_path=args.raw_data,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        day_start=args.day_start,
        day_end=args.day_end,
    )

    report_path = os.path.join(args.output_dir, "p128_multimodel_ledger_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        status = result["status"]
        print(f"\n=== P128: Multi-Model Ledger === Status: {status}")
        print(f"  Models loaded: {result['model_count']}")
        for name, info in result.get("models", {}).items():
            print(f"    {name}: {info.get('status','?')} (trusted={info.get('trusted','?')})")
        print(f"  Total rows: {result['total_rows']}")
        print(f"  Ledger: {result.get('ledger_path', 'N/A')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
