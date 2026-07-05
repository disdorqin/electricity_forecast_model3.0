"""
scripts/run_p38_fused_full_chain.py — P38: Full chain with fused prediction.

Runs the complete pipeline: load trained models → predict → fuse → evaluate.
Single-command end-to-end fused day-ahead prediction.

Usage::

    python -m scripts.run_p38_fused_full_chain --target-day 2026-06-30

Options::

    --target-day YYYY-MM-DD  Target day to predict (default: 2026-06-30).
    --work-dir PATH          Model artifacts dir.
    --force                  Overwrite existing outputs.
    --json                   Output JSON report.
    --strict                 Exit non-zero on failures.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p31_p40_multimodel_fusion")
_RAW_DATA_PATH = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.1/data/shandong_pmos_hourly.csv"

_MODEL_NAMES = [
    ("lightgbm_cfg05_dayahead", "cfg05_dayahead_lgbm"),
    ("best_two_average", "best_two_average"),
    ("stage3_business_fixed", "stage3_business_fixed"),
    ("catboost_sota", "catboost_sota"),
    ("catboost_spike_residual", "catboost_spike_residual"),
]


def _import_source_modules(source_repo: str):
    import importlib as _il
    if source_repo not in sys.path:
        sys.path.insert(0, source_repo)
    data_loader = _il.import_module("src.common.data_loader")
    feature_builder = _il.import_module("src.common.feature_builder_dayahead")
    return data_loader, feature_builder


def _fill_v3_columns(df):
    """Same as P31 v3 fill."""
    df = df.copy()
    for c in ["price_volatility_24h", "price_volatility_168h"]:
        if c not in df.columns:
            df[c] = df["y"].shift(1).rolling(24 if "24" in c else 168, min_periods=1).std().fillna(0)
    for c in ["renewable_penetration_rank_30d", "load_ramp_rank_30d"]:
        if c not in df.columns:
            df[c] = 0.5
    for c_src, c_dst in [("bidding_space", "bidding_space_change_24h"), ("net_load", "net_load_change_24h")]:
        if c_dst not in df.columns and c_src in df.columns:
            df[c_dst] = df[c_src].diff(24).fillna(0)
    if "renewable_change_24h" not in df.columns:
        renew = df.get("wind", 0) + df.get("solar", 0)
        df["renewable_change_24h"] = renew.diff(24).fillna(0)
    for c in ["is_spring_festival_exact", "days_to_spring_festival_exact", "days_after_spring_festival_exact"]:
        if c not in df.columns:
            src = c.replace("_exact", "_window") if "spring" in c else c.replace("_exact", "")
            df[c] = df.get(src if src in df.columns else c.replace("_exact", ""), 0)
    for c_src, c_dst in [("bidding_space", "hour_x_bidding_space"), ("net_load", "hour_x_net_load")]:
        if c_dst not in df.columns and c_src in df.columns:
            df[c_dst] = df["hour"] * df[c_src]
    for c in ["period_x_bidding_space", "period_x_renewable_penetration"]:
        if c not in df.columns:
            df[c] = 0
    for c in [
        "price_volatility_24h", "price_volatility_168h", "renewable_penetration_rank_30d",
        "load_ramp_rank_30d", "bidding_space_change_24h", "net_load_change_24h",
        "renewable_change_24h", "is_spring_festival_exact", "days_to_spring_festival_exact",
        "days_after_spring_festival_exact", "hour_x_bidding_space", "hour_x_net_load",
        "period_x_bidding_space", "period_x_renewable_penetration",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def _get_model_feature_cols(model_key: str):
    """Get feature columns for a model."""
    if model_key in ("cfg05_dayahead_lgbm",):
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        return list(CFG05_FEATURE_COLUMNS)
    if model_key in ("best_two_average", "catboost_spike_residual"):
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        return list(CFG05_FEATURE_COLUMNS)
    if model_key == "stage3_business_fixed":
        return [
            "hour", "month", "day_of_week", "is_weekend",
            "lag_price_target", "lag_price_week",
            "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
            "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
            "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
            "morning_trend", "is_info_fresh",
            "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
            "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
            "same_hour_max_7d", "same_hour_min_7d",
            "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
            "is_spring_festival_window", "days_to_spring_festival",
            "days_after_spring_festival", "is_month_start", "is_month_end",
        ]
    if model_key == "catboost_sota":
        return [
            "hour", "month", "day_of_week", "is_weekend",
            "lag_price_target", "lag_price_week",
            "load", "wind", "solar", "interconnect",
            "bidding_space", "space_ratio",
            "net_load", "solar_ratio", "net_load_sq",
            "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
            "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
        ]
    return None


def run_fused_full_chain(
    target_day: Optional[str] = None,
    work_dir: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run full chain: predict all models → fuse → evaluate."""
    target_day = target_day or "2026-06-30"
    work_dir = work_dir or _DEFAULT_WORK_DIR
    source_repo = os.path.join(".local_artifacts", "source_repos", "epf-sota-experiment")

    result: dict[str, Any] = {
        "phase": "P38",
        "target_day": target_day,
        "models": {},
        "fusion": {},
        "p38_status": "P38_NOT_STARTED",
        "reason_codes": [],
    }

    # Load data + features
    try:
        data_loader, feature_builder = _import_source_modules(source_repo)
        df = data_loader.load_data(_RAW_DATA_PATH, target="dayahead")
        df_feat = feature_builder.build_features_dayahead(df, use_extended=True)
        df_feat = _fill_v3_columns(df_feat)
        result["reason_codes"].append(f"FEATURES:{len(df_feat)}rows")
    except Exception as e:
        result["reason_codes"].append(f"DATA_FAILED:{e}")
        result["p38_status"] = "P38_DATA_FAILED"
        return result

    # Load period BGEW weights
    weights_path = os.path.join(work_dir, "period_bgew_weights.json")
    if os.path.isfile(weights_path):
        with open(weights_path) as f:
            weights = json.load(f)
        result["reason_codes"].append("WEIGHTS_LOADED")
    else:
        weights = None
        result["reason_codes"].append("WEIGHTS_MISSING_USING_EQUAL")

    # Predict each model
    all_preds = []
    for ledger_name, model_key in _MODEL_NAMES:
        model_dir = os.path.join(work_dir, "models", model_key)
        mresult = _predict_single(ledger_name, model_key, model_dir, df_feat, target_day)
        result["models"][ledger_name] = mresult
        if mresult["success"]:
            all_preds.append(mresult["df"])
            result["reason_codes"].append(f"{ledger_name}:OK({mresult['rows']}rows)")
        else:
            result["reason_codes"].append(f"{ledger_name}:FAIL:{mresult.get('error','')}")

    if len(all_preds) < 2:
        result["p38_status"] = "P38_INSUFFICIENT_MODELS"
        return result

    # Fuse predictions
    combined = pd.concat(all_preds, ignore_index=True)
    combined["period"] = combined["hour_business"].apply(
        lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
    )

    fused_rows = []
    for (bd, hb), group in combined.groupby(["business_day", "hour_business"]):
        period = group["period"].iloc[0]
        if weights and period in weights:
            w = weights[period]
        else:
            # Equal weight fallback
            n = len(group)
            w = {r["model_name"]: 1.0 / n for _, r in group.iterrows()}

        fused_pred = 0.0
        total_w = 0.0
        included = []
        for _, row in group.iterrows():
            mw = w.get(row["model_name"], 0)
            if mw > 0:
                fused_pred += mw * row["y_pred"]
                total_w += mw
                included.append(row["model_name"])

        if total_w > 0:
            fused_pred = fused_pred / total_w
        else:
            continue

        fused_rows.append({
            "target_day": target_day,
            "business_day": bd,
            "hour_business": hb,
            "period": period,
            "fused_price": round(fused_pred, 4),
            "weights_json": json.dumps(w),
            "included_models": ";".join(sorted(set(included))),
            "fusion_method": "bgew_skeleton",
        })

    fused_df = pd.DataFrame(fused_rows)
    result["fusion"]["rows"] = len(fused_df)

    # Save
    fusion_path = os.path.join(work_dir, "fused_full_chain_output.csv")
    try:
        fused_df.to_csv(fusion_path, index=False)
        result["fusion"]["path"] = fusion_path
        result["reason_codes"].append(f"FUSION_SAVED:{fusion_path}({len(fused_df)}rows)")
    except Exception as e:
        result["reason_codes"].append(f"FUSION_SAVE_FAILED:{e}")

    # Status
    if len(fused_df) >= 24:
        result["p38_status"] = "P38_FULL_CHAIN_COMPLETE"
    elif len(fused_df) > 0:
        result["p38_status"] = "P38_FULL_CHAIN_PARTIAL"
    else:
        result["p38_status"] = "P38_FULL_CHAIN_FAILED"

    return result


def _predict_single(
    ledger_name: str, model_key: str, model_dir: str,
    df_feat: pd.DataFrame, target_day: str,
) -> dict[str, Any]:
    """Run prediction for a single model."""
    result: dict[str, Any] = {"success": False, "rows": 0, "error": None, "df": None}

    if not os.path.isdir(model_dir):
        result["error"] = f"model_dir_missing:{model_dir}"
        return result

    try:
        # Restrict features
        feat_cols = _get_model_feature_cols(model_key)
        if feat_cols:
            keep = ["ds", "y"] + [c for c in feat_cols if c in df_feat.columns]
            df_model = df_feat[keep].copy()
        else:
            df_model = df_feat

        if model_key == "cfg05_dayahead_lgbm":
            from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter
            adapter = CFG05DayaheadAdapter()
            adapter.load()
            adapter._load_model(model_dir)
        else:
            from models.adapters.multimodel_pool import create_adapter
            adapter = create_adapter(model_key)
            adapter.load()
            adapter._load_artifacts(model_dir)

        pred_df = adapter.predict(df=df_model, target_date=target_day, model_dir=model_dir)
        if len(pred_df) == 0:
            result["error"] = "empty_prediction"
            return result

        # Replace model_name in output with ledger name
        pred_df["model_name"] = ledger_name
        result["success"] = True
        result["rows"] = len(pred_df)
        result["df"] = pred_df
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 60)
    print("P38 — Full Chain Fused Prediction")
    print("=" * 60)
    print(f"  Target day:       {result['target_day']}")
    print()
    for model_name, mres in result.get("models", {}).items():
        status = "OK" if mres.get("success") else f"FAIL:{mres.get('error','')}"
        rows = mres.get("rows", 0)
        print(f"  {model_name:<30} {status:<30} rows={rows}")
    print()
    f = result.get("fusion", {})
    print(f"  Fusion rows:      {f.get('rows', 0)}")
    print(f"  Fusion path:      {f.get('path', 'N/A')}")
    print(f"  Status:           {result['p38_status']}")
    print()
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P38: Full chain fused prediction.")
    parser.add_argument("--target-day", type=str, default="2026-06-30")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_fused_full_chain(
        target_day=args.target_day, work_dir=args.work_dir, force=args.force,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)
    if args.strict and "COMPLETE" not in result["p38_status"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
