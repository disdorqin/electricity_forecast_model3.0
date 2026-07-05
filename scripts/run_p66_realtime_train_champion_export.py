"""
scripts/run_p66_realtime_train_champion_export.py — P66: Realtime Training/Champion/Export.

Trains realtime model if needed, selects champion, exports online pack.
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

REALTIME_DEEP_READY = "REALTIME_DEEP_READY"
REALTIME_DEEP_READY_FAST_DEV = "REALTIME_DEEP_READY_FAST_DEV"
REALTIME_DEEP_BLOCKED_NO_ARTIFACT = "REALTIME_DEEP_BLOCKED_NO_ARTIFACT"
REALTIME_DEEP_BLOCKED_TRAIN_FAILED = "REALTIME_DEEP_BLOCKED_TRAIN_FAILED"
REALTIME_DEEP_BLOCKED_LEAKAGE = "REALTIME_DEEP_BLOCKED_LEAKAGE"


def run_p66_realtime_train_champion_export(
    raw_data: str = "",
    realtime_source_repo: str = "",
    sgdfnet_root: str = "",
    start_day: str = "",
    end_day: str = "",
    work_dir: str = "",
    device: str = "cpu",
    reuse_artifacts: bool = False,
) -> dict[str, Any]:
    """Train realtime model, select champion, export online pack."""
    result: dict[str, Any] = {
        "p66_status": REALTIME_DEEP_BLOCKED_NO_ARTIFACT,
        "model_type": "unknown",
        "champion": {},
        "online_pack_path": None,
        "reason_codes": [],
    }

    os.makedirs(work_dir, exist_ok=True)

    try:
        from models.adapters.realtime_deep_adapter import RealtimeDeepAdapter
        adapter = RealtimeDeepAdapter(
            source_repo_path=realtime_source_repo,
            raw_data_path=raw_data,
            sgdfnet_root=sgdfnet_root,
            work_dir=work_dir,
        )

        # Check environment
        env = adapter.check_environment()
        result["environment"] = env

        # Train if needed
        if not reuse_artifacts:
            train_result = adapter.train_if_needed(force=True)
            result["training"] = train_result

        # Select champion
        champion = adapter.select_champion()
        result["champion"] = champion
        result["model_type"] = champion.get("model_type", "unknown")

        # Export online pack (needs da_predictions)
        # For now, create a minimal pack
        if raw_data and os.path.isfile(raw_data):
            try:
                raw_df = pd.read_csv(raw_data, encoding="gbk")
                raw_df["ds"] = pd.to_datetime(raw_df["时刻"])
                from data.business_day import add_business_time_columns
                raw_df = add_business_time_columns(raw_df, timestamp_col="ds")

                # Filter to date range
                if start_day:
                    raw_df = raw_df[raw_df["business_day"] >= start_day]
                if end_day:
                    raw_df = raw_df[raw_df["business_day"] <= end_day]

                # Create minimal da_predictions from raw data
                da_pred = pd.DataFrame({
                    "business_day": raw_df["business_day"].values,
                    "ds": raw_df["ds"].values,
                    "hour_business": raw_df["hour_business"].values,
                    "period": raw_df["period"].values,
                    "y_pred": raw_df["日前电价"].values,
                })

                export = adapter.export_online_pack(
                    da_predictions=da_pred,
                    output_dir=os.path.join(work_dir, "online_pack"),
                )
                result["online_pack_path"] = export.get("output_path")
                result["export_status"] = export.get("status")
            except Exception as e:
                result["reason_codes"].append(f"EXPORT_ERROR:{e}")

        result["p66_status"] = champion.get("status", REALTIME_DEEP_BLOCKED_NO_ARTIFACT)

    except Exception as e:
        result["p66_status"] = REALTIME_DEEP_BLOCKED_TRAIN_FAILED
        result["reason_codes"].append(f"ERROR:{e}")

    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="P66: Realtime train/champion/export")
    p.add_argument("--raw-data", type=str, default="")
    p.add_argument("--realtime-source-repo", type=str, default="")
    p.add_argument("--sgdfnet-root", type=str, default="")
    p.add_argument("--start-day", type=str, default="")
    p.add_argument("--end-day", type=str, default="")
    p.add_argument("--work-dir", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--reuse-artifacts", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = run_p66_realtime_train_champion_export(
        raw_data=args.raw_data,
        realtime_source_repo=args.realtime_source_repo,
        sgdfnet_root=args.sgdfnet_root,
        start_day=args.start_day,
        end_day=args.end_day,
        work_dir=args.work_dir or os.path.join(".local_artifacts", "realtime"),
        device=args.device,
        reuse_artifacts=args.reuse_artifacts,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"P66 Status: {result['p66_status']}")
        print(f"Model type: {result['model_type']}")
    return 0 if "READY" in result["p66_status"] else 1


if __name__ == "__main__":
    sys.exit(main())
