"""
scripts/run_p94_realtime_pooled_learner.py — Run realtime 30D pooled learner.

Usage::

    python -m scripts.run_p94_realtime_pooled_learner \\
        --realtime-predictions .local_artifacts/p93/realtime_prediction_ledger.csv \\
        --realtime-actuals .local_artifacts/ledger/realtime_actual_ledger.csv \\
        --target-day 2026-07-03 \\
        --output-dir .local_artifacts/p94 \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def run_realtime_pooled_learner(
    realtime_predictions: str = "",
    realtime_actuals: str = "",
    target_day: str = "",
    alpha: float = 0.05,
    output_dir: str = ".local_artifacts/p94",
    hard_reject_bad_assist: bool = False,
) -> dict:
    """Run realtime 30D pooled learner.

    Returns dict with weights and status.
    """
    from fusion.unified_weight_learner import train_unified_weights

    os.makedirs(output_dir, exist_ok=True)

    if not realtime_predictions or not os.path.isfile(realtime_predictions):
        return {"status": "BLOCKED", "reason": f"RT predictions not found: {realtime_predictions}"}
    if not realtime_actuals or not os.path.isfile(realtime_actuals):
        return {"status": "BLOCKED", "reason": f"RT actuals not found: {realtime_actuals}"}

    rt_pred = pd.read_csv(realtime_predictions)
    rt_act = pd.read_csv(realtime_actuals)

    # Use learner_policy override for realtime only
    learner_policy = {
        "dayahead": "period_regime_bgew",
        "realtime": "pooled_30d_bgew",
    }

    result = train_unified_weights(
        realtime_predictions=rt_pred,
        realtime_actuals=rt_act,
        target_day=target_day,
        alpha=alpha,
        learner_policy=learner_policy,
        hard_reject_bad_assist=hard_reject_bad_assist,
    )

    # Save weights
    if result.get("realtime_weights") is not None:
        weights_path = os.path.join(output_dir, "realtime_pooled_weights.csv")
        result["realtime_weights"].to_csv(weights_path, index=False)
        result["weights_path"] = weights_path

    # Summary
    weights_info = ""
    if result.get("realtime_weights") is not None:
        wdf = result["realtime_weights"]
        model_weights = dict(zip(wdf["model_name"], wdf["weight"]))
        weights_info = model_weights

    return {
        "status": result["status"],
        "realtime_status": "TRAINED" if result.get("realtime_weights") is not None else "FAILED",
        "training_days": result.get("training_days", 0),
        "weights": weights_info,
        "reason_codes": result.get("reason_codes", []),
        "weights_path": result.get("weights_path", ""),
        "target_day": target_day,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P94: Run realtime 30D pooled learner")
    p.add_argument("--realtime-predictions", type=str, required=True)
    p.add_argument("--realtime-actuals", type=str, required=True)
    p.add_argument("--target-day", type=str, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output-dir", type=str, default=".local_artifacts/p94")
    p.add_argument("--hard-reject-bad-assist", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = run_realtime_pooled_learner(
        realtime_predictions=args.realtime_predictions,
        realtime_actuals=args.realtime_actuals,
        target_day=args.target_day,
        alpha=args.alpha,
        output_dir=args.output_dir,
        hard_reject_bad_assist=args.hard_reject_bad_assist,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"P94 Realtime Pooled Learner: {result['status']}")
        print(f"{'='*60}")
        print(f"  Training days: {result['training_days']}")
        print(f"  Weights: {result['weights']}")
        if result.get("reason_codes"):
            print(f"  Reasons: {result['reason_codes']}")
        print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
