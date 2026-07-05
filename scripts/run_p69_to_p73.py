"""
scripts/run_p69_residual_correction_full_chain.py — P69: Residual Correction.
scripts/run_p70_unified_weight_learner.py — P70: Unified Weight Learner.
scripts/run_p71_unified_fusion_engine.py — P71: Unified Fusion Engine.
scripts/run_p72_final_classifier_engine.py — P72: Classifier Engine.
scripts/run_p73_final_output_builder.py — P73: Final Output Builder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


# ── P69 ───────────────────────────────────────────────────────────────

def run_p69_residual_correction(
    dayahead_path: str = "",
    realtime_path: str = "",
    actual_path: str = "",
    work_dir: str = "",
) -> dict[str, Any]:
    """Run residual correction for both tasks."""
    import pandas as pd
    from residuals.residual_correction_engine import run_full_chain_residual_correction

    da_pred = pd.read_csv(dayahead_path) if dayahead_path and os.path.isfile(dayahead_path) else None
    rt_pred = pd.read_csv(realtime_path) if realtime_path and os.path.isfile(realtime_path) else None

    return run_full_chain_residual_correction(
        dayahead_predictions=da_pred,
        realtime_predictions=rt_pred,
        actual_ledger_path=actual_path,
        work_dir=work_dir,
    )


# ── P70 ───────────────────────────────────────────────────────────────

def run_p70_unified_weight_learner(
    dayahead_pred_path: str = "",
    realtime_pred_path: str = "",
    dayahead_actual_path: str = "",
    realtime_actual_path: str = "",
    target_day: str = "",
) -> dict[str, Any]:
    """Run unified weight learner."""
    import pandas as pd
    from fusion.unified_weight_learner import train_unified_weights

    da_pred = pd.read_csv(dayahead_pred_path) if dayahead_pred_path and os.path.isfile(dayahead_pred_path) else None
    rt_pred = pd.read_csv(realtime_pred_path) if realtime_pred_path and os.path.isfile(realtime_pred_path) else None
    da_actual = pd.read_csv(dayahead_actual_path) if dayahead_actual_path and os.path.isfile(dayahead_actual_path) else None
    rt_actual = pd.read_csv(realtime_actual_path) if realtime_actual_path and os.path.isfile(realtime_actual_path) else None

    return train_unified_weights(
        dayahead_predictions=da_pred,
        realtime_predictions=rt_pred,
        dayahead_actuals=da_actual,
        realtime_actuals=rt_actual,
        target_day=target_day,
    )


# ── P71 ───────────────────────────────────────────────────────────────

def run_p71_unified_fusion(
    dayahead_pred_path: str = "",
    realtime_pred_path: str = "",
    target_day: str = "",
) -> dict[str, Any]:
    """Run unified fusion engine."""
    import pandas as pd
    from fusion.unified_fusion_engine import run_unified_fusion

    da_pred = pd.read_csv(dayahead_pred_path) if dayahead_pred_path and os.path.isfile(dayahead_pred_path) else None
    rt_pred = pd.read_csv(realtime_pred_path) if realtime_pred_path and os.path.isfile(realtime_pred_path) else None

    return run_unified_fusion(
        dayahead_predictions=da_pred,
        realtime_predictions=rt_pred,
        target_day=target_day,
    )


# ── P72 ───────────────────────────────────────────────────────────────

def run_p72_classifier(
    dayahead_fused_path: str = "",
    realtime_fused_path: str = "",
    work_dir: str = "",
) -> dict[str, Any]:
    """Run final classifier."""
    import pandas as pd
    from classifiers.final_classifier_engine import run_final_classifier

    da_fused = pd.read_csv(dayahead_fused_path) if dayahead_fused_path and os.path.isfile(dayahead_fused_path) else None
    rt_fused = pd.read_csv(realtime_fused_path) if realtime_fused_path and os.path.isfile(realtime_fused_path) else None

    return run_final_classifier(
        dayahead_fused=da_fused,
        realtime_fused=rt_fused,
        work_dir=work_dir,
    )


# ── P73 ───────────────────────────────────────────────────────────────

def run_p73_final_output(
    dayahead_fused_path: str = "",
    realtime_fused_path: str = "",
    target_day: str = "",
    work_dir: str = "",
) -> dict[str, Any]:
    """Build final output."""
    import pandas as pd
    from delivery.final_output_builder import build_final_output, save_final_output

    da_fused = pd.read_csv(dayahead_fused_path) if dayahead_fused_path and os.path.isfile(dayahead_fused_path) else None
    rt_fused = pd.read_csv(realtime_fused_path) if realtime_fused_path and os.path.isfile(realtime_fused_path) else None

    result = build_final_output(
        dayahead_fused=da_fused,
        realtime_fused=rt_fused,
        target_day=target_day,
    )

    if result.get("output") is not None:
        paths = save_final_output(result["output"], work_dir)
        result["output_paths"] = paths

    return result
