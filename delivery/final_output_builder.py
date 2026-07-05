"""
delivery/final_output_builder.py — P73: Final Output Builder.

Constructs the final delivery CSV and JSON with all required columns.

Output schema:
  business_day, ds, hour_business, period,
  dayahead_price, realtime_price,
  dayahead_model_or_fusion, realtime_model_or_fusion,
  dayahead_confidence, realtime_confidence,
  residual_correction_applied, classifier_action,
  negative_risk, spike_risk, uncertainty_score,
  delivery_status, reason_codes

Strict rules:
  - 24 rows per target day
  - No y_true, actual, label, eval_residual in output
  - No NaN in price columns
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────
FINAL_OUTPUT_COLUMNS = [
    "business_day",
    "ds",
    "hour_business",
    "period",
    "dayahead_price",
    "realtime_price",
    "dayahead_model_or_fusion",
    "realtime_model_or_fusion",
    "dayahead_confidence",
    "realtime_confidence",
    "residual_correction_applied",
    "classifier_action",
    "negative_risk",
    "spike_risk",
    "uncertainty_score",
    "delivery_status",
    "reason_codes",
]

FORBIDDEN_COLUMNS = [
    "y_true",
    "actual",
    "label",
    "residual_from_y_true",
    "future_actual",
    "eval_residual",
]

# ── Status constants ──────────────────────────────────────────────────
OUTPUT_BUILT = "OUTPUT_BUILT"
OUTPUT_BLOCKED = "OUTPUT_BLOCKED"
OUTPUT_DEGRADED = "OUTPUT_DEGRADED"


def build_final_output(
    dayahead_fused: Optional[pd.DataFrame] = None,
    realtime_fused: Optional[pd.DataFrame] = None,
    dayahead_classified: Optional[pd.DataFrame] = None,
    realtime_classified: Optional[pd.DataFrame] = None,
    residual_info: Optional[dict] = None,
    target_day: str = "",
    delivery_status: str = "NORMAL",
    reason_codes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the final output DataFrame and save to CSV/JSON.

    Parameters
    ----------
    dayahead_fused : DataFrame
        Fused day-ahead predictions.
    realtime_fused : DataFrame
        Fused realtime predictions.
    dayahead_classified : DataFrame
        Classified day-ahead predictions.
    realtime_classified : DataFrame
        Classified realtime predictions.
    residual_info : dict
        Residual correction info.
    target_day : str
        Target day.
    delivery_status : str
        Overall delivery status.
    reason_codes : list[str]
        Reason codes for the run.

    Returns
    -------
    dict with output DataFrame, path, and status.
    """
    result: dict[str, Any] = {
        "status": OUTPUT_BLOCKED,
        "output": None,
        "rows": 0,
        "reason_codes": list(reason_codes or []),
    }

    # Build base DataFrame with 24 hours
    output = _build_base_output(target_day)

    # Merge day-ahead prices
    if dayahead_fused is not None and len(dayahead_fused) > 0:
        output = _merge_price_column(
            output, dayahead_fused, "dayahead_price", "dayahead_model_or_fusion"
        )
        result["reason_codes"].append("DAYAHEAD_PRICES_MERGED")

    # Merge realtime prices
    if realtime_fused is not None and len(realtime_fused) > 0:
        output = _merge_price_column(
            output, realtime_fused, "realtime_price", "realtime_model_or_fusion"
        )
        result["reason_codes"].append("REALTIME_PRICES_MERGED")

    # Merge classifier outputs
    if dayahead_classified is not None and len(dayahead_classified) > 0:
        output = _merge_classifier_output(output, dayahead_classified)

    # Add confidence columns
    if "dayahead_confidence" not in output.columns:
        output["dayahead_confidence"] = 0.8
    if "realtime_confidence" not in output.columns:
        output["realtime_confidence"] = 0.5

    # Add residual info
    residual_applied = False
    if residual_info:
        da_status = residual_info.get("dayahead", {}).get("status", "")
        residual_applied = "APPLIED" in da_status
    output["residual_correction_applied"] = residual_applied

    # Add delivery status and reason codes
    output["delivery_status"] = delivery_status
    output["reason_codes"] = ";".join(result["reason_codes"]) if result["reason_codes"] else ""

    # Validate
    issues = _validate_output(output)
    if issues:
        result["reason_codes"].extend(issues)
        if any("NaN" in i for i in issues):
            result["status"] = OUTPUT_DEGRADED
        else:
            result["status"] = OUTPUT_BLOCKED
            result["output"] = output
            return result

    # Check forbidden columns
    for col in FORBIDDEN_COLUMNS:
        if col in output.columns:
            output = output.drop(columns=[col])

    result["output"] = output
    result["rows"] = len(output)
    result["status"] = OUTPUT_BUILT if len(output) == 24 else OUTPUT_DEGRADED
    return result


def _build_base_output(target_day: str) -> pd.DataFrame:
    """Build base DataFrame with 24 business hours."""
    rows = []
    for h in range(1, 25):
        period = "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")
        # Timestamp: target_day at hour h (business convention)
        if target_day:
            ds = pd.Timestamp(target_day) + pd.Timedelta(hours=h - 1)
            if h == 24:
                ds = pd.Timestamp(target_day) + pd.Timedelta(hours=23)
            business_day = target_day
        else:
            ds = pd.NaT
            business_day = ""

        rows.append({
            "business_day": business_day,
            "ds": ds,
            "hour_business": h,
            "period": period,
        })
    return pd.DataFrame(rows)


def _merge_price_column(
    output: pd.DataFrame,
    fused: pd.DataFrame,
    price_col: str,
    model_col: str,
) -> pd.DataFrame:
    """Merge a price column from fused predictions into output."""
    if "hour_business" in fused.columns and "hour_business" in output.columns:
        price_map = dict(zip(fused["hour_business"], fused[price_col])) if price_col in fused.columns else {}
        model_map = dict(zip(fused["hour_business"], fused[model_col])) if model_col in fused.columns else {}
        output[price_col] = output["hour_business"].map(price_map)
        output[model_col] = output["hour_business"].map(model_map)
    return output


def _merge_classifier_output(
    output: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    """Merge classifier outputs into the final output."""
    if "hour_business" not in classified.columns:
        return output

    for col in ["classifier_action", "negative_risk", "spike_risk",
                "uncertainty_score", "delivery_warning_level"]:
        if col in classified.columns:
            col_map = dict(zip(classified["hour_business"], classified[col]))
            output[col] = output["hour_business"].map(col_map)

    return output


def _validate_output(output: pd.DataFrame) -> list[str]:
    """Validate the final output."""
    issues = []

    # Check 24 rows
    if len(output) != 24:
        issues.append(f"ROW_COUNT:{len(output)}_expected_24")

    # Check hour_business 1..24
    if "hour_business" in output.columns:
        expected_hours = set(range(1, 25))
        actual_hours = set(output["hour_business"].dropna().astype(int))
        if actual_hours != expected_hours:
            issues.append(f"HOUR_RANGE:{sorted(actual_hours)}")

    # Check no NaN in price columns
    for col in ["dayahead_price", "realtime_price"]:
        if col in output.columns and output[col].isna().any():
            issues.append(f"NaN_IN_{col}")

    # Check forbidden columns
    for col in FORBIDDEN_COLUMNS:
        if col in output.columns:
            issues.append(f"FORBIDDEN_COLUMN:{col}")

    return issues


def save_final_output(
    output: pd.DataFrame,
    output_dir: str,
) -> dict[str, str]:
    """Save final output to CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "final_output.csv")
    output.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, "final_output.json")
    output.to_json(json_path, orient="records", indent=2, default_handler=str)

    return {"csv": csv_path, "json": json_path}
