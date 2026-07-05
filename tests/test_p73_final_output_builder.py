"""P73 — Final Output Builder unit tests."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from delivery.final_output_builder import (
    FINAL_OUTPUT_COLUMNS,
    FORBIDDEN_COLUMNS,
    OUTPUT_BLOCKED,
    OUTPUT_BUILT,
    OUTPUT_DEGRADED,
    build_final_output,
    save_final_output,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def da_fused():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "dayahead_price": np.random.uniform(100, 400, 24),
        "dayahead_model_or_fusion": ["cfg05"] * 24,
    })


@pytest.fixture
def rt_fused():
    return pd.DataFrame({
        "business_day": ["2026-06-01"] * 24,
        "hour_business": list(range(1, 25)),
        "ds": pd.date_range("2026-06-01", periods=24, freq="h"),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "realtime_price": np.random.uniform(100, 400, 24),
        "realtime_model_or_fusion": ["rt_da_anchor"] * 24,
    })


@pytest.fixture
def da_classified(da_fused):
    df = da_fused.copy()
    df["classifier_action"] = "NORMAL"
    df["negative_risk"] = 0.0
    df["spike_risk"] = 0.0
    df["uncertainty_score"] = 0.3
    return df


@pytest.fixture
def rt_classified(rt_fused):
    df = rt_fused.copy()
    df["classifier_action"] = "NORMAL"
    df["negative_risk"] = 0.0
    df["spike_risk"] = 0.0
    df["uncertainty_score"] = 0.3
    return df


# ── Constants ─────────────────────────────────────────────────────────────────


class TestOutputConstants:
    def test_final_output_columns_count(self):
        assert len(FINAL_OUTPUT_COLUMNS) == 17

    def test_forbidden_columns_count(self):
        assert len(FORBIDDEN_COLUMNS) == 6

    def test_y_true_forbidden(self):
        assert "y_true" in FORBIDDEN_COLUMNS

    def test_business_day_in_output(self):
        assert "business_day" in FINAL_OUTPUT_COLUMNS

    def test_dayahead_price_in_output(self):
        assert "dayahead_price" in FINAL_OUTPUT_COLUMNS

    def test_realtime_price_in_output(self):
        assert "realtime_price" in FINAL_OUTPUT_COLUMNS


# ── build_final_output ────────────────────────────────────────────────────────


class TestBuildFinalOutput:
    def test_no_inputs(self):
        result = build_final_output()
        assert isinstance(result, dict)
        assert "status" in result
        assert "output" in result

    def test_24_rows_always_created(self):
        result = build_final_output()
        output = result["output"]
        assert len(output) == 24

    def test_hour_business_1_to_24(self):
        result = build_final_output()
        output = result["output"]
        assert set(output["hour_business"]) == set(range(1, 25))

    def test_with_dayahead_fused(self, da_fused):
        result = build_final_output(dayahead_fused=da_fused, target_day="2026-06-01")
        assert result["status"] in (OUTPUT_BUILT, OUTPUT_DEGRADED)

    def test_with_both_fused(self, da_fused, rt_fused):
        result = build_final_output(
            dayahead_fused=da_fused,
            realtime_fused=rt_fused,
            target_day="2026-06-01",
        )
        assert result["status"] in (OUTPUT_BUILT, OUTPUT_DEGRADED)

    def test_with_classified(self, da_classified, rt_classified):
        result = build_final_output(
            dayahead_classified=da_classified,
            realtime_classified=rt_classified,
            target_day="2026-06-01",
        )
        output = result["output"]
        if "classifier_action" in output.columns:
            assert output["classifier_action"].notna().any()

    def test_period_mapping(self):
        result = build_final_output(target_day="2026-06-01")
        output = result["output"]
        # Hours 1-8 → "1_8"
        h1_8 = output[output["hour_business"].isin(range(1, 9))]
        assert (h1_8["period"] == "1_8").all()
        # Hours 9-16 → "9_16"
        h9_16 = output[output["hour_business"].isin(range(9, 17))]
        assert (h9_16["period"] == "9_16").all()
        # Hours 17-24 → "17_24"
        h17_24 = output[output["hour_business"].isin(range(17, 25))]
        assert (h17_24["period"] == "17_24").all()

    def test_rows_key(self, da_fused):
        result = build_final_output(dayahead_fused=da_fused, target_day="2026-06-01")
        assert result["rows"] == 24

    def test_reason_codes_list(self):
        result = build_final_output()
        assert isinstance(result["reason_codes"], list)

    def test_residual_info_applied(self, da_fused):
        residual_info = {"dayahead": {"status": "RESIDUAL_CORRECTION_APPLIED"}}
        result = build_final_output(
            dayahead_fused=da_fused,
            residual_info=residual_info,
            target_day="2026-06-01",
        )
        output = result["output"]
        if "residual_correction_applied" in output.columns:
            assert output["residual_correction_applied"].any()

    def test_confidence_defaults(self, da_fused, rt_fused):
        result = build_final_output(
            dayahead_fused=da_fused,
            realtime_fused=rt_fused,
            target_day="2026-06-01",
        )
        output = result["output"]
        if "dayahead_confidence" in output.columns:
            assert (output["dayahead_confidence"] == 0.8).any()
        if "realtime_confidence" in output.columns:
            assert (output["realtime_confidence"] == 0.5).any()


# ── save_final_output ─────────────────────────────────────────────────────────


class TestSaveFinalOutput:
    def test_save_creates_files(self, tmp_path):
        output = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
            "dayahead_price": [200.0] * 24,
        })
        result = save_final_output(output, str(tmp_path))
        assert "csv" in result
        assert "json" in result
        assert os.path.isfile(result["csv"])
        assert os.path.isfile(result["json"])

    def test_csv_readable(self, tmp_path):
        output = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        result = save_final_output(output, str(tmp_path))
        df = pd.read_csv(result["csv"])
        assert len(df) == 24

    def test_json_readable(self, tmp_path):
        output = pd.DataFrame({
            "business_day": ["2026-06-01"] * 24,
            "hour_business": list(range(1, 25)),
        })
        result = save_final_output(output, str(tmp_path))
        df = pd.read_json(result["json"])
        assert len(df) == 24
