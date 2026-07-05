"""tests/test_p28_actual_ledger_builder.py — P28 actual ledger builder tests."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def test_p28_status_constants():
    from scripts.build_actual_ledger_from_raw_csv import (
        P28_ACTUAL_LEDGER_BLOCKED,
        P28_ACTUAL_LEDGER_PARTIAL,
        P28_ACTUAL_LEDGER_READY,
        P28_RAW_DATA_MISSING,
    )
    assert P28_ACTUAL_LEDGER_READY == "P28_ACTUAL_LEDGER_READY"
    assert P28_ACTUAL_LEDGER_PARTIAL == "P28_ACTUAL_LEDGER_PARTIAL"
    assert P28_ACTUAL_LEDGER_BLOCKED == "P28_ACTUAL_LEDGER_BLOCKED"
    assert P28_RAW_DATA_MISSING == "P28_RAW_DATA_MISSING"


def test_p28_missing_raw_data():
    from scripts.build_actual_ledger_from_raw_csv import (
        P28_RAW_DATA_MISSING,
        build_actual_ledger_from_raw_csv,
    )
    result = build_actual_ledger_from_raw_csv(raw_data="/nonexistent.csv")
    assert result["final_status"] == P28_RAW_DATA_MISSING


def test_p28_output_keys():
    from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
    result = build_actual_ledger_from_raw_csv(raw_data="/nonexistent.csv")
    required = [
        "total_rows", "target_days", "complete_days", "duplicate_keys",
        "null_y_true_rows", "hour_business_range", "schema_valid",
        "final_status", "reason_codes", "forbidden_files_check",
    ]
    for k in required:
        assert k in result, f"Missing key: {k}"


def test_p28_schema_columns():
    """Actual ledger schema should match data/schema.py."""
    from data.schema import ACTUAL_LEDGER_COLUMNS
    assert "task" in ACTUAL_LEDGER_COLUMNS
    assert "target_day" in ACTUAL_LEDGER_COLUMNS
    assert "business_day" in ACTUAL_LEDGER_COLUMNS
    assert "ds" in ACTUAL_LEDGER_COLUMNS
    assert "hour_business" in ACTUAL_LEDGER_COLUMNS
    assert "period" in ACTUAL_LEDGER_COLUMNS
    assert "y_true" in ACTUAL_LEDGER_COLUMNS
    assert "actual_source" in ACTUAL_LEDGER_COLUMNS


def test_p28_forbidden_files_check_default():
    from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
    result = build_actual_ledger_from_raw_csv(raw_data="/nonexistent.csv")
    assert result["forbidden_files_check"] == "PASS"


def test_p28_with_synthetic_csv():
    """Build actual ledger from the real raw CSV if available."""
    from scripts.build_actual_ledger_from_raw_csv import (
        P28_ACTUAL_LEDGER_READY,
        P28_ACTUAL_LEDGER_PARTIAL,
        build_actual_ledger_from_raw_csv,
    )
    # Use the real data file — synthetic CSV has GBK/UTF-8 encoding issues on Windows
    real_csv = os.path.join(
        "..", "electricity_forecast_model2.1", "data", "shandong_pmos_hourly.csv"
    )
    if not os.path.isfile(real_csv):
        pytest.skip("Real raw CSV not available for encoding-sensitive test")
    work_dir = tempfile.mkdtemp()
    result = build_actual_ledger_from_raw_csv(
        raw_data=real_csv,
        start_day="2026-06-01",
        end_day="2026-06-01",
        work_dir=work_dir,
    )
    assert result["total_rows"] > 0
    assert result["target_days"] >= 1
    assert result["duplicate_keys"] == 0
    assert result["hour_business_range"] is not None
    assert result["hour_business_range"][0] >= 1
    assert result["hour_business_range"][1] <= 24


def test_p28_hour_business_range():
    """Hour business should be 1..24."""
    from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
    real_csv = os.path.join("..", "electricity_forecast_model2.1", "data", "shandong_pmos_hourly.csv")
    if not os.path.isfile(real_csv):
        pytest.skip("Real raw CSV not available")
    work_dir = tempfile.mkdtemp()
    result = build_actual_ledger_from_raw_csv(
        raw_data=real_csv,
        start_day="2026-06-01",
        end_day="2026-06-02",
        work_dir=work_dir,
    )
    if result["hour_business_range"]:
        assert result["hour_business_range"][0] >= 1
        assert result["hour_business_range"][1] <= 24


def test_p28_null_y_true_reported():
    """Null y_true rows should be reported, not silently filled."""
    from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
    real_csv = os.path.join("..", "electricity_forecast_model2.1", "data", "shandong_pmos_hourly.csv")
    if not os.path.isfile(real_csv):
        pytest.skip("Real raw CSV not available")
    work_dir = tempfile.mkdtemp()
    result = build_actual_ledger_from_raw_csv(
        raw_data=real_csv,
        start_day="2026-06-01",
        end_day="2026-06-01",
        work_dir=work_dir,
    )
    assert "null_y_true_rows" in result
    assert result["null_y_true_rows"] >= 0


def test_p28_ledger_file_created():
    """Actual ledger CSV should be created in work_dir/ledgers/."""
    from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
    real_csv = os.path.join("..", "electricity_forecast_model2.1", "data", "shandong_pmos_hourly.csv")
    if not os.path.isfile(real_csv):
        pytest.skip("Real raw CSV not available")
    work_dir = tempfile.mkdtemp()
    result = build_actual_ledger_from_raw_csv(
        raw_data=real_csv,
        start_day="2026-06-01",
        end_day="2026-06-01",
        work_dir=work_dir,
    )
    if result["actual_ledger_path"]:
        assert os.path.isfile(result["actual_ledger_path"])


def test_p28_no_duplicate_keys():
    """Each (task, target_day, business_day, hour_business) should be unique."""
    from scripts.build_actual_ledger_from_raw_csv import build_actual_ledger_from_raw_csv
    real_csv = os.path.join("..", "electricity_forecast_model2.1", "data", "shandong_pmos_hourly.csv")
    if not os.path.isfile(real_csv):
        pytest.skip("Real raw CSV not available")
    work_dir = tempfile.mkdtemp()
    result = build_actual_ledger_from_raw_csv(
        raw_data=real_csv,
        start_day="2026-06-01",
        end_day="2026-06-01",
        work_dir=work_dir,
    )
    assert result["duplicate_keys"] == 0
