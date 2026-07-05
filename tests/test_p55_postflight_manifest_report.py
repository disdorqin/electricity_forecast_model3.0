"""
tests/test_p55_postflight_manifest_report.py — P55 tests (15+).

Tests: postflight validation checks, manifest creation/read/write,
delivery report generation, terminal report.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ──────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────


def _make_valid_csv(path: str, target_date: str = "2026-07-05") -> str:
    """Create a valid 24-row output CSV for testing."""
    rows = []
    for h in range(1, 25):
        row = {
            "business_day": target_date,
            "hour_business": h,
            "y_pred": 100.0 + h * 0.5,
            "ds": f"{target_date} {h:02d}:00:00",
        }
        rows.append(row)
    # Hour 24 → D+1 00:00:00 convention
    rows[-1]["ds"] = "2026-07-06 00:00:00"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _make_csv_with_rows(path: str, n_rows: int) -> str:
    """Create a CSV with a given number of rows."""
    rows = []
    for i in range(n_rows):
        row = {
            "business_day": "2026-07-05",
            "hour_business": (i % 24) + 1,
            "y_pred": 100.0,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _sample_profile_def(delivery_allowed: bool = True) -> dict:
    return {
        "delivery_allowed": delivery_allowed,
        "allowed_models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
        "excluded_models": {"stage3_business_fixed": "SUSPECT_LEAKAGE"},
    }


def _sample_manifest() -> dict:
    return {
        "run_id": "test-run-001",
        "target_day": "2026-07-05",
        "profile": "trusted_delivery",
        "started_at": "2026-07-05T08:00:00",
        "completed_at": "2026-07-05T08:30:00",
        "status": "PASS",
        "delivery_status": "DELIVERY_READY",
        "selected_training_days": 30,
        "trusted_models": ["lightgbm_cfg05_dayahead", "catboost_spike_residual"],
        "quarantined_models": ["stage3_business_fixed"],
        "fusion_method": "BGEW",
        "fallback": {
            "fallback_used": False,
            "fallback_method": "",
        },
        "postflight": {
            "status": "PASS",
            "checks": {"check_1": {"passed": True, "detail": "ok"}},
            "errors": [],
            "summary": {"total": 1, "passed": 1, "failed": 0, "warned": 0},
            "output_path": "/tmp/final_output.csv",
            "submission_ready_path": "/tmp/final_output_submission_ready.csv",
        },
        "metrics": {"sMAPE": 9.23, "MAE": 15.4},
        "warnings": [],
        "errors": [],
    }


# ──────────────────────────────────────────────────────
# Postflight Tests (8)
# ──────────────────────────────────────────────────────


class TestPostflight:
    """Tests for delivery.postflight.run_postflight."""

    def test_postflight_pass_for_valid_output(self):
        """PASS for valid 24-row output."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            _make_valid_csv(csv_path)
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="trusted_delivery",
                profile_def=_sample_profile_def(),
                work_dir=tmp,
            )
            assert result["status"] in ("PASS", "WARN")
            assert result["summary"]["total"] >= 10
            assert result["summary"]["failed"] == 0

    def test_postflight_fail_for_23_rows(self):
        """FAIL for 23 rows."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            _make_csv_with_rows(csv_path, 23)
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="trusted_delivery",
            )
            twenty_four = result["checks"].get("twenty_four_rows", {})
            assert twenty_four.get("passed") is False
            assert "23" in twenty_four.get("detail", "")

    def test_postflight_fail_for_nan(self):
        """FAIL for NaN in y_pred."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            rows = []
            for h in range(1, 25):
                val = 100.0 if h != 12 else None
                rows.append({
                    "business_day": "2026-07-05",
                    "hour_business": h,
                    "y_pred": val,
                })
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="trusted_delivery",
            )
            nan_check = result["checks"].get("no_nan_in_predictions", {})
            assert nan_check.get("passed") is False

    def test_postflight_fail_for_duplicates(self):
        """FAIL for duplicate hour_business."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            rows = []
            for h in range(1, 25):
                rows.append({
                    "business_day": "2026-07-05",
                    "hour_business": h if h < 24 else 23,  # duplicate last
                    "y_pred": 100.0,
                })
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="trusted_delivery",
            )
            dup_check = result["checks"].get("no_duplicate_hours", {})
            assert dup_check.get("passed") is False

    def test_postflight_fail_for_missing_file(self):
        """FAIL for missing file."""
        from delivery.postflight import run_postflight
        result = run_postflight(
            output_path="/nonexistent/path.csv",
            target_date="2026-07-05",
            profile_name="trusted_delivery",
        )
        exists_check = result["checks"].get("file_exists_readable", {})
        assert exists_check.get("passed") is False

    def test_postflight_fail_for_non_delivery_profile(self):
        """FAIL for non-delivery profile."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            _make_valid_csv(csv_path)
            profile_def = _sample_profile_def(delivery_allowed=False)
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="balanced_candidate",
                profile_def=profile_def,
            )
            delivery_check = result["checks"].get("profile_delivery_allowed", {})
            assert delivery_check.get("passed") is False

    def test_postflight_detects_quarantined_model_usage(self):
        """Detects quarantined model in allowed_models."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            _make_valid_csv(csv_path)
            profile_def = {
                "delivery_allowed": True,
                "allowed_models": ["stage3_business_fixed"],  # quarantined!
                "excluded_models": {"stage3_business_fixed": "SUSPECT_LEAKAGE"},
            }
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="trusted_delivery",
                profile_def=profile_def,
            )
            quarantine_check = result["checks"].get("no_quarantined_models", {})
            assert quarantine_check.get("passed") is False

    def test_postflight_detects_merge_suffixes(self):
        """Detects _x/_y suffix columns from bad merge."""
        from delivery.postflight import run_postflight
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "final_output.csv")
            rows = []
            for h in range(1, 25):
                rows.append({
                    "business_day": "2026-07-05",
                    "hour_business": h,
                    "y_pred": 100.0,
                    "price_x": 101.0,
                    "price_y": 102.0,
                })
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            result = run_postflight(
                output_path=csv_path,
                target_date="2026-07-05",
                profile_name="trusted_delivery",
            )
            merge_check = result["checks"].get("no_merge_suffixes", {})
            assert merge_check.get("passed") is False


# ──────────────────────────────────────────────────────
# Manifest Tests (5)
# ──────────────────────────────────────────────────────


class TestManifest:
    """Tests for delivery.manifest module."""

    def test_create_manifest_builds_correct_structure(self):
        from delivery.manifest import create_manifest
        manifest = create_manifest(
            run_id="test-001",
            target_day="2026-07-05",
            profile="trusted_delivery",
            started_at="2026-07-05T08:00:00",
            completed_at="2026-07-05T08:30:00",
            status="PASS",
            delivery_status="DELIVERY_READY",
            selected_training_days=30,
            trusted_models=["model_a", "model_b"],
            quarantined_models=["model_c"],
            fusion_method="BGEW",
            fallback_used=False,
            fallback_method="",
            postflight={"status": "PASS", "checks": {}},
            metrics={"sMAPE": 9.23},
            warnings=[],
            errors=[],
        )
        assert manifest["run_id"] == "test-001"
        assert manifest["profile"] == "trusted_delivery"
        assert manifest["fusion_method"] == "BGEW"
        assert manifest["fallback"] == {"fallback_used": False, "fallback_method": ""}
        assert manifest["postflight"]["status"] == "PASS"
        assert manifest["metrics"]["sMAPE"] == 9.23

    def test_write_manifest_writes_valid_json(self):
        from delivery.manifest import create_manifest, write_manifest
        manifest = create_manifest(
            run_id="test-002",
            target_day="2026-07-05",
            profile="trusted_delivery",
            started_at="2026-07-05T08:00:00",
            completed_at="2026-07-05T08:30:00",
            status="PASS",
            delivery_status="DELIVERY_READY",
            selected_training_days=30,
            trusted_models=["a"],
            quarantined_models=[],
            fusion_method="equal_weight",
            fallback_used=True,
            fallback_method="simple_average",
            postflight={},
            metrics={},
            warnings=[],
            errors=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(manifest, tmp)
            assert os.path.isfile(path)
            assert path.endswith("run_manifest.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["run_id"] == "test-002"
            assert data["fallback"]["fallback_used"] is True
            assert data["fallback"]["fallback_method"] == "simple_average"

    def test_read_manifest_round_trips_correctly(self):
        from delivery.manifest import create_manifest, write_manifest, read_manifest
        original = create_manifest(
            run_id="roundtrip-001",
            target_day="2026-07-06",
            profile="balanced_candidate",
            started_at="2026-07-06T08:00:00",
            completed_at="2026-07-06T08:45:00",
            status="WARN",
            delivery_status="DELIVERY_DEGRADED",
            selected_training_days=20,
            trusted_models=["x"],
            quarantined_models=["y", "z"],
            fusion_method="BGEW",
            fallback_used=True,
            fallback_method="equal_weight",
            postflight={"status": "WARN", "checks": {}},
            metrics={"sMAPE": 10.5},
            warnings=["low data"],
            errors=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest(original, tmp)
            loaded = read_manifest(path)
            assert loaded == original
            assert loaded["run_id"] == "roundtrip-001"
            assert loaded["quarantined_models"] == ["y", "z"]

    def test_read_manifest_raises_on_missing_file(self):
        from delivery.manifest import read_manifest
        with pytest.raises(FileNotFoundError):
            read_manifest("/nonexistent/manifest.json")

    def test_manifest_contains_all_required_keys(self):
        from delivery.manifest import create_manifest, validate_manifest_keys
        manifest = create_manifest(
            run_id="key-test",
            target_day="2026-07-05",
            profile="trusted_delivery",
            started_at="2026-07-05T08:00:00",
            completed_at="2026-07-05T08:30:00",
            status="PASS",
            delivery_status="DELIVERY_READY",
            selected_training_days=30,
            trusted_models=[],
            quarantined_models=[],
            fusion_method="BGEW",
            fallback_used=False,
            fallback_method="",
            postflight={},
            metrics={},
            warnings=[],
            errors=[],
        )
        missing = validate_manifest_keys(manifest)
        assert missing == [], f"Missing keys: {missing}"


# ──────────────────────────────────────────────────────
# Report Tests (5)
# ──────────────────────────────────────────────────────


class TestReport:
    """Tests for delivery.report module."""

    def test_generate_delivery_report_creates_both_files(self):
        from delivery.report import generate_delivery_report
        manifest = _sample_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_delivery_report(manifest, tmp)
            assert os.path.isfile(paths["markdown_path"])
            assert os.path.isfile(paths["json_path"])
            assert paths["markdown_path"].endswith("delivery_report.md")
            assert paths["json_path"].endswith("delivery_report.json")

    def test_generate_delivery_report_has_all_required_sections(self):
        from delivery.report import generate_delivery_report
        manifest = _sample_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_delivery_report(manifest, tmp)
            with open(paths["markdown_path"], "r", encoding="utf-8") as f:
                content = f.read()
            required_sections = [
                "P55 Delivery Report",
                "Delivery Status",
                "Training Summary",
                "Model Pool",
                "Postflight Results",
                "Metrics",
                "Output Files",
            ]
            for section in required_sections:
                assert section in content, f"Missing section: {section}"

    def test_generate_delivery_report_json_has_manifest(self):
        from delivery.report import generate_delivery_report
        manifest = _sample_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_delivery_report(manifest, tmp)
            with open(paths["json_path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "manifest" in data
            assert data["manifest"]["run_id"] == "test-run-001"
            assert "report_data" in data
            assert "generated_at" in data

    def test_print_terminal_report_runs_without_error(self):
        from delivery.report import print_terminal_report
        manifest = _sample_manifest()
        # Should not raise any exception
        print_terminal_report(manifest)

    def test_build_report_data_contains_expected_fields(self):
        from delivery.report import _build_report_data
        manifest = _sample_manifest()
        data = _build_report_data(manifest)
        expected_fields = [
            "generated_at", "run_id", "target_day", "profile", "status",
            "delivery_status", "fusion_method", "fallback_used",
            "postflight_status", "postflight_checks_total",
            "trusted_models", "quarantined_models", "metrics",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
        assert data["run_id"] == "test-run-001"


# ──────────────────────────────────────────────────────
# Individual Check Tests (3)
# ──────────────────────────────────────────────────────


class TestIndividualChecks:
    """Tests that individual postflight checks can be called separately."""

    def test_individual_file_exists_check(self):
        from delivery.postflight import _check_exists
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "test.csv")
            pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)
            result = _check_exists(csv_path)
            assert result["passed"] is True
            result2 = _check_exists("/nonexistent.csv")
            assert result2["passed"] is False

    def test_individual_24_rows_check(self):
        from delivery.postflight import _check_24_rows
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "test.csv")
            _make_csv_with_rows(csv_path, 24)
            assert _check_24_rows(csv_path)["passed"] is True
            _make_csv_with_rows(csv_path, 10)
            assert _check_24_rows(csv_path)["passed"] is False

    def test_individual_hour_business_range_check(self):
        from delivery.postflight import _check_hour_business_range
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "test.csv")
            _make_valid_csv(csv_path)
            assert _check_hour_business_range(csv_path)["passed"] is True
            # Bad range
            rows = [{"hour_business": h} for h in [0, 25]]
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            assert _check_hour_business_range(csv_path)["passed"] is False
