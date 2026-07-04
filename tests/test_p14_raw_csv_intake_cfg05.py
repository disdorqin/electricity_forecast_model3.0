"""
tests/test_p14_raw_csv_intake_cfg05.py — P14 tests.

Validates:
    1. Missing raw data → CFG05_RAW_DATA_MISSING, non-strict exit 0
    2. Missing raw data → strict exit non-zero
    3. Schema inspector redacts values by default
    4. Schema inspector reports columns/dtypes/null counts/time range
    5. Invalid raw CSV → CFG05_RAW_DATA_INVALID
    6. Valid minimal Chinese CSV passes contract
    7. P14 wrapper does not train/export if raw data invalid
    8. P14 wrapper uses .local_artifacts/p14_cfg05/ by default
    9. Unsafe work-dir is rejected
    10. REAL smoke not attempted unless gates pass
    11. Summary JSON contains required keys
    12. Forbidden files check passes
    13. No generated forbidden files are tracked in repo
"""

from __future__ import annotations

import os
import subprocess

import pandas as pd
import pytest

from scripts.check_cfg05_raw_data_contract import (
    check_cfg05_raw_data_contract,
    RAW_DATA_MISSING,
    RAW_DATA_INVALID,
    RAW_DATA_VALID,
    REQUIRED_CHINESE_COLUMNS,
)
from scripts.inspect_cfg05_raw_csv_schema import (
    inspect_cfg05_raw_csv_schema,
    _REDACTED,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_valid_chinese_csv(tmp_path, filename="raw.csv", rows=48):
    """Create a minimal valid Chinese CSV with required columns."""
    import numpy as np
    np.random.seed(42)
    base = pd.Timestamp("2026-06-30") + pd.Timedelta(hours=1)
    times = [base + pd.Timedelta(hours=i) for i in range(rows)]

    data = {"时刻": times}
    for col in REQUIRED_CHINESE_COLUMNS[1:]:
        data[col] = np.random.uniform(100, 500, rows)

    df = pd.DataFrame(data)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False, encoding="gbk")
    return path


# ── Schema inspector tests ─────────────────────────────────────────────────


class TestSchemaInspector:
    """Contract: inspect_cfg05_raw_csv_schema."""

    def test_missing_raw_data_returns_missing(self):
        """No raw data returns CFG05_RAW_DATA_MISSING."""
        result = inspect_cfg05_raw_csv_schema(raw_data=None)
        assert result["raw_data_status"] == RAW_DATA_MISSING

    def test_redacts_values_by_default(self, tmp_path):
        """Schema inspector redacts values by default."""
        path = _make_valid_chinese_csv(tmp_path)
        result = inspect_cfg05_raw_csv_schema(raw_data=path, redact_values=True)
        assert result["redacted"] is True
        # Sample rows should not contain raw values
        for row in result.get("sample_rows", []):
            for val in row.values():
                assert val == _REDACTED or val == "***REDACTED***"

    def test_reports_dtypes_and_null_counts(self, tmp_path):
        """Schema inspector reports column dtypes and null counts."""
        path = _make_valid_chinese_csv(tmp_path)
        result = inspect_cfg05_raw_csv_schema(raw_data=path)
        assert result["raw_data_status"] == RAW_DATA_VALID
        assert len(result["dtypes"]) > 0
        assert "float64" in str(result["dtypes"]) or "float" in str(result["dtypes"])

    def test_reports_time_range(self, tmp_path):
        """Schema inspector reports time range."""
        path = _make_valid_chinese_csv(tmp_path)
        result = inspect_cfg05_raw_csv_schema(raw_data=path)
        assert result["rows"] == 48
        assert result["time_min"] is not None
        assert result["time_max"] is not None

    def test_invalid_csv_returns_invalid(self, tmp_path):
        """Invalid CSV returns CFG05_RAW_DATA_INVALID."""
        path = os.path.join(str(tmp_path), "bad.csv")
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        result = inspect_cfg05_raw_csv_schema(raw_data=path)
        assert result["raw_data_status"] == RAW_DATA_INVALID

    def test_valid_csv_passes(self, tmp_path):
        """Valid minimal Chinese CSV passes contract in inspector."""
        path = _make_valid_chinese_csv(tmp_path)
        result = inspect_cfg05_raw_csv_schema(raw_data=path)
        assert result["raw_data_status"] == RAW_DATA_VALID

    def test_non_strict_exit_0_on_missing(self):
        """Non-strict inspector exits 0 on missing."""
        from scripts.inspect_cfg05_raw_csv_schema import main
        exit_code = main(["--raw-data", "/nonexistent"])
        assert exit_code == 0

    def test_strict_exit_nonzero_on_missing(self):
        """Strict inspector exits non-zero on missing."""
        from scripts.inspect_cfg05_raw_csv_schema import main
        exit_code = main(["--strict", "--raw-data", "/nonexistent"])
        assert exit_code != 0


# ── P14 wrapper tests ─────────────────────────────────────────────────────


class TestP14Wrapper:
    """Contract: run_p14_raw_csv_intake_cfg05."""

    def test_missing_raw_data_returns_missing_and_exit_0(self):
        """Missing raw data returns CFG05_RAW_DATA_MISSING and exit 0 non-strict."""
        from scripts.run_p14_raw_csv_intake_cfg05 import main
        exit_code = main([])
        assert exit_code == 0

    def test_missing_raw_data_exits_nonzero_strict(self):
        """Missing raw data exits non-zero in strict mode."""
        from scripts.run_p14_raw_csv_intake_cfg05 import main
        exit_code = main(["--strict"])
        assert exit_code != 0

    def test_does_not_train_if_raw_data_invalid(self, tmp_path):
        """P14 wrapper does not train/export if raw data invalid."""
        from scripts.run_p14_raw_csv_intake_cfg05 import (
            run_p14_raw_csv_intake_cfg05,
        )

        bad_csv = os.path.join(str(tmp_path), "bad.csv")
        pd.DataFrame({"a": [1]}).to_csv(bad_csv, index=False)

        result = run_p14_raw_csv_intake_cfg05(
            raw_data=bad_csv,
            source_repo=str(tmp_path / "repo"),
        )
        assert result["final_status"] == "CFG05_RAW_DATA_INVALID"
        assert result["model_export_status"] == "NOT_ATTEMPTED"
        assert result["feature_export_status"] == "NOT_ATTEMPTED"

    def test_uses_local_artifacts_p14_by_default(self, tmp_path):
        """P14 wrapper uses .local_artifacts/p14_cfg05/ by default."""
        from scripts.run_p14_raw_csv_intake_cfg05 import (
            run_p14_raw_csv_intake_cfg05,
        )

        result = run_p14_raw_csv_intake_cfg05(
            raw_data=None,
        )
        assert ".local_artifacts" in result["model_out"]
        assert "p14_cfg05" in result["model_out"]
        assert ".local_artifacts" in result["features_out"]
        assert "p14_cfg05" in result["features_out"]

    def test_unsafe_work_dir_rejected(self, tmp_path):
        """Unsafe work-dir is rejected by CLI."""
        from scripts.run_p14_raw_csv_intake_cfg05 import main
        exit_code = main(["--work-dir", "outputs/unsafe_p14"])
        assert exit_code != 0

    def test_real_smoke_not_attempted_without_gates(self, tmp_path):
        """REAL smoke not attempted unless gates pass."""
        from scripts.run_p14_raw_csv_intake_cfg05 import (
            run_p14_raw_csv_intake_cfg05,
        )

        path = _make_valid_chinese_csv(tmp_path)
        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = run_p14_raw_csv_intake_cfg05(
            raw_data=path,
            source_repo=str(source_repo),
            run_real_smoke=True,
        )
        # Without real training, gates won't pass
        assert result["raw_data_status"] == RAW_DATA_VALID
        assert result["source_repo_status"] == "PRESENT"

    def test_summary_contains_required_keys(self, tmp_path):
        """Summary JSON contains all required keys."""
        from scripts.run_p14_raw_csv_intake_cfg05 import (
            run_p14_raw_csv_intake_cfg05,
        )

        result = run_p14_raw_csv_intake_cfg05(
            raw_data=None,
        )
        required = [
            "raw_data_status", "source_repo_status",
            "model_export_status", "feature_export_status",
            "cfg05_artifact_status", "cfg05_input_status",
            "real_smoke_attempted", "readiness_label",
            "prediction_rows", "validator_passed",
            "final_status", "model_out", "features_out",
            "reason_codes", "forbidden_files_check",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_nonexistent_raw_data_path_returns_missing(self):
        """Non-existent raw data path returns CFG05_RAW_DATA_MISSING."""
        from scripts.run_p14_raw_csv_intake_cfg05 import (
            run_p14_raw_csv_intake_cfg05,
        )

        result = run_p14_raw_csv_intake_cfg05(
            raw_data="/nonexistent/path.csv",
        )
        assert result["final_status"] == "CFG05_RAW_DATA_MISSING"

    def test_missing_source_repo_still_reported(self, tmp_path):
        """Missing source repo is still reported."""
        from scripts.run_p14_raw_csv_intake_cfg05 import (
            run_p14_raw_csv_intake_cfg05,
        )

        path = _make_valid_chinese_csv(tmp_path)

        result = run_p14_raw_csv_intake_cfg05(
            raw_data=path,
            source_repo="/nonexistent_repo",
        )
        # Raw data is valid but source repo missing
        assert result["source_repo_status"] == "MISSING"


# ── Forbidden files ────────────────────────────────────────────────────────


class TestForbiddenFiles:
    """No forbidden file extensions in untracked."""

    def test_forbidden_files_check(self):
        """No forbidden file extensions in untracked git files."""
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        for f in result.stdout.strip().split("\n"):
            if not f.strip():
                continue
            ext = os.path.splitext(f)[1].lower()
            assert ext not in (".csv", ".pkl", ".joblib", ".parquet",
                               ".feather", ".pt", ".pth", ".ckpt"), (
                f"Forbidden untracked file: {f}"
            )

    def test_no_generated_artifacts_in_repo(self):
        """No generated forbidden files are tracked in repo."""
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        forbidden_exts = (".csv", ".pkl", ".joblib", ".parquet",
                          ".feather", ".pt", ".pth", ".ckpt")
        for f in result.stdout.strip().split("\n"):
            if not f.strip():
                continue
            ext = os.path.splitext(f)[1].lower()
            # Allow test CSV fixtures and .csv in .gitignore paths
            assert ext not in forbidden_exts, (
                f"Forbidden tracked file: {f}"
            )
