"""
tests/test_p13_cfg05_raw_data_contract_export.py — P13 tests.

Validates:
    1. Missing raw data → CFG05_RAW_DATA_MISSING, no crash
    2. CSV missing required Chinese columns → CFG05_RAW_DATA_INVALID
    3. Valid CSV with all required columns → contract passes
    4. Invalid timestamp in 时刻 is reported
    5. Non-numeric price/load columns are reported
    6. Source repo missing → blocker in orchestration
    7. Work-dir outside .local_artifacts/ is rejected
    8. Train/export does not run if raw data invalid
    9. Train/export summary contains required keys
    10. Generated model/features paths are under ignored local work-dir
    11. Placeholder/invalid model never becomes REAL_READY
    12. REAL smoke not attempted unless artifact + input gates pass
    13. Non-strict exits 0 on blocker
    14. Strict exits nonzero on blocker
    15. Forbidden files check passes
    16. Mock success: training placeholder avoids false REAL_READY
"""

from __future__ import annotations

import os
import json
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


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_valid_chinese_csv(tmp_path, filename="raw.csv", rows=48, extra_cols=None):
    """Create a minimal valid Chinese CSV with required columns."""
    import numpy as np
    np.random.seed(42)
    base = pd.Timestamp("2026-06-30") + pd.Timedelta(hours=1)
    times = [base + pd.Timedelta(hours=i) for i in range(rows)]

    data = {"时刻": times}
    for col in REQUIRED_CHINESE_COLUMNS[1:]:
        data[col] = np.random.uniform(100, 500, rows)

    if extra_cols:
        data.update(extra_cols)

    df = pd.DataFrame(data)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False, encoding="gbk")
    return path


# ── Raw data contract tests ────────────────────────────────────────────────


class TestRawDataContract:
    """Contract: check_cfg05_raw_data_contract."""

    def test_missing_raw_data_returns_missing(self):
        """No raw data provided returns CFG05_RAW_DATA_MISSING."""
        result = check_cfg05_raw_data_contract(raw_data=None)
        assert result["raw_data_status"] == RAW_DATA_MISSING

    def test_nonexistent_file_returns_missing(self):
        """Non-existent file path returns CFG05_RAW_DATA_MISSING."""
        result = check_cfg05_raw_data_contract(raw_data="/nonexistent/path.csv")
        assert result["raw_data_status"] == RAW_DATA_MISSING

    def test_missing_chinese_columns_returns_invalid(self, tmp_path):
        """CSV missing required Chinese columns returns CFG05_RAW_DATA_INVALID."""
        path = os.path.join(str(tmp_path), "bad.csv")
        pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_csv(path, index=False)

        result = check_cfg05_raw_data_contract(raw_data=path)
        assert result["raw_data_status"] == RAW_DATA_INVALID
        assert len(result["missing_columns"]) > 0

    def test_invalid_timestamp_reported(self, tmp_path):
        """Invalid timestamps in 时刻 are reported."""
        path = os.path.join(str(tmp_path), "bad_ts.csv")
        data = {"时刻": ["not_a_date", "also_bad", "2026-07-01 01:00:00"]}
        for col in REQUIRED_CHINESE_COLUMNS[1:]:
            data[col] = [100, 200, 300]
        pd.DataFrame(data).to_csv(path, index=False, encoding="gbk")

        result = check_cfg05_raw_data_contract(raw_data=path)
        assert "UNPARSEABLE_TIMESTAMPS" in str(result.get("reason_codes", []))

    def test_non_numeric_columns_reported(self, tmp_path):
        """Non-numeric price/load columns are reported."""
        path = os.path.join(str(tmp_path), "non_num.csv")
        base = pd.Timestamp("2026-07-01")
        data = {"时刻": [base + pd.Timedelta(hours=i) for i in range(5)]}
        for col in REQUIRED_CHINESE_COLUMNS[1:]:
            data[col] = ["abc", "def", "ghi", "jkl", "mno"]
        pd.DataFrame(data).to_csv(path, index=False, encoding="gbk")

        result = check_cfg05_raw_data_contract(raw_data=path)
        assert "NON_NUMERIC" in str(result.get("reason_codes", [])) or \
               result["raw_data_status"] == RAW_DATA_INVALID

    def test_valid_csv_contract_passes(self, tmp_path):
        """Valid Chinese CSV with all required columns passes."""
        path = _make_valid_chinese_csv(tmp_path)

        result = check_cfg05_raw_data_contract(raw_data=path)
        assert result["raw_data_status"] == RAW_DATA_VALID
        assert result["rows"] == 48
        assert result["time_min"] is not None
        assert result["time_max"] is not None

    def test_utf8_encoding_works(self, tmp_path):
        """UTF-8 encoded CSV is also readable."""
        import numpy as np
        np.random.seed(42)
        base = pd.Timestamp("2026-07-01")
        data = {"时刻": [base + pd.Timedelta(hours=i) for i in range(24)]}
        for col in REQUIRED_CHINESE_COLUMNS[1:]:
            data[col] = np.random.uniform(100, 500, 24)

        path = os.path.join(str(tmp_path), "utf8.csv")
        pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8")

        result = check_cfg05_raw_data_contract(raw_data=path)
        assert result["raw_data_status"] == RAW_DATA_VALID

    def test_contract_result_contains_required_keys(self, tmp_path):
        """Contract result contains all expected keys."""
        path = _make_valid_chinese_csv(tmp_path)
        result = check_cfg05_raw_data_contract(raw_data=path)

        required = [
            "raw_data_status", "raw_data_path", "rows",
            "columns_present", "missing_columns",
            "time_min", "time_max", "null_counts", "reason_codes",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_non_strict_exit_0_on_missing(self):
        """Non-strict main exits 0 even when raw data missing."""
        from scripts.check_cfg05_raw_data_contract import main
        exit_code = main(["--raw-data", "/nonexistent"])
        assert exit_code == 0

    def test_strict_exit_nonzero_on_missing(self):
        """Strict main exits non-zero when raw data missing."""
        from scripts.check_cfg05_raw_data_contract import main
        exit_code = main(["--strict", "--raw-data", "/nonexistent"])
        assert exit_code != 0

    def test_strict_exit_nonzero_on_invalid(self, tmp_path):
        """Strict main exits non-zero when raw data invalid."""
        path = os.path.join(str(tmp_path), "bad.csv")
        pd.DataFrame({"a": [1]}).to_csv(path, index=False)
        from scripts.check_cfg05_raw_data_contract import main
        exit_code = main(["--strict", "--raw-data", path])
        assert exit_code != 0


# ── Train/export tests ─────────────────────────────────────────────────────


class TestTrainExportCfg05Local:
    """Contract: train_export_cfg05_local."""

    def test_source_repo_missing_returns_blocker(self, tmp_path):
        """Missing source repo returns blocker without crash."""
        from scripts.train_export_cfg05_local import train_export_cfg05_local

        result = train_export_cfg05_local(
            source_repo=str(tmp_path / "nonexistent"),
        )
        assert result["source_repo_status"] == "MISSING"
        assert not result["train_done"]

    def test_raw_data_missing_skips_training(self, tmp_path):
        """Train/export does not run if raw data invalid."""
        from scripts.train_export_cfg05_local import train_export_cfg05_local

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = train_export_cfg05_local(
            source_repo=str(source_repo),
            raw_data=None,
        )
        assert result["raw_data_status"] in (RAW_DATA_MISSING, "NOT_CHECKED")
        assert not result["train_done"]

    def test_summary_contains_required_keys(self, tmp_path):
        """Train/export summary contains required keys."""
        from scripts.train_export_cfg05_local import train_export_cfg05_local

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = train_export_cfg05_local(
            source_repo=str(source_repo),
        )
        required = [
            "source_repo", "source_repo_status",
            "raw_data_path", "raw_data_status",
            "model_out", "features_out",
            "train_rows", "train_done",
            "model_saved", "features_saved",
            "cfg05_artifact_status", "cfg05_input_status",
            "reason_codes",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_model_features_paths_under_local_workdir(self, tmp_path):
        """Generated paths are under ignored local work-dir when defaults used."""
        from scripts.train_export_cfg05_local import (
            train_export_cfg05_local,
            _DEFAULT_WORK_DIR,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = train_export_cfg05_local(
            source_repo=str(source_repo),
            work_dir=str(tmp_path / ".local_artifacts" / "p13_cfg05"),
        )
        # Paths should be under the specified work-dir
        assert result["model_out"].startswith(str(tmp_path / ".local_artifacts" / "p13_cfg05"))
        assert result["features_out"].startswith(str(tmp_path / ".local_artifacts" / "p13_cfg05"))

    def test_unsafe_work_dir_rejected(self, tmp_path):
        """Work-dir under outputs/ is rejected by CLI."""
        from scripts.train_export_cfg05_local import main

        exit_code = main([
            "--work-dir", "outputs/unsafe_p13",
            "--source-repo", str(tmp_path / "repo"),
        ])
        assert exit_code != 0


# ── Orchestration tests ────────────────────────────────────────────────────


class TestP13Orchestration:
    """Contract: run_p13_cfg05_raw_data_to_real_smoke."""

    def test_no_raw_data_returns_missing(self, tmp_path):
        """No raw data returns CFG05_RAW_DATA_MISSING."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo=str(source_repo),
            raw_data=None,
        )
        assert result["final_status"] == "CFG05_RAW_DATA_MISSING"
        assert result["raw_data_status"] == RAW_DATA_MISSING

    def test_invalid_raw_data_returns_invalid(self, tmp_path):
        """Invalid raw data returns CFG05_RAW_DATA_INVALID."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        bad_csv = os.path.join(str(tmp_path), "bad.csv")
        pd.DataFrame({"a": [1]}).to_csv(bad_csv, index=False)

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo=str(source_repo),
            raw_data=bad_csv,
        )
        assert "CFG05_RAW_DATA_INVALID" in (result["final_status"] or "")
        assert result["raw_data_status"] == RAW_DATA_INVALID

    def test_source_repo_missing_returns_blocker(self):
        """Missing source repo returns blocker."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo="/nonexistent_repo_xyz",
        )
        assert result["source_repo_status"] == "MISSING"

    def test_placeholder_model_not_real_ready(self, tmp_path):
        """Placeholder/invalid model never becomes REAL_READY."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo=str(source_repo),
            raw_data=None,
            run_real_smoke=True,
        )
        assert result["real_smoke_attempted"] is False
        assert result["final_status"] != "CFG05_REAL_READY_LOCAL"

    def test_real_smoke_not_attempted_without_gates(self, tmp_path):
        """REAL smoke not attempted unless artifact + input gates pass."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo=str(source_repo),
            raw_data=None,
            run_real_smoke=True,
        )
        assert result["real_smoke_attempted"] is False

    def test_non_strict_exit_0_on_blocker(self):
        """Non-strict mode exits 0 even with blocker."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import main

        exit_code = main(["--source-repo", "/nonexistent"])
        assert exit_code == 0

    def test_strict_exit_nonzero_on_blocker(self):
        """Strict mode exits non-zero with blocker."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import main

        exit_code = main(["--strict", "--source-repo", "/nonexistent"])
        assert exit_code != 0

    def test_summary_contains_required_keys(self, tmp_path):
        """Orchestration summary contains all required keys."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo=str(source_repo),
        )

        required = [
            "source_repo_status", "raw_data_status",
            "model_export_status", "feature_export_status",
            "cfg05_artifact_status", "cfg05_input_status",
            "real_smoke_attempted", "readiness_label",
            "prediction_rows", "validator_passed",
            "final_status", "model_out", "features_out",
            "reason_codes", "forbidden_files_check",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_mock_export_avoids_false_real_ready(self, tmp_path, monkeypatch):
        """Mock training/export creates placeholder but does not claim REAL_READY."""
        from scripts.run_p13_cfg05_raw_data_to_real_smoke import (
            run_p13_cfg05_raw_data_to_real_smoke,
        )

        source_repo = tmp_path / "repo"
        source_repo.mkdir()

        work_dir = os.path.join(str(tmp_path), ".local_artifacts", "p13_cfg05")
        os.makedirs(work_dir, exist_ok=True)
        model_out = os.path.join(work_dir, "cfg05_model.txt")
        features_out = os.path.join(work_dir, "cfg05_features_2026-07-01.csv")

        # Create a placeholder model file (not a real LightGBM model)
        with open(model_out, "w") as f:
            f.write("not a real lightgbm model\n")

        # Create a placeholder features CSV with correct columns
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        cols = ["ds"] + list(CFG05_FEATURE_COLUMNS)
        import numpy as np
        np.random.seed(42)
        base = pd.Timestamp("2026-07-01") + pd.Timedelta(hours=1)
        df = pd.DataFrame({c: np.random.rand(24) for c in cols})
        df["ds"] = [base + pd.Timedelta(hours=i) for i in range(24)]
        df.to_csv(features_out, index=False)

        # Monkeypatch train/export to return success for these pre-existing files
        def mock_export(**kwargs):
            return {
                "source_repo": str(source_repo),
                "source_repo_status": "PRESENT",
                "raw_data_path": None,
                "raw_data_status": RAW_DATA_VALID,
                "model_out": model_out,
                "features_out": features_out,
                "train_rows": 1000,
                "train_done": True,
                "model_saved": True,
                "features_saved": True,
                "cfg05_artifact_status": "PRESENT",  # not LOADABLE — placeholder
                "cfg05_input_status": "SCHEMA_READY",
                "reason_codes": ["MOCK_TRAINING_SUCCESS"],
            }

        monkeypatch.setattr(
            "scripts.run_p13_cfg05_raw_data_to_real_smoke.train_export_cfg05_local",
            mock_export,
        )

        result = run_p13_cfg05_raw_data_to_real_smoke(
            source_repo=str(source_repo),
            raw_data=_make_valid_chinese_csv(tmp_path),
            run_real_smoke=True,
        )
        # Placeholder model means artifact not LOADABLE → smoke not attempted
        assert result["real_smoke_attempted"] is False or result["final_status"] != "CFG05_REAL_READY_LOCAL"


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
