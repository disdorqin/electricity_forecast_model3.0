"""
tests/test_p24_artifact_upgrade_scan.py — P24 tests.

Tests P24 artifact upgrade scan with synthetic data and tmp_path.
Minimum 10 tests covering:
  1. scan does not mark placeholder as REAL
  2. code-only findings NOT counted as REAL
  3. no artifacts found → P24_NO_REAL_ARTIFACTS_FOUND
  4. actual model file found → P24_UPGRADE_ARTIFACTS_FOUND or P24_UPGRADE_PARTIAL
  5. summary keys present
  6. p5m scan with no directory → ARTIFACT_MISSING
  7. extrempriceclf scan with code-only dir → ARTIFACT_CODE_ONLY
  8. actual ledger existence check
  9. bgew training status
  10. upgrade recommendations populated
  11. _is_trained_model_file correctly classifies files
"""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_p24_artifact_upgrade_scan import (
    run_p24_artifact_upgrade_scan,
    scan_p5m_artifacts,
    scan_extrempriceclf_artifacts,
    scan_actual_ledger,
    _is_trained_model_file,
    _path_is_safe,
    P24_UPGRADE_ARTIFACTS_FOUND,
    P24_UPGRADE_PARTIAL,
    P24_NO_REAL_ARTIFACTS_FOUND,
    ARTIFACT_REAL,
    ARTIFACT_MISSING,
    ARTIFACT_CODE_ONLY,
)


# ── _is_trained_model_file tests ───────────────────────────────────────────

class TestIsTrainedModelFile:
    def test_pkl_is_trained(self, tmp_path):
        f = tmp_path / "model.pkl"
        f.write_bytes(b"fake pickle data")
        assert _is_trained_model_file(str(f)) is True

    def test_joblib_is_trained(self, tmp_path):
        f = tmp_path / "model.joblib"
        f.write_bytes(b"fake joblib data")
        assert _is_trained_model_file(str(f)) is True

    def test_pt_is_trained(self, tmp_path):
        f = tmp_path / "model.pt"
        f.write_bytes(b"fake pytorch data")
        assert _is_trained_model_file(str(f)) is True

    def test_model_txt_is_trained(self, tmp_path):
        f = tmp_path / "cfg05_model.txt"
        f.write_text("fake lightgbm model")
        assert _is_trained_model_file(str(f)) is True

    def test_py_is_not_trained(self, tmp_path):
        f = tmp_path / "model.py"
        f.write_text("# python code")
        assert _is_trained_model_file(str(f)) is False

    def test_csv_is_not_trained(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3")
        assert _is_trained_model_file(str(f)) is False

    def test_readme_is_not_trained(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# README")
        assert _is_trained_model_file(str(f)) is False

    def test_nonexistent_file(self):
        assert _is_trained_model_file("/nonexistent/model.pkl") is False

    def test_plain_txt_is_not_trained(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("just some notes")
        assert _is_trained_model_file(str(f)) is False


# ── Scan component tests ───────────────────────────────────────────────────

class TestP5MScan:
    def test_no_directory_missing(self):
        result = scan_p5m_artifacts(
            extra_scan_dirs=["/nonexistent/dir"],
            include_defaults=False,
        )
        assert result["status"] == ARTIFACT_MISSING
        assert result["n_artifacts"] == 0

    def test_directory_with_model_pkl(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "residual_model.pkl").write_bytes(b"fake")
        result = scan_p5m_artifacts(
            extra_scan_dirs=[str(tmp_path)],
            include_defaults=False,
        )
        assert result["status"] == ARTIFACT_REAL
        assert result["n_artifacts"] >= 1

    def test_directory_with_code_only(self, tmp_path):
        code_dir = tmp_path / "src"
        code_dir.mkdir()
        (code_dir / "model.py").write_text("# code only")
        result = scan_p5m_artifacts(
            extra_scan_dirs=[str(tmp_path)],
            include_defaults=False,
        )
        # Code-only should NOT be counted as REAL
        assert result["status"] == ARTIFACT_MISSING
        assert result["n_artifacts"] == 0


class TestExtremPriceScan:
    def test_no_directory_missing(self):
        result = scan_extrempriceclf_artifacts(
            extra_scan_dirs=["/nonexistent/dir"],
            include_defaults=False,
        )
        assert result["status"] == ARTIFACT_MISSING

    def test_code_only_directory(self, tmp_path):
        code_dir = tmp_path / "ExtremPriceClf"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("# training code")
        (code_dir / "README.md").write_text("# docs")
        result = scan_extrempriceclf_artifacts(
            extra_scan_dirs=[str(tmp_path)],
            include_defaults=False,
        )
        assert result["status"] == ARTIFACT_CODE_ONLY
        assert result["n_artifacts"] == 0

    def test_real_model_found(self, tmp_path):
        model_dir = tmp_path / "ExtremPriceClf"
        model_dir.mkdir()
        (model_dir / "classifier.pkl").write_bytes(b"fake model")
        result = scan_extrempriceclf_artifacts(
            extra_scan_dirs=[str(tmp_path)],
            include_defaults=False,
        )
        assert result["status"] == ARTIFACT_REAL
        assert result["n_artifacts"] >= 1


class TestActualLedgerScan:
    def test_ledger_missing(self, tmp_path):
        result = scan_actual_ledger(work_dir=str(tmp_path))
        assert result["status"] == ARTIFACT_MISSING
        assert result["exists"] is False

    def test_ledger_exists(self, tmp_path):
        ledger_dir = tmp_path / "ledgers"
        ledger_dir.mkdir()
        (ledger_dir / "actual_ledger.csv").write_text("task,ds,y_true\n")
        result = scan_actual_ledger(work_dir=str(tmp_path))
        assert result["status"] == ARTIFACT_REAL
        assert result["exists"] is True


# ── Full scan orchestration tests ──────────────────────────────────────────

class TestFullScan:
    def test_no_artifacts_found(self, tmp_path):
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=["/nonexistent/p5m"],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        assert result["final_status"] == P24_NO_REAL_ARTIFACTS_FOUND
        assert result["real_artifact_count"] == 0

    def test_one_artifact_partial(self, tmp_path):
        # Create one real artifact (actual_ledger)
        ledger_dir = tmp_path / "ledgers"
        ledger_dir.mkdir()
        (ledger_dir / "actual_ledger.csv").write_text("task,ds,y_true\n")
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=["/nonexistent/p5m"],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        assert result["final_status"] == P24_UPGRADE_PARTIAL
        assert result["real_artifact_count"] == 1

    def test_multiple_artifacts_found(self, tmp_path):
        # Create actual_ledger
        ledger_dir = tmp_path / "ledgers"
        ledger_dir.mkdir()
        (ledger_dir / "actual_ledger.csv").write_text("task,ds,y_true\n")
        # Create p5m model
        p5m_dir = tmp_path / "p5m_models"
        p5m_dir.mkdir()
        (p5m_dir / "residual.pkl").write_bytes(b"fake")
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=[str(p5m_dir)],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        assert result["final_status"] == P24_UPGRADE_ARTIFACTS_FOUND
        assert result["real_artifact_count"] >= 2

    def test_summary_keys_present(self, tmp_path):
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=["/nonexistent/p5m"],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        required_keys = [
            "p5m_pack_status", "p5m_artifacts_found", "p5m_n_artifacts",
            "extrempriceclf_status", "extrempriceclf_artifacts_found",
            "extrempriceclf_n_artifacts",
            "actual_ledger_status", "actual_ledger_path", "actual_ledger_exists",
            "bgew_training_status", "upgrade_recommendations",
            "real_artifact_count", "final_status", "reason_codes",
            "forbidden_files_check",
        ]
        for key in required_keys:
            assert key in result, f"Missing summary key: {key}"

    def test_placeholder_not_marked_real(self, tmp_path):
        # Create only code files, no trained models
        code_dir = tmp_path / "code_only"
        code_dir.mkdir()
        (code_dir / "model.py").write_text("# code")
        (code_dir / "config.yaml").write_text("key: value")
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=[str(code_dir)],
            extra_extremprice_dirs=[str(code_dir)],
            include_defaults=False,
        )
        # Code-only findings should NOT be counted as REAL
        assert result["p5m_pack_status"] != ARTIFACT_REAL
        assert result["extrempriceclf_status"] != ARTIFACT_REAL
        assert result["real_artifact_count"] == 0

    def test_bgew_training_status_blocked(self, tmp_path):
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=["/nonexistent/p5m"],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        assert result["bgew_training_status"] == "BGEW_TRAINING_BLOCKED_NO_ACTUAL_LEDGER"

    def test_bgew_training_status_feasible(self, tmp_path):
        ledger_dir = tmp_path / "ledgers"
        ledger_dir.mkdir()
        (ledger_dir / "actual_ledger.csv").write_text("task,ds,y_true\n")
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=["/nonexistent/p5m"],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        assert result["bgew_training_status"] == "BGEW_TRAINING_FEASIBLE"

    def test_upgrade_recommendations_populated(self, tmp_path):
        result = run_p24_artifact_upgrade_scan(
            work_dir=str(tmp_path),
            extra_p5m_dirs=["/nonexistent/p5m"],
            extra_extremprice_dirs=["/nonexistent/extrem"],
            include_defaults=False,
        )
        assert len(result["upgrade_recommendations"]) >= 3


class TestPathSafety:
    def test_safe_path(self):
        assert _path_is_safe(".local_artifacts/test") is True

    def test_unsafe_data(self):
        assert _path_is_safe("data/raw") is False
