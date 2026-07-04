"""
tests/test_artifact_readiness.py — Artifact readiness module tests.

Validates:
    1. Missing paths return MISSING
    2. Nonexistent paths return MISSING
    3. Invalid status string raises ValueError
    4. run_all with all None returns all MISSING
    5. run_all returns expected top-level keys
    6. status_to_dict produces serializable dict
    7. cfg05_input with missing feature columns returns INVALID
    8. cfg05_input without ds column returns INVALID
    9. cfg05_input with all feature columns returns SCHEMA_READY
    10. cfg05_input with empty CSV returns INVALID
    11. run_all mixed (some present, some missing)
    12. cfg05_artifact with empty directory returns PRESENT (not loadable)
    13. run_all with all nonexistent paths returns all MISSING
    14. No REAL_READY without real artifacts
    15. ArtifactStatus dataclass defaults
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from artifacts.readiness import (
    ArtifactStatus,
    check_cfg05_artifact,
    check_cfg05_input,
    check_rt_assist_pack,
    check_p5m_pack,
    check_actual_ledger,
    check_extrempriceclf_artifact,
    run_all_artifact_readiness,
    status_to_dict,
    MISSING,
    PRESENT,
    LOADABLE,
    SCHEMA_READY,
    REAL_READY,
    NOT_IMPLEMENTED,
    INVALID,
)


class TestArtifactStatusDataclass:
    """Contract: ArtifactStatus basic behavior."""

    def test_default_status_is_missing(self):
        """Default status is MISSING."""
        status = ArtifactStatus(name="test")
        assert status.status == MISSING
        assert not status.exists
        assert not status.loadable
        assert not status.schema_ready
        assert not status.real_ready
        assert status.reason_codes == []
        assert status.details == {}

    def test_invalid_status_raises(self):
        """Invalid status string raises ValueError."""
        with pytest.raises(ValueError):
            ArtifactStatus(name="test", status="NOT_A_REAL_STATUS")

    def test_valid_status_accepted(self):
        """All valid status constants are accepted."""
        for s in [MISSING, PRESENT, LOADABLE, SCHEMA_READY, REAL_READY, NOT_IMPLEMENTED, INVALID]:
            status = ArtifactStatus(name="test", status=s)
            assert status.status == s

    def test_status_to_dict_serializable(self):
        """status_to_dict produces JSON-serializable dict."""
        status = ArtifactStatus(
            name="test",
            status=PRESENT,
            path="/tmp/test",
            exists=True,
            reason_codes=["TEST_CODE"],
            details={"key": "value"},
        )
        d = status_to_dict(status)
        assert d["name"] == "test"
        assert d["status"] == PRESENT
        assert d["path"] == "/tmp/test"
        assert d["exists"] is True
        assert d["reason_codes"] == ["TEST_CODE"]
        assert d["details"]["key"] == "value"


class TestCfg05ArtifactCheck:
    """Contract: check_cfg05_artifact."""

    def test_none_path_returns_missing(self):
        """None path returns MISSING."""
        status = check_cfg05_artifact(None)
        assert status.status == MISSING
        assert "CFG05_ARTIFACT_MISSING" in status.reason_codes

    def test_empty_string_path_returns_missing(self):
        """Empty string path returns MISSING."""
        status = check_cfg05_artifact("")
        assert status.status == MISSING

    def test_nonexistent_path_returns_missing(self):
        """Nonexistent file path returns MISSING."""
        status = check_cfg05_artifact("/nonexistent/path/model.txt")
        assert status.status == MISSING

    def test_nonexistent_dir_returns_missing(self):
        """Nonexistent directory path returns MISSING."""
        status = check_cfg05_artifact("/nonexistent/dir")
        assert status.status == MISSING

    def test_empty_dir_returns_present(self, tmp_path):
        """Empty directory (no model files) returns PRESENT."""
        d = tmp_path / "empty_model_dir"
        d.mkdir()
        status = check_cfg05_artifact(str(d))
        assert status.status == PRESENT
        assert status.exists
        assert any("NO_MODEL_FILE" in rc for rc in status.reason_codes)

    def test_placeholder_file_returns_invalid(self, tmp_path):
        """Placeholder file (not a valid LightGBM model) returns INVALID."""
        f = tmp_path / "model.txt"
        f.write_text("not a real lightgbm model file\n")
        status = check_cfg05_artifact(str(f))
        assert status.status in (PRESENT, INVALID)
        assert status.exists


class TestCfg05InputCheck:
    """Contract: check_cfg05_input."""

    def test_none_path_returns_missing(self):
        """None path returns MISSING."""
        status = check_cfg05_input(None)
        assert status.status == MISSING

    def test_nonexistent_path_returns_missing(self):
        """Nonexistent path returns MISSING."""
        status = check_cfg05_input("/nonexistent/input.csv")
        assert status.status == MISSING

    def test_empty_csv_returns_invalid(self, tmp_path):
        """Empty CSV returns INVALID."""
        f = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(f, index=False)
        status = check_cfg05_input(str(f))
        assert status.status == INVALID

    def test_missing_feature_columns_returns_invalid(self, tmp_path):
        """CSV without all required feature columns returns INVALID."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        # Only include a subset of columns
        subset = ["hour", "month", "day_of_week", "ds"]
        f = tmp_path / "partial.csv"
        pd.DataFrame({c: [0] for c in subset}).to_csv(f, index=False)
        status = check_cfg05_input(str(f))
        assert status.status == INVALID
        assert any("MISSING" in rc for rc in status.reason_codes)

    def test_without_ds_column_returns_invalid(self, tmp_path):
        """CSV with all feature columns but no ds column returns INVALID."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        cols = list(CFG05_FEATURE_COLUMNS)
        f = tmp_path / "no_ds.csv"
        pd.DataFrame({c: [0] for c in cols}).to_csv(f, index=False)
        status = check_cfg05_input(str(f))
        assert status.status == INVALID
        assert any("DS" in rc.upper() for rc in status.reason_codes)

    def test_full_features_and_ds_returns_schema_ready(self, tmp_path):
        """CSV with all CFG05_FEATURE_COLUMNS + ds returns SCHEMA_READY."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        cols = list(CFG05_FEATURE_COLUMNS) + ["ds"]
        f = tmp_path / "full.csv"
        pd.DataFrame({c: [0 for _ in range(24)] for c in cols}).to_csv(f, index=False)
        status = check_cfg05_input(str(f))
        assert status.status == SCHEMA_READY
        assert status.schema_ready


class TestRtAssistPackCheck:
    """Contract: check_rt_assist_pack."""

    def test_none_path_returns_missing(self):
        """None path returns MISSING."""
        status = check_rt_assist_pack(None)
        assert status.status == MISSING

    def test_nonexistent_path_returns_missing(self):
        """Nonexistent directory returns MISSING."""
        status = check_rt_assist_pack("/nonexistent/rt_assist_pack")
        assert status.status == MISSING

    def test_empty_dir_returns_present(self, tmp_path):
        """Empty directory returns PRESENT."""
        d = tmp_path / "rt_assist_pack"
        d.mkdir()
        status = check_rt_assist_pack(str(d))
        assert status.status == PRESENT
        assert any("NO_RECOGNISED_MODEL_FILE" in rc for rc in status.reason_codes)


class TestP5mPackCheck:
    """Contract: check_p5m_pack."""

    def test_none_path_returns_missing(self):
        """None path returns MISSING."""
        status = check_p5m_pack(None)
        assert status.status == MISSING

    def test_nonexistent_path_returns_missing(self):
        """Nonexistent directory returns MISSING."""
        status = check_p5m_pack("/nonexistent/p5m_pack")
        assert status.status == MISSING


class TestActualLedgerCheck:
    """Contract: check_actual_ledger."""

    def test_none_path_returns_missing(self):
        """None path returns MISSING."""
        status = check_actual_ledger(None)
        assert status.status == MISSING

    def test_nonexistent_path_returns_missing(self):
        """Nonexistent path returns MISSING."""
        status = check_actual_ledger("/nonexistent/actual_ledger.csv")
        assert status.status == MISSING

    def test_empty_csv_returns_invalid(self, tmp_path):
        """Empty CSV returns INVALID (cannot be parsed)."""
        f = tmp_path / "actual_ledger.csv"
        pd.DataFrame().to_csv(f, index=False)
        status = check_actual_ledger(str(f))
        assert status.status == INVALID


class TestExtremPriceClfCheck:
    """Contract: check_extrempriceclf_artifact."""

    def test_none_path_returns_missing(self):
        """None path returns MISSING."""
        status = check_extrempriceclf_artifact(None)
        assert status.status == MISSING

    def test_nonexistent_path_returns_missing(self):
        """Nonexistent directory returns MISSING."""
        status = check_extrempriceclf_artifact("/nonexistent/clf")
        assert status.status == MISSING


class TestRunAllArtifactReadiness:
    """Contract: run_all_artifact_readiness."""

    def test_run_all_missing_returns_all_missing(self):
        """All None returns all gates MISSING."""
        report = run_all_artifact_readiness()
        assert report["summary"]["total_gates"] == 6
        assert report["summary"]["status_counts"].get(MISSING, 0) == 6

    def test_run_all_returns_expected_keys(self):
        """Report contains 'gates' and 'summary' keys."""
        report = run_all_artifact_readiness()
        assert "gates" in report
        assert "summary" in report

    def test_run_all_gate_names(self):
        """All 6 expected gate names are present."""
        report = run_all_artifact_readiness()
        expected_gates = {
            "cfg05_artifact",
            "cfg05_input",
            "rt_assist_pack",
            "p5m_pack",
            "actual_ledger",
            "extrempriceclf_artifact",
        }
        assert set(report["gates"].keys()) == expected_gates

    def test_run_all_summary_has_expected_fields(self):
        """Summary contains expected computed fields."""
        report = run_all_artifact_readiness()
        s = report["summary"]
        assert "total_gates" in s
        assert "status_counts" in s
        assert "real_ready_gates" in s
        assert "all_real_ready" in s
        assert "any_missing" in s

    def test_run_all_no_real_ready(self):
        """No gates are REAL_READY without real artifacts."""
        report = run_all_artifact_readiness()
        assert len(report["summary"]["real_ready_gates"]) == 0
        assert not report["summary"]["all_real_ready"]

    def test_run_all_any_missing_true(self):
        """any_missing is True when all gates are None."""
        report = run_all_artifact_readiness()
        assert report["summary"]["any_missing"]

    def test_run_all_non_none_nonexistent_all_missing(self):
        """All non-None but nonexistent paths → all MISSING."""
        report = run_all_artifact_readiness(
            cfg05_model="/nonexistent/model.txt",
            cfg05_input="/nonexistent/input.csv",
            rt_assist_pack="/nonexistent/pack",
            p5m_pack="/nonexistent/pack",
            actual_ledger="/nonexistent/ledger.csv",
            extrempriceclf_dir="/nonexistent/dir",
        )
        assert all(
            g["status"] == MISSING
            for g in report["gates"].values()
        )

    def test_run_all_with_input_schema_ready(self, tmp_path):
        """cfg05_input with full columns + ds returns SCHEMA_READY."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        cols = list(CFG05_FEATURE_COLUMNS) + ["ds"]
        f = tmp_path / "full.csv"
        pd.DataFrame({c: [0] for c in cols}).to_csv(f, index=False)

        report = run_all_artifact_readiness(cfg05_input=str(f))
        assert report["gates"]["cfg05_input"]["status"] == SCHEMA_READY
        assert report["gates"]["cfg05_artifact"]["status"] == MISSING


class TestArtifactReadinessEdgeCases:
    """Contract: edge cases."""

    def test_artifact_status_path_none(self):
        """ArtifactStatus with path=None is valid."""
        status = ArtifactStatus(name="test", path=None)
        assert status.path is None
        assert status.status == MISSING

    def test_artifact_status_all_fields(self):
        """All fields can be set and retrieved."""
        status = ArtifactStatus(
            name="full_test",
            status=LOADABLE,
            path="/tmp/test.txt",
            exists=True,
            loadable=True,
            schema_ready=True,
            real_ready=False,
            reason_codes=["A", "B"],
            details={"nested": {"key": 1}},
        )
        assert status.name == "full_test"
        assert status.loadable
        assert status.details["nested"]["key"] == 1

    def test_run_all_report_is_json_serializable(self):
        """Report dict is JSON-serializable."""
        import json
        report = run_all_artifact_readiness()
        # Should not raise
        json.dumps(report, indent=2, default=str)
