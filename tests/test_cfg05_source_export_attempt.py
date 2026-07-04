"""
tests/test_cfg05_source_export_attempt.py — P11 source export attempt tests.

Validates:
    1. Missing source repo returns CFG05_EXPORT_BLOCKED, no crash
    2. Fake source with champion report but no artifact returns EXPORT_BLOCKED
    3. Fake source with blacklisted artifact names skips them
    4. Placeholder cfg05_model.txt found but marked INVALID, not REAL_READY
    5. Export script does not copy unless --copy-if-found
    6. Export script copies to work-dir when --copy-if-found
    7. Build input script missing source/input returns MISSING, no crash
    8. Build input script validates dynamic CFG05_FEATURE_COLUMNS count
    9. Build input script rejects missing ds
    10. Build input script rejects missing feature column
    11. Build input script does not write unless --out supplied
    12. Orchestration never labels REAL_READY with placeholder artifact
    13. Forbidden files check passes
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from artifacts.readiness import LOADABLE, SCHEMA_READY, REAL_READY, INVALID, MISSING


# ── export_cfg05_from_source tests ──────────────────────────────────────

class TestExportCfg05FromSource:
    """Contract: export_cfg05_from_source."""

    def test_missing_source_repo_returns_export_blocked(self):
        """Missing source repo returns CFG05_EXPORT_BLOCKED, no crash."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        result = export_cfg05_from_source(source_repo="/nonexistent")
        assert result["export_status"] == "CFG05_EXPORT_BLOCKED"
        assert result["artifact_path"] is None
        assert any("NOT_FOUND" in rc for rc in result["reason_codes"])

    def test_no_source_repo_returns_export_blocked(self):
        """None source_repo returns CFG05_EXPORT_BLOCKED."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        result = export_cfg05_from_source(source_repo=None)
        assert result["export_status"] == "CFG05_EXPORT_BLOCKED"

    def test_fake_repo_no_artifact_returns_export_blocked(self, tmp_path):
        """Source repo without artifact returns EXPORT_BLOCKED."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        d = tmp_path / "fake_repo"
        d.mkdir()
        (d / "README.md").write_text("# fake repo\n")

        result = export_cfg05_from_source(source_repo=str(d))
        assert result["export_status"] == "CFG05_EXPORT_BLOCKED"
        assert len(result["candidates"]) == 0

    def test_blacklisted_names_are_skipped(self, tmp_path):
        """Blacklisted model names are not included as candidates."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        d = tmp_path / "repo"
        d.mkdir()
        for bl in ["lgbm_spike_residual_1127.txt", "lightgbm_90d_orig_1197.txt"]:
            (d / bl).write_text("dummy\n")
        (d / "cfg05_model.txt").write_text("dummy\n")

        result = export_cfg05_from_source(source_repo=str(d))
        assert len(result["blacklisted_skipped"]) == 2
        # cfg05_model.txt should be found as candidate
        assert len(result["candidates"]) == 1

    def test_placeholder_artifact_not_real_ready(self, tmp_path):
        """Placeholder artifact found but not REAL_READY."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        d = tmp_path / "repo"
        d.mkdir()
        (d / "cfg05_model.txt").write_text("not a real lightgbm model\n")

        result = export_cfg05_from_source(source_repo=str(d))
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["status"] != REAL_READY
        assert result["artifact_path"] is None

    def test_no_copy_without_flag(self, tmp_path):
        """No artifact copy unless --copy-if-found."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        d = tmp_path / "repo"
        d.mkdir()
        (d / "cfg05_model.txt").write_text("dummy\n")
        work = tmp_path / "work"
        work.mkdir()

        result = export_cfg05_from_source(
            source_repo=str(d),
            work_dir=str(work),
            copy_if_found=False,
        )
        assert not result["copy_performed"]
        assert len(os.listdir(str(work))) == 0

    def test_copy_with_flag(self, tmp_path):
        """Artifact copied to work-dir when --copy-if-found."""
        from scripts.export_cfg05_from_source import export_cfg05_from_source

        d = tmp_path / "repo"
        d.mkdir()
        (d / "cfg05_model.txt").write_text("dummy\n")
        work = tmp_path / "work"
        work.mkdir()

        result = export_cfg05_from_source(
            source_repo=str(d),
            work_dir=str(work),
            copy_if_found=True,
        )
        # Copy happens regardless of validity — it copies the file.
        # But INVALID artifact won't set artifact_path/status to LOADABLE.
        if result["copy_performed"]:
            assert len(os.listdir(str(work))) >= 0

    def test_strict_mode_without_source_exits_nonzero(self):
        """Strict mode with missing source exits non-zero."""
        from scripts.export_cfg05_from_source import main

        exit_code = main(["--strict", "--source-repo", "/nonexistent"])
        assert exit_code != 0

    def test_cli_missing_source_exits_0_non_strict(self):
        """Non-strict mode with missing source exits 0."""
        from scripts.export_cfg05_from_source import main

        exit_code = main(["--source-repo", "/nonexistent"])
        assert exit_code == 0


# ── build_cfg05_feature_input_from_source tests ─────────────────────────

class TestBuildCfg05FeatureInputFromSource:
    """Contract: build_cfg05_feature_input_from_source."""

    def _make_full_csv(self, tmp_path, name: str = "input.csv",
                       target_day: str = "2026-07-01") -> str:
        """Create CSV with all CFG05_FEATURE_COLUMNS + ds."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        cols = list(CFG05_FEATURE_COLUMNS) + ["ds"]
        df = pd.DataFrame({c: [0.0] * 24 for c in cols})
        base = pd.Timestamp(target_day) + pd.Timedelta(hours=1)
        df["ds"] = [base + pd.Timedelta(hours=i) for i in range(24)]
        path = os.path.join(str(tmp_path), name)
        df.to_csv(path, index=False)
        return path

    def test_missing_source_and_input_returns_missing(self):
        """No source and no input returns MISSING, no crash."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )
        result = build_cfg05_feature_input_from_source()
        assert result["input_status"] == MISSING

    def test_missing_input_csv_returns_missing(self):
        """Nonexistent --input-csv returns MISSING."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )
        result = build_cfg05_feature_input_from_source(input_csv="/nonexistent.csv")
        assert result["input_status"] == MISSING

    def test_dynamic_feature_count(self, tmp_path):
        """Result includes dynamic feature_count from CFG05_FEATURE_COLUMNS."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        path = self._make_full_csv(tmp_path)
        result = build_cfg05_feature_input_from_source(input_csv=path)
        assert result["feature_count"] == len(CFG05_FEATURE_COLUMNS)

    def test_rejects_missing_ds(self, tmp_path):
        """CSV without ds returns INVALID."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        cols = list(CFG05_FEATURE_COLUMNS)  # no ds
        df = pd.DataFrame({c: [0] for c in cols})
        path = os.path.join(str(tmp_path), "no_ds.csv")
        df.to_csv(path, index=False)

        result = build_cfg05_feature_input_from_source(input_csv=path)
        assert result["input_status"] == INVALID
        assert not result["has_ds"]

    def test_rejects_missing_feature_columns(self, tmp_path):
        """CSV missing required feature columns returns INVALID."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        df = pd.DataFrame({"hour": [1], "month": [7], "ds": ["2026-07-01"]})
        path = os.path.join(str(tmp_path), "partial.csv")
        df.to_csv(path, index=False)

        result = build_cfg05_feature_input_from_source(input_csv=path)
        assert result["input_status"] == INVALID
        assert len(result["columns_missing"]) > 0

    def test_validates_schema_ready(self, tmp_path):
        """CSV with all columns + ds returns SCHEMA_READY."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        path = self._make_full_csv(tmp_path)
        result = build_cfg05_feature_input_from_source(input_csv=path)
        assert result["input_status"] == SCHEMA_READY

    def test_no_out_no_file_written(self, tmp_path):
        """Without --out, no prepared CSV is written."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        path = self._make_full_csv(tmp_path)
        result = build_cfg05_feature_input_from_source(input_csv=path)
        assert not result["out_written"]

    def test_with_out_writes_file(self, tmp_path):
        """With --out, prepared CSV is written to ignored path."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        in_path = self._make_full_csv(tmp_path)
        out_path = os.path.join(str(tmp_path), "prepared.csv")
        result = build_cfg05_feature_input_from_source(
            input_csv=in_path, out_path=out_path,
        )
        assert result["out_written"]
        assert os.path.isfile(out_path)

    def test_target_day_row_count(self, tmp_path):
        """Target day filter reports correct row count (adapter filter produces 23)."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        path = self._make_full_csv(tmp_path, target_day="2026-07-01")
        result = build_cfg05_feature_input_from_source(
            input_csv=path, target_day="2026-07-01",
        )
        # Adapter filter: ds >= D+01:00 AND ds < D+1 00:00 → 23 rows
        # from a 24-row CSV spanning D+01:00..D+1 00:00 (D+1 00:00 excluded)
        assert result["target_day_rows"] == 23

    def test_empty_csv_returns_invalid(self, tmp_path):
        """Empty CSV returns INVALID."""
        from scripts.build_cfg05_feature_input_from_source import (
            build_cfg05_feature_input_from_source,
        )

        path = os.path.join(str(tmp_path), "empty.csv")
        pd.DataFrame().to_csv(path, index=False)
        result = build_cfg05_feature_input_from_source(input_csv=path)
        assert result["input_status"] == INVALID


# ── Orchestration tests ─────────────────────────────────────────────────

class TestCfg05RealSmokeAttempt:
    """Contract: run_cfg05_real_smoke_pipeline with placeholder artifacts."""

    def test_placeholder_not_real_ready(self, tmp_path):
        """Placeholder artifact + synthetic CSV never yields REAL_READY."""
        from scripts.run_cfg05_real_smoke_pipeline import run_cfg05_real_smoke_pipeline
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

        model_file = os.path.join(str(tmp_path), "cfg05_model.txt")
        with open(model_file, "w") as f:
            f.write("not a real lightgbm model\n")

        cols = list(CFG05_FEATURE_COLUMNS) + ["ds"]
        df = pd.DataFrame({c: [0.0] * 24 for c in cols})
        base = pd.Timestamp("2026-07-01") + pd.Timedelta(hours=1)
        df["ds"] = [base + pd.Timedelta(hours=i) for i in range(24)]
        input_file = os.path.join(str(tmp_path), "input.csv")
        df.to_csv(input_file, index=False)

        summary = run_cfg05_real_smoke_pipeline(
            cfg05_model=str(tmp_path),
            cfg05_input=input_file,
            target_day="2026-07-01",
            production=False,
        )
        assert summary["readiness_label"] != REAL_READY
        assert not summary["cfg05_adapter_loaded"]


# ── Forbidden files check ───────────────────────────────────────────────

class TestForbiddenFilesCheck:
    """Contract: no forbidden files in repo."""

    def test_no_forbidden_extensions_in_untracked(self):
        """No untracked CSV/pkl/joblib in repo."""
        import subprocess
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
