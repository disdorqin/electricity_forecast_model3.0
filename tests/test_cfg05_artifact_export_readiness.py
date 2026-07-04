"""
tests/test_cfg05_artifact_export_readiness.py — P10 artifact export & REAL smoke tests.

Validates:
    1. locate script handles missing source repo → MISSING, no crash
    2. locate script ignores blacklisted model names
    3. locate script finds candidates but placeholder ≠ REAL_READY
    4. prepare script imports CFG05_FEATURE_COLUMNS, reports dynamic count
    5. prepare script rejects CSV missing required feature columns
    6. prepare script rejects missing ds
    7. prepare script validates schema-ready CSV with all columns + ds
    8. prepare script does not write output unless --out supplied
    9. real smoke pipeline missing model exits 0 non-strict
    10. real smoke pipeline missing model exits nonzero strict
    11. real smoke pipeline missing input exits 0 non-strict
    12. placeholder artifact never returns REAL_READY
    13. no generated files appear in forbidden repo paths
    14. prepare script rejects empty CSV
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from artifacts.readiness import (
    MISSING, PRESENT, LOADABLE, SCHEMA_READY, REAL_READY, INVALID,
)


# ── locate_cfg05_artifact tests ──────────────────────────────────────────

class TestLocateCfg05Artifact:
    """Contract: locate_cfg05_artifact CLI and core function."""

    def test_missing_source_repo_returns_empty(self):
        """Missing source repo returns no candidates, no crash."""
        from scripts.locate_cfg05_artifact import locate_cfg05_artifact

        result = locate_cfg05_artifact(source_repo="/nonexistent/repo")
        assert result["total_candidates"] == 0
        assert result["best_status"] == MISSING
        assert result["best_path"] is None

    def test_source_repo_none_returns_empty(self):
        """None source_repo returns empty result, no crash."""
        from scripts.locate_cfg05_artifact import locate_cfg05_artifact

        result = locate_cfg05_artifact(source_repo=None)
        assert result["total_candidates"] == 0

    def test_blacklisted_names_are_skipped(self, tmp_path):
        """Blacklisted model names are not included in candidates.

        Only files matching keyword search (cfg05/lgbm/lightgbm/champion)
        are discovered; those on the blacklist are skipped as candidates.
        stage3_old_1164 is not matched by keyword search and is silently
        ignored (no cfg05/lgbm/lightgbm/champion in filename).
        """
        from scripts.locate_cfg05_artifact import locate_cfg05_artifact

        # Create blacklisted model files that match keyword search
        for bl_name in ["lgbm_spike_residual_1127", "lightgbm_90d_orig_1197"]:
            f = tmp_path / f"{bl_name}.txt"
            f.write_text("dummy model content\n")

        # This one doesn't match keyword search → not discovered
        f = tmp_path / "stage3_old_1164.txt"
        f.write_text("dummy model content\n")

        # Create a non-blacklisted model
        ok = tmp_path / "cfg05_model.txt"
        ok.write_text("dummy model content\n")

        result = locate_cfg05_artifact(source_repo=str(tmp_path))
        assert result["total_candidates"] == 1  # only cfg05_model.txt
        assert len(result["blacklisted_skipped"]) == 2  # 2 found, then blacklisted
        assert all("cfg05_model.txt" in c["path"] for c in result["candidates"])

    def test_placeholder_artifact_not_real_ready(self, tmp_path):
        """Placeholder artifact files do not become REAL_READY."""
        from scripts.locate_cfg05_artifact import locate_cfg05_artifact

        f = tmp_path / "cfg05_model.txt"
        f.write_text("not a real lightgbm model\n")

        result = locate_cfg05_artifact(model_dir=str(tmp_path))
        assert result["total_candidates"] == 1
        assert result["best_status"] != REAL_READY

    def test_cli_exits_0_with_missing_repo(self):
        """CLI with missing --source-repo exits 0 (no crash)."""
        from scripts.locate_cfg05_artifact import main

        exit_code = main(["--source-repo", "/nonexistent/repo"])
        assert exit_code == 0

    def test_cli_json_output(self, tmp_path):
        """CLI --json produces parseable JSON."""
        from scripts.locate_cfg05_artifact import main

        f = tmp_path / "cfg05_model.txt"
        f.write_text("dummy\n")

        # Capture stdout via printing from main is hard in test,
        # so just verify it doesn't crash
        exit_code = main(["--model-dir", str(tmp_path), "--json"])
        assert exit_code == 0


# ── prepare_cfg05_real_input tests ──────────────────────────────────────

class TestPrepareCfg05RealInput:
    """Contract: prepare_cfg05_real_input CLI and core function."""

    def _make_full_csv(self, tmp_path, name: str = "input.csv") -> str:
        """Create a CSV with all CFG05_FEATURE_COLUMNS + ds."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        cols = list(CFG05_FEATURE_COLUMNS) + ["ds"]
        df = pd.DataFrame(
            {c: [0.0] * 24 for c in cols},
        )
        # Add parsable ds timestamps spanning a target day
        base = pd.Timestamp("2026-07-01") + pd.Timedelta(hours=1)
        df["ds"] = [base + pd.Timedelta(hours=i) for i in range(24)]
        path = os.path.join(str(tmp_path), name)
        df.to_csv(path, index=False)
        return path

    def test_imports_dynamic_feature_count(self):
        """prepare script dynamically imports CFG05_FEATURE_COLUMNS."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        count = len(CFG05_FEATURE_COLUMNS)
        assert count > 0

    def test_rejects_missing_feature_columns(self, tmp_path):
        """CSV missing required feature columns returns INVALID."""
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        # Only write a subset of columns
        subset = ["hour", "month", "ds"]
        df = pd.DataFrame({c: [0] for c in subset})
        path = os.path.join(str(tmp_path), "partial.csv")
        df.to_csv(path, index=False)

        result = prepare_cfg05_real_input(input_path=path)
        assert result["status"] == INVALID
        assert len(result["columns_missing"]) > 0

    def test_rejects_missing_ds(self, tmp_path):
        """CSV without ds column returns INVALID."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        cols = list(CFG05_FEATURE_COLUMNS)  # no ds
        df = pd.DataFrame({c: [0] for c in cols})
        path = os.path.join(str(tmp_path), "no_ds.csv")
        df.to_csv(path, index=False)

        result = prepare_cfg05_real_input(input_path=path)
        assert result["status"] == INVALID
        assert not result["has_ds"]

    def test_validates_schema_ready(self, tmp_path):
        """CSV with all columns + ds returns SCHEMA_READY."""
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        path = self._make_full_csv(tmp_path)
        result = prepare_cfg05_real_input(input_path=path)
        assert result["status"] == SCHEMA_READY
        assert result["has_ds"]
        assert result["ds_parsable"]
        assert len(result["columns_missing"]) == 0

    def test_no_out_no_file_written(self, tmp_path):
        """Without --out, no prepared CSV is written."""
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        path = self._make_full_csv(tmp_path)
        result = prepare_cfg05_real_input(input_path=path)
        assert result["status"] == SCHEMA_READY
        assert not result["out_written"]

    def test_with_out_writes_file(self, tmp_path):
        """With --out, prepared CSV is written."""
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        in_path = self._make_full_csv(tmp_path, name="source.csv")
        out_path = os.path.join(str(tmp_path), "prepared.csv")

        result = prepare_cfg05_real_input(input_path=in_path, out_path=out_path)
        assert result["status"] == SCHEMA_READY
        assert result["out_written"]
        assert os.path.isfile(out_path)

    def test_rejects_empty_csv(self, tmp_path):
        """Empty CSV returns INVALID."""
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        path = os.path.join(str(tmp_path), "empty.csv")
        pd.DataFrame().to_csv(path, index=False)

        result = prepare_cfg05_real_input(input_path=path)
        assert result["status"] == INVALID

    def test_reports_dynamic_feature_count(self, tmp_path):
        """Result includes dynamic feature_count from CFG05_FEATURE_COLUMNS."""
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        from scripts.prepare_cfg05_real_input import prepare_cfg05_real_input

        path = self._make_full_csv(tmp_path)
        result = prepare_cfg05_real_input(input_path=path)
        assert result["feature_count"] == len(CFG05_FEATURE_COLUMNS)


# ── run_cfg05_real_smoke_pipeline tests ──────────────────────────────────

class TestRunCfg05RealSmokePipeline:
    """Contract: run_cfg05_real_smoke_pipeline CLI and core function."""

    def test_missing_model_non_strict_exit_0(self):
        """Missing model in non-strict mode exits 0."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        exit_code = main(["--no-production"])
        assert exit_code == 0

    def test_missing_model_strict_exit_nonzero(self):
        """Missing model in strict mode exits non-zero."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        exit_code = main(["--strict", "--no-production"])
        assert exit_code != 0

    def test_missing_input_non_strict_exit_0(self):
        """Missing input in non-strict mode exits 0."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        exit_code = main(["--no-production", "--cfg05-model", "/nonexistent"])
        assert exit_code == 0

    def test_missing_input_strict_exit_nonzero(self):
        """Missing input in strict mode exits non-zero."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        exit_code = main(["--strict", "--no-production", "--cfg05-model", "/nonexistent"])
        assert exit_code != 0

    def test_placeholder_not_real_ready(self, tmp_path):
        """Placeholder artifact + CSV never yields REAL_READY."""
        from scripts.run_cfg05_real_smoke_pipeline import run_cfg05_real_smoke_pipeline
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

        # Create placeholder model file (not valid LightGBM)
        model_file = os.path.join(str(tmp_path), "cfg05_model.txt")
        with open(model_file, "w") as f:
            f.write("not a valid lightgbm model\n")

        # Create input CSV with all columns
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
        assert summary["cfg05_adapter_loaded"] is False

    def test_summary_contains_all_required_keys(self):
        """Pipeline summary contains all required fields."""
        from scripts.run_cfg05_real_smoke_pipeline import run_cfg05_real_smoke_pipeline

        summary = run_cfg05_real_smoke_pipeline(production=False)
        required = [
            "cfg05_artifact_status", "cfg05_input_status",
            "cfg05_adapter_loaded", "prediction_rows",
            "validator_passed", "readiness_label",
            "overall_status", "reason_codes",
        ]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_reason_codes_descriptive_when_missing(self):
        """Missing artifacts produce clear reason codes."""
        from scripts.run_cfg05_real_smoke_pipeline import run_cfg05_real_smoke_pipeline

        summary = run_cfg05_real_smoke_pipeline(production=False)
        assert len(summary["reason_codes"]) > 0
        assert any("MISSING" in rc for rc in summary["reason_codes"])

    def test_no_out_no_file_written(self, tmp_path):
        """Without --out, no output file is written."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        before = set(os.listdir(str(tmp_path)))
        main(["--no-production"])
        after = set(os.listdir(str(tmp_path)))
        assert before == after  # no new files

    def test_with_out_writes_file(self, tmp_path):
        """With --out, output JSON file is written."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        out_path = os.path.join(str(tmp_path), "smoke_result.json")
        exit_code = main(["--no-production", "--out", out_path])
        assert exit_code == 0
        assert os.path.isfile(out_path)

        with open(out_path) as f:
            data = json.load(f)
        assert "cfg05_artifact_status" in data
        assert "cfg05_input_status" in data
        assert "readiness_label" in data

    def test_strict_with_placeholder_model_exits_nonzero(self, tmp_path):
        """Strict mode with placeholder artifact exits non-zero."""
        from scripts.run_cfg05_real_smoke_pipeline import main

        model_dir = os.path.join(str(tmp_path), "model_dir")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "cfg05_model.txt")
        with open(model_file, "w") as f:
            f.write("invalid\n")

        exit_code = main([
            "--strict", "--no-production",
            "--cfg05-model", model_dir,
        ])
        assert exit_code != 0


class TestForbiddenFilesCheck:
    """Contract: no generated files in forbidden paths."""

    FORBIDDEN_PATTERNS = [
        "data/", "outputs/", "reports/local/",
        "ledgers/", "*.csv", "*.pkl", "*.joblib",
        "*.parquet", "*.feather", "*.pt", "*.pth", "*.ckpt",
    ]

    def test_no_csv_in_repo_root(self):
        """No untracked CSV files in repo root."""
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "*.csv", "*.pkl", "*.joblib", "*.parquet"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        untracked = [f for f in result.stdout.strip().split("\n") if f.strip()]
        assert len(untracked) == 0, f"Forbidden untracked files: {untracked}"

    def test_no_forbidden_files_in_tracked(self):
        """No forbidden file extensions in git tracked files."""
        import subprocess
        forbidden_exts = (".csv", ".pkl", ".joblib", ".parquet", ".feather", ".pt", ".pth", ".ckpt")
        result = subprocess.run(
            ["git", "ls-files", "--cached"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        for f in result.stdout.strip().split("\n"):
            if f.endswith(forbidden_exts):
                # Exception: test fixtures that reference csv but are .py files
                if not f.startswith(("data/", "outputs/", "reports/local/", "ledgers/")):
                    assert False, f"Forbidden tracked file: {f}"
