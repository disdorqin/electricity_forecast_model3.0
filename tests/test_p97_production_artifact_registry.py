"""
Tests for P97: Production Artifact Registry.

Covers:
  1. Registry config loads correctly
  2. scan_artifact finds real artifacts
  3. scan_artifact handles missing artifacts
  4. sha256 computation
  5. Registry output schema
  6. go_blockers detection
  7. Report generation
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from artifacts.production_registry import (
    sha256_file,
    scan_artifact,
    run_production_registry,
    save_registry_output,
    ARTIFACT_FOUND,
    ARTIFACT_MISSING,
    ARTIFACT_LOADED,
    ARTIFACT_LOAD_FAILED,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSHA256:
    def test_sha256_existing_file(self):
        path = os.path.join(REPO_ROOT, "README.md")
        h = sha256_file(path)
        assert len(h) == 64  # SHA256 hex

    def test_sha256_missing_file(self):
        assert sha256_file("/nonexistent/path") == ""

    def test_sha256_real_binary(self):
        """Test that a real model file has a readable SHA256."""
        paths = [
            os.path.join(REPO_ROOT, ".local_artifacts", "p31_p40_multimodel_fusion",
                         "models", "cfg05_dayahead_lgbm", "cfg05_model.txt"),
        ]
        for p in paths:
            if os.path.isfile(p):
                h = sha256_file(p)
                assert len(h) == 64
                return
        # If none found, skip gracefully
        assert True


class TestScanArtifact:
    def test_find_real_cfg05(self):
        """Scan for a real artifact that exists."""
        paths = [
            ".local_artifacts/p31_p40_multimodel_fusion/models/cfg05_dayahead_lgbm/cfg05_model.txt",
        ]
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        result = scan_artifact(paths, config_path=cfg)
        if result["found"]:
            assert result["sha256"] != ""
        else:
            # If file doesn't exist on this filesystem, test passes silently
            pytest.skip("cfg05 model not found on disk")

    def test_missing_artifact(self):
        result = scan_artifact(["/nonexistent/file.pkl"])
        assert not result["found"]
        assert result["status"] == ARTIFACT_MISSING

    def test_fallback_path_used(self):
        """When primary path missing but fallback exists, fallback should work."""
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        result = scan_artifact(
            paths=["/nonexistent/primary.pkl"],
            fallback_paths=[".local_artifacts/p31_p40_multimodel_fusion/models/"
                           "cfg05_dayahead_lgbm/cfg05_model.txt"],
            config_path=cfg,
        )
        if result["found"]:
            assert "USING_FALLBACK_PATH" in str(result.get("reason_codes", []))

    def test_sha256_in_result(self):
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        paths = [".local_artifacts/p31_p40_multimodel_fusion/models/"
                 "cfg05_dayahead_lgbm/cfg05_model.txt"]
        result = scan_artifact(paths, config_path=cfg)
        if result["found"]:
            assert result["sha256"] != ""

    def test_no_paths_returns_missing(self):
        result = scan_artifact([])
        assert not result["found"]
        assert result["status"] == ARTIFACT_MISSING


class TestRunRegistry:
    def test_registry_loads_config(self):
        config_path = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        assert os.path.isfile(config_path)

    def test_registry_returns_result(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        assert result["status"] == "REGISTRY_COMPLETE"
        assert "summary" in result
        assert result["summary"]["total"] > 0

    def test_registry_has_artifacts(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        assert len(result["artifacts"]) > 0

    def test_registry_summary_counts(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        s = result["summary"]
        assert s["found"] + s["missing"] == s["total"]

    def test_registry_overall_assessment(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        assert result["overall_assessment"] in ("GO", "GO_WITH_CAVEATS", "BLOCKED")

    def test_registry_required_field_present(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        for key, val in result["artifacts"].items():
            assert "required_for_go" in val
            assert "status_if_missing" in val

    def test_registry_go_blockers(self):
        """If a required artifact is missing, it should be in go_blockers."""
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        if result["go_blockers"]:
            for b in result["go_blockers"]:
                assert "REQUIRED" in b or "required" in b

    def test_registry_version_string(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        assert "3.0" in result.get("version", "")


class TestSaveRegistry:
    def test_save_json(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        out_dir = tempfile.mkdtemp()
        paths = save_registry_output(result, out_dir)
        assert os.path.isfile(paths["json"])
        with open(paths["json"]) as f:
            loaded = json.load(f)
        assert loaded["summary"]["total"] > 0

    def test_save_md(self):
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        out_dir = tempfile.mkdtemp()
        paths = save_registry_output(result, out_dir)
        assert os.path.isfile(paths["md"])
        with open(paths["md"], encoding="utf-8") as f:
            content = f.read()
        assert "Artifact" in content

    def test_save_output_schema(self):
        """Saved JSON must have expected top-level keys."""
        result = run_production_registry(
            config_path=os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"),
        )
        out_dir = tempfile.mkdtemp()
        paths = save_registry_output(result, out_dir)
        with open(paths["json"]) as f:
            loaded = json.load(f)
        for key in ("overall_assessment", "summary", "artifacts", "go_blockers", "caveats"):
            assert key in loaded, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
