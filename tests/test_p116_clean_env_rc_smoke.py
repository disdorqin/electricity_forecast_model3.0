"""P116: Clean Environment RC Smoke — tests that work without any local artifacts."""
from __future__ import annotations
import json, os, subprocess, pytest
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestMainExists:
    def test_main_py_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "main.py"))

class TestVersion:
    def test_version_file(self):
        with open(os.path.join(REPO_ROOT, "VERSION")) as f:
            v = f.read().strip()
        assert v == "3.0.0-rc1"

class TestConfigFiles:
    def test_production_artifacts_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "config", "production_artifacts.yaml"))

    def test_model_sets_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "config", "model_sets.yaml"))

    def test_runtime_profiles_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "config", "runtime_profiles.yaml"))

class TestCertification:
    def test_certification_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "production_certification.json"))

    def test_certification_valid_json(self):
        with open(os.path.join(REPO_ROOT, "production_certification.json"), encoding="utf-8") as f:
            c = json.load(f)
        assert "final_verdict" in c
        assert "components" in c

    def test_certification_not_go(self):
        with open(os.path.join(REPO_ROOT, "production_certification.json"), encoding="utf-8") as f:
            c = json.load(f)
        assert "CAVEATS" in c["final_verdict"]

class TestClientDocs:
    def test_client_delivery_note(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_DELIVERY_NOTE.md"))

    def test_client_runbook(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_RUNBOOK.md"))

    def test_client_caveats(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_CAVEATS.md"))

    def test_client_demo_commands(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "docs", "CLIENT_DEMO_COMMANDS.md"))

class TestReadme:
    def test_readme_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "README.md"))

    def test_readme_has_delivery_commands(self):
        path = os.path.join(REPO_ROOT, "README.md")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "main.py" in content or "python" in content

class TestRunbookCommands:
    def test_runbook_has_command(self):
        with open(os.path.join(REPO_ROOT, "docs", "CLIENT_RUNBOOK.md"), encoding="utf-8") as f:
            content = f.read()
        assert "main.py" in content

class TestForbiddenFiles:
    def test_no_local_artifacts_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", ".local_artifacts"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.stdout.strip() == ""

    def test_no_artifact_binaries_tracked(self):
        for ext in ["*.csv", "*.pkl", "*.joblib", "*.cbm", "*.pt", "*.pth"]:
            result = subprocess.run(
                ["git", "ls-files", ext],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            # Only allow .csv in specific source-coded directories
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if "data/" in line or ".local_artifacts/" in line or "outputs/" in line:
                        assert False, f"Forbidden file tracked: {line}"

class TestMissingArtifacts:
    def test_missing_artifacts_produce_caveats(self):
        """System should not crash when artifacts are missing."""
        from artifacts.production_registry import run_production_registry
        cfg = os.path.join(REPO_ROOT, "config", "production_artifacts.yaml")
        result = run_production_registry(cfg)
        assert result["status"] == "REGISTRY_COMPLETE"
        assert result["overall_assessment"] in ("GO_WITH_CAVEATS", "BLOCKED")
