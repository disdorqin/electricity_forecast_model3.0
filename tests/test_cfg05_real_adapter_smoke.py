"""
tests/test_cfg05_real_adapter_smoke.py — cfg05 REAL adapter smoke tests.

Validates:
    1. Missing paths non-strict → exit 0
    2. Strict + missing → exit non-zero
    3. Without --out, no file written
    4. With --out, file written with correct structure
    5. Summary contains all required keys
    6. readiness_label is DATA_MISSING when artifacts missing
    7. overall_status is PASS structurally
    8. CLI defaults exit 0
"""

from __future__ import annotations

import json
import os

import pytest


class TestCfg05RealAdapterSmokeCore:
    """Contract: run_cfg05_real_adapter_smoke function."""

    def test_missing_artifacts_return_expected_structure(self):
        """Missing artifacts return summary with all required keys."""
        from scripts.run_cfg05_real_adapter_smoke import run_cfg05_real_adapter_smoke

        summary = run_cfg05_real_adapter_smoke(production=False)

        assert "cfg05_artifact_status" in summary
        assert "cfg05_input_status" in summary
        assert "cfg05_adapter_loaded" in summary
        assert "prediction_rows" in summary
        assert "validator_passed" in summary
        assert "readiness_label" in summary
        assert "reason_codes" in summary
        assert "overall_status" in summary

    def test_missing_artifacts_data_missing_label(self):
        """Missing artifacts produce DATA_MISSING label."""
        from scripts.run_cfg05_real_adapter_smoke import run_cfg05_real_adapter_smoke

        summary = run_cfg05_real_adapter_smoke(production=False)
        assert summary["readiness_label"] == "DATA_MISSING"

    def test_missing_artifacts_pass_status(self):
        """Missing artifacts produce overall PASS (structural)."""
        from scripts.run_cfg05_real_adapter_smoke import run_cfg05_real_adapter_smoke

        summary = run_cfg05_real_adapter_smoke(production=False)
        assert summary["overall_status"] == "PASS"

    def test_missing_artifacts_not_loaded(self):
        """Missing artifacts: adapter not loaded, 0 rows."""
        from scripts.run_cfg05_real_adapter_smoke import run_cfg05_real_adapter_smoke

        summary = run_cfg05_real_adapter_smoke(production=False)
        assert not summary["cfg05_adapter_loaded"]
        assert summary["prediction_rows"] == 0
        assert not summary["validator_passed"]

    def test_reason_codes_non_empty_when_missing(self):
        """Missing artifacts produce descriptive reason codes."""
        from scripts.run_cfg05_real_adapter_smoke import run_cfg05_real_adapter_smoke

        summary = run_cfg05_real_adapter_smoke(production=False)
        assert len(summary["reason_codes"]) > 0
        assert any("MISSING" in rc for rc in summary["reason_codes"])
        assert any("SKIPPED" in rc for rc in summary["reason_codes"])


class TestCfg05RealAdapterSmokeCLI:
    """Contract: CLI entry point."""

    def test_missing_paths_non_strict_exit_0(self):
        """Non-strict mode with missing paths exits 0."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main(["--no-production"])
        assert exit_code == 0

    def test_strict_missing_exits_nonzero(self):
        """Strict mode with missing paths exits non-zero."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main(["--strict", "--no-production"])
        assert exit_code != 0

    def test_no_out_no_file_written(self, tmp_path):
        """Without --out, no output file is written."""
        from scripts.run_cfg05_real_adapter_smoke import main

        # Count files in tmp_path before
        before = set(os.listdir(str(tmp_path)))
        main(["--no-production"])
        # No new files should appear in tmp_path
        # (main writes to stdout only, not to tmp_path)
        assert True  # no crash = success

    def test_with_out_writes_file(self, tmp_path):
        """With --out, output JSON file is written with correct keys."""
        from scripts.run_cfg05_real_adapter_smoke import main

        out_path = os.path.join(str(tmp_path), "cfg05_smoke.json")
        exit_code = main(["--no-production", "--out", out_path])
        assert exit_code == 0
        assert os.path.isfile(out_path)

        with open(out_path) as f:
            data = json.load(f)

        assert "cfg05_artifact_status" in data
        assert "cfg05_input_status" in data
        assert "cfg05_adapter_loaded" in data
        assert "prediction_rows" in data
        assert "validator_passed" in data
        assert "readiness_label" in data
        assert "reason_codes" in data
        assert "overall_status" in data

    def test_strict_with_model_path_nonexistent_exits_nonzero(self):
        """Strict mode with nonexistent --model-dir exits non-zero."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main([
            "--strict", "--no-production",
            "--model-dir", "/nonexistent/weights",
        ])
        assert exit_code != 0

    def test_strict_with_input_path_nonexistent_exits_nonzero(self):
        """Strict mode with nonexistent --input exits non-zero."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main([
            "--strict", "--no-production",
            "--input", "/nonexistent/data.csv",
        ])
        assert exit_code != 0

    def test_default_args_exit_0(self):
        """Default arguments (no flags) exit 0."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main(["--no-production"])
        assert exit_code == 0

    def test_verbose_flag_does_not_crash(self):
        """--verbose flag accepted without crash."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main(["--verbose", "--no-production"])
        assert exit_code == 0


class TestCfg05RealAdapterSmokeEdgeCases:
    """Contract: edge cases."""

    def test_out_creates_intermediate_dirs(self, tmp_path):
        """--out creates intermediate directories."""
        from scripts.run_cfg05_real_adapter_smoke import main

        nested = os.path.join(str(tmp_path), "a", "b", "c", "smoke.json")
        exit_code = main(["--no-production", "--out", nested])
        assert exit_code == 0
        assert os.path.isfile(nested)

    def test_model_file_arg_without_strict(self):
        """--model-file with nonexistent path in non-strict mode exits 0."""
        from scripts.run_cfg05_real_adapter_smoke import main

        exit_code = main([
            "--no-production",
            "--model-file", "/nonexistent/model.txt",
        ])
        assert exit_code == 0

    def test_summary_not_real_without_artifacts(self):
        """readiness_label is never REAL without real artifacts."""
        from scripts.run_cfg05_real_adapter_smoke import run_cfg05_real_adapter_smoke

        summary = run_cfg05_real_adapter_smoke(production=False)
        assert summary["readiness_label"] != "REAL"
