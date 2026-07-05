"""
tests/test_p60_final_safety_freeze_audit.py — P60 Safety Freeze Audit tests.

Tests 8 groups:
  1. run_audit() returns correct structure
  2. All module existence checks pass
  3. All API contract checks pass
  4. All constants checks pass
  5. main() with --json flag
  6. main() with --strict flag
  7. Summary structure is correct
  8. Overall status is set correctly
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def audit_report():
    """Run the full audit against the real project once per module."""
    from scripts.run_p60_final_safety_freeze_audit import run_audit

    return run_audit()


# ===========================================================================
# 1. Structure
# ===========================================================================


class TestAuditStructure:
    """run_audit() returns the expected top-level structure."""

    def test_returns_dict(self, audit_report):
        assert isinstance(audit_report, dict)

    def test_has_phase_key(self, audit_report):
        assert audit_report["phase"] == "P60"

    def test_has_timestamp_key(self, audit_report):
        ts = audit_report["timestamp"]
        assert isinstance(ts, str)
        assert ts.startswith("2026-")

    def test_has_checks_list(self, audit_report):
        checks = audit_report["checks"]
        assert isinstance(checks, list)
        assert len(checks) == 24

    def test_has_summary_key(self, audit_report):
        assert "summary" in audit_report

    def test_has_overall_status_key(self, audit_report):
        assert "overall_status" in audit_report

    def test_every_check_is_dict_with_required_keys(self, audit_report):
        for c in audit_report["checks"]:
            assert isinstance(c, dict), f"check {c} is not a dict"
            assert "check" in c, f"check missing 'check' key: {c}"
            assert "status" in c, f"check missing 'status' key: {c}"
            assert c["status"] in ("PASS", "FAIL", "WARNING"), (
                f"unexpected status {c['status']!r} in {c['check']}"
            )
            assert "detail" in c, f"check missing 'detail' key: {c['check']}"


# ===========================================================================
# 2. Module existence checks (checks 1-6)
# ===========================================================================


class TestModuleExistence:
    """All module existence checks MUST pass."""

    def _first_six(self, report):
        return report["checks"][:6]

    def test_all_six_pass(self, audit_report):
        for c in self._first_six(audit_report):
            assert c["status"] == "PASS", f"{c['check']}: {c['detail']}"

    def test_safety_leakage_sentinel(self, audit_report):
        assert audit_report["checks"][0]["status"] == "PASS"

    def test_fusion_adaptive_training_days(self, audit_report):
        assert audit_report["checks"][1]["status"] == "PASS"

    def test_delivery_fallback_ladder(self, audit_report):
        assert audit_report["checks"][2]["status"] == "PASS"

    def test_delivery_postflight(self, audit_report):
        assert audit_report["checks"][3]["status"] == "PASS"

    def test_delivery_manifest(self, audit_report):
        assert audit_report["checks"][4]["status"] == "PASS"

    def test_delivery_report(self, audit_report):
        assert audit_report["checks"][5]["status"] == "PASS"


# ===========================================================================
# 3. API contract checks (checks 7-14)
# ===========================================================================


class TestAPIContract:
    """All API contract checks MUST pass."""

    def _contract_checks(self, report):
        return report["checks"][6:14]

    def test_all_eight_pass(self, audit_report):
        for c in self._contract_checks(audit_report):
            assert c["status"] == "PASS", f"{c['check']}: {c['detail']}"

    def test_run_leakage_sentinel_callable(self, audit_report):
        assert audit_report["checks"][6]["status"] == "PASS"

    def test_check_model_leakage_callable(self, audit_report):
        assert audit_report["checks"][7]["status"] == "PASS"

    def test_is_delivery_allowed_callable(self, audit_report):
        assert audit_report["checks"][8]["status"] == "PASS"

    def test_select_complete_training_days_callable(self, audit_report):
        assert audit_report["checks"][9]["status"] == "PASS"

    def test_run_fallback_ladder_callable(self, audit_report):
        assert audit_report["checks"][10]["status"] == "PASS"

    def test_run_postflight_callable(self, audit_report):
        assert audit_report["checks"][11]["status"] == "PASS"

    def test_create_manifest_callable(self, audit_report):
        assert audit_report["checks"][12]["status"] == "PASS"

    def test_generate_delivery_report_callable(self, audit_report):
        assert audit_report["checks"][13]["status"] == "PASS"


# ===========================================================================
# 4. Constants checks (checks 15-18)
# ===========================================================================


class TestConstants:
    """All constants checks MUST pass."""

    def test_corr_threshold(self, audit_report):
        assert audit_report["checks"][14]["status"] == "PASS"

    def test_within_1pct_threshold(self, audit_report):
        assert audit_report["checks"][15]["status"] == "PASS"

    def test_valid_periods_defined(self, audit_report):
        assert audit_report["checks"][16]["status"] == "PASS"

    def test_default_cfg05_floor(self, audit_report):
        assert audit_report["checks"][17]["status"] == "PASS"


# ===========================================================================
# 5. main() with --json
# ===========================================================================


class TestMainJson:
    """main() with --json produces valid JSON on stdout."""

    def test_main_json_produces_valid_json(self, capsys):
        from scripts.run_p60_final_safety_freeze_audit import main

        exit_code = main(["--json"])
        captured = capsys.readouterr()
        assert exit_code == 0
        data = json.loads(captured.out)
        assert data["phase"] == "P60"
        assert "checks" in data
        assert "summary" in data
        assert "overall_status" in data

    def test_main_json_includes_all_checks(self, capsys):
        from scripts.run_p60_final_safety_freeze_audit import main

        main(["--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["checks"]) == 24

    def test_main_json_summary_structure(self, capsys):
        from scripts.run_p60_final_safety_freeze_audit import main

        main(["--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        s = data["summary"]
        assert "total" in s
        assert "passed" in s
        assert "failed" in s
        assert "warnings" in s


# ===========================================================================
# 6. main() with --strict
# ===========================================================================


class TestMainStrict:
    """main() with --strict returns 0 when all checks pass."""

    def test_main_strict_returns_zero_when_all_pass(self):
        from scripts.run_p60_final_safety_freeze_audit import main

        exit_code = main(["--strict", "--json"])
        # Since the real project should have all checks pass, --strict -> 0
        assert exit_code == 0

    def test_main_strict_returns_nonzero_on_failure(self):
        from scripts.run_p60_final_safety_freeze_audit import main

        # Use a tmp work_dir with no modules; checks will fail
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(["--strict", "--work-dir", tmp, "--json"])
            assert exit_code == 1


# ===========================================================================
# 7. Summary structure
# ===========================================================================


class TestSummary:
    """Audit report summary has correct structure and totals."""

    def test_summary_has_all_keys(self, audit_report):
        s = audit_report["summary"]
        assert "total" in s
        assert "passed" in s
        assert "failed" in s
        assert "warnings" in s

    def test_total_is_24(self, audit_report):
        assert audit_report["summary"]["total"] == 24

    def test_passed_failed_warnings_sum_to_total(self, audit_report):
        s = audit_report["summary"]
        assert s["passed"] + s["failed"] + s["warnings"] == s["total"]

    def test_summary_values_are_nonnegative(self, audit_report):
        s = audit_report["summary"]
        assert s["total"] >= 0
        assert s["passed"] >= 0
        assert s["failed"] >= 0
        assert s["warnings"] >= 0

    def test_summary_values_are_ints(self, audit_report):
        s = audit_report["summary"]
        assert isinstance(s["total"], int)
        assert isinstance(s["passed"], int)
        assert isinstance(s["failed"], int)
        assert isinstance(s["warnings"], int)


# ===========================================================================
# 8. Overall status
# ===========================================================================


class TestOverallStatus:
    """Overall status reflects audit outcome."""

    def test_overall_status_is_string(self, audit_report):
        assert isinstance(audit_report["overall_status"], str)

    def test_overall_status_valid_value(self, audit_report):
        valid = {"P60_SAFETY_FREEZE_PASS", "P60_SAFETY_FREEZE_FAILED"}
        assert audit_report["overall_status"] in valid

    def test_overall_pass_when_no_failures(self, audit_report):
        if audit_report["summary"]["failed"] == 0:
            assert audit_report["overall_status"] == "P60_SAFETY_FREEZE_PASS"

    def test_overall_fail_when_any_failure(self, audit_report):
        if audit_report["summary"]["failed"] > 0:
            assert audit_report["overall_status"] == "P60_SAFETY_FREEZE_FAILED"


# ===========================================================================
# 9. Edge-case / sanity tests
# ===========================================================================


class TestEdgeCases:
    """Audit behaves correctly with missing project, AST issues, etc."""

    def test_run_audit_on_empty_dir_has_exact_checks(self):
        """Running against an empty directory produces 24 checks."""
        from scripts.run_p60_final_safety_freeze_audit import run_audit

        with tempfile.TemporaryDirectory() as tmp:
            report = run_audit(work_dir=tmp)
            assert len(report["checks"]) == 24

    def test_run_audit_on_empty_dir_all_fail(self):
        """When modules are missing, module existence checks fail."""
        from scripts.run_p60_final_safety_freeze_audit import run_audit

        with tempfile.TemporaryDirectory() as tmp:
            report = run_audit(work_dir=tmp)
            # First 6 checks are file existence - all should FAIL
            for c in report["checks"][:6]:
                assert c["status"] == "FAIL", f"{c['check']} should FAIL on empty dir"

    def test_custom_work_dir_respected(self):
        """--work-dir flag is correctly passed to run_audit()."""
        from scripts.run_p60_final_safety_freeze_audit import run_audit

        with tempfile.TemporaryDirectory() as tmp:
            report = run_audit(work_dir=tmp)
            assert report["summary"]["failed"] > 0

    def test_main_accepts_empty_argv(self):
        """main() with no arguments returns 0."""
        from scripts.run_p60_final_safety_freeze_audit import main

        exit_code = main([])
        assert exit_code == 0

    def test_main_work_dir_flag(self):
        """main() with --work-dir pointing to empty dir returns 0 (no --strict)."""
        from scripts.run_p60_final_safety_freeze_audit import main

        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(["--work-dir", tmp])
            assert exit_code == 0  # no --strict, so always 0


# ===========================================================================
# 10. Helper function tests (AST internals)
# ===========================================================================


class TestASTHelpers:
    """Direct tests for the AST inspection helpers."""

    def _import_helpers(self):
        from scripts.run_p60_final_safety_freeze_audit import (
            _ast_extract_value,
            _count_function_params,
            _function_exists_in,
            _function_returns_int,
            _get_module_constant,
        )

        return _ast_extract_value, _count_function_params, _function_exists_in, _function_returns_int, _get_module_constant

    def test_ast_extract_value_constant(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        node = ast.Constant(value=42)
        assert _ast_extract_value(node) == 42

    def test_ast_extract_value_list(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        node = ast.List(elts=[ast.Constant(value="a"), ast.Constant(value="b")])
        assert _ast_extract_value(node) == ["a", "b"]

    def test_ast_extract_value_nested_list(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        inner = ast.List(elts=[ast.Constant(value=1)])
        node = ast.List(elts=[inner])
        assert _ast_extract_value(node) == [[1]]

    def test_ast_extract_value_dict(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        node = ast.Dict(
            keys=[ast.Constant(value="k")],
            values=[ast.Constant(value="v")],
        )
        assert _ast_extract_value(node) == {"k": "v"}

    def test_ast_extract_value_negated_number(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=5))
        assert _ast_extract_value(node) == -5

    def test_ast_extract_value_name(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        node = ast.Name(id="None")
        assert _ast_extract_value(node) == "None"

    def test_ast_extract_value_unknown(self):
        _ast_extract_value, _, _, _, _ = self._import_helpers()
        import ast

        node = ast.Lambda()
        assert _ast_extract_value(node) is None

    def test_get_module_constant_normal(self, tmp_path):
        _, _, _, _, _get_module_constant = self._import_helpers()

        src = tmp_path / "demo.py"
        src.write_text("MY_CONST = 42\n", encoding="utf-8")
        assert _get_module_constant(str(src), "MY_CONST") == 42
        assert _get_module_constant(str(src), "NONEXIST") is None

    def test_get_module_constant_annotated(self, tmp_path):
        _, _, _, _, _get_module_constant = self._import_helpers()

        src = tmp_path / "annot.py"
        src.write_text("VAL: int = 99\n", encoding="utf-8")
        assert _get_module_constant(str(src), "VAL") == 99

    def test_get_module_constant_nonexistent_file(self):
        _, _, _, _, _get_module_constant = self._import_helpers()
        assert _get_module_constant("/nonexistent/path.py", "X") is None

    def test_count_function_params_simple(self, tmp_path):
        _, _count_function_params, _, _, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def foo(a, b, c): pass\n", encoding="utf-8")
        assert _count_function_params(str(src), "foo") == 3

    def test_count_function_params_with_self(self, tmp_path):
        _, _count_function_params, _, _, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("class X:\n    def method(self, a, b): pass\n", encoding="utf-8")
        assert _count_function_params(str(src), "method") == 2

    def test_count_function_params_no_args(self, tmp_path):
        _, _count_function_params, _, _, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def bar(): pass\n", encoding="utf-8")
        assert _count_function_params(str(src), "bar") == 0

    def test_count_function_params_not_found(self, tmp_path):
        _, _count_function_params, _, _, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def foo(): pass\n", encoding="utf-8")
        assert _count_function_params(str(src), "nope") is None

    def test_function_exists_in(self, tmp_path):
        _, _, _function_exists_in, _, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def my_func(): pass\n", encoding="utf-8")
        assert _function_exists_in(str(src), "my_func") is True
        assert _function_exists_in(str(src), "other") is False

    def test_function_returns_int_true(self, tmp_path):
        _, _, _, _function_returns_int, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def main() -> int: return 0\n", encoding="utf-8")
        assert _function_returns_int(str(src), "main") is True

    def test_function_returns_int_false(self, tmp_path):
        _, _, _, _function_returns_int, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def main() -> str: return ''\n", encoding="utf-8")
        assert _function_returns_int(str(src), "main") is False

    def test_function_returns_int_no_annotation(self, tmp_path):
        _, _, _, _function_returns_int, _ = self._import_helpers()

        src = tmp_path / "mod.py"
        src.write_text("def main(): pass\n", encoding="utf-8")
        assert _function_returns_int(str(src), "main") is False

    def test_function_returns_int_nonexistent_file(self):
        _, _, _, _function_returns_int, _ = self._import_helpers()
        assert _function_returns_int("/nonexistent.py", "main") is False
