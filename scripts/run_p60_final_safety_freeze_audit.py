"""
scripts/run_p60_final_safety_freeze_audit.py — P60: Safety freeze meta-audit.

Runs 24 safety checks across all P52-P57 modules without importing them
(the audit verifies that modules exist, expose the right APIs, and that
their contracts are internally consistent).

Checks
------
Module existence (6):       1-6
API contract (8):           7-14
Constants (4):              15-18
Consistency (4):            19-22
Pipeline integration (2):   23-24

Usage::

    python -m scripts.run_p60_final_safety_freeze_audit
    python -m scripts.run_p60_final_safety_freeze_audit --json
    python -m scripts.run_p60_final_safety_freeze_audit --strict
    python -m scripts.run_p60_final_safety_freeze_audit --work-dir /path/to/project
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AST helpers (meta-audit uses source inspection, never imports real modules)
# ---------------------------------------------------------------------------


def _ast_extract_value(node: ast.AST):
    """Extract a Python literal value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_extract_value(elt) for elt in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_extract_value(node.operand)
        return -inner if isinstance(inner, (int, float)) else None
    if isinstance(node, ast.Tuple):
        return tuple(_ast_extract_value(elt) for elt in node.elts)
    if isinstance(node, ast.Set):
        return {_ast_extract_value(elt) for elt in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _ast_extract_value(k): _ast_extract_value(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }
    if isinstance(node, ast.Name):
        # Treat bare names as string references (e.g. True, False, None)
        return node.id
    return None


def _get_module_constant(path: str, name: str):
    """Return the value of a module-level assignment or annotated assignment.

    Handles both ``NAME = value`` and ``NAME: type = value``.
    Returns ``None`` if the assignment is not found or unparseable.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _ast_extract_value(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                if node.value is not None:  # may be bare annotation
                    return _ast_extract_value(node.value)
    return None


def _count_function_params(path: str, func_name: str) -> int | None:
    """Count callable (non-self) parameters of a top-level function.

    Returns ``None`` if the file or function is not found.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            args = node.args.args
            # Exclude ``self`` (methods)
            if args and args[0].arg == "self":
                args = args[1:]
            # Exclude ``*args`` / ``**kwargs`` from count
            return len(args)
    return None


def _function_exists_in(path: str, func_name: str) -> bool:
    """Check whether *func_name* is defined as a top-level function in *path*."""
    return _count_function_params(path, func_name) is not None


def _function_returns_int(path: str, func_name: str) -> bool:
    """Check whether *func_name* has a ``-> int`` return annotation."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            if node.returns is None:
                return False
            # ``-> int``  ->  ast.Name(id='int')
            if isinstance(node.returns, ast.Name) and node.returns.id == "int":
                return True
            # ``-> "int"`` (string annotation edge case)
            if isinstance(node.returns, ast.Constant) and node.returns.value == "int":
                return True
    return False


def _file_contains_text(path: str, *texts: str) -> bool:
    """Check whether *path* exists and contains all given substrings."""
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    return all(t in content for t in texts)


# ---------------------------------------------------------------------------
# Audit result helpers
# ---------------------------------------------------------------------------

_CHECK_RESULT = dict  # alias: {"check": str, "status": str, "detail": str}

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARNING"

_OVERALL_PASS = "P60_SAFETY_FREEZE_PASS"
_OVERALL_FAIL = "P60_SAFETY_FREEZE_FAILED"


def _ok(name: str, detail: str = "") -> _CHECK_RESULT:
    return {"check": name, "status": _PASS, "detail": detail}


def _fail(name: str, detail: str = "") -> _CHECK_RESULT:
    return {"check": name, "status": _FAIL, "detail": detail}


def _warn(name: str, detail: str = "") -> _CHECK_RESULT:
    return {"check": name, "status": _WARN, "detail": detail}


# ---------------------------------------------------------------------------
# Audit class
# ---------------------------------------------------------------------------


class SafetyFreezeAudit:
    """Meta-audit that inspects the project tree for P52-P57 contract health.

    Parameters
    ----------
    work_dir : str, optional
        Project root directory.  Auto-resolved from the script location when
        ``None``.
    """

    def __init__(self, work_dir: str | None = None) -> None:
        if work_dir is not None:
            self._root = os.path.abspath(work_dir)
        else:
            # Resolve from scripts/run_p60_*  ->  project root
            self._root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._checks: list[_CHECK_RESULT] = []

    # -- path helpers -------------------------------------------------------

    def _p(self, *parts: str) -> str:
        return os.path.join(self._root, *parts)

    def _module_path(self, dotted: str) -> str:
        """Convert ``safety.leakage_sentinel`` to absolute file path."""
        return self._p(*(dotted.split("."))) + ".py"

    # -- check methods ------------------------------------------------------

    # Module existence (checks 1-6) ----------------------------------------

    def check_01_leakage_sentinel_exists(self) -> _CHECK_RESULT:
        """safety/leakage_sentinel.py exists."""
        p = self._p("safety", "leakage_sentinel.py")
        ok = os.path.isfile(p)
        return (_ok if ok else _fail)(
            "safety/leakage_sentinel.py exists",
            os.path.relpath(p, self._root) if ok else "MISSING",
        )

    def check_02_adaptive_training_days_exists(self) -> _CHECK_RESULT:
        """fusion/adaptive_training_days.py exists."""
        p = self._p("fusion", "adaptive_training_days.py")
        ok = os.path.isfile(p)
        return (_ok if ok else _fail)(
            "fusion/adaptive_training_days.py exists",
            os.path.relpath(p, self._root) if ok else "MISSING",
        )

    def check_03_fallback_ladder_exists(self) -> _CHECK_RESULT:
        """delivery/fallback_ladder.py exists."""
        p = self._p("delivery", "fallback_ladder.py")
        ok = os.path.isfile(p)
        return (_ok if ok else _fail)(
            "delivery/fallback_ladder.py exists",
            os.path.relpath(p, self._root) if ok else "MISSING",
        )

    def check_04_postflight_exists(self) -> _CHECK_RESULT:
        """delivery/postflight.py exists."""
        p = self._p("delivery", "postflight.py")
        ok = os.path.isfile(p)
        return (_ok if ok else _fail)(
            "delivery/postflight.py exists",
            os.path.relpath(p, self._root) if ok else "MISSING",
        )

    def check_05_manifest_exists(self) -> _CHECK_RESULT:
        """delivery/manifest.py exists."""
        p = self._p("delivery", "manifest.py")
        ok = os.path.isfile(p)
        return (_ok if ok else _fail)(
            "delivery/manifest.py exists",
            os.path.relpath(p, self._root) if ok else "MISSING",
        )

    def check_06_report_exists(self) -> _CHECK_RESULT:
        """delivery/report.py exists."""
        p = self._p("delivery", "report.py")
        ok = os.path.isfile(p)
        return (_ok if ok else _fail)(
            "delivery/report.py exists",
            os.path.relpath(p, self._root) if ok else "MISSING",
        )

    # API contract (checks 7-14) -------------------------------------------

    def check_07_run_leakage_sentinel_callable(self) -> _CHECK_RESULT:
        """safety.leakage_sentinel.run_leakage_sentinel callable with 3 args."""
        src = self._module_path("safety.leakage_sentinel")
        n = _count_function_params(src, "run_leakage_sentinel")
        if n is None:
            return _fail("run_leakage_sentinel callable with 3 args", "function not found")
        if n >= 3:
            return _ok("run_leakage_sentinel callable with 3 args", f"found with {n} params")
        return _fail("run_leakage_sentinel callable with 3 args", f"only {n} params found")

    def check_08_check_model_leakage_callable(self) -> _CHECK_RESULT:
        """safety.leakage_sentinel.check_model_leakage callable with 4 args."""
        src = self._module_path("safety.leakage_sentinel")
        n = _count_function_params(src, "check_model_leakage")
        if n is None:
            return _fail("check_model_leakage callable with 4 args", "function not found")
        if n >= 4:
            return _ok("check_model_leakage callable with 4 args", f"found with {n} params")
        return _fail("check_model_leakage callable with 4 args", f"only {n} params found")

    def check_09_is_delivery_allowed_callable(self) -> _CHECK_RESULT:
        """safety.leakage_sentinel.is_delivery_allowed callable with 3 args."""
        src = self._module_path("safety.leakage_sentinel")
        n = _count_function_params(src, "is_delivery_allowed")
        if n is None:
            return _fail("is_delivery_allowed callable with 3 args", "function not found")
        if n >= 3:
            return _ok("is_delivery_allowed callable with 3 args", f"found with {n} params")
        return _fail("is_delivery_allowed callable with 3 args", f"only {n} params found")

    def check_10_select_complete_training_days_callable(self) -> _CHECK_RESULT:
        """fusion.adaptive_training_days.select_complete_training_days is callable."""
        src = self._module_path("fusion.adaptive_training_days")
        ok = _function_exists_in(src, "select_complete_training_days")
        return (_ok if ok else _fail)(
            "select_complete_training_days is callable",
            "function found" if ok else "function MISSING",
        )

    def check_11_run_fallback_ladder_callable(self) -> _CHECK_RESULT:
        """delivery.fallback_ladder.run_fallback_ladder callable with 5 args."""
        src = self._module_path("delivery.fallback_ladder")
        n = _count_function_params(src, "run_fallback_ladder")
        if n is None:
            return _fail("run_fallback_ladder callable with 5 args", "function not found")
        if n >= 5:
            return _ok("run_fallback_ladder callable with 5 args", f"found with {n} params")
        return _fail("run_fallback_ladder callable with 5 args", f"only {n} params found")

    def check_12_run_postflight_callable(self) -> _CHECK_RESULT:
        """delivery.postflight.run_postflight callable with 3 args."""
        src = self._module_path("delivery.postflight")
        n = _count_function_params(src, "run_postflight")
        if n is None:
            return _fail("run_postflight callable with 3 args", "function not found")
        if n >= 3:
            return _ok("run_postflight callable with 3 args", f"found with {n} params")
        return _fail("run_postflight callable with 3 args", f"only {n} params found")

    def check_13_create_manifest_callable(self) -> _CHECK_RESULT:
        """delivery.manifest.create_manifest is callable."""
        src = self._module_path("delivery.manifest")
        ok = _function_exists_in(src, "create_manifest")
        return (_ok if ok else _fail)(
            "create_manifest is callable",
            "function found" if ok else "function MISSING",
        )

    def check_14_generate_delivery_report_callable(self) -> _CHECK_RESULT:
        """delivery.report.generate_delivery_report is callable."""
        src = self._module_path("delivery.report")
        ok = _function_exists_in(src, "generate_delivery_report")
        return (_ok if ok else _fail)(
            "generate_delivery_report is callable",
            "function found" if ok else "function MISSING",
        )

    # Constants (checks 15-18) ---------------------------------------------

    def check_15_corr_threshold(self) -> _CHECK_RESULT:
        """Leakage sentinel has CORR_THRESHOLD == 0.995."""
        src = self._module_path("safety.leakage_sentinel")
        val = _get_module_constant(src, "CORR_THRESHOLD")
        if val is None:
            return _fail("CORR_THRESHOLD == 0.995", "CORR_THRESHOLD not found")
        ok = val == 0.995
        return (_ok if ok else _fail)(
            "CORR_THRESHOLD == 0.995",
            f"found CORR_THRESHOLD = {val!r}" if not ok else "CORR_THRESHOLD = 0.995",
        )

    def check_16_within_1pct_threshold(self) -> _CHECK_RESULT:
        """Leakage sentinel has WITHIN_1PCT_THRESHOLD == 0.80."""
        src = self._module_path("safety.leakage_sentinel")
        val = _get_module_constant(src, "WITHIN_1PCT_THRESHOLD")
        if val is None:
            return _fail("WITHIN_1PCT_THRESHOLD == 0.80", "WITHIN_1PCT_THRESHOLD not found")
        ok = val == 0.80
        return (_ok if ok else _fail)(
            "WITHIN_1PCT_THRESHOLD == 0.80",
            f"found WITHIN_1PCT_THRESHOLD = {val!r}" if not ok else "WITHIN_1PCT_THRESHOLD = 0.80",
        )

    def check_17_valid_periods_defined(self) -> _CHECK_RESULT:
        """Regime BGEW has VALID_PERIODS defined."""
        src = self._module_path("fusion.trust_gated_regime_bgew")
        val = _get_module_constant(src, "VALID_PERIODS")
        if val is None:
            return _fail("VALID_PERIODS defined", "VALID_PERIODS not found in trust_gated_regime_bgew")
        return _ok("VALID_PERIODS defined", f"found VALID_PERIODS = {val}")

    def check_18_default_cfg05_floor(self) -> _CHECK_RESULT:
        """Regime BGEW has DEFAULT_CFG05_FLOOR == 0.30."""
        src = self._module_path("fusion.trust_gated_regime_bgew")
        val = _get_module_constant(src, "DEFAULT_CFG05_FLOOR")
        if val is None:
            return _fail("DEFAULT_CFG05_FLOOR == 0.30", "DEFAULT_CFG05_FLOOR not found")
        ok = val == 0.30
        return (_ok if ok else _fail)(
            "DEFAULT_CFG05_FLOOR == 0.30",
            f"found DEFAULT_CFG05_FLOOR = {val!r}" if not ok else "DEFAULT_CFG05_FLOOR = 0.30",
        )

    # Consistency (checks 19-22) -------------------------------------------

    def check_19_runner_script_with_p57_flags(self) -> _CHECK_RESULT:
        """P47/P57 runner exists with --fusion-engine, --strict-no-leakage, etc."""
        runner = self._p("scripts", "run_delivery_local_chain.py")
        if not os.path.isfile(runner):
            return _fail("runner script with P57 flags", "scripts/run_delivery_local_chain.py MISSING")
        has_flags = _file_contains_text(
            runner,
            "--fusion-engine",
            "--strict-no-leakage",
            "--allow-degraded",
            "--profile",
        )
        if not has_flags:
            return _fail(
                "runner script with P57 flags",
                "runner exists but missing one or more expected CLI flags",
            )
        return _ok("runner script with P57 flags", "runner exists with --fusion-engine, --strict-no-leakage, etc.")

    def check_20_test_files_exist(self) -> _CHECK_RESULT:
        """Test files exist for P52-P57."""
        expected = [
            self._p("tests", "test_p52_adaptive_training_days.py"),
            self._p("tests", "test_p53_leakage_sentinel.py"),
            self._p("tests", "test_p54_fallback_ladder.py"),
            self._p("tests", "test_p55_postflight_manifest_report.py"),
            self._p("tests", "test_p56_trust_gated_regime_bgew.py"),
        ]
        missing = [os.path.relpath(p, self._root) for p in expected if not os.path.isfile(p)]
        if missing:
            return _fail("test files exist for P52-P57", f"missing: {missing}")
        return _ok("test files exist for P52-P57", "all 5 test files present")

    def check_21_report_docs_exist(self) -> _CHECK_RESULT:
        """Report docs exist for P52-P57."""
        expected = [
            self._p("docs", "reports", "p52_adaptive_training_days_report.md"),
            self._p("docs", "reports", "p53_leakage_sentinel_report.md"),
            self._p("docs", "reports", "p54_fallback_ladder_report.md"),
            self._p("docs", "reports", "p55_postflight_manifest_report.md"),
            self._p("docs", "reports", "p56_trust_gated_regime_bgew_report.md"),
        ]
        missing = [os.path.relpath(p, self._root) for p in expected if not os.path.isfile(p)]
        if missing:
            return _fail("report docs exist for P52-P57", f"missing: {missing}")
        return _ok("report docs exist for P52-P57", "all 5 report docs present")

    def check_22_stage3_suspect_leakage_in_config(self) -> _CHECK_RESULT:
        """Stage3 listed as SUSPECT_LEAKAGE in any source config."""
        config_yaml = self._p("config", "fusion_profiles.yaml")
        if not os.path.isfile(config_yaml):
            return _fail("Stage3 is SUSPECT_LEAKAGE in config", "config/fusion_profiles.yaml MISSING")
        found = _file_contains_text(config_yaml, "stage3_business_fixed", "SUSPECT_LEAKAGE")
        if found:
            return _ok("Stage3 is SUSPECT_LEAKAGE in config", "found in config/fusion_profiles.yaml")
        return _fail(
            "Stage3 is SUSPECT_LEAKAGE in config",
            "stage3_business_fixed: SUSPECT_LEAKAGE not found in config files",
        )

    # Pipeline integration (checks 23-24) ----------------------------------

    def check_23_valid_periods_has_right_periods(self) -> _CHECK_RESULT:
        """trust_gated_regime_bgew VALID_PERIODS contains ['1_8', '9_16', '17_24']."""
        src = self._module_path("fusion.trust_gated_regime_bgew")
        val = _get_module_constant(src, "VALID_PERIODS")
        if val is None:
            return _fail("VALID_PERIODS contains correct periods", "VALID_PERIODS not found")
        expected = ["1_8", "9_16", "17_24"]
        if val == expected:
            return _ok("VALID_PERIODS contains correct periods", f"VALID_PERIODS = {val}")
        return _fail("VALID_PERIODS contains correct periods", f"expected {expected}, got {val}")

    def check_24_runner_main_returns_int(self) -> _CHECK_RESULT:
        """Runner main() returns int (-> int annotation)."""
        runner = self._p("scripts", "run_delivery_local_chain.py")
        if not os.path.isfile(runner):
            return _fail("runner main() returns int", "scripts/run_delivery_local_chain.py MISSING")
        ok = _function_returns_int(runner, "main")
        return (_ok if ok else _fail)(
            "runner main() returns int",
            "main() has -> int annotation" if ok else "main() missing -> int annotation",
        )

    # -- run all -----------------------------------------------------------

    def run_all(self) -> list[dict]:
        """Execute all 24 checks and return a list of results."""
        self._checks = [
            self.check_01_leakage_sentinel_exists(),
            self.check_02_adaptive_training_days_exists(),
            self.check_03_fallback_ladder_exists(),
            self.check_04_postflight_exists(),
            self.check_05_manifest_exists(),
            self.check_06_report_exists(),
            self.check_07_run_leakage_sentinel_callable(),
            self.check_08_check_model_leakage_callable(),
            self.check_09_is_delivery_allowed_callable(),
            self.check_10_select_complete_training_days_callable(),
            self.check_11_run_fallback_ladder_callable(),
            self.check_12_run_postflight_callable(),
            self.check_13_create_manifest_callable(),
            self.check_14_generate_delivery_report_callable(),
            self.check_15_corr_threshold(),
            self.check_16_within_1pct_threshold(),
            self.check_17_valid_periods_defined(),
            self.check_18_default_cfg05_floor(),
            self.check_19_runner_script_with_p57_flags(),
            self.check_20_test_files_exist(),
            self.check_21_report_docs_exist(),
            self.check_22_stage3_suspect_leakage_in_config(),
            self.check_23_valid_periods_has_right_periods(),
            self.check_24_runner_main_returns_int(),
        ]
        return self._checks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_audit(work_dir: str | None = None) -> dict[str, Any]:
    """Run the full P60 safety freeze meta-audit.

    Parameters
    ----------
    work_dir : str, optional
        Project root directory.  Auto-resolved when ``None``.

    Returns
    -------
    dict
        Audit report with keys: ``phase``, ``timestamp``, ``checks``,
        ``summary``, ``overall_status``.
    """
    auditor = SafetyFreezeAudit(work_dir=work_dir)
    checks = auditor.run_all()

    passed = sum(1 for c in checks if c["status"] == _PASS)
    failed = sum(1 for c in checks if c["status"] == _FAIL)
    warnings = sum(1 for c in checks if c["status"] == _WARN)
    total = len(checks)

    overall = _OVERALL_PASS if failed == 0 else _OVERALL_FAIL

    return {
        "phase": "P60",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
        },
        "overall_status": overall,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P60: Safety freeze meta-audit (24 checks across P52-P57)."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Project root directory (auto-detected when omitted).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output report as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Return non-zero exit code if any check FAILs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = run_audit(work_dir=args.work_dir)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human_report(report)

    if args.strict and report["overall_status"] == _OVERALL_FAIL:
        return 1
    return 0


def _print_human_report(report: dict[str, Any]) -> None:
    """Print a human-readable audit summary."""
    checks = report["checks"]
    summary = report["summary"]

    print("=" * 60)
    print(f"  P60 — Safety Freeze Audit  ({report['timestamp']})")
    print("=" * 60)

    for c in checks:
        status_symbol = "PASS" if c["status"] == _PASS else "FAIL" if c["status"] == _FAIL else "WARN"
        print(f"  [{status_symbol}] {c['check']}")
        if c.get("detail"):
            print(f"         {c['detail']}")

    print()
    print(f"  Total:   {summary['total']}")
    print(f"  Passed:  {summary['passed']}")
    print(f"  Failed:  {summary['failed']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Status:  {report['overall_status']}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
