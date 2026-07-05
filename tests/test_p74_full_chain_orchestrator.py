"""P74 — Full Chain Orchestrator (run_full_chain) unit tests."""
from __future__ import annotations

import os

import pytest

from scripts.run_full_chain import (
    FULL_CHAIN_DELIVERY_GO,
    FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
    FULL_CHAIN_DELIVERY_NO_GO,
    _parse_args,
    run_full_chain,
)


# ── CLI Parsing ───────────────────────────────────────────────────────────────


class TestCLIParsing:
    def test_default_args(self):
        args = _parse_args([])
        assert args.profile == "trusted_delivery"
        assert args.fusion_engine == "period_bgew"

    def test_raw_data_flag(self):
        args = _parse_args(["--raw-data", "/path/to/data.csv"])
        assert args.raw_data == "/path/to/data.csv"

    def test_strict_flag(self):
        args = _parse_args(["--strict"])
        assert args.strict is True

    def test_strict_no_leakage_flag(self):
        args = _parse_args(["--strict-no-leakage"])
        assert args.strict_no_leakage is True

    def test_fast_dev_run_flag(self):
        args = _parse_args(["--fast-dev-run"])
        assert args.fast_dev_run is True

    def test_target_dates(self):
        args = _parse_args([
            "--target-start", "2026-02-01",
            "--target-end", "2026-02-28",
        ])
        assert args.target_start == "2026-02-01"
        assert args.target_end == "2026-02-28"

    def test_device_flag(self):
        args = _parse_args(["--device", "gpu"])
        assert args.device == "gpu"

    def test_json_flag(self):
        args = _parse_args(["--json"])
        assert args.json is True

    def test_profile_flag(self):
        args = _parse_args(["--profile", "balanced_candidate"])
        assert args.profile == "balanced_candidate"

    def test_work_dir_flag(self):
        args = _parse_args(["--work-dir", "/tmp/test_run"])
        assert args.work_dir == "/tmp/test_run"


# ── Status Constants ──────────────────────────────────────────────────────────


class TestStatusConstants:
    def test_go_constant(self):
        assert FULL_CHAIN_DELIVERY_GO == "FULL_CHAIN_DELIVERY_GO"

    def test_go_with_caveats(self):
        assert FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS == "FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS"

    def test_no_go(self):
        assert FULL_CHAIN_DELIVERY_NO_GO == "FULL_CHAIN_DELIVERY_NO_GO"


# ── run_full_chain (structural / no-data) ─────────────────────────────────────


class TestRunFullChainStructure:
    def test_returns_dict(self):
        result = run_full_chain(fast_dev_run=True)
        assert isinstance(result, dict)

    def test_has_run_id(self):
        result = run_full_chain(fast_dev_run=True)
        assert "run_id" in result

    def test_has_started_at(self):
        result = run_full_chain(fast_dev_run=True)
        assert "started_at" in result

    def test_has_completed_at(self):
        result = run_full_chain(fast_dev_run=True)
        assert "completed_at" in result

    def test_has_elapsed_seconds(self):
        result = run_full_chain(fast_dev_run=True)
        assert "elapsed_seconds" in result

    def test_has_steps_dict(self):
        result = run_full_chain(fast_dev_run=True)
        assert "steps" in result
        assert isinstance(result["steps"], dict)

    def test_has_step_order(self):
        result = run_full_chain(fast_dev_run=True)
        assert "step_order" in result
        assert isinstance(result["step_order"], list)

    def test_has_overall_status(self):
        result = run_full_chain(fast_dev_run=True)
        assert result["overall_status"] in (
            FULL_CHAIN_DELIVERY_GO,
            FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
            FULL_CHAIN_DELIVERY_NO_GO,
        )

    def test_has_metrics(self):
        result = run_full_chain(fast_dev_run=True)
        assert "metrics" in result

    def test_has_errors_list(self):
        result = run_full_chain(fast_dev_run=True)
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_has_warnings_list(self):
        result = run_full_chain(fast_dev_run=True)
        assert "warnings" in result

    def test_profile_echoed(self):
        result = run_full_chain(profile="trusted_delivery", fast_dev_run=True)
        assert result["profile"] == "trusted_delivery"

    def test_fusion_engine_echoed(self):
        result = run_full_chain(fusion_engine="period_bgew", fast_dev_run=True)
        assert result["fusion_engine"] == "period_bgew"

    def test_step_order_contains_profile_load(self):
        result = run_full_chain(fast_dev_run=True)
        assert "profile_load" in result["step_order"]

    def test_step_order_contains_raw_data_check(self):
        result = run_full_chain(fast_dev_run=True)
        assert "raw_data_check" in result["step_order"]

    def test_step_order_contains_claim_guard(self):
        result = run_full_chain(fast_dev_run=True)
        assert "claim_guard" in result["step_order"]

    def test_no_raw_data_fails_gracefully(self):
        result = run_full_chain(raw_data="/nonexistent/data.csv", fast_dev_run=True)
        assert isinstance(result, dict)
        assert result["overall_status"] in (
            FULL_CHAIN_DELIVERY_NO_GO,
            FULL_CHAIN_DELIVERY_GO_WITH_CAVEATS,
        )

    def test_output_files_dict(self):
        result = run_full_chain(fast_dev_run=True)
        assert "output_files" in result
        assert isinstance(result["output_files"], dict)
