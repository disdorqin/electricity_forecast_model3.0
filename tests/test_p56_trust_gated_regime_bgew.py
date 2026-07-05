"""
tests/test_p56_trust_gated_regime_bgew.py — Contract tests for P56
Trust-Gated Adaptive Regime BGEW.

Tests cover:
    1. TRUSTED models allowed through gate
    2. SUSPECT_LEAKAGE blocked
    3. CONSERVATIVE_QUARANTINE blocked in trusted_delivery
    4. CONSERVATIVE_QUARANTINE allowed in balanced_candidate
    5. Regime classification: normal
    6. Regime classification: negative_risk
    7. Regime classification: low_price
    8. Regime classification: high_spike
    9. Weight normalization respects min/max bounds
    10. cfg05_floor enforced
    11. Falls back to period_bgew when insufficient regime training days
    12. Falls back to equal_weight
    13. Builds valid 24H output
    14. Empty trusted models handled
    15. All warnings captured
    16. Weights within [0.05, 0.75] in full pipeline
    17. Error handling: empty ledgers, missing target predictions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion.trust_gated_regime_bgew import (
    # Public API
    run_trust_gated_regime_bgew,
    classify_regime,
    # Constants
    ALL_REGIMES,
    REGIME_NORMAL,
    REGIME_LOW_PRICE,
    REGIME_NEGATIVE_RISK,
    REGIME_HIGH_SPIKE,
    TRUST_STATE_TRUSTED,
    TRUST_STATE_SUSPECT_LEAKAGE,
    TRUST_STATE_CONSERVATIVE_QUARANTINE,
    # Helpers
    _apply_trust_gate,
    _normalize_weights,
    _build_24h_output,
    _compute_smape,
)


# ── Test data helpers ────────────────────────────────────────────────


def _make_prediction_ledger(
    n_training_days: int = 15,
    models: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic prediction ledger.

    Generates *n_training_days* of 24-hour predictions ending on
    2026-07-03 so training is strictly before target date 2026-07-04.
    """
    if models is None:
        models = ["cfg05", "catboost_spike_residual"]

    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-06-19")
    rows: list[pd.DataFrame] = []

    for day_offset in range(n_training_days):
        bd = base_date + pd.Timedelta(days=day_offset)
        target_day = bd
        timestamps = pd.date_range(f"{bd.date()} 01:00", periods=24, freq="h")

        for model in models:
            prices = rng.uniform(80, 250, 24)
            rows.append(pd.DataFrame({
                "task": ["dayahead"] * 24,
                "model_name": [model] * 24,
                "target_day": [target_day] * 24,
                "business_day": [bd] * 24,
                "ds": timestamps,
                "hour_business": list(range(1, 25)),
                "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                "y_pred": prices,
            }))

    return pd.concat(rows, ignore_index=True)


def _make_actual_ledger(
    n_training_days: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic actual ledger."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2026-06-19")
    rows: list[pd.DataFrame] = []

    for day_offset in range(n_training_days):
        bd = base_date + pd.Timedelta(days=day_offset)
        target_day = bd
        timestamps = pd.date_range(f"{bd.date()} 01:00", periods=24, freq="h")

        prices = rng.uniform(80, 250, 24)
        rows.append(pd.DataFrame({
            "task": ["dayahead"] * 24,
            "target_day": [target_day] * 24,
            "business_day": [bd] * 24,
            "ds": timestamps,
            "hour_business": list(range(1, 25)),
            "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
            "y_true": prices,
        }))

    return pd.concat(rows, ignore_index=True)


def _make_target_preds(
    target_date: str = "2026-07-04",
    models: list[str] | None = None,
    seed: int = 99,
) -> pd.DataFrame:
    """Create synthetic target-day predictions."""
    if models is None:
        models = ["cfg05", "catboost_spike_residual"]

    rng = np.random.default_rng(seed)
    target = pd.Timestamp(target_date)
    timestamps = pd.date_range(f"{target_date} 01:00", periods=24, freq="h")
    rows: list[pd.DataFrame] = []

    for model in models:
        prices = rng.uniform(80, 250, 24)
        rows.append(pd.DataFrame({
            "task": ["dayahead"] * 24,
            "model_name": [model] * 24,
            "target_day": [target] * 24,
            "business_day": [target - pd.Timedelta(days=1)] * 24,
            "ds": timestamps,
            "hour_business": list(range(1, 25)),
            "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
            "y_pred": prices,
        }))

    return pd.concat(rows, ignore_index=True)


# ── 1. Trust Gate ────────────────────────────────────────────────────


class TestTrustGate:
    """Contract: _apply_trust_gate basic behavior."""

    def test_trusted_models_allowed(self):
        """TRUSTED models always pass the gate."""
        allowed, blocked, _warnings = _apply_trust_gate(
            model_names=["cfg05", "catboost_spike_residual"],
            trusted_models=["cfg05", "catboost_spike_residual"],
            profile_name="trusted_delivery",
        )
        assert "cfg05" in allowed
        assert "catboost_spike_residual" in allowed
        assert len(blocked) == 0

    def test_suspect_leakage_blocked(self):
        """SUSPECT_LEAKAGE models are blocked."""
        allowed, blocked, _warnings = _apply_trust_gate(
            model_names=["cfg05", "stage3"],
            trusted_models=["cfg05"],
            profile_name="trusted_delivery",
            model_trust_states={"stage3": TRUST_STATE_SUSPECT_LEAKAGE},
        )
        assert "cfg05" in allowed
        assert "stage3" in blocked

    def test_conservative_quarantine_blocked_in_trusted_delivery(self):
        """CONSERVATIVE_QUARANTINE blocked in trusted_delivery."""
        allowed, blocked, _warnings = _apply_trust_gate(
            model_names=["cfg05", "catboost_spike_residual"],
            trusted_models=["cfg05"],
            profile_name="trusted_delivery",
            model_trust_states={
                "catboost_spike_residual": TRUST_STATE_CONSERVATIVE_QUARANTINE,
            },
        )
        assert "cfg05" in allowed
        assert "catboost_spike_residual" in blocked

    def test_conservative_quarantine_allowed_in_balanced(self):
        """CONSERVATIVE_QUARANTINE allowed in balanced_candidate."""
        allowed, blocked, _warnings = _apply_trust_gate(
            model_names=["cfg05", "catboost_spike_residual"],
            trusted_models=["cfg05"],
            profile_name="balanced_candidate",
            model_trust_states={
                "catboost_spike_residual": TRUST_STATE_CONSERVATIVE_QUARANTINE,
            },
        )
        assert "cfg05" in allowed
        assert "catboost_spike_residual" in allowed


# ── 2. Regime Classification ─────────────────────────────────────────


class TestRegimeClassification:
    """Contract: classify_regime produces correct labels."""

    def test_normal(self):
        """High median below p90, positive prices -> normal."""
        assert classify_regime(200.0, 300.0, 150.0) == REGIME_NORMAL

    def test_negative_risk_via_ensemble(self):
        """Negative ensemble median -> negative_risk."""
        assert classify_regime(-10.0, 200.0, 50.0) == REGIME_NEGATIVE_RISK

    def test_negative_risk_via_historical(self):
        """Negative historical median -> negative_risk."""
        assert classify_regime(150.0, 200.0, -5.0) == REGIME_NEGATIVE_RISK

    def test_low_price(self):
        """Ensemble median < 100 CNY -> low_price."""
        assert classify_regime(50.0, 200.0, 80.0) == REGIME_LOW_PRICE

    def test_high_spike(self):
        """Ensemble median > recent_p90 -> high_spike."""
        assert classify_regime(350.0, 300.0, 150.0) == REGIME_HIGH_SPIKE

    def test_exact_boundary_low_price(self):
        """Ensemble median = 100 is NOT low_price."""
        assert classify_regime(100.0, 200.0, 80.0) == REGIME_NORMAL

    def test_exact_boundary_high_spike(self):
        """Ensemble median = p90 is NOT high_spike."""
        assert classify_regime(200.0, 200.0, 80.0) == REGIME_NORMAL


# ── 3. Weight Normalization ──────────────────────────────────────────


class TestWeightNormalization:
    """Contract: _normalize_weights respects bounds."""

    def test_min_max_bounds_respected(self):
        """Weights stay within [0.05, 0.75]."""
        result = _normalize_weights(
            {"cfg05": 0.9, "catboost": 0.1},
            min_w=0.05, max_w=0.75,
        )
        for w in result.values():
            assert 0.05 <= w <= 0.75 + 1e-10
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_cfg05_floor_enforced(self):
        """cfg05 gets at least 30% in trusted_delivery."""
        result = _normalize_weights(
            {"cfg05": 0.1, "catboost": 0.9},
            min_w=0.05, max_w=0.75,
            cfg05_floor=0.30, profile_name="trusted_delivery",
        )
        assert result["cfg05"] >= 0.30 - 1e-10
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_cfg05_floor_not_applied_without_cfg05(self):
        """No cfg05 floor when cfg05 absent."""
        result = _normalize_weights(
            {"catboost": 1.0},
            cfg05_floor=0.30, profile_name="trusted_delivery",
        )
        assert abs(result["catboost"] - 1.0) < 1e-6

    def test_renormalize_after_clipping(self):
        """Renormalization preserves sum=1 after clipping."""
        result = _normalize_weights(
            {"cfg05": 100.0, "catboost": 1.0},
            min_w=0.05, max_w=0.75,
        )
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_empty_weights(self):
        """Empty input returns empty output."""
        assert _normalize_weights({}) == {}

    def test_zero_total_gets_equal_fallback(self):
        """All-zero scores produce equal weights."""
        result = _normalize_weights({"a": 0.0, "b": 0.0})
        assert abs(result["a"] - 0.5) < 1e-6
        assert abs(result["b"] - 0.5) < 1e-6


# ── 4. Output Builder ────────────────────────────────────────────────


class TestOutputBuilder:
    """Contract: _build_24h_output produces valid output."""

    def test_output_has_correct_structure(self):
        """24 rows with expected columns."""
        prices = list(np.random.default_rng(42).uniform(80, 200, 24))
        output = _build_24h_output("2026-07-04", prices, "regime_bgew")

        assert len(output) == 24
        for col in ["business_day", "ds", "hour_business", "period",
                     "dayahead_price", "realtime_price"]:
            assert col in output.columns

        assert list(output["hour_business"]) == list(range(1, 25))
        assert list(output["dayahead_price"]) == prices
        assert output["realtime_price"].isna().all()

    def test_empty_prices(self):
        """Empty prices produce empty DataFrame."""
        output = _build_24h_output("2026-07-04", [], "regime_bgew")
        assert len(output) == 0


# ── 5. sMAPE Computation ─────────────────────────────────────────────


class TestSmapeComputation:
    """Contract: _compute_smape basic behavior."""

    def test_perfect_prediction(self):
        """Perfect prediction gives 0 sMAPE."""
        assert abs(_compute_smape(
            np.array([100.0, 150.0]),
            np.array([100.0, 150.0]),
        )) < 1e-6

    def test_off_by_10_percent(self):
        """~10% error yields ~9.52% sMAPE."""
        smape = _compute_smape(np.array([110.0]), np.array([100.0]))
        assert abs(smape - 9.5238) < 0.1

    def test_zero_actual_does_not_break(self):
        """Zero actual is handled gracefully."""
        smape = _compute_smape(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
        assert np.isfinite(smape)
        assert smape > 0


# ── 6. Full Pipeline ─────────────────────────────────────────────────


class TestFullPipeline:
    """Contract: run_trust_gated_regime_bgew end-to-end."""

    def _run(self, n_training_days: int = 15, **kw):
        """Helper: run fusion with default test data."""
        pred_ledger = _make_prediction_ledger(n_training_days=n_training_days)
        actual_ledger = _make_actual_ledger(n_training_days=n_training_days)
        target_preds = _make_target_preds()
        full_ledger = pd.concat([pred_ledger, target_preds], ignore_index=True)

        return run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=["cfg05", "catboost_spike_residual"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=full_ledger,
            actual_ledger=actual_ledger,
            **kw,
        )

    def test_24h_output_produced(self):
        """Full pipeline produces 24-row output."""
        result = self._run(n_training_days=15)
        assert result["success"]
        assert result["output"] is not None
        assert len(result["output"]) == 24

    def test_weights_present(self):
        """Weights dict sums to 1."""
        result = self._run(n_training_days=15)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-6

    def test_weights_within_bounds(self):
        """Weights within [0.05, 0.75]."""
        result = self._run(n_training_days=15)
        for w in result["weights"].values():
            assert 0.05 <= w <= 0.75 + 1e-10

    def test_method_regime_bgew_with_sufficient_data(self):
        """15 training days -> regime_bgew."""
        result = self._run(n_training_days=15)
        assert result["method"] == "regime_bgew"

    def test_fallback_to_period_bgew(self):
        """< 10 days -> period_bgew."""
        result = self._run(
            n_training_days=7,
            min_training_days_for_regime=10,
            min_training_days_for_period=5,
        )
        assert result["method"] == "period_bgew"
        assert any(
            fb["level"] == 2 and fb["method"] == "period_bgew"
            for fb in result["fallback_chain"]
        )

    def test_fallback_to_equal_weight(self):
        """< 5 days -> equal_weight."""
        result = self._run(
            n_training_days=3,
            min_training_days_for_regime=10,
            min_training_days_for_period=5,
        )
        assert result["method"] == "equal_weight"
        assert any(
            fb["level"] == 3 and fb["method"] == "equal_weight"
            for fb in result["fallback_chain"]
        )

    def test_empty_trusted_models(self):
        """Empty trusted_models -> failure."""
        pred_ledger = _make_prediction_ledger(n_training_days=5)
        actual_ledger = _make_actual_ledger(n_training_days=5)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=[],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=pred_ledger,
            actual_ledger=actual_ledger,
        )
        assert not result["success"]
        assert result["delivery_status"] == "TRUST_GATE_BLOCKED"

    def test_regime_reported(self):
        """Regime is detected and reported."""
        result = self._run(n_training_days=15)
        assert result["regime"] in ALL_REGIMES
        assert "ensemble_median" in result["regime_details"]

    def test_training_days_reported(self):
        """training_days_used > 0."""
        result = self._run(n_training_days=15)
        assert result["training_days_used"] > 0

    def test_fallback_chain_populated(self):
        """Fallback chain is non-empty."""
        result = self._run(n_training_days=15)
        assert len(result["fallback_chain"]) >= 1

    def test_warnings_with_blocked_models(self):
        """Blocked models produce warnings."""
        pred_ledger = _make_prediction_ledger(n_training_days=5)
        actual_ledger = _make_actual_ledger(n_training_days=5)
        target_preds = _make_target_preds()
        full_ledger = pd.concat([pred_ledger, target_preds], ignore_index=True)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=["cfg05"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=full_ledger,
            actual_ledger=actual_ledger,
            model_trust_states={
                "catboost_spike_residual": TRUST_STATE_SUSPECT_LEAKAGE,
            },
        )
        # There should be warnings; catboost_spike_residual is in the ledger
        # for the target date and should be gated -> blocked -> warning
        assert len(result["warnings"]) > 0, "Expected warnings from gate"
        all_warnings = " ".join(result["warnings"])
        assert "blocked" in all_warnings.lower() or "SUSPECT_LEAKAGE" in all_warnings

    def test_all_trusted_profiles(self):
        """Delivery status is DELIVERY_READY for regime_bgew."""
        result = self._run(n_training_days=15)
        if result["method"] in ("regime_bgew", "period_bgew"):
            assert result["delivery_status"] == "DELIVERY_READY"

    def test_fused_prices_reasonable(self):
        """Fused prices are in a reasonable range (not NaN)."""
        result = self._run(n_training_days=15)
        if result["success"]:
            prices = result["fused_prices"]
            assert all(np.isfinite(p) for p in prices)
            assert all(0 < p < 500 for p in prices)


# ── 7. Error handling ────────────────────────────────────────────────


class TestErrorHandling:
    """Contract: graceful error handling."""

    def test_empty_prediction_ledger(self):
        """Empty prediction ledger -> error."""
        empty = pd.DataFrame(columns=[
            "task", "model_name", "target_day", "business_day",
            "ds", "hour_business", "period", "y_pred",
        ])
        actual_ledger = _make_actual_ledger(n_training_days=5)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=["cfg05"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=empty,
            actual_ledger=actual_ledger,
        )
        assert not result["success"]
        assert "empty" in " ".join(result["errors"]).lower()

    def test_empty_actual_ledger(self):
        """Empty actual ledger -> error."""
        pred_ledger = _make_prediction_ledger(n_training_days=5)
        empty = pd.DataFrame(columns=[
            "task", "target_day", "business_day",
            "ds", "hour_business", "period", "y_true",
        ])

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=["cfg05"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=pred_ledger,
            actual_ledger=empty,
        )
        assert not result["success"]
        assert "empty" in " ".join(result["errors"]).lower()

    def test_no_target_predictions(self):
        """No target predictions -> error."""
        pred_ledger = _make_prediction_ledger(n_training_days=5)
        actual_ledger = _make_actual_ledger(n_training_days=5)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=["cfg05"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=pred_ledger,
            actual_ledger=actual_ledger,
        )
        assert not result["success"]
        assert any("predictions" in e.lower() for e in result["errors"])


# ── 8. Profile behavior ──────────────────────────────────────────────


class TestProfileBehavior:
    """Contract: profile_name affects trust gate."""

    def test_balanced_allows_conservative_quarantine(self):
        """balanced_candidate allows CONSERVATIVE_QUARANTINE models."""
        pred_ledger = _make_prediction_ledger(
            n_training_days=5, models=["cfg05", "catboost_spike_residual"],
        )
        actual_ledger = _make_actual_ledger(n_training_days=5)
        target_preds = _make_target_preds(models=["cfg05", "catboost_spike_residual"])
        full_ledger = pd.concat([pred_ledger, target_preds], ignore_index=True)

        result = run_trust_gated_regime_bgew(
            target_date="2026-07-04",
            trusted_models=["cfg05"],
            prediction_ledger_path="",
            actual_ledger_path="",
            prediction_ledger=full_ledger,
            actual_ledger=actual_ledger,
            profile_name="balanced_candidate",
            model_trust_states={
                "catboost_spike_residual": TRUST_STATE_CONSERVATIVE_QUARANTINE,
            },
        )
        assert result["success"]
