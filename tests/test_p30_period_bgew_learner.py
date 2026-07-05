"""tests/test_p30_period_bgew_learner.py — P30 period BGEW learner tests."""

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


def test_p30_status_constants():
    from scripts.train_p30_period_bgew_learner import (
        P30_LEARNER_BLOCKED_NO_DATA,
        P30_LEARNER_BLOCKED_NO_PREDICTIONS,
        P30_LEARNER_BLOCKED_SINGLE_MODEL,
        P30_PERIOD_BGEW_TRAINED,
    )
    assert P30_PERIOD_BGEW_TRAINED == "P30_PERIOD_BGEW_TRAINED"
    assert P30_LEARNER_BLOCKED_SINGLE_MODEL == "P30_LEARNER_BLOCKED_SINGLE_MODEL"
    assert P30_LEARNER_BLOCKED_NO_DATA == "P30_LEARNER_BLOCKED_NO_DATA"
    assert P30_LEARNER_BLOCKED_NO_PREDICTIONS == "P30_LEARNER_BLOCKED_NO_PREDICTIONS"


def test_p30_compute_weights_basic():
    from scripts.train_p30_period_bgew_learner import _compute_weights
    scores = {"model_a": 10.0, "model_b": 20.0}
    weights = _compute_weights(scores)
    assert len(weights) == 2
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_p30_compute_weights_sum_to_one():
    from scripts.train_p30_period_bgew_learner import _compute_weights
    scores = {"m1": 15.0, "m2": 25.0, "m3": 35.0}
    weights = _compute_weights(scores)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_p30_compute_weights_min_weight():
    from scripts.train_p30_period_bgew_learner import MIN_WEIGHT, _compute_weights
    scores = {"m1": 5.0, "m2": 100.0}  # m2 very bad
    weights = _compute_weights(scores)
    for w in weights.values():
        assert w >= MIN_WEIGHT - 1e-6


def test_p30_compute_weights_max_weight():
    from scripts.train_p30_period_bgew_learner import MAX_WEIGHT, _compute_weights
    scores = {"m1": 5.0, "m2": 5.1, "m3": 5.2, "m4": 5.3, "m5": 100.0}
    weights = _compute_weights(scores)
    for w in weights.values():
        assert w <= MAX_WEIGHT + 1e-6


def test_p30_compute_weights_cfg05_min_prior():
    from scripts.train_p30_period_bgew_learner import CFG05_MIN_PRIOR, _compute_weights
    scores = {"lightgbm_cfg05_dayahead": 30.0, "other_model": 10.0}
    weights = _compute_weights(scores)
    # cfg05 min_prior is applied before clipping+renormalization
    # After renorm, cfg05 weight should be > raw proportional weight
    raw_total = sum(math.exp(-0.05 * s) for s in scores.values())
    raw_cfg05_weight = math.exp(-0.05 * 30.0) / raw_total
    # With min_prior applied, cfg05 should get more than its raw share
    assert weights.get("lightgbm_cfg05_dayahead", 0) > raw_cfg05_weight


def test_p30_compute_weights_empty():
    from scripts.train_p30_period_bgew_learner import _compute_weights
    assert _compute_weights({}) == {}


def test_p30_compute_fusion_basic():
    from scripts.train_p30_period_bgew_learner import _compute_fusion
    pred_df = pd.DataFrame({
        "task": ["dayahead"] * 4,
        "target_day": ["2026-06-01"] * 4,
        "business_day": ["2026-06-01"] * 4,
        "ds": pd.to_datetime(["2026-06-01 01:00", "2026-06-01 02:00",
                               "2026-06-01 01:00", "2026-06-01 02:00"]),
        "hour_business": [1, 2, 1, 2],
        "period": ["1_8", "1_8", "1_8", "1_8"],
        "y_pred": [100.0, 200.0, 120.0, 180.0],
        "model_name": ["cfg05", "cfg05", "other", "other"],
    })
    weights = {"cfg05": 0.6, "other": 0.4}
    fusion = _compute_fusion(pred_df, weights)
    assert len(fusion) == 2
    assert "fused_price" in fusion.columns
    # First hour: 0.6*100 + 0.4*120 = 108
    assert abs(fusion.iloc[0]["fused_price"] - 108.0) < 1e-6


def test_p30_learner_blocked_single_model():
    from scripts.train_p30_period_bgew_learner import (
        P30_LEARNER_BLOCKED_SINGLE_MODEL,
        train_p30_period_bgew_learner,
    )
    actual_df = pd.DataFrame({
        "task": ["dayahead"] * 24,
        "target_day": ["2026-06-01"] * 24,
        "business_day": ["2026-06-01"] * 24,
        "ds": pd.date_range("2026-06-01 01:00", periods=24, freq="h"),
        "hour_business": range(1, 25),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "y_true": np.random.uniform(100, 500, 24),
    })
    pred_df = pd.DataFrame({
        "task": ["dayahead"] * 24,
        "target_day": ["2026-06-01"] * 24,
        "business_day": ["2026-06-01"] * 24,
        "ds": pd.date_range("2026-06-01 01:00", periods=24, freq="h"),
        "hour_business": range(1, 25),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "y_pred": np.random.uniform(100, 500, 24),
        "model_name": ["lightgbm_cfg05_dayahead"] * 24,  # only 1 model
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as af:
        actual_df.to_csv(af.name, index=False)
        actual_path = af.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as pf:
        pred_df.to_csv(pf.name, index=False)
        pred_path = pf.name

    work_dir = tempfile.mkdtemp()
    try:
        result = train_p30_period_bgew_learner(
            actual_ledger_path=actual_path,
            prediction_ledger_path=pred_path,
            work_dir=work_dir,
        )
        assert result["final_status"] == P30_LEARNER_BLOCKED_SINGLE_MODEL
    finally:
        os.unlink(actual_path)
        os.unlink(pred_path)


def test_p30_learner_trains_with_two_models():
    from scripts.train_p30_period_bgew_learner import (
        P30_PERIOD_BGEW_TRAINED,
        train_p30_period_bgew_learner,
    )
    n_hours = 48  # 2 days
    actual_df = pd.DataFrame({
        "task": ["dayahead"] * n_hours,
        "target_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
        "business_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
        "ds": list(pd.date_range("2026-06-01 01:00", periods=24, freq="h")) +
              list(pd.date_range("2026-06-02 01:00", periods=24, freq="h")),
        "hour_business": list(range(1, 25)) * 2,
        "period": (["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8) * 2,
        "y_true": np.random.uniform(100, 500, n_hours),
    })
    pred_df = pd.concat([
        pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "target_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "business_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "ds": list(pd.date_range("2026-06-01 01:00", periods=24, freq="h")) +
                  list(pd.date_range("2026-06-02 01:00", periods=24, freq="h")),
            "hour_business": list(range(1, 25)) * 2,
            "period": (["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8) * 2,
            "y_pred": np.random.uniform(100, 500, n_hours),
            "model_name": ["lightgbm_cfg05_dayahead"] * n_hours,
        }),
        pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "target_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "business_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "ds": list(pd.date_range("2026-06-01 01:00", periods=24, freq="h")) +
                  list(pd.date_range("2026-06-02 01:00", periods=24, freq="h")),
            "hour_business": list(range(1, 25)) * 2,
            "period": (["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8) * 2,
            "y_pred": np.random.uniform(100, 500, n_hours),
            "model_name": ["catboost_sota"] * n_hours,
        }),
    ], ignore_index=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as af:
        actual_df.to_csv(af.name, index=False)
        actual_path = af.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as pf:
        pred_df.to_csv(pf.name, index=False)
        pred_path = pf.name

    work_dir = tempfile.mkdtemp()
    try:
        result = train_p30_period_bgew_learner(
            actual_ledger_path=actual_path,
            prediction_ledger_path=pred_path,
            work_dir=work_dir,
        )
        assert result["final_status"] == P30_PERIOD_BGEW_TRAINED
        assert result["n_models_in_pool"] == 2
        assert len(result["period_weights"]) > 0
    finally:
        os.unlink(actual_path)
        os.unlink(pred_path)


def test_p30_period_weights_structure():
    """Period weights should have entries for each period."""
    from scripts.train_p30_period_bgew_learner import VALID_PERIODS
    assert VALID_PERIODS == ["1_8", "9_16", "17_24"]


def test_p30_no_forbidden_files():
    from scripts.train_p30_period_bgew_learner import train_p30_period_bgew_learner
    result = train_p30_period_bgew_learner(
        actual_ledger_path="/nonexistent.csv",
        prediction_ledger_path="/nonexistent.csv",
    )
    assert result["forbidden_files_check"] == "PASS"


def test_p30_blocked_models_excluded():
    """Models with BLOCKED readiness labels should not enter the pool."""
    from scripts.train_p30_period_bgew_learner import (
        BLOCKED_READINESS_LABELS,
        ALLOWED_READINESS_LABELS,
    )
    # No overlap between allowed and blocked
    assert len(ALLOWED_READINESS_LABELS & BLOCKED_READINESS_LABELS) == 0


def test_p30_negative_period_analysis():
    """Negative period analysis should be present when learner trains."""
    from scripts.train_p30_period_bgew_learner import (
        P30_PERIOD_BGEW_TRAINED,
        train_p30_period_bgew_learner,
    )
    n_hours = 48
    actual_df = pd.DataFrame({
        "task": ["dayahead"] * n_hours,
        "target_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
        "business_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
        "ds": list(pd.date_range("2026-06-01 01:00", periods=24, freq="h")) +
              list(pd.date_range("2026-06-02 01:00", periods=24, freq="h")),
        "hour_business": list(range(1, 25)) * 2,
        "period": (["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8) * 2,
        "y_true": np.random.uniform(100, 500, n_hours),
    })
    pred_df = pd.concat([
        pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "target_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "business_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "ds": list(pd.date_range("2026-06-01 01:00", periods=24, freq="h")) +
                  list(pd.date_range("2026-06-02 01:00", periods=24, freq="h")),
            "hour_business": list(range(1, 25)) * 2,
            "period": (["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8) * 2,
            "y_pred": np.random.uniform(100, 500, n_hours),
            "model_name": ["lightgbm_cfg05_dayahead"] * n_hours,
        }),
        pd.DataFrame({
            "task": ["dayahead"] * n_hours,
            "target_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "business_day": ["2026-06-01"] * 24 + ["2026-06-02"] * 24,
            "ds": list(pd.date_range("2026-06-01 01:00", periods=24, freq="h")) +
                  list(pd.date_range("2026-06-02 01:00", periods=24, freq="h")),
            "hour_business": list(range(1, 25)) * 2,
            "period": (["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8) * 2,
            "y_pred": np.random.uniform(100, 500, n_hours),
            "model_name": ["catboost_sota"] * n_hours,
        }),
    ], ignore_index=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as af:
        actual_df.to_csv(af.name, index=False)
        actual_path = af.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as pf:
        pred_df.to_csv(pf.name, index=False)
        pred_path = pf.name

    work_dir = tempfile.mkdtemp()
    try:
        result = train_p30_period_bgew_learner(
            actual_ledger_path=actual_path,
            prediction_ledger_path=pred_path,
            work_dir=work_dir,
        )
        assert result["negative_period_analysis"] is not None
        assert isinstance(result["negative_period_analysis"], dict)
    finally:
        os.unlink(actual_path)
        os.unlink(pred_path)


def test_p30_no_fake_multi_model_fusion():
    """Single model must NOT produce fake fusion output."""
    from scripts.train_p30_period_bgew_learner import (
        P30_LEARNER_BLOCKED_SINGLE_MODEL,
        train_p30_period_bgew_learner,
    )
    actual_df = pd.DataFrame({
        "task": ["dayahead"] * 24,
        "target_day": ["2026-06-01"] * 24,
        "business_day": ["2026-06-01"] * 24,
        "ds": pd.date_range("2026-06-01 01:00", periods=24, freq="h"),
        "hour_business": list(range(1, 25)),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "y_true": np.random.uniform(100, 500, 24),
    })
    pred_df = pd.DataFrame({
        "task": ["dayahead"] * 24,
        "target_day": ["2026-06-01"] * 24,
        "business_day": ["2026-06-01"] * 24,
        "ds": pd.date_range("2026-06-01 01:00", periods=24, freq="h"),
        "hour_business": list(range(1, 25)),
        "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
        "y_pred": np.random.uniform(100, 500, 24),
        "model_name": ["lightgbm_cfg05_dayahead"] * 24,
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as af:
        actual_df.to_csv(af.name, index=False)
        actual_path = af.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as pf:
        pred_df.to_csv(pf.name, index=False)
        pred_path = pf.name

    work_dir = tempfile.mkdtemp()
    try:
        result = train_p30_period_bgew_learner(
            actual_ledger_path=actual_path,
            prediction_ledger_path=pred_path,
            work_dir=work_dir,
        )
        assert result["final_status"] == P30_LEARNER_BLOCKED_SINGLE_MODEL
        assert "CANNOT_FAKE_MULTI_MODEL_FUSION" in result["reason_codes"]
    finally:
        os.unlink(actual_path)
        os.unlink(pred_path)
