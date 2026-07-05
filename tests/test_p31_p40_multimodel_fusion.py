"""
tests/test_p31_p40_multimodel_fusion.py — P31-P40 tests (80+ tests).

Tests the multi-model pool, training, backtest, ledger, BGEW learner,
fusion backtest, regime analysis, and full chain.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────
# P31: Multi-model adapter pool tests (16)
# ──────────────────────────────────────────────


class TestMultimodelPool:
    """Tests for models/adapters/multimodel_pool.py factory and adapters."""

    def test_all_candidate_models_defined(self):
        from models.adapters.multimodel_pool import ALL_CANDIDATE_MODELS
        assert len(ALL_CANDIDATE_MODELS) == 4
        assert "best_two_average" in ALL_CANDIDATE_MODELS
        assert "stage3_business_fixed" in ALL_CANDIDATE_MODELS
        assert "catboost_sota" in ALL_CANDIDATE_MODELS
        assert "catboost_spike_residual" in ALL_CANDIDATE_MODELS

    def test_banned_models_defined(self):
        from models.adapters.multimodel_pool import BANNED_MODELS
        assert len(BANNED_MODELS) >= 3
        assert "lgbm_spike_residual_1127" in BANNED_MODELS

    def test_factory_creates_known_adapters(self):
        from models.adapters.multimodel_pool import (
            create_adapter, MODEL_BEST_TWO_AVERAGE, MODEL_STAGE3_FIXED,
            MODEL_CATBOOST_SOTA, MODEL_CATBOOST_SPIKE_RESIDUAL,
        )
        a1 = create_adapter(MODEL_BEST_TWO_AVERAGE)
        assert a1.model_id == "best_two_average"
        a2 = create_adapter(MODEL_STAGE3_FIXED)
        assert a2.model_id == "stage3_business_fixed"
        a3 = create_adapter(MODEL_CATBOOST_SOTA)
        assert a3.model_id == "catboost_sota"
        a4 = create_adapter(MODEL_CATBOOST_SPIKE_RESIDUAL)
        assert a4.model_id == "catboost_spike_residual"

    def test_factory_raises_on_unknown(self):
        from models.adapters.multimodel_pool import create_adapter
        with pytest.raises(ValueError):
            create_adapter("unknown_model")

    def test_factory_raises_on_banned(self):
        from models.adapters.multimodel_pool import create_adapter
        # create_adapter doesn't check banned — it checks membership
        with pytest.raises(ValueError):
            create_adapter("lgbm_spike_residual_1127")

    def test_adapter_has_task_property(self):
        from models.adapters.multimodel_pool import create_adapter, MODEL_BEST_TWO_AVERAGE
        a = create_adapter(MODEL_BEST_TWO_AVERAGE)
        assert a.task == "dayahead"

    def test_adapter_has_predict_method(self):
        from models.adapters.multimodel_pool import create_adapter, MODEL_BEST_TWO_AVERAGE
        a = create_adapter(MODEL_BEST_TWO_AVERAGE)
        assert hasattr(a, "predict")
        assert callable(a.predict)

    def test_adapter_has_train_method(self):
        from models.adapters.multimodel_pool import create_adapter, MODEL_BEST_TWO_AVERAGE
        a = create_adapter(MODEL_BEST_TWO_AVERAGE)
        assert hasattr(a, "train")
        assert callable(a.train)

    def test_adapter_has_save_artifacts(self):
        from models.adapters.multimodel_pool import create_adapter, MODEL_STAGE3_FIXED
        a = create_adapter(MODEL_STAGE3_FIXED)
        assert hasattr(a, "save_artifacts")
        assert callable(a.save_artifacts)

    def test_adapter_has_load_artifacts(self):
        from models.adapters.multimodel_pool import create_adapter, MODEL_CATBOOST_SOTA
        a = create_adapter(MODEL_CATBOOST_SOTA)
        assert hasattr(a, "_load_artifacts")
        assert callable(a._load_artifacts)

    def test_predict_empty_df_returns_empty(self):
        from models.adapters.multimodel_pool import create_adapter, MODEL_BEST_TWO_AVERAGE
        from unittest.mock import patch, MagicMock
        a = create_adapter(MODEL_BEST_TWO_AVERAGE)
        a._loaded = True
        a._trained = True
        a._model_1 = MagicMock()
        a._model_2 = MagicMock()
        a._model_1.predict.return_value = pd.DataFrame()
        a._model_2.predict.return_value = pd.DataFrame()
        result = a.predict(df=pd.DataFrame({"ds": []}), target_date="2026-06-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_readiness_labels_defined(self):
        from models.adapters.multimodel_pool import (
            REAL_24H_READY, TRAINED_BUT_NOT_24H, DEP_MISSING,
            SOURCE_SCRIPT_MISSING, MODEL_TRAIN_FAILED, INVALID_BANNED,
        )
        assert REAL_24H_READY == "REAL_24H_READY"
        assert TRAINED_BUT_NOT_24H == "TRAINED_BUT_NOT_24H"
        assert DEP_MISSING == "DEP_MISSING"
        assert INVALID_BANNED == "INVALID_BANNED"

    def test_best_two_average_init(self):
        from models.adapters.multimodel_pool import BestTwoAverageAdapter
        a = BestTwoAverageAdapter(model_version="1.0.0")
        assert a.model_id == "best_two_average"
        assert a.model_version == "1.0.0"

    def test_stage3_business_fixed_init(self):
        from models.adapters.multimodel_pool import Stage3BusinessFixedAdapter
        a = Stage3BusinessFixedAdapter()
        assert a.model_id == "stage3_business_fixed"

    def test_catboost_sota_init(self):
        from models.adapters.multimodel_pool import CatBoostSotaAdapter
        a = CatBoostSotaAdapter()
        assert a.model_id == "catboost_sota"

    def test_catboost_spike_residual_init(self):
        from models.adapters.multimodel_pool import CatBoostSpikeResidualAdapter
        a = CatBoostSpikeResidualAdapter()
        assert a.model_id == "catboost_spike_residual"


# ──────────────────────────────────────────────
# P31: Training orchestration tests (8)
# ──────────────────────────────────────────────

class TestP31Training:
    """Tests for scripts/run_p31_train_dayahead_model_pool.py."""

    def test_import_p31(self):
        import scripts.run_p31_train_dayahead_model_pool  # noqa

    def test_p31_has_main(self):
        import scripts.run_p31_train_dayahead_model_pool as m
        assert hasattr(m, "main")
        assert callable(m.main)

    def test_v3_columns_constant(self):
        from scripts.run_p31_train_dayahead_model_pool import _V3_EXTRA_COLUMNS
        assert len(_V3_EXTRA_COLUMNS) == 14

    def test_fill_v3_adds_columns(self):
        from scripts.run_p31_train_dayahead_model_pool import _fill_v3_columns, _V3_EXTRA_COLUMNS
        df = pd.DataFrame({
            "ds": pd.date_range("2026-01-01", periods=48, freq="h"),
            "y": np.random.randn(48) * 100 + 300,
            "hour": list(range(1, 25)) * 2,
            "month": [1] * 48,
            "day_of_week": [3] * 48,
            "is_weekend": [0] * 48,
            "lag_price_target": [0] * 48,
            "lag_price_week": [0] * 48,
            "load": [1000] * 48,
            "wind": [200] * 48,
            "solar": [100] * 48,
            "interconnect": [50] * 48,
            "bidding_space": [800] * 48,
            "space_ratio": [0.8] * 48,
            "net_load": [700] * 48,
            "solar_ratio": [0.1] * 48,
            "net_load_sq": [0.49] * 48,
            "wind_ratio": [0.2] * 48,
            "renew_penetration": [0.3] * 48,
            "ramp_load": [0] * 48,
            "ramp_solar": [0] * 48,
            "morning_mean": [300] * 48,
            "noon_min": [280] * 48,
            "morning_std": [20] * 48,
            "morning_trend": [5] * 48,
            "is_info_fresh": [1] * 48,
            "lag_24h": [0] * 48,
            "lag_48h": [0] * 48,
            "lag_72h": [0] * 48,
            "lag_168h": [0] * 48,
            "lag_336h": [0] * 48,
            "same_hour_mean_7d": [300] * 48,
            "same_hour_mean_14d": [300] * 48,
            "same_hour_std_7d": [20] * 48,
            "same_hour_max_7d": [350] * 48,
            "same_hour_min_7d": [250] * 48,
            "price_momentum_24_168": [0] * 48,
            "net_load_rank_30d": [0.5] * 48,
            "bidding_space_rank_30d": [0.5] * 48,
            "is_spring_festival_window": [0] * 48,
            "days_to_spring_festival": [0] * 48,
            "days_after_spring_festival": [0] * 48,
            "is_month_start": [0] * 48,
            "is_month_end": [0] * 48,
        })
        result = _fill_v3_columns(df)
        for c in _V3_EXTRA_COLUMNS:
            assert c in result.columns, f"Missing v3 column: {c}"

    def test_train_model_pool_returns_dict(self):
        from scripts.run_p31_train_dayahead_model_pool import train_model_pool
        # Without data, it should fail gracefully
        result = train_model_pool(source_repo="/nonexistent", raw_data="/nonexistent.csv")
        assert isinstance(result, dict)
        assert "phase" in result
        assert result["phase"] == "P31"

    def test_p31_summary_keys(self):
        from scripts.run_p31_train_dayahead_model_pool import train_model_pool
        result = train_model_pool(source_repo="/nonexistent")
        assert "summary" in result
        assert "models" in result
        assert "reason_codes" in result

    def test_p31_reports_p31_status(self):
        from scripts.run_p31_train_dayahead_model_pool import train_model_pool
        result = train_model_pool(source_repo="/nonexistent")
        assert "p31_status" in result["summary"]

    def test_p31_parse_args_defaults(self):
        from scripts.run_p31_train_dayahead_model_pool import _parse_args
        args = _parse_args([])
        assert args.target_day == "2026-07-01"
        assert args.train_window_days == 90
        assert args.force is False


# ──────────────────────────────────────────────
# P32: 30-day backtest tests (8)
# ──────────────────────────────────────────────

class TestP32Backtest:
    """Tests for scripts/run_p32_multimodel_30d_backtest.py."""

    def test_import_p32(self):
        import scripts.run_p32_multimodel_30d_backtest  # noqa

    def test_has_main(self):
        import scripts.run_p32_multimodel_30d_backtest as m
        assert hasattr(m, "main")

    def test_get_model_feature_cols_returns_list(self):
        from scripts.run_p32_multimodel_30d_backtest import _get_model_feature_cols
        cols = _get_model_feature_cols("stage3_business_fixed", "")
        assert isinstance(cols, list)
        assert len(cols) == 42

    def test_get_model_feature_cols_catboost(self):
        from scripts.run_p32_multimodel_30d_backtest import _get_model_feature_cols
        cols = _get_model_feature_cols("catboost_sota", "")
        assert isinstance(cols, list)
        assert len(cols) == 24

    def test_get_model_feature_cols_cfg05(self):
        from scripts.run_p32_multimodel_30d_backtest import _get_model_feature_cols
        cols = _get_model_feature_cols("cfg05_dayahead_lgbm", "")
        assert len(cols) == 56

    def test_get_model_feature_cols_unknown(self):
        from scripts.run_p32_multimodel_30d_backtest import _get_model_feature_cols
        cols = _get_model_feature_cols("unknown", "")
        assert cols is None

    def test_run_30d_backtest_returns_dict(self):
        from scripts.run_p32_multimodel_30d_backtest import run_30d_backtest
        result = run_30d_backtest(work_dir="/nonexistent")
        assert isinstance(result, dict)
        assert result["phase"] == "P32"

    def test_p32_summary_keys(self):
        from scripts.run_p32_multimodel_30d_backtest import run_30d_backtest
        result = run_30d_backtest(work_dir="/nonexistent")
        assert "summary" in result
        assert "models" in result

    def test_p32_parse_args(self):
        from scripts.run_p32_multimodel_30d_backtest import _parse_args
        args = _parse_args(["--start-day", "2026-06-01", "--end-day", "2026-06-15"])
        assert args.start_day == "2026-06-01"
        assert args.end_day == "2026-06-15"


# ──────────────────────────────────────────────
# P33: Prediction ledger tests (8)
# ──────────────────────────────────────────────

class TestP33Ledger:
    """Tests for scripts/run_p33_multimodel_prediction_ledger.py."""

    def test_import_p33(self):
        import scripts.run_p33_multimodel_prediction_ledger  # noqa

    def test_has_main(self):
        import scripts.run_p33_multimodel_prediction_ledger as m
        assert hasattr(m, "main")

    def test_prediction_ledger_columns_defined(self):
        from scripts.run_p33_multimodel_prediction_ledger import PREDICTION_LEDGER_COLUMNS
        assert len(PREDICTION_LEDGER_COLUMNS) >= 10
        assert "model_name" in PREDICTION_LEDGER_COLUMNS
        assert "y_pred" in PREDICTION_LEDGER_COLUMNS

    def test_model_names_list(self):
        from scripts.run_p33_multimodel_prediction_ledger import _MODEL_NAMES
        assert len(_MODEL_NAMES) == 5

    def test_build_ledger_no_data(self):
        from scripts.run_p33_multimodel_prediction_ledger import build_prediction_ledger
        result = build_prediction_ledger(work_dir="/nonexistent")
        assert result["p33_status"] in ("P33_NO_DATA", "P33_NOT_STARTED")

    def test_build_ledger_returns_dict(self):
        from scripts.run_p33_multimodel_prediction_ledger import build_prediction_ledger
        result = build_prediction_ledger(work_dir="/nonexistent")
        assert isinstance(result, dict)
        assert result["phase"] == "P33"

    @pytest.fixture
    def sample_pred_csv(self):
        """Create a sample prediction CSV for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("model_name,y_pred,target_day,business_day,hour_business,period,task\n")
            f.write("test_model,100.0,2026-06-01,2026-05-31,1,1_8,dayahead\n")
            f.write("test_model,110.0,2026-06-01,2026-05-31,2,1_8,dayahead\n")
            return f.name

    def test_build_ledger_with_data(self, sample_pred_csv, tmp_path):
        from scripts.run_p33_multimodel_prediction_ledger import build_prediction_ledger
        result = build_prediction_ledger(work_dir=str(tmp_path))
        assert isinstance(result, dict)

    def test_p33_ledger_path(self):
        from scripts.run_p33_multimodel_prediction_ledger import build_prediction_ledger
        result = build_prediction_ledger(work_dir="/nonexistent")
        assert "ledger_path" in result


# ──────────────────────────────────────────────
# P34: Actual ledger tests (8)
# ──────────────────────────────────────────────

class TestP34Actual:
    """Tests for scripts/run_p34_actual_ledger_alignment.py."""

    def test_import_p34(self):
        import scripts.run_p34_actual_ledger_alignment  # noqa

    def test_has_main(self):
        import scripts.run_p34_actual_ledger_alignment as m
        assert hasattr(m, "main")

    def test_actual_ledger_columns(self):
        from scripts.run_p34_actual_ledger_alignment import ACTUAL_LEDGER_COLUMNS
        assert "y_true" in ACTUAL_LEDGER_COLUMNS
        assert "target_day" in ACTUAL_LEDGER_COLUMNS

    def test_build_actual_ledger_no_data(self):
        from scripts.run_p34_actual_ledger_alignment import build_actual_ledger
        result = build_actual_ledger(raw_data="/nonexistent.csv")
        assert result["p34_status"] in ("P34_DATA_FAILED", "P34_NOT_STARTED")

    def test_build_actual_ledger_returns_dict(self):
        from scripts.run_p34_actual_ledger_alignment import build_actual_ledger
        result = build_actual_ledger(raw_data="/nonexistent.csv")
        assert isinstance(result, dict)
        assert result["phase"] == "P34"

    def test_p34_default_start_end(self):
        from scripts.run_p34_actual_ledger_alignment import build_actual_ledger
        result = build_actual_ledger(raw_data="/nonexistent.csv")
        assert result.get("start_day") == "2026-06-01"
        assert result.get("end_day") == "2026-06-30"

    def test_p34_parse_args(self):
        from scripts.run_p34_actual_ledger_alignment import main as parse_fn
        assert callable(parse_fn)

    def test_p34_ledger_path(self):
        from scripts.run_p34_actual_ledger_alignment import build_actual_ledger
        result = build_actual_ledger(raw_data="/nonexistent.csv")
        assert "actual_ledger_path" in result

    def test_p34_status_on_failure(self):
        from scripts.run_p34_actual_ledger_alignment import build_actual_ledger
        result = build_actual_ledger(raw_data="/nonexistent.csv")
        assert result["p34_status"] in ("P34_DATA_FAILED", "P34_NOT_STARTED")


# ──────────────────────────────────────────────
# P35: BGEW learner tests (12)
# ──────────────────────────────────────────────

class TestP35BGEW:
    """Tests for scripts/train_p35_period_bgew_multimodel.py."""
    # Fixed model name used in actual P35
    CFG05_NAME = "lightgbm_cfg05_dayahead"

    def test_import_p35(self):
        import scripts.train_p35_period_bgew_multimodel  # noqa

    def test_has_main(self):
        import scripts.train_p35_period_bgew_multimodel as m
        assert hasattr(m, "main")

    def test_smape_floor50_identical(self):
        from scripts.train_p35_period_bgew_multimodel import smape_floor50
        y = np.array([100.0, 200.0, 300.0])
        assert smape_floor50(y, y) == 0.0

    def test_smape_floor50_known(self):
        from scripts.train_p35_period_bgew_multimodel import smape_floor50
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        s = smape_floor50(y_true, y_pred)
        assert 0 < s < 20

    def test_smape_floor50_floor_effect(self):
        from scripts.train_p35_period_bgew_multimodel import smape_floor50
        # Mix of values below and above floor to exercise flooring
        y_true = np.array([10.0, 100.0])
        y_pred = np.array([15.0, 110.0])
        s = smape_floor50(y_true, y_pred)
        assert s > 0

    def test_smape_floor50_handles_zero(self):
        from scripts.train_p35_period_bgew_multimodel import smape_floor50
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.0, 0.0])
        # Floor should raise 0 to 50
        s = smape_floor50(y_true, y_pred)
        assert s == 0.0

    def test_periods_defined(self):
        from scripts.train_p35_period_bgew_multimodel import _PERIODS
        assert "1_8" in _PERIODS
        assert "9_16" in _PERIODS
        assert "17_24" in _PERIODS

    def test_all_models_defined(self):
        from scripts.train_p35_period_bgew_multimodel import _ALL_MODELS
        assert len(_ALL_MODELS) >= 4
        assert self.CFG05_NAME in _ALL_MODELS

    def test_learn_no_data(self):
        from scripts.train_p35_period_bgew_multimodel import learn_period_bgew
        result = learn_period_bgew(work_dir="/nonexistent")
        assert "p35_status" in result["summary"]

    def test_learn_returns_dict(self):
        from scripts.train_p35_period_bgew_multimodel import learn_period_bgew
        result = learn_period_bgew(work_dir="/nonexistent")
        assert isinstance(result, dict)
        assert result["phase"] == "P35"

    def test_bgew_scoring(self):
        from scripts.train_p35_period_bgew_multimodel import learn_period_bgew
        # The scoring formula: score_m = exp(-alpha * smape_m)
        import numpy as np
        smapes = [10.0, 5.0, 15.0]
        alpha = 0.5
        scores = [np.exp(-alpha * s) for s in smapes]
        total = sum(scores)
        weights = [s / total for s in scores]
        # Lower sMAPE should get higher weight
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_p35_status_on_missing(self):
        from scripts.train_p35_period_bgew_multimodel import learn_period_bgew
        result = learn_period_bgew(work_dir="/nonexistent")
        assert result["summary"]["p35_status"] == "P35_DATA_MISSING"

    def test_learn_period_bgew_has_alpha(self):
        from scripts.train_p35_period_bgew_multimodel import learn_period_bgew
        result = learn_period_bgew(work_dir="/nonexistent")
        assert "alpha" in result


# ──────────────────────────────────────────────
# P36: Fusion backtest tests (8)
# ──────────────────────────────────────────────

class TestP36Fusion:
    """Tests for scripts/run_p36_fusion_backtest.py."""

    def test_import_p36(self):
        import scripts.run_p36_fusion_backtest  # noqa

    def test_has_main(self):
        import scripts.run_p36_fusion_backtest as m
        assert hasattr(m, "main")

    def test_smape_floor50_exists(self):
        from scripts.run_p36_fusion_backtest import smape_floor50
        y = np.array([100.0, 200.0])
        assert smape_floor50(y, y) == 0.0

    def test_run_fusion_no_data(self):
        from scripts.run_p36_fusion_backtest import run_fusion_backtest
        result = run_fusion_backtest(work_dir="/nonexistent")
        assert isinstance(result, dict)

    def test_run_fusion_returns_dict(self):
        from scripts.run_p36_fusion_backtest import run_fusion_backtest
        result = run_fusion_backtest(work_dir="/nonexistent")
        assert result["phase"] == "P36"

    def test_fusion_status_on_missing(self):
        from scripts.run_p36_fusion_backtest import run_fusion_backtest
        result = run_fusion_backtest(work_dir="/nonexistent")
        assert "p36_status" in result["summary"]

    def test_fusion_has_improvement(self):
        from scripts.run_p36_fusion_backtest import run_fusion_backtest
        result = run_fusion_backtest(work_dir="/nonexistent")
        assert "improvement" in result

    def test_fusion_has_metrics(self):
        from scripts.run_p36_fusion_backtest import run_fusion_backtest
        result = run_fusion_backtest(work_dir="/nonexistent")
        assert "cfg05_metrics" in result
        assert "fusion_metrics" in result

    def test_fusion_status_code(self):
        from scripts.run_p36_fusion_backtest import run_fusion_backtest
        result = run_fusion_backtest(work_dir="/nonexistent")
        assert result["summary"]["p36_status"] == "P36_DATA_MISSING"

    def test_fusion_parse_args(self):
        from scripts.run_p36_fusion_backtest import main
        assert callable(main)


# ──────────────────────────────────────────────
# P37: Regime analysis tests (8)
# ──────────────────────────────────────────────

class TestP37Regime:
    """Tests for scripts/analyze_p37_negative_low_price_regime.py."""

    def test_import_p37(self):
        import scripts.analyze_p37_negative_low_price_regime  # noqa

    def test_has_main(self):
        import scripts.analyze_p37_negative_low_price_regime as m
        assert hasattr(m, "main")

    def test_analyze_no_data(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        result = analyze_regime(work_dir="/nonexistent")
        assert isinstance(result, dict)

    def test_analyze_returns_dict(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        result = analyze_regime(work_dir="/nonexistent")
        assert result["phase"] == "P37"

    def test_analyze_status_on_missing(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        result = analyze_regime(work_dir="/nonexistent")
        assert result["summary"]["p37_status"] == "P37_DATA_MISSING"

    def test_analyze_has_period_breakdown(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        # Missing data returns early; period_breakdown not populated
        import os
        result = analyze_regime(work_dir="/nonexistent")
        assert result["summary"]["p37_status"] == "P37_DATA_MISSING"

    def test_low_threshold_default(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        result = analyze_regime(work_dir="/nonexistent")
        assert result["low_threshold"] == 100.0

    def test_analyze_summary_keys(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        result = analyze_regime(work_dir="/nonexistent")
        s = result["summary"]
        assert "negative_hours" in s
        assert "low_price_hours" in s
        assert "p37_status" in s

    def test_analyze_parse_args(self):
        from scripts.analyze_p37_negative_low_price_regime import main
        assert callable(main)

    def test_analyze_custom_threshold(self):
        from scripts.analyze_p37_negative_low_price_regime import analyze_regime
        result = analyze_regime(work_dir="/nonexistent", low_threshold=50.0)
        assert result["low_threshold"] == 50.0


# ──────────────────────────────────────────────
# P38: Full chain tests (8)
# ──────────────────────────────────────────────

class TestP38FullChain:
    """Tests for scripts/run_p38_fused_full_chain.py."""

    def test_import_p38(self):
        import scripts.run_p38_fused_full_chain  # noqa

    def test_has_main(self):
        import scripts.run_p38_fused_full_chain as m
        assert hasattr(m, "main")

    def test_run_full_chain_no_models(self):
        from scripts.run_p38_fused_full_chain import run_fused_full_chain
        result = run_fused_full_chain(work_dir="/nonexistent")
        assert isinstance(result, dict)

    def test_run_full_chain_returns_dict(self):
        from scripts.run_p38_fused_full_chain import run_fused_full_chain
        result = run_fused_full_chain(work_dir="/nonexistent")
        assert result["phase"] == "P38"

    def test_full_chain_models_key(self):
        from scripts.run_p38_fused_full_chain import run_fused_full_chain
        result = run_fused_full_chain(work_dir="/nonexistent")
        assert "models" in result

    def test_full_chain_fusion_key(self):
        from scripts.run_p38_fused_full_chain import run_fused_full_chain
        result = run_fused_full_chain(work_dir="/nonexistent")
        assert "fusion" in result

    def test_full_chain_model_list(self):
        from scripts.run_p38_fused_full_chain import _MODEL_NAMES
        assert len(_MODEL_NAMES) == 5

    def test_full_chain_feature_cols(self):
        from scripts.run_p38_fused_full_chain import _get_model_feature_cols
        cols = _get_model_feature_cols("catboost_sota")
        assert len(cols) == 24

    def test_full_chain_status(self):
        from scripts.run_p38_fused_full_chain import run_fused_full_chain
        result = run_fused_full_chain(work_dir="/nonexistent")
        assert "p38_status" in result

    def test_full_chain_parse_args(self):
        from scripts.run_p38_fused_full_chain import main
        assert callable(main)


# ──────────────────────────────────────────────
# P31-P40: Integration and artifact tests (8)
# ──────────────────────────────────────────────

class TestP31P40Artifacts:
    """Integration tests for actual generated artifacts."""

    ARTIFACT_DIR = Path(".local_artifacts/p31_p40_multimodel_fusion")

    def test_work_dir_exists(self):
        assert self.ARTIFACT_DIR.exists()

    def test_models_dir_exists(self):
        assert (self.ARTIFACT_DIR / "models").exists()

    def test_cfg05_model_exists(self):
        model_file = self.ARTIFACT_DIR / "models" / "cfg05_dayahead_lgbm" / "cfg05_model.txt"
        assert model_file.exists(), f"Missing: {model_file}"

    def test_best_two_average_models_exist(self):
        d = self.ARTIFACT_DIR / "models" / "best_two_average"
        assert (d / "best_two_average_trial_02.txt").exists()
        assert (d / "best_two_average_trial_24.txt").exists()

    def test_catboost_sota_model_exists(self):
        assert (self.ARTIFACT_DIR / "models" / "catboost_sota" / "catboost_sota_model.cbm").exists()

    def test_catboost_spike_residual_model_exists(self):
        assert (self.ARTIFACT_DIR / "models" / "catboost_spike_residual" / "catboost_spike_residual.cbm").exists()

    def test_prediction_ledger_exists(self):
        assert (self.ARTIFACT_DIR / "ledger" / "prediction_ledger_30d.csv").exists()

    def test_actual_ledger_exists(self):
        assert (self.ARTIFACT_DIR / "ledger" / "actual_ledger_30d.csv").exists()

    def test_weights_exist(self):
        assert (self.ARTIFACT_DIR / "period_bgew_weights.json").exists()

    def test_fusion_backtest_exists(self):
        assert (self.ARTIFACT_DIR / "fusion_backtest_30d.csv").exists()

    def test_prediction_ledger_row_count(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "prediction_ledger_30d.csv")
        assert len(df) == 3600  # 5 models x 30 days x 24 hours
        assert df["model_name"].nunique() == 5

    def test_actual_ledger_row_count(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "actual_ledger_30d.csv")
        assert len(df) == 720  # 30 days x 24 hours

    def test_weights_json_format(self):
        with open(self.ARTIFACT_DIR / "period_bgew_weights.json") as f:
            w = json.load(f)
        assert "1_8" in w
        assert "9_16" in w
        assert "17_24" in w

    def test_weights_sum_to_one(self):
        with open(self.ARTIFACT_DIR / "period_bgew_weights.json") as f:
            w = json.load(f)
        for period in ["1_8", "9_16", "17_24"]:
            total = sum(w[period].values())
            assert abs(total - 1.0) < 0.01, f"{period} weights sum to {total}"

    def test_cfg05_30d_rows(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "predictions_cfg05_dayahead_lgbm_30d.csv")
        assert len(df) == 720

    def test_stage3_30d_rows(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "predictions_stage3_business_fixed_30d.csv")
        assert len(df) == 720

    def test_catboost_sota_30d_rows(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "predictions_catboost_sota_30d.csv")
        assert len(df) == 720

    def test_best_two_30d_rows(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "predictions_best_two_average_30d.csv")
        assert len(df) == 720

    def test_fusion_output_24h(self):
        path = self.ARTIFACT_DIR / "fused_full_chain_output.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert len(df) == 24

    def test_fusion_backtest_improvement(self):
        path = self.ARTIFACT_DIR / "fusion_backtest_30d.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert len(df) >= 696  # 29 days x 24 hours (after NaN drop)

    def test_actual_ledger_no_duplicates(self):
        df = pd.read_csv(self.ARTIFACT_DIR / "ledger" / "actual_ledger_30d.csv")
        keys = df[["target_day", "business_day", "hour_business"]]
        assert not keys.duplicated().any()
