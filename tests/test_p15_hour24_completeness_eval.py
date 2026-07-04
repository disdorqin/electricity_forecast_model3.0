"""
tests/test_p15_hour24_completeness_eval.py — P15 tests.

Validates:
    1. Shared day-ahead window helper (get_dayahead_window, day_ahead_mask,
       filter_dayahead, get_business_day_info)
    2. Hour-24 completeness checker (COMPLETE_24H, INCOMPLETE_23H,
       MISSING_HOURS, DUPLICATE_HOURS, INVALID)
    3. Metric computation (sMAPE_floor50, MAE, RMSE)
    4. P15 pipeline structural tests (missing data, unsafe paths,
       forbidden files)
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from artifacts.dayahead_window import (
    day_ahead_mask,
    filter_dayahead,
    get_business_day_info,
    get_dayahead_window,
)
from scripts.check_cfg05_hour24_completeness import (
    COMPLETE_24H,
    DUPLICATE_HOURS,
    INCOMPLETE_23H,
    INVALID,
    MISSING_HOURS,
    check_cfg05_hour24_completeness,
)
from scripts.run_p15_cfg05_24h_smoke_and_eval import (
    compute_mae,
    compute_metrics,
    compute_rmse,
    compute_smape_floor50,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_24h_csv(
    tmp_path,
    target_day: str = "2026-06-30",
    filename: str = "test_24h.csv",
    missing_hours: list[int] | None = None,
    add_duplicate_hour: int | None = None,
    include_hour24: bool = True,
    extra_col: str | None = None,
) -> str:
    """Create a CSV with 24-hour day-ahead rows for a target day.

    Parameters
    ----------
    tmp_path : pytest tmp_path
    target_day : str
        The business day (YYYY-MM-DD).
    filename : str
        CSV filename.
    missing_hours : list[int], optional
        Hours to exclude (1..24).
    add_duplicate_hour : int, optional
        If set, duplicate this hour row.
    include_hour24 : bool
        Whether to include hour 24 (D+1 00:00).
    extra_col : str, optional
        Extra column name to add.

    Returns
    -------
    str
        Path to created CSV.
    """
    target_dt = pd.Timestamp(target_day)
    missing_hours = missing_hours or []

    rows = []
    for h in range(1, 25):
        if h in missing_hours:
            continue
        if h == 24 and not include_hour24:
            continue

        # D 01:00 → D+1 00:00
        ts = target_dt + pd.Timedelta(hours=h)

        row = {
            "ds": ts,
            "hour_business": h,
            "y_pred": 300 + h * 10,
        }
        if extra_col:
            row[extra_col] = f"val_{h}"
        rows.append(row)

        if add_duplicate_hour and h == add_duplicate_hour:
            # Add a duplicate row with slightly different ds
            dup_ts = ts + pd.Timedelta(minutes=1)
            dup_row = dict(row)
            dup_row["ds"] = dup_ts
            rows.append(dup_row)

    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), filename)
    df.to_csv(path, index=False)
    return path


def _make_24h_feature_csv(
    tmp_path,
    target_day: str = "2026-06-30",
    missing_hours: list[int] | None = None,
) -> str:
    """Create a CSV that looks like a cfg05 feature CSV with 24-hour window."""
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

    missing_hours = missing_hours or []
    target_dt = pd.Timestamp(target_day)

    rows = []
    for h in range(1, 25):
        if h in missing_hours:
            continue
        ts = target_dt + pd.Timedelta(hours=h)
        row = {"ds": ts}
        for col in CFG05_FEATURE_COLUMNS:
            row[col] = 0.5
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(str(tmp_path), "test_features_24h.csv")
    df.to_csv(path, index=False)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Shared day-ahead window helper tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDayaheadWindowHelper:
    """Contract: artifacts/dayahead_window.py."""

    def test_get_dayahead_window_correct_bounds(self):
        """get_dayahead_window returns D+01:00 and D+1+01:00."""
        start, end = get_dayahead_window("2026-06-30")
        assert start == pd.Timestamp("2026-06-30 01:00:00")
        assert end == pd.Timestamp("2026-07-01 01:00:00")

    def test_get_dayahead_window_hour24_included(self):
        """D+1 00:00 falls within [start, end)."""
        start, end = get_dayahead_window("2026-06-30")
        hour24 = pd.Timestamp("2026-07-01 00:00:00")
        assert start <= hour24 < end  # hour 24 is INCLUDED

    def test_day_ahead_mask_24_rows(self):
        """day_ahead_mask returns exactly 24 rows for a complete day."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-06-30 01:00", "2026-07-01 00:00", freq="1h"),
        })
        assert len(df) == 24
        mask = day_ahead_mask(df, "2026-06-30")
        assert mask.sum() == 24

    def test_day_ahead_mask_excludes_d_plus_1_01(self):
        """day_ahead_mask excludes D+1 01:00."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-06-30 01:00", "2026-07-01 02:00", freq="1h"),
        })
        mask = day_ahead_mask(df, "2026-06-30")
        assert mask.sum() == 24  # only 24 rows, D+1 01:00 is excluded

    def test_filter_dayahead_sorted(self):
        """filter_dayahead returns sorted DataFrame."""
        df = pd.DataFrame({
            "ds": [
                "2026-07-01 00:00",
                "2026-06-30 03:00",
                "2026-06-30 01:00",
                "2026-06-30 02:00",
            ],
        })
        result = filter_dayahead(df, "2026-06-30")
        ds_seq = pd.to_datetime(result["ds"])
        assert ds_seq.is_monotonic_increasing

    def test_get_business_day_info_hour24(self):
        """get_business_day_info: D+1 00:00 → hour_business=24,
        business_day=D."""
        ds = pd.Series([pd.Timestamp("2026-07-01 00:00")])
        info = get_business_day_info(ds)
        assert info["hour_business"].iloc[0] == 24
        assert info["business_day"].iloc[0] == pd.Timestamp("2026-06-30").date()

    def test_get_business_day_info_hour1(self):
        """get_business_day_info: D 01:00 → hour_business=1,
        business_day=D."""
        ds = pd.Series([pd.Timestamp("2026-06-30 01:00")])
        info = get_business_day_info(ds)
        assert info["hour_business"].iloc[0] == 1
        assert info["business_day"].iloc[0] == pd.Timestamp("2026-06-30").date()

    def test_get_business_day_info_periods(self):
        """get_business_day_info assigns correct periods."""
        ds = pd.Series([
            pd.Timestamp("2026-06-30 04:00"),  # hour 4 → 1_8
            pd.Timestamp("2026-06-30 12:00"),  # hour 12 → 9_16
            pd.Timestamp("2026-06-30 20:00"),  # hour 20 → 17_24
        ])
        info = get_business_day_info(ds)
        assert list(info["period"]) == ["1_8", "9_16", "17_24"]


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Hour-24 completeness checker tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHour24CompletenessChecker:
    """Contract: check_cfg05_hour24_completeness."""

    def test_missing_input_file(self):
        """Missing input file returns INVALID with INPUT_FILE_MISSING."""
        result = check_cfg05_hour24_completeness(
            input_path="/nonexistent/file.csv",
            target_day="2026-06-30",
        )
        assert result["completeness_status"] == INVALID
        assert "INPUT_FILE_MISSING" in result["reason_codes"]

    def test_no_target_day(self, tmp_path):
        """No target day returns INVALID with TARGET_DAY_NOT_PROVIDED."""
        path = _make_24h_csv(tmp_path)
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day=None,
        )
        assert result["completeness_status"] == INVALID
        assert "TARGET_DAY_NOT_PROVIDED" in result["reason_codes"]

    def test_complete_24h(self, tmp_path):
        """24 rows with hours 1..24 → COMPLETE_24H."""
        path = _make_24h_csv(tmp_path)
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day="2026-06-30",
        )
        assert result["completeness_status"] == COMPLETE_24H
        assert result["row_count"] == 24
        assert len(result["missing_hours"]) == 0

    def test_complete_24h_feature_csv(self, tmp_path):
        """Feature CSV with 24 rows → COMPLETE_24H (no hour_business column)."""
        path = _make_24h_feature_csv(tmp_path)
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day="2026-06-30",
        )
        assert result["completeness_status"] == COMPLETE_24H
        assert result["row_count"] == 24

    def test_incomplete_23h_only_hour24_missing(self, tmp_path):
        """23 rows, only hour 24 missing → INCOMPLETE_23H."""
        path = _make_24h_csv(tmp_path, include_hour24=False)
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day="2026-06-30",
        )
        assert result["completeness_status"] == INCOMPLETE_23H
        assert result["missing_hours"] == [24]
        assert result["row_count"] == 23

    def test_missing_hours_middle(self, tmp_path):
        """Missing hour 5 and 10 → MISSING_HOURS."""
        path = _make_24h_csv(tmp_path, missing_hours=[5, 10])
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day="2026-06-30",
        )
        assert result["completeness_status"] == MISSING_HOURS
        assert 5 in result["missing_hours"]
        assert 10 in result["missing_hours"]

    def test_duplicate_hours(self, tmp_path):
        """Duplicate hour 12 → DUPLICATE_HOURS."""
        path = _make_24h_csv(tmp_path, add_duplicate_hour=12)
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day="2026-06-30",
        )
        assert result["completeness_status"] == DUPLICATE_HOURS
        assert 12 in result["duplicate_hours"]

    def test_no_rows_in_window(self, tmp_path):
        """CSV with timestamps outside the target window → MISSING_HOURS."""
        df = pd.DataFrame({
            "ds": pd.date_range("2026-05-01", periods=10, freq="1h"),
        })
        path = os.path.join(str(tmp_path), "outside.csv")
        df.to_csv(path, index=False)
        result = check_cfg05_hour24_completeness(
            input_path=path,
            target_day="2026-06-30",
        )
        assert result["completeness_status"] in (MISSING_HOURS, INVALID)
        assert any("NO_ROWS_IN_TARGET_WINDOW" in rc for rc in result["reason_codes"])

    def test_strict_mode_exit_nonzero(self, tmp_path):
        """Strict mode exits non-zero for MISSING_HOURS."""
        from scripts.check_cfg05_hour24_completeness import main

        path = _make_24h_csv(tmp_path, missing_hours=[1])
        exit_code = main([
            "--input", path,
            "--target-day", "2026-06-30",
            "--strict",
        ])
        assert exit_code != 0

    def test_non_strict_exit_zero_on_missing(self, tmp_path):
        """Non-strict exits 0 even with missing hours."""
        from scripts.check_cfg05_hour24_completeness import main

        path = _make_24h_csv(tmp_path, missing_hours=[1])
        exit_code = main([
            "--input", path,
            "--target-day", "2026-06-30",
        ])
        assert exit_code == 0


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Metric computation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMetrics:
    """Contract: metrics in run_p15_cfg05_24h_smoke_and_eval."""

    def test_smape_floor50_identical(self):
        """sMAPE_floor50 = 0 when predictions equal actuals."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])
        smape = compute_smape_floor50(y_true, y_pred)
        assert smape == 0.0

    def test_smape_floor50_known_value(self):
        """sMAPE_floor50 with known values."""
        # y_true=[100,100], y_pred=[120,80]
        # sMAPE = 200% * mean(|120-100|/(120+100) + |80-100|/(80+100))
        #       = 2 * mean(20/220 + 20/180) * 100
        #       = 2 * mean(0.0909 + 0.1111) * 100
        #       = 2 * 0.1010 * 100
        #       = 20.20%
        # With floor50: no values below 50, so same result
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([120.0, 80.0])
        smape = compute_smape_floor50(y_true, y_pred)
        expected = 200 * np.mean([20 / 220, 20 / 180])
        assert abs(smape - expected) < 1e-10

    def test_smape_floor50_applies_floor(self):
        """sMAPE_floor50 floors values below 50."""
        y_true = np.array([10.0, 200.0])  # 10 gets floored to 50
        y_pred = np.array([15.0, 200.0])  # 15 gets floored to 50
        # After floor: y_true=[50,200], y_pred=[50,200]
        # sMAPE = 200% * mean(0/100 + 0/400) = 0
        smape = compute_smape_floor50(y_true, y_pred)
        assert smape == 0.0

    def test_mae(self):
        """MAE computed correctly."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 290])
        mae = compute_mae(y_true, y_pred)
        assert mae == 10.0

    def test_rmse(self):
        """RMSE computed correctly."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 290])
        # errors: 10, -10, -10
        # mse = (100 + 100 + 100) / 3 = 100
        # rmse = 10
        rmse = compute_rmse(y_true, y_pred)
        assert rmse == 10.0

    def test_compute_metrics(self):
        """compute_metrics returns dict with all keys."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 290])
        metrics = compute_metrics(y_true, y_pred)
        assert "sMAPE_floor50" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "n_observations" in metrics
        assert metrics["n_observations"] == 3

    def test_compute_metrics_empty(self):
        """compute_metrics with empty arrays returns nan."""
        metrics = compute_metrics(np.array([]), np.array([]))
        assert np.isnan(metrics["sMAPE_floor50"])
        assert np.isnan(metrics["MAE"])
        assert np.isnan(metrics["RMSE"])
        assert metrics["n_observations"] == 0


# ══════════════════════════════════════════════════════════════════════════════
#  4.  P15 pipeline structural tests
# ══════════════════════════════════════════════════════════════════════════════


class TestP15Pipeline:
    """Contract: run_p15_cfg05_24h_smoke_and_eval."""

    def test_missing_raw_data_returns_missing(self):
        """Missing raw data returns CFG05_RAW_DATA_MISSING."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import (
            run_p15_cfg05_24h_smoke_and_eval,
        )

        result = run_p15_cfg05_24h_smoke_and_eval(raw_data=None)
        assert result["final_status"] == "CFG05_RAW_DATA_MISSING"

    def test_nonexistent_raw_data_path_returns_missing(self):
        """Non-existent raw data path returns CFG05_RAW_DATA_MISSING."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import (
            run_p15_cfg05_24h_smoke_and_eval,
        )

        result = run_p15_cfg05_24h_smoke_and_eval(raw_data="/nonexistent/path.csv")
        assert result["final_status"] == "CFG05_RAW_DATA_MISSING"

    def test_missing_raw_data_exit_0_non_strict(self):
        """Missing raw data exits 0 non-strict."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import main

        exit_code = main([])
        assert exit_code == 0

    def test_missing_raw_data_exits_nonzero_strict(self):
        """Missing raw data exits non-zero in strict mode."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import main

        exit_code = main(["--strict"])
        assert exit_code != 0

    def test_missing_source_repo_reported(self, tmp_path):
        """Missing source repo is reported."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import (
            run_p15_cfg05_24h_smoke_and_eval,
        )

        result = run_p15_cfg05_24h_smoke_and_eval(
            raw_data=os.path.join(str(tmp_path), "dummy.csv"),
            source_repo="/nonexistent_repo",
        )

        # Create dummy CSV so raw_data check passes
        pd.DataFrame({"时刻": ["2026-06-30 01:00"], "日前电价": [300]}).to_csv(
            os.path.join(str(tmp_path), "dummy.csv"), index=False, encoding="gbk",
        )

        result = run_p15_cfg05_24h_smoke_and_eval(
            raw_data=os.path.join(str(tmp_path), "dummy.csv"),
            source_repo="/nonexistent_repo",
        )
        assert result["source_repo_status"] == "MISSING"
        assert result["final_status"] == "CFG05_SOURCE_REPO_MISSING"

    def test_uses_local_artifacts_p15_by_default(self):
        """P15 uses .local_artifacts/p15_cfg05/ by default."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import (
            run_p15_cfg05_24h_smoke_and_eval,
        )

        result = run_p15_cfg05_24h_smoke_and_eval(raw_data=None)
        assert ".local_artifacts" in result["model_path"]
        assert "p15_cfg05" in result["model_path"]
        assert ".local_artifacts" in result["features_path"]
        assert "p15_cfg05" in result["features_path"]

    def test_unsafe_work_dir_rejected(self):
        """Unsafe work-dir is rejected by CLI."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import main

        exit_code = main(["--work-dir", "outputs/unsafe_p15"])
        assert exit_code != 0

    def test_unsafe_model_path_rejected(self):
        """Unsafe model path is rejected by CLI."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import main

        exit_code = main(["--model", "data/unsafe_model.txt"])
        assert exit_code != 0

    def test_unsafe_features_path_rejected(self):
        """Unsafe features path is rejected by CLI."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import main

        exit_code = main(["--features", "outputs/unsafe_features.csv"])
        assert exit_code != 0

    def test_summary_contains_required_keys(self):
        """Summary dict contains all required keys."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import (
            run_p15_cfg05_24h_smoke_and_eval,
        )

        result = run_p15_cfg05_24h_smoke_and_eval(raw_data=None)
        required = [
            "raw_data_status", "source_repo_status",
            "train_attempted", "train_done",
            "model_path", "features_path",
            "primary_prediction", "eval_attempted",
            "final_status", "reason_codes",
            "hour24_fix_applied",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

        pp = result["primary_prediction"]
        pp_required = [
            "target_day", "attempted", "prediction_rows",
            "feature_hours_status", "prediction_hours_status",
            "validator_passed",
        ]
        for key in pp_required:
            assert key in pp, f"Missing primary_prediction key: {key}"

    def test_hour24_fix_applied_flag(self):
        """hour24_fix_applied is True by default."""
        from scripts.run_p15_cfg05_24h_smoke_and_eval import (
            run_p15_cfg05_24h_smoke_and_eval,
        )

        result = run_p15_cfg05_24h_smoke_and_eval(raw_data=None)
        assert result["hour24_fix_applied"] is True

    def test_forbidden_files_check(self):
        """No forbidden file extensions in untracked files."""
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        for f in result.stdout.strip().split("\n"):
            if not f.strip():
                continue
            ext = os.path.splitext(f)[1].lower()
            assert ext not in (
                ".csv", ".pkl", ".joblib", ".parquet",
                ".feather", ".pt", ".pth", ".ckpt",
            ), f"Forbidden untracked file: {f}"

    def test_no_generated_artifacts_in_repo(self):
        """No generated forbidden files are tracked in repo."""
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        forbidden_exts = (
            ".pkl", ".joblib", ".parquet",
            ".feather", ".pt", ".pth", ".ckpt",
        )
        # Allow .csv test fixtures if tracked
        for f in result.stdout.strip().split("\n"):
            if not f.strip():
                continue
            ext = os.path.splitext(f)[1].lower()
            assert ext not in forbidden_exts, (
                f"Forbidden tracked file: {f}"
            )
