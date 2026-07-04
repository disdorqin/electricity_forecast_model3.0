"""
scripts/train_export_cfg05_local.py — Local cfg05 LightGBM training & export.

Loads raw Chinese CSV via source repo data_loader, builds 42-column features
via source feature_builder_dayahead, trains a LightGBM model with 3.0 CFG05_PARAMS,
saves model file and feature input CSV to .local_artifacts/p13_cfg05/.

Usage::

    python -m scripts.train_export_cfg05_local \\
        --raw-data /path/to/raw.csv --target-day 2026-07-01

    python -m scripts.train_export_cfg05_local \\
        --raw-data /path/to/raw.csv --target-day 2026-07-01 \\
        --force --json --strict

Options::

    --source-repo PATH          Path to epf-sota-experiment (default: .local_artifacts/...).
    --raw-data PATH             Path to raw Chinese CSV (required).
    --target-day YYYY-MM-DD     Target day for prediction (default: 2026-07-01).
    --train-window-days N       Training window in days (default: 90).
    --work-dir PATH             Local work dir (default: .local_artifacts/p13_cfg05).
    --model-out PATH            Output model path (default: work-dir/cfg05_model.txt).
    --features-out PATH         Output feature CSV path (default: auto).
    --force                     Overwrite existing output files.
    --json                      Output JSON report.
    --strict                    Exit non-zero on any failure.
    --verbose, -v               Increase log verbosity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from typing import Any, Optional

import pandas as pd

from artifacts.readiness import check_cfg05_artifact, check_cfg05_input, LOADABLE, SCHEMA_READY
from scripts.check_cfg05_raw_data_contract import (
    check_cfg05_raw_data_contract,
    RAW_DATA_VALID,
)

logger = logging.getLogger(__name__)

# ── Safe paths ─────────────────────────────────────────────────────────────
_ALLOWED_WORK_DIRS = (".local_artifacts",)
_FORBIDDEN_PATH_PARTS = ("data/", "outputs/", "ledgers/", "reports/local/")

_DEFAULT_SOURCE_REPO = os.path.join(
    ".local_artifacts", "source_repos", "epf-sota-experiment",
)
_DEFAULT_WORK_DIR = os.path.join(".local_artifacts", "p13_cfg05")


def _path_is_safe(path: str) -> bool:
    """Check path is under an ignored, allowed directory."""
    norm = path.replace("\\", "/")
    if any(f in norm for f in _FORBIDDEN_PATH_PARTS):
        return False
    if not os.path.isabs(norm):
        return any(norm.startswith(a) for a in _ALLOWED_WORK_DIRS)
    return True


def _import_source_module(source_repo: str, module_name: str):
    """Import a module from the source repo by name.

    Parameters
    ----------
    source_repo : str
        Path to epf-sota-experiment.
    module_name : str
        Dotted module path within source repo (e.g. 'src.common.data_loader').

    Returns
    -------
    The imported module.
    """
    full_path = os.path.join(source_repo, *module_name.split(".")) + ".py"
    if not os.path.isfile(full_path):
        raise ImportError(f"Source module not found: {full_path}")
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {full_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_export_cfg05_local(
    source_repo: Optional[str] = None,
    raw_data: Optional[str] = None,
    target_day: Optional[str] = None,
    train_window_days: int = 90,
    work_dir: Optional[str] = None,
    model_out: Optional[str] = None,
    features_out: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Train and export cfg05 model + features from raw Chinese CSV.

    Parameters
    ----------
    source_repo : str, optional
        Path to epf-sota-experiment.
    raw_data : str, optional
        Path to raw Chinese CSV.
    target_day : str, optional
        Target day in YYYY-MM-DD.
    train_window_days : int
        Training window in days (default 90).
    work_dir : str, optional
        Local work dir.
    model_out : str, optional
        Output model path.
    features_out : str, optional
        Output feature CSV path.
    force : bool
        Overwrite existing files.

    Returns
    -------
    dict with train/export summary.
    """
    source_repo = source_repo or _DEFAULT_SOURCE_REPO
    target_day = target_day or "2026-07-01"
    work_dir = work_dir or _DEFAULT_WORK_DIR
    os.makedirs(work_dir, exist_ok=True)

    model_out = model_out or os.path.join(work_dir, "cfg05_model.txt")
    features_out = features_out or os.path.join(
        work_dir, f"cfg05_features_{target_day}.csv",
    )

    result: dict[str, Any] = {
        "source_repo": source_repo,
        "source_repo_status": "NOT_CHECKED",
        "raw_data_path": raw_data,
        "raw_data_status": "NOT_CHECKED",
        "model_out": model_out,
        "features_out": features_out,
        "train_rows": 0,
        "train_done": False,
        "model_saved": False,
        "features_saved": False,
        "cfg05_artifact_status": None,
        "cfg05_input_status": None,
        "reason_codes": [],
    }

    # ── Step 1: Check source repo ──
    if not os.path.isdir(source_repo):
        result["source_repo_status"] = "MISSING"
        result["reason_codes"].append(f"SOURCE_REPO_MISSING: {source_repo}")
        return result

    result["source_repo_status"] = "PRESENT"
    result["reason_codes"].append(f"SOURCE_REPO_FOUND: {source_repo}")

    # ── Step 2: Check raw data contract ──
    contract = check_cfg05_raw_data_contract(raw_data=raw_data)
    result["raw_data_status"] = contract["raw_data_status"]
    if contract["reason_codes"]:
        result["reason_codes"].extend(
            [f"RAW_DATA:{rc}" for rc in contract["reason_codes"]]
        )

    if contract["raw_data_status"] != RAW_DATA_VALID:
        result["reason_codes"].append("TRAIN_SKIPPED_RAW_DATA_INVALID")
        return result

    # ── Step 3: Import source modules ──
    try:
        # Add source repo to sys.path for internal imports
        if source_repo not in sys.path:
            sys.path.insert(0, source_repo)

        data_loader = _import_source_module(source_repo, "src.common.data_loader")
        feature_builder = _import_source_module(
            source_repo, "src.common.feature_builder_dayahead"
        )
        lgb_adapter_cls = _import_source_module(
            source_repo, "src.models.lightgbm_dayahead_adapter"
        ).LightGBMDayaheadAdapter

        result["reason_codes"].append("SOURCE_MODULES_IMPORTED")
    except Exception as e:
        result["reason_codes"].append(f"SOURCE_MODULE_IMPORT_FAILED: {e}")
        return result

    # ── Step 4: Load raw data ──
    try:
        df = data_loader.load_data(raw_data, target="dayahead")
        logger.info("Raw data loaded: %d rows, columns=%s", len(df), list(df.columns))
        result["reason_codes"].append(f"RAW_DATA_LOADED: {len(df)} rows")
    except Exception as e:
        result["reason_codes"].append(f"RAW_DATA_LOAD_FAILED: {e}")
        return result

    # ── Step 5: Build features ──
    try:
        df_feat = feature_builder.build_features_dayahead(df, use_extended=True)
        logger.info(
            "Features built: %d rows, %d columns",
            len(df_feat), len(df_feat.columns),
        )
        result["reason_codes"].append(f"FEATURES_BUILT: {len(df_feat)} rows")
    except Exception as e:
        result["reason_codes"].append(f"FEATURE_BUILD_FAILED: {e}")
        return result

    # ── Step 6: Verify CFG05_FEATURE_COLUMNS ──
    from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS

    missing_feats = [c for c in CFG05_FEATURE_COLUMNS if c not in df_feat.columns]
    if missing_feats:
        result["reason_codes"].append(
            f"FEATURE_BUILD_MISSING_{len(missing_feats)}_CFG05_COLUMNS"
        )
        return result

    result["reason_codes"].append(
        f"ALL_{len(CFG05_FEATURE_COLUMNS)}_CFG05_FEATURES_PRESENT"
    )

    # ── Step 7: No-leakage split ──
    target_dt = pd.Timestamp(target_day)
    train_start = target_dt - pd.Timedelta(days=train_window_days)
    train_end = target_dt - pd.Timedelta(hours=1)

    train_mask = (
        (df_feat["ds"] >= train_start) & (df_feat["ds"] < train_end)
    )
    train_df = df_feat[train_mask].copy()
    result["train_rows"] = len(train_df)

    if len(train_df) < 100:
        result["reason_codes"].append(
            f"TRAIN_DATA_INSUFFICIENT: {len(train_df)} rows < 100"
        )
        result["reason_codes"].append("CFG05_LOCAL_TRAIN_FAILED")
        return result

    # Feature input rows: target_day + 1 hour to target_day + 1 day
    feat_start = target_dt + pd.Timedelta(hours=1)
    feat_end = target_dt + pd.Timedelta(days=1)
    feat_mask = (
        (df_feat["ds"] >= feat_start) & (df_feat["ds"] < feat_end)
    )
    feat_df = df_feat[feat_mask].copy()
    logger.info(
        "Train: %d rows, Feature input: %d rows",
        len(train_df), len(feat_df),
    )

    # ── Step 8: Train LightGBM ──
    try:
        import lightgbm as lgb
    except ImportError:
        result["reason_codes"].append("LIGHTGBM_NOT_INSTALLED")
        result["reason_codes"].append("CFG05_LOCAL_TRAIN_FAILED")
        return result

    from models.adapters.cfg05_dayahead_lgbm import CFG05_PARAMS

    try:
        # Prepare train data
        X_train = train_df[CFG05_FEATURE_COLUMNS].fillna(0).values
        y_train = train_df["y"].values

        # Use CFG05_PARAMS as base, with added training controls
        params = dict(CFG05_PARAMS)
        params["verbosity"] = -1
        n_estimators = params.pop("n_estimators", 2000)

        booster = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            num_boost_round=n_estimators,
            callbacks=[lgb.log_evaluation(0)],
        )
        result["train_done"] = True
        result["reason_codes"].append(
            f"TRAINING_COMPLETED: {booster.best_iteration} iterations"
        )
    except Exception as e:
        result["reason_codes"].append(f"TRAINING_FAILED: {e}")
        result["reason_codes"].append("CFG05_LOCAL_TRAIN_FAILED")
        return result

    # ── Step 9: Save model ──
    if not force and os.path.exists(model_out):
        result["reason_codes"].append(
            f"MODEL_OUT_EXISTS_SKIP: {model_out} (use --force to overwrite)"
        )
    else:
        try:
            os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
            booster.save_model(model_out)
            result["model_saved"] = True
            result["reason_codes"].append(f"MODEL_SAVED: {model_out}")
        except Exception as e:
            result["reason_codes"].append(f"MODEL_SAVE_FAILED: {e}")
            result["reason_codes"].append("CFG05_LOCAL_EXPORT_FAILED")
            return result

    # ── Step 10: Save feature input CSV ──
    if len(feat_df) > 0:
        out_cols = ["ds"] + list(CFG05_FEATURE_COLUMNS)
        available_out = [c for c in out_cols if c in feat_df.columns]
        if "ds" not in available_out:
            result["reason_codes"].append("FEATURE_OUTPUT_MISSING_DS")
            result["reason_codes"].append("CFG05_INPUT_EXPORT_FAILED")
            return result

        if not force and os.path.exists(features_out):
            result["reason_codes"].append(
                f"FEATURES_OUT_EXISTS_SKIP: {features_out} (use --force to overwrite)"
            )
        else:
            try:
                feat_df[available_out].to_csv(features_out, index=False)
                result["features_saved"] = True
                result["reason_codes"].append(f"FEATURES_SAVED: {features_out}")
            except Exception as e:
                result["reason_codes"].append(f"FEATURES_SAVE_FAILED: {e}")
                result["reason_codes"].append("CFG05_INPUT_EXPORT_FAILED")
                return result
    else:
        result["reason_codes"].append("FEATURE_INPUT_EMPTY_NO_ROWS_FOR_TARGET_DAY")
        result["reason_codes"].append("CFG05_INPUT_EXPORT_FAILED")
        return result

    # ── Step 11: Run readiness checks on exported files ──
    if result["model_saved"]:
        artifact_status = check_cfg05_artifact(model_out)
        result["cfg05_artifact_status"] = artifact_status.status
        result["reason_codes"].extend(
            [f"ARTIFACT:{rc}" for rc in artifact_status.reason_codes]
        )
    else:
        result["cfg05_artifact_status"] = "NOT_SAVED"

    if result["features_saved"]:
        input_status = check_cfg05_input(features_out)
        result["cfg05_input_status"] = input_status.status
        result["reason_codes"].extend(
            [f"INPUT:{rc}" for rc in input_status.reason_codes]
        )
    else:
        result["cfg05_input_status"] = "NOT_SAVED"

    return result


def _print_report(result: dict[str, Any]) -> None:
    """Print human-readable train/export report."""
    print("=" * 60)
    print("cfg05 Local Train / Export Report")
    print("=" * 60)
    print(f"  Source repo:      {result['source_repo']} ({result['source_repo_status']})")
    print(f"  Raw data:         {result['raw_data_path'] or 'N/A'} ({result['raw_data_status']})")
    print(f"  Train rows:       {result['train_rows']}")
    print(f"  Train done:       {result['train_done']}")
    print(f"  Model saved:      {result['model_saved']} -> {result['model_out']}")
    print(f"  Features saved:   {result['features_saved']} -> {result['features_out']}")
    if result["cfg05_artifact_status"]:
        print(f"  Artifact status:  {result['cfg05_artifact_status']}")
    if result["cfg05_input_status"]:
        print(f"  Input status:     {result['cfg05_input_status']}")
    print()
    print("  Reason codes:")
    for rc in result.get("reason_codes", []):
        print(f"    -> {rc}")
    print("=" * 60)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export cfg05 model + features from raw Chinese CSV.",
    )
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to epf-sota-experiment.")
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Path to raw Chinese CSV (required).")
    parser.add_argument("--target-day", type=str, default="2026-07-01",
                        help="Target day (YYYY-MM-DD).")
    parser.add_argument("--train-window-days", type=int, default=90,
                        help="Training window in days.")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Local work dir.")
    parser.add_argument("--model-out", type=str, default=None,
                        help="Output model path.")
    parser.add_argument("--features-out", type=str, default=None,
                        help="Output feature CSV path.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite existing output files.")
    parser.add_argument("--json", action="store_true", default=False,
                        help="Output JSON report.")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Exit non-zero on any failure.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Increase verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Validate work-dir safety
    work_dir = args.work_dir or _DEFAULT_WORK_DIR
    if not _path_is_safe(work_dir):
        logger.error("Unsafe work-dir: %s", work_dir)
        return 1

    model_out = args.model_out
    if model_out and not _path_is_safe(model_out):
        logger.error("Unsafe model-out path: %s", model_out)
        return 1

    features_out = args.features_out
    if features_out and not _path_is_safe(features_out):
        logger.error("Unsafe features-out path: %s", features_out)
        return 1

    result = train_export_cfg05_local(
        source_repo=args.source_repo,
        raw_data=args.raw_data,
        target_day=args.target_day,
        train_window_days=args.train_window_days,
        work_dir=work_dir,
        model_out=model_out,
        features_out=features_out,
        force=args.force,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_report(result)

    if args.strict:
        if result["train_done"] and result["model_saved"] and result["features_saved"]:
            logger.info("Train/export: PASS")
            return 0
        else:
            logger.error("Train/export: FAIL")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
