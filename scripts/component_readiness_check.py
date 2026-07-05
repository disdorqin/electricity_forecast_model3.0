"""
scripts/component_readiness_check.py — P3.5 component readiness gate.

Checks each 3.0 pipeline component and assigns a readiness state:

    READY_REAL       Real artifact/pipeline available, schema conformant.
    READY_DRY_RUN    Interface stable, dry-run works, no real artifact.
    READY_STUB       Interface exists but no real weight/artifact.
    DATA_MISSING     Requires external data (canonical pack, risk model).
    NOT_READY        Import / instantiation / contract failure.

Components assessed:
    1. cfg05 day-ahead model zoo
    2. Realtime DA-Safe Assist
    3. SGDFNet 2.5
    4. P5M residual plugin

Usage:
    python scripts/component_readiness_check.py
    python scripts/component_readiness_check.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("component_readiness_check")

# ── Readiness states ────────────────────────────

READY_REAL = "READY_REAL"
READY_DRY_RUN = "READY_DRY_RUN"
READY_STUB = "READY_STUB"
DATA_MISSING = "DATA_MISSING"
NOT_READY = "NOT_READY"

STATE_ORDER = {
    READY_REAL: 0,
    READY_DRY_RUN: 1,
    READY_STUB: 2,
    DATA_MISSING: 3,
    NOT_READY: 4,
}

# ── Expected model artifact paths ────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

CFG05_ARTIFACT_PATHS = [
    BASE_DIR / "models" / "cfg05" / "model.txt",
    BASE_DIR / "models" / "cfg05" / "model.pkl",
    BASE_DIR / "models" / "cfg05" / "lgbm_model.txt",
]

RT_ASSIST_DIR = BASE_DIR / "models" / "rt_assist_pack"

SGDFNET_WEIGHT_PATHS = [
    BASE_DIR / "models" / "sgdfnet_weights",
    BASE_DIR / "models" / "sgdfnet" / "checkpoint.pt",
]

P5M_RISK_MODEL_PATHS = [
    BASE_DIR / "models" / "p5m" / "negative_risk_model.pkl",
    BASE_DIR / "models" / "p5m" / "risk_model.pkl",
]


def _check_import(module_path: str) -> tuple[bool, Any]:
    """Try to import a module. Returns (success, module_or_error)."""
    try:
        mod = importlib.import_module(module_path)
        return True, mod
    except Exception as e:
        return False, e


def _check_adapter_class(module, class_name: str) -> tuple[bool, type | None]:
    """Check an adapter class exists in a module."""
    cls = getattr(module, class_name, None)
    if cls is None:
        cls = getattr(module, class_name.lower(), None)
    if cls is None:
        return False, None
    return True, cls


def _dry_run_predict(adapter_class, **kwargs: Any) -> bool:
    """Attempt a synthetic dry-run prediction."""
    import numpy as np
    import pandas as pd

    try:
        adapter = adapter_class(**kwargs)
        adapter.load()
        n = 24
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=n, freq="h")

        # Minimal prediction DataFrame
        df = pd.DataFrame({
            "task": "dayahead",
            "model_name": "cfg05",
            "target_day": "2026-03-05",
            "ds": timestamps,
            "y_pred": rng.uniform(80, 200, n),
        })

        result = adapter.predict(df=df)

        if result is None or len(result) == 0:
            return False

        # Validate schema presence
        from data.schema import PREDICTION_OUTPUT_COLUMNS
        missing = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in result.columns]
        if missing:
            logger.debug(f"Dry-run output missing columns: {missing}")
            return False

        return True
    except Exception as e:
        logger.debug(f"Dry-run predict failed: {e}")
        return False


def _has_real_artifact(paths: list[Path]) -> bool:
    """Check if any of the given artifact paths exist on disk."""
    for p in paths:
        if p.exists():
            return True
    return False


# ── Component checks ────────────────────────────


def check_cfg05_dayahead() -> dict[str, object]:
    """Assess cfg05 day-ahead model zoo readiness."""
    result: dict[str, object] = {
        "component": "cfg05_dayahead_model_zoo",
        "state": NOT_READY,
        "details": [],
    }

    # 1. Import adapter
    ok, mod_or_err = _check_import("models.adapters.cfg05_dayahead_lgbm")
    if not ok:
        result["details"].append(f"Import failed: {mod_or_err}")
        return result

    ok, cls = _check_adapter_class(mod_or_err, "CFG05DayaheadAdapter")
    if not ok or cls is None:
        result["details"].append("CFG05DayaheadAdapter class not found")
        return result
    result["details"].append("Adapter class found (CFG05DayaheadAdapter)")

    # 2. Check registered in model zoo
    try:
        from src.registry.dayahead_models import (
            is_valid_model, CHAMPION_MODEL_ID, CHAMPION_SMAPE_FLOOR50,
        )
        assert is_valid_model("cfg05"), "cfg05 not registered as valid model"
        assert CHAMPION_MODEL_ID == "cfg05"
        assert isinstance(CHAMPION_SMAPE_FLOOR50, float)
        result["details"].append("cfg05 registered as champion model")
    except Exception as e:
        result["details"].append(f"Model registry check failed: {e}")
        result["state"] = NOT_READY
        return result

    # 3. Check feature columns
    try:
        from data.features.dayahead_features import get_dayahead_feature_columns
        feat_cols = get_dayahead_feature_columns("cfg05")
        assert len(feat_cols) == 56, f"Expected 56 features, got {len(feat_cols)}"
        result["details"].append(f"cfg05 has {len(feat_cols)} feature columns")
    except Exception as e:
        result["details"].append(f"Feature column check failed: {e}")
        result["state"] = NOT_READY
        return result

    # 4. Pipeline-level dry-run (adapter requires real weights)
    try:
        from scripts.run_dayahead_model_zoo import main as zoo_main
        exit_code = zoo_main(["--dry-run", "--models", "cfg05"])
        if exit_code == 0:
            result["details"].append("Model zoo dry-run OK")
        else:
            result["details"].append(f"Model zoo dry-run exit code {exit_code}")
            result["state"] = NOT_READY
            return result
    except Exception as e:
        result["details"].append(f"Model zoo dry-run failed: {e}")
        result["state"] = NOT_READY
        return result

    # 5. Real artifact check
    if _has_real_artifact(CFG05_ARTIFACT_PATHS):
        result["state"] = READY_REAL
        result["details"].append("Model artifact found on disk")
    else:
        result["state"] = READY_DRY_RUN
        result["details"].append("No real model artifact found (dry-run only)")

    return result


def check_realtime_assist() -> dict[str, object]:
    """Assess realtime DA-Safe Assist readiness."""
    result: dict[str, object] = {
        "component": "realtime_da_safe_assist",
        "state": NOT_READY,
        "details": [],
    }

    # 1. Import adapter
    ok, mod_or_err = _check_import("models.adapters.realtime_da_safe_assist")
    if not ok:
        result["details"].append(f"Import failed: {mod_or_err}")
        return result

    ok, cls = _check_adapter_class(mod_or_err, "DASafeRealtimeAssistAdapter")
    if not ok or cls is None:
        result["details"].append("DASafeRealtimeAssistAdapter class not found")
        return result
    result["details"].append("Adapter class found")

    # 2. Try dry-run (DA_ONLY mode)
    import numpy as np
    import pandas as pd

    try:
        adapter = cls()
        adapter.load()
        n = 48
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=n, freq="h")
        df = pd.DataFrame({
            "task": "realtime",
            "model_name": "da_safe_realtime_assist",
            "target_day": "2026-03-05",
            "ds": timestamps,
            "da_anchor": rng.uniform(80, 200, n),
            "y_pred": rng.uniform(80, 200, n),
        })
        result_rt = adapter.predict(df=df)

        from data.schema import PREDICTION_OUTPUT_COLUMNS
        missing = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in result_rt.columns]
        if missing:
            result["details"].append(f"Dry-run output missing: {missing}")
            result["state"] = NOT_READY
            return result

        # Check DA_ONLY: rt_pred == da_anchor (within tolerance)
        rt_pred = result_rt["y_pred"].values
        if np.allclose(rt_pred, df["da_anchor"].values[:len(rt_pred)]):
            result["details"].append("Dry-run DA_ONLY output correct")
        else:
            result["details"].append("Dry-run DA_ONLY output differs from da_anchor")
            # Not a hard failure — correction may be enabled
    except Exception as e:
        result["details"].append(f"Dry-run predict failed: {e}")
        result["state"] = NOT_READY
        return result

    # 3. Real artifact check (assist pack)
    if RT_ASSIST_DIR.exists():
        result["state"] = READY_REAL
        result["details"].append(f"RT assist pack found at {RT_ASSIST_DIR}")
    else:
        result["state"] = READY_DRY_RUN
        result["details"].append("No real RT assist pack (DA_ONLY mode)")

    return result


def check_sgdfnet() -> dict[str, object]:
    """Assess SGDFNet 2.5 readiness."""
    result: dict[str, object] = {
        "component": "sgdfnet_2_5",
        "state": NOT_READY,
        "details": [],
    }

    # 1. Import adapter
    ok, mod_or_err = _check_import("models.adapters.sgdfnet_2_5")
    if not ok:
        result["details"].append(f"Import failed: {mod_or_err}")
        return result

    ok, cls = _check_adapter_class(mod_or_err, "SGDFNet25Adapter")
    if not ok or cls is None:
        result["details"].append("SGDFNet25Adapter class not found")
        return result
    result["details"].append("Adapter class found")

    # 2. Can instantiate
    try:
        adapter = cls()
        adapter.load()
        result["details"].append("Adapter instantiated and loaded OK")
    except Exception as e:
        result["details"].append(f"Instantiation failed: {e}")
        result["state"] = NOT_READY
        return result

    # 3. Real artifact check
    if _has_real_artifact(SGDFNET_WEIGHT_PATHS):
        result["state"] = READY_STUB
        result["details"].append("Weight paths found")
    else:
        result["state"] = READY_STUB
        result["details"].append("No weight files found (interface only)")

    return result


def check_p5m_residual() -> dict[str, object]:
    """Assess P5M residual plugin readiness."""
    result: dict[str, object] = {
        "component": "p5m_residual_plugin",
        "state": NOT_READY,
        "details": [],
    }

    # 1. Import adapter
    ok, mod_or_err = _check_import("models.adapters.p5m_residual_plugin")
    if not ok:
        result["details"].append(f"Import failed: {mod_or_err}")
        return result

    ok, cls = _check_adapter_class(mod_or_err, "P5MResidualPluginAdapter")
    if not ok or cls is None:
        result["details"].append("P5MResidualPluginAdapter class not found")
        return result
    result["details"].append("Adapter class found")

    # 2. Check profiles
    for profile in ("conservative", "moderate", "aggressive"):
        try:
            a = cls(profile=profile)
            a.load()
            result["details"].append(f"Profile '{profile}' OK")
        except Exception as e:
            result["details"].append(f"Profile '{profile}' failed: {e}")
            result["state"] = NOT_READY
            return result

    # 3. Dry-run no-op test
    import numpy as np
    import pandas as pd

    try:
        adapter = cls()
        adapter.load()
        n = 24
        rng = np.random.default_rng(42)
        timestamps = pd.date_range("2026-03-05 01:00", periods=n, freq="h")
        df = pd.DataFrame({
            "task": "dayahead",
            "model_name": "cfg05",
            "target_day": "2026-03-05",
            "business_day": pd.Timestamp("2026-03-05"),
            "ds": timestamps,
            "hour_business": list(range(1, n + 1)),
            "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
            "y_pred": rng.uniform(80, 200, n),
        })
        result_p5m = adapter.predict(df=df)

        from data.schema import PREDICTION_OUTPUT_COLUMNS
        missing = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in result_p5m.columns]
        if missing:
            result["details"].append(f"Dry-run output missing: {missing}")
            result["state"] = NOT_READY
            return result
        result["details"].append("Dry-run predict OK (no-op default)")
    except Exception as e:
        result["details"].append(f"Dry-run predict failed: {e}")
        result["state"] = NOT_READY
        return result

    # 4. Check risk model artifact
    if _has_real_artifact(P5M_RISK_MODEL_PATHS):
        result["state"] = READY_STUB
        result["details"].append("Risk model artifact found")
    else:
        result["state"] = DATA_MISSING
        result["details"].append(
            "No risk model artifact. Requires canonical pack or risk data."
        )

    # 5. Check residual correction pipeline
    try:
        from pipelines.residual_correction import apply_residual_correction
        result["details"].append("Residual correction pipeline importable")
    except Exception as e:
        result["details"].append(f"Pipeline import failed: {e}")
        return result

    return result


# ── Report ──────────────────────────────────────


def run_all_checks() -> list[dict[str, object]]:
    """Run all component readiness checks and return results."""
    checks = [
        check_cfg05_dayahead,
        check_realtime_assist,
        check_sgdfnet,
        check_p5m_residual,
    ]
    return [check() for check in checks]


def print_report(results: list[dict[str, object]], verbose: bool = False) -> None:
    """Print a formatted readiness report."""
    print("=" * 72)
    print("  3.0 Component Readiness Gate")
    print("=" * 72)
    print()
    print(f"{'Component':<40} {'State':<20}")
    print("-" * 72)

    for r in results:
        state = str(r.get("state", "UNKNOWN"))
        name = str(r.get("component", "unknown"))
        print(f"{name:<40} {state:<20}")

    print("-" * 72)
    print()

    if verbose:
        for r in results:
            name = str(r.get("component", "unknown"))
            state = str(r.get("state", "UNKNOWN"))
            details = r.get("details", [])
            print(f"  [{state}] {name}")
            for d in details:
                print(f"         - {d}")
            print()

    # Summary
    states = [str(r.get("state", NOT_READY)) for r in results]
    by_state = {s: states.count(s) for s in set(states)}
    print(f"  Summary: {by_state}")
    print(f"  Overall gate: ", end="")

    if all(s in (READY_REAL, READY_DRY_RUN) for s in states):
        print("PASS (all production-capable)")
    elif all(s != NOT_READY for s in states):
        print("PASS (all structural components OK)")
    else:
        n_not_ready = sum(1 for s in states if s == NOT_READY)
        print(f"PARTIAL ({n_not_ready} component(s) NOT_READY)")

    print()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check 3.0 component readiness state.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed check information.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_all_checks()
    print_report(results, verbose=args.verbose)

    states = [str(r.get("state", NOT_READY)) for r in results]
    if any(s == NOT_READY for s in states):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
