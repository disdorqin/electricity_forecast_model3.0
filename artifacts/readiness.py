"""
artifacts/readiness.py — Artifact readiness verification.

Defines status codes and check functions for every real artifact gate
in the electricity_forecast_model3.0 system:

1. cfg05 LightGBM model artifact
2. cfg05 input / feature schema
3. RT assist pack
4. P5M residual pack / risk data
5. Actual ledger
6. ExtremPriceClf artifact

Status codes:

MISSING
    Path not provided or does not exist.
PRESENT
    Path exists but not yet loaded or verified.
LOADABLE
    Adapter can load the artifact without error.
SCHEMA_READY
    Input schema / features are compatible.
REAL_READY
    Artifact loaded, real non-synthetic inference produced,
    and validator passed.
NOT_IMPLEMENTED
    Artifact exists but real inference path is still a stub.
INVALID
    Path exists but loading or validation failed.

Usage::

    from artifacts.readiness import (
        check_cfg05_artifact, check_cfg05_input,
        run_all_artifact_readiness, ArtifactStatus,
    )

    status = check_cfg05_artifact("/path/to/model.txt")
    print(status.status, status.reason_codes)

    report = run_all_artifact_readiness(cfg05_model="/path/to/model.txt")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────────────

MISSING = "MISSING"
PRESENT = "PRESENT"
LOADABLE = "LOADABLE"
SCHEMA_READY = "SCHEMA_READY"
REAL_READY = "REAL_READY"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
INVALID = "INVALID"

_VALID_STATUSES = frozenset({
    MISSING, PRESENT, LOADABLE, SCHEMA_READY,
    REAL_READY, NOT_IMPLEMENTED, INVALID,
})


# ── Data model ────────────────────────────────────────────────────────


@dataclass
class ArtifactStatus:
    """Status of a single artifact readiness gate.

    Attributes
    ----------
    name : str
        Human-readable artifact name (e.g. "cfg05_artifact").
    status : str
        One of the status constants.
    path : str or None
        Path that was checked.
    exists : bool
        Whether the path exists on disk.
    loadable : bool
        Whether the artifact could be loaded.
    schema_ready : bool
        Whether input schema / features are compatible.
    real_ready : bool
        Whether real inference was produced and validated.
    reason_codes : list[str]
        Audit trail for this status.
    details : dict
        Extra details (e.g. file size, adapter version).
    """

    name: str
    status: str = MISSING
    path: Optional[str] = None
    exists: bool = False
    loadable: bool = False
    schema_ready: bool = False
    real_ready: bool = False
    reason_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {_VALID_STATUSES}")


def status_to_dict(status: ArtifactStatus) -> dict[str, Any]:
    """Convert ArtifactStatus to a JSON-serializable dict."""
    return asdict(status)


# ── Shared helpers ────────────────────────────────────────────────────


def _path_exists(path: Optional[str]) -> bool:
    """Check if a path exists (None or empty → False)."""
    if not path:
        return False
    return os.path.exists(path)


def _check_extension(path: str, expected_ext: str) -> bool:
    """Check if a file has the expected extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() == expected_ext.lower()


# ── cfg05 artifact check ──────────────────────────────────────────────


def check_cfg05_artifact(model_path: Optional[str]) -> ArtifactStatus:
    """Check cfg05 LightGBM model artifact readiness.

    Parameters
    ----------
    model_path : str or None
        Path to cfg05 model file (*.txt) or directory containing model.txt.

    Returns
    -------
    ArtifactStatus
    """
    status = ArtifactStatus(
        name="cfg05_artifact",
        path=model_path,
    )

    if not _path_exists(model_path):
        status.reason_codes.append("CFG05_ARTIFACT_MISSING")
        return status

    assert model_path is not None  # narrowed by _path_exists
    status.exists = True
    status.status = PRESENT

    # Determine actual file path (directory or file)
    if os.path.isdir(model_path):
        candidates = [
            os.path.join(model_path, "cfg05_model.txt"),
            os.path.join(model_path, "model.txt"),
            os.path.join(model_path, "lightgbm_cfg05_dayahead.txt"),
        ]
        model_file = None
        for c in candidates:
            if os.path.isfile(c):
                model_file = c
                break
        if model_file is None:
            status.reason_codes.append(
                f"CFG05_ARTIFACT_DIR_EXISTS_BUT_NO_MODEL_FILE_FOUND"
            )
            status.details["tried"] = candidates
            return status
    elif os.path.isfile(model_path):
        model_file = model_path
    else:
        status.reason_codes.append("CFG05_ARTIFACT_PATH_NOT_FILE_OR_DIR")
        return status

    status.details["model_file"] = model_file
    status.details["file_size_bytes"] = os.path.getsize(model_file)

    # Try loading the artifact
    try:
        import lightgbm as lgb

        _ = lgb.Booster(model_file=model_file)
        status.loadable = True
        status.status = LOADABLE
        status.reason_codes.append("CFG05_ARTIFACT_LOADABLE")
    except ImportError:
        status.reason_codes.append("LIGHTGBM_NOT_INSTALLED")
        status.status = INVALID
        return status
    except Exception as e:
        status.loadable = False
        status.status = INVALID
        status.reason_codes.append(f"CFG05_ARTIFACT_LOAD_FAILED: {e}")
        return status

    # Check adapter import
    try:
        from models.adapters.cfg05_dayahead_lgbm import CFG05DayaheadAdapter
        adapter = CFG05DayaheadAdapter()
        adapter.load()
        status.details["adapter_version"] = adapter.model_version
        status.reason_codes.append("CFG05_ADAPTER_LOADED")
    except Exception as e:
        status.reason_codes.append(f"CFG05_ADAPTER_LOAD_FAILED: {e}")
        return status

    # Still LOADABLE — REAL_READY requires input + prediction + validation
    return status


def check_cfg05_input(input_path: Optional[str]) -> ArtifactStatus:
    """Check cfg05 input / feature schema readiness.

    Parameters
    ----------
    input_path : str or None
        Path to a CSV with cfg05 feature columns.

    Returns
    -------
    ArtifactStatus
    """
    status = ArtifactStatus(
        name="cfg05_input",
        path=input_path,
    )

    if not _path_exists(input_path):
        status.reason_codes.append("CFG05_INPUT_MISSING")
        return status

    assert input_path is not None
    status.exists = True
    status.status = PRESENT

    try:
        df = pd.read_csv(input_path)
        if len(df) == 0:
            status.reason_codes.append("CFG05_INPUT_EMPTY")
            status.status = INVALID
            return status

        status.details["rows"] = len(df)
        status.details["columns"] = list(df.columns)

        # Check for required feature columns
        from models.adapters.cfg05_dayahead_lgbm import CFG05_FEATURE_COLUMNS
        missing = [c for c in CFG05_FEATURE_COLUMNS if c not in df.columns]
        present = [c for c in CFG05_FEATURE_COLUMNS if c in df.columns]

        status.details["feature_columns_present"] = len(present)
        status.details["feature_columns_missing"] = len(missing)

        if missing:
            status.reason_codes.append(
                f"CFG05_INPUT_MISSING_{len(missing)}_FEATURE_COLUMNS"
            )
            status.status = INVALID
            return status

        # Check required timestamp column
        if "ds" not in df.columns:
            status.reason_codes.append("CFG05_INPUT_MISSING_DS_COLUMN")
            status.status = INVALID
            return status

        status.schema_ready = True
        status.status = SCHEMA_READY
        status.reason_codes.append("CFG05_INPUT_SCHEMA_READY")

    except Exception as e:
        status.status = INVALID
        status.reason_codes.append(f"CFG05_INPUT_LOAD_FAILED: {e}")

    return status


# ── RT assist pack check ──────────────────────────────────────────────


def check_rt_assist_pack(pack_dir: Optional[str]) -> ArtifactStatus:
    """Check RT assist pack artifact readiness.

    Looks for typical RT assist model files:
    - ``model.pkl``, ``rt_assist_model.pkl``, ``model.pt``
    - ``rt_assist_pack/`` directory with metadata

    Parameters
    ----------
    pack_dir : str or None
        Path to RT assist pack directory.

    Returns
    -------
    ArtifactStatus
    """
    status = ArtifactStatus(
        name="rt_assist_pack",
        path=pack_dir,
    )

    if not _path_exists(pack_dir):
        status.reason_codes.append("RT_ASSIST_PACK_MISSING")
        return status

    assert pack_dir is not None
    status.exists = True

    if not os.path.isdir(pack_dir):
        status.reason_codes.append("RT_ASSIST_PACK_PATH_NOT_DIR")
        status.status = INVALID
        return status

    # Scan for recognisable model files
    model_candidates = [
        os.path.join(pack_dir, "model.pkl"),
        os.path.join(pack_dir, "rt_assist_model.pkl"),
        os.path.join(pack_dir, "model.pt"),
        os.path.join(pack_dir, "rt_assist_model.pt"),
    ]
    found = [p for p in model_candidates if os.path.isfile(p)]

    status.details["found_model_files"] = found
    status.details["directory_contents"] = os.listdir(pack_dir)

    if not found:
        status.reason_codes.append("RT_ASSIST_PACK_NO_RECOGNISED_MODEL_FILE")
        status.status = PRESENT
        return status

    status.status = LOADABLE
    status.reason_codes.append("RT_ASSIST_PACK_LOADABLE")

    # Try adapter import (not full load — the RT assist adapter may need
    # torch or other heavy deps)
    try:
        from models.adapters.realtime_da_safe_assist import DASafeRealtimeAssistAdapter
        adapter = DASafeRealtimeAssistAdapter()
        adapter.load()
        status.details["adapter_version"] = adapter.model_version
        status.reason_codes.append("RT_ASSIST_ADAPTER_LOADED")
    except Exception as e:
        status.reason_codes.append(f"RT_ASSIST_ADAPTER_LOAD_FAILED: {e}")
        return status

    # NOT_IMPLEMENTED because real inference requires full feature pipeline
    status.status = NOT_IMPLEMENTED
    status.reason_codes.append("RT_ASSIST_REAL_INFERENCE_NOT_IMPLEMENTED")

    return status


# ── P5M pack check ────────────────────────────────────────────────────


def check_p5m_pack(pack_dir: Optional[str]) -> ArtifactStatus:
    """Check P5M residual / risk pack readiness.

    Looks for:
    - ``negative_risk_model.pkl`` or ``p5m_model.pkl``
    - ``risk_data.csv`` or ``risk_config.json``

    Parameters
    ----------
    pack_dir : str or None
        Path to P5M pack directory.

    Returns
    -------
    ArtifactStatus
    """
    status = ArtifactStatus(
        name="p5m_pack",
        path=pack_dir,
    )

    if not _path_exists(pack_dir):
        status.reason_codes.append("P5M_PACK_MISSING")
        return status

    assert pack_dir is not None
    status.exists = True

    if not os.path.isdir(pack_dir):
        status.reason_codes.append("P5M_PACK_PATH_NOT_DIR")
        status.status = INVALID
        return status

    contents = os.listdir(pack_dir)
    status.details["directory_contents"] = contents

    model_files = [f for f in contents if f.endswith((".pkl", ".joblib", ".pt"))]
    risk_files = [f for f in contents if "risk" in f.lower() or "config" in f.lower()]

    status.details["model_files"] = model_files
    status.details["risk_files"] = risk_files

    if not model_files:
        status.reason_codes.append("P5M_PACK_NO_MODEL_FILES")
        status.status = PRESENT
        return status

    status.status = LOADABLE
    status.reason_codes.append("P5M_PACK_LOADABLE")

    # Try adapter import
    try:
        from models.adapters.p5m_residual_plugin import P5MResidualPluginAdapter
        adapter = P5MResidualPluginAdapter()
        adapter.load()
        status.details["adapter_version"] = adapter.model_version
        status.reason_codes.append("P5M_ADAPTER_LOADED")
    except Exception as e:
        status.reason_codes.append(f"P5M_ADAPTER_LOAD_FAILED: {e}")
        return status

    # REAL_READY requires risk data + real correction producing delta != 0
    if risk_files:
        status.reason_codes.append("P5M_PACK_RISK_FILES_PRESENT")
        status.status = NOT_IMPLEMENTED
        status.reason_codes.append("P5M_REAL_CORRECTION_NOT_VERIFIED")
    else:
        status.reason_codes.append("P5M_PACK_NO_RISK_FILES")
        status.status = NOT_IMPLEMENTED

    return status


# ── Actual ledger check ────────────────────────────────────────────────


def check_actual_ledger(
    ledger_path: Optional[str],
    min_days: int = 7,
) -> ArtifactStatus:
    """Check actual ledger readiness for BGEW training.

    Parameters
    ----------
    ledger_path : str or None
        Path to actual ledger CSV.
    min_days : int
        Minimum number of unique business days required (default 7).

    Returns
    -------
    ArtifactStatus
    """
    status = ArtifactStatus(
        name="actual_ledger",
        path=ledger_path,
    )

    if not _path_exists(ledger_path):
        status.reason_codes.append("ACTUAL_LEDGER_MISSING")
        return status

    assert ledger_path is not None
    status.exists = True
    status.status = PRESENT

    try:
        df = pd.read_csv(ledger_path)
        if len(df) == 0:
            status.reason_codes.append("ACTUAL_LEDGER_EMPTY")
            return status

        status.details["rows"] = len(df)

        # Check required columns
        from data.schema import ACTUAL_LEDGER_COLUMNS
        missing = [c for c in ACTUAL_LEDGER_COLUMNS if c not in df.columns]
        if missing:
            status.reason_codes.append(f"ACTUAL_LEDGER_MISSING_COLUMNS: {missing}")
            status.status = INVALID
            return status

        # Count unique business days
        if "business_day" in df.columns:
            unique_days = pd.to_datetime(df["business_day"]).nunique()
            status.details["unique_business_days"] = int(unique_days)

            if unique_days < min_days:
                status.reason_codes.append(
                    f"ACTUAL_LEDGER_ONLY_{unique_days}_DAYS_NEED_{min_days}"
                )
                status.status = LOADABLE
                return status

        # Check y_true no NaN
        null_y = df["y_true"].isna().sum()
        if null_y > 0:
            status.reason_codes.append(f"ACTUAL_LEDGER_{null_y}_NULL_Y_TRUE")
            status.status = INVALID
            return status

        status.schema_ready = True
        status.status = SCHEMA_READY
        status.reason_codes.append("ACTUAL_LEDGER_SCHEMA_READY")

    except Exception as e:
        status.status = INVALID
        status.reason_codes.append(f"ACTUAL_LEDGER_LOAD_FAILED: {e}")

    return status


# ── ExtremPriceClf artifact check ─────────────────────────────────────


def check_extrempriceclf_artifact(model_dir: Optional[str]) -> ArtifactStatus:
    """Check ExtremPriceClf artifact readiness.

    Looks for recognised Extreme Price Radar classifier artifacts:
    - ``ExtremPriceClf``, ``ExtremPriceClf.pkl``
    - ``extreme_price_radar/classifier.py``
    - ``classifier.pkl``, ``classifier.pt``

    Parameters
    ----------
    model_dir : str or None
        Path to classifier model directory.

    Returns
    -------
    ArtifactStatus
    """
    status = ArtifactStatus(
        name="extrempriceclf_artifact",
        path=model_dir,
    )

    if not _path_exists(model_dir):
        status.reason_codes.append("EXTREMPRICECLF_ARTIFACT_MISSING")
        return status

    assert model_dir is not None
    status.exists = True

    if not os.path.isdir(model_dir):
        status.reason_codes.append("EXTREMPRICECLF_PATH_NOT_DIR")
        status.status = INVALID
        return status

    contents = os.listdir(model_dir)
    status.details["directory_contents"] = contents

    # Recognised artifact patterns
    artifact_names = [
        "ExtremPriceClf", "ExtremPriceClf.pkl",
        "extreme_price_radar", "classifier.pkl", "classifier.pt",
    ]
    found = [a for a in artifact_names if a in contents or
             any(f.startswith(a.replace(".pkl", "").replace(".pt", ""))
                 for f in contents)]

    if not found:
        status.reason_codes.append("EXTREMPRICECLF_NO_RECOGNISED_ARTIFACT")
        status.status = PRESENT
        return status

    status.details["found_artifacts"] = found
    status.status = LOADABLE

    # Try the adapter from extreme package
    from extreme.negative_classifier import NegativeClassifierAdapter
    adapter = NegativeClassifierAdapter()
    try:
        adapter.load(model_dir=model_dir)
        if adapter._artifact_found:
            status.reason_codes.append("EXTREMPRICECLF_ADAPTER_LOADED")
        else:
            status.reason_codes.append("EXTREMPRICECLF_ADAPTER_NO_ARTIFACT")
            status.status = PRESENT
            return status
    except Exception as e:
        status.reason_codes.append(f"EXTREMPRICECLF_ADAPTER_LOAD_FAILED: {e}")
        status.status = INVALID
        return status

    # NOT_IMPLEMENTED because real ExtremPriceClf inference is still stub
    status.status = NOT_IMPLEMENTED
    status.reason_codes.append("EXTREMPRICECLF_REAL_INFERENCE_NOT_IMPLEMENTED")

    return status


# ── Aggregate check ───────────────────────────────────────────────────


def run_all_artifact_readiness(
    cfg05_model: Optional[str] = None,
    cfg05_input: Optional[str] = None,
    rt_assist_pack: Optional[str] = None,
    p5m_pack: Optional[str] = None,
    actual_ledger: Optional[str] = None,
    extrempriceclf_dir: Optional[str] = None,
    actual_ledger_min_days: int = 7,
) -> dict[str, Any]:
    """Run all artifact readiness checks and return a comprehensive report.

    Parameters
    ----------
    cfg05_model : str, optional
        Path to cfg05 model file or directory.
    cfg05_input : str, optional
        Path to cfg05 input CSV.
    rt_assist_pack : str, optional
        Path to RT assist pack directory.
    p5m_pack : str, optional
        Path to P5M pack directory.
    actual_ledger : str, optional
        Path to actual ledger CSV.
    extrempriceclf_dir : str, optional
        Path to ExtremPriceClf model directory.
    actual_ledger_min_days : int
        Minimum actual ledger days for readiness (default 7).

    Returns
    -------
    dict
        ``{"gates": {name: ArtifactStatus dict, ...}, "summary": {...}}``
    """
    gates: dict[str, ArtifactStatus] = {}

    gates["cfg05_artifact"] = check_cfg05_artifact(cfg05_model)
    gates["cfg05_input"] = check_cfg05_input(cfg05_input)
    gates["rt_assist_pack"] = check_rt_assist_pack(rt_assist_pack)
    gates["p5m_pack"] = check_p5m_pack(p5m_pack)
    gates["actual_ledger"] = check_actual_ledger(actual_ledger, min_days=actual_ledger_min_days)
    gates["extrempriceclf_artifact"] = check_extrempriceclf_artifact(extrempriceclf_dir)

    # Compute summary
    status_counts: dict[str, int] = {}
    for gate in gates.values():
        s = gate.status
        status_counts[s] = status_counts.get(s, 0) + 1

    real_ready_gates = [
        name for name, gate in gates.items()
        if gate.status == REAL_READY
    ]

    summary = {
        "total_gates": len(gates),
        "status_counts": status_counts,
        "real_ready_gates": real_ready_gates,
        "all_real_ready": len(real_ready_gates) == len(gates),
        "any_missing": any(g.status == MISSING for g in gates.values()),
    }

    return {
        "gates": {name: status_to_dict(g) for name, g in gates.items()},
        "summary": summary,
    }
