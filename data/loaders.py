"""
data/loaders.py — Unified data loading layer.

Provides ``load_table()`` as the single entry point for reading
tabular data from .csv, .xlsx, .xls files with:
  - Multi-encoding fallback (utf-8 → utf-8-sig → gbk → gb18030)
  - Chinese column name mapping to canonical English names
  - Optional datetime parsing and business-time column addition
  - Structured metadata about transformations applied

Usage:
    df, meta = load_table("path/to/data.csv")
    df, meta = load_table("path/to/data.xlsx", add_business_time=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from data.business_day import add_business_time_columns

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Chinese-to-canonical column name mapping
# ──────────────────────────────────────────────

CN_COLUMN_MAP: dict[str, str] = {
    # Timestamp columns
    "时间": "ds",
    "日期时间": "ds",
    "timestamp": "ds",
    "times": "ds",
    "date_time": "ds",
    # Day-ahead price columns
    "日前价格": "da_anchor",
    "日前电价": "da_anchor",
    "da_price": "da_anchor",
    # Real-time price columns
    "实时价格": "rt_actual",
    "实时电价": "rt_actual",
    "rt_price": "rt_actual",
    # Prediction columns
    "预测价格": "y_pred",
    "预测值": "y_pred",
    "预测电价": "y_pred",
    # Actual price (eval only)
    "实际价格": "y_true",
    "实际电价": "y_true",
    "actual_price": "y_true",
    # Load
    "负荷": "load",
    "load_actual": "load",
    # Wind
    "风电": "wind",
    "wind_actual": "wind",
    # Solar
    "光伏": "solar",
    "solar_actual": "solar",
    # Bidding space
    "竞价空间": "bidding_space",
    "bidding": "bidding_space",
    # Net load
    "净负荷": "net_load",
}

# ──────────────────────────────────────────────
# Allowed encoding list (tried in order)
# ──────────────────────────────────────────────

ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

# ──────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────


def load_table(
    path: str,
    *,
    parse_dates: bool = True,
    add_business_time: bool = False,
    timestamp_col: str = "ds",
    encoding: Optional[str] = None,
    sheet_name: Optional[str] = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a table from CSV or Excel into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the file. Supports ``.csv``, ``.xlsx``, ``.xls``.
    parse_dates : bool
        If True (default), attempt to parse the timestamp column as datetime.
    add_business_time : bool
        If True, call ``add_business_time_columns()`` to add ``business_day``,
        ``hour_business``, ``period`` columns.
    timestamp_col : str
        Name of the timestamp column to use for business-time inference.
        Default: ``"ds"``.
    encoding : str, optional
        If provided, use this encoding only (no fallback). Used for testing.
    sheet_name : str, optional
        For Excel files, the sheet to read. Defaults to first sheet.
    **kwargs
        Additional arguments forwarded to ``pd.read_csv`` or ``pd.read_excel``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - DataFrame with loaded data
        - Metadata dict with keys:
            - ``path``: original file path
            - ``rows``: row count
            - ``columns``: column name list
            - ``encoding_used``: encoding that succeeded (CSV only)
            - ``cn_mappings``: list of (cn_name, canonical_name) tuples applied
            - ``added_business_time``: whether business-time columns were added
            - ``parse_dates``: whether date parsing was attempted
            - ``errors``: list of warning/error messages

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    meta: dict[str, Any] = {
        "path": str(p.resolve()),
        "encoding_used": None,
        "cn_mappings": [],
        "added_business_time": False,
        "parse_dates": parse_dates,
        "errors": [],
    }

    ext = p.suffix.lower()

    # ── Read raw data ──────────────────────────
    if ext == ".csv":
        # Detect BOM before reading
        bom_found = _detect_bom(p)
        if bom_found and encoding is None:
            # Force utf-8-sig for BOM files so metadata reflects it
            df, enc = _read_csv_with_fallback(p, encoding="utf-8-sig")
            meta["encoding_used"] = enc
        else:
            df, enc = _read_csv_with_fallback(p, encoding)
            meta["encoding_used"] = enc
    elif ext in (".xlsx", ".xls"):
        df = _read_excel(p, sheet_name, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension: '{ext}'. "
            f"Supported: .csv, .xlsx, .xls"
        )

    # ── Apply Chinese column name mapping ──────
    df, mappings = _apply_cn_column_map(df)
    meta["cn_mappings"] = mappings

    # ── Normalise common column names ──────────
    df = _normalise_common_columns(df)

    # ── Parse dates ────────────────────────────
    if parse_dates and timestamp_col in df.columns:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            n_null = df[timestamp_col].isna().sum()
            if n_null > 0:
                meta["errors"].append(
                    f"{n_null} rows failed datetime parsing in '{timestamp_col}'"
                )
        except Exception as exc:
            meta["errors"].append(f"Date parsing failed: {exc}")

    # ── Add business-time columns ──────────────
    if add_business_time:
        if timestamp_col in df.columns:
            df = add_business_time_columns(df, timestamp_col=timestamp_col)
            meta["added_business_time"] = True
        else:
            msg = (
                f"Cannot add business-time columns: '{timestamp_col}' not found. "
                f"Available: {list(df.columns)}"
            )
            meta["errors"].append(msg)
            logger.warning(msg)

    meta["rows"] = len(df)
    meta["columns"] = list(df.columns)

    return df, meta


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _detect_bom(path: Path) -> bool:
    """Check if a file starts with a UTF-8 BOM (\\xef\\xbb\\xbf)."""
    try:
        with open(path, "rb") as f:
            header = f.read(3)
        return header == b"\xef\xbb\xbf"
    except Exception:
        return False


def _read_csv_with_fallback(
    path: Path,
    encoding: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """Read a CSV file trying multiple encodings.

    Returns (DataFrame, encoding_used).
    """
    if encoding is not None:
        # Single-encoding mode (used for testing)
        return pd.read_csv(path, encoding=encoding), encoding

    encodings_to_try = ENCODING_CANDIDATES
    last_error: Exception | None = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.debug(f"CSV loaded with encoding: {enc}")
            return df, enc
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            # Non-encoding errors (e.g. empty file) should propagate
            raise e

    raise UnicodeDecodeError(
        f"encoding",
        b"",
        0,
        0,
        f"CSV could not be decoded with any of {encodings_to_try}. "
        f"Last error: {last_error}",
    )


def _read_excel(
    path: Path,
    sheet_name: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel file."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "openpyxl is required for .xlsx files. Install with: pip install openpyxl"
        )

    try:
        import xlrd  # noqa: F401
    except ImportError:
        raise ImportError(
            "xlrd is required for .xls files. Install with: pip install xlrd"
        )

    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def _apply_cn_column_map(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Map Chinese column names to canonical English names.

    Returns (renamed_df, list_of_(original, canonical)_tuples).
    """
    mappings: list[tuple[str, str]] = []
    rename_dict: dict[str, str] = {}

    for col in df.columns:
        if col in CN_COLUMN_MAP:
            canonical = CN_COLUMN_MAP[col]
            rename_dict[col] = canonical
            mappings.append((col, canonical))

    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df, mappings


def _normalise_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise common column naming variations (lowercase, strip)."""
    rename = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower != col:
            # Only normalise if the lowercased version exists in CN_COLUMN_MAP
            if col_lower in CN_COLUMN_MAP or col_lower in (
                "ds", "da_anchor", "rt_actual", "y_pred", "y_true",
                "load", "wind", "solar", "bidding_space", "net_load",
                "hour", "month", "day_of_week",
            ):
                rename[col] = col_lower
    if rename:
        df = df.rename(columns=rename)
    return df


def list_mapped_columns() -> dict[str, str]:
    """Return the Chinese→canonical column mapping dictionary."""
    return dict(CN_COLUMN_MAP)
