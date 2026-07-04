"""
artifacts/dayahead_window.py — Canonical day-ahead business-day window.

Business day D includes 24 hourly rows:

    D 01:00
    D 02:00
    ...
    D 23:00
    D+1 00:00

The canonical filter is::

    ds >= D + 1h   AND   ds < D + 25h   (i.e. D+1 01:00)

This ensures D+1 00:00 is included while D+1 01:00 is excluded (it belongs
to business day D+1).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def get_dayahead_window(target_day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end_exclusive) for the canonical 24-hour day-ahead window.

    Parameters
    ----------
    target_day : str
        Business day in ``YYYY-MM-DD`` format.

    Returns
    -------
    tuple (start, end_exclusive)
        ``start`` = target_day + 1 hour
        ``end_exclusive`` = target_day + 25 hours (target_day + 1 day + 1 hour)
    """
    target_dt = pd.Timestamp(target_day)
    start = target_dt + pd.Timedelta(hours=1)
    end_exclusive = target_dt + pd.Timedelta(days=1, hours=1)
    return start, end_exclusive


def day_ahead_mask(df: pd.DataFrame, target_day: str) -> pd.Series:
    """Return boolean mask for the canonical 24-hour day-ahead window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``ds`` column.
    target_day : str
        Business day in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.Series[bool]
        Mask selecting rows in [target_day+1h, target_day+25h).
    """
    ds = pd.to_datetime(df["ds"])
    start, end = get_dayahead_window(target_day)
    return (ds >= start) & (ds < end)


def filter_dayahead(df: pd.DataFrame, target_day: str) -> pd.DataFrame:
    """Filter a DataFrame to the canonical 24-hour day-ahead window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``ds`` column.
    target_day : str
        Business day in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Filtered copy sorted by ds.
    """
    mask = day_ahead_mask(df, target_day)
    result = df[mask].copy()
    result = result.sort_values("ds").reset_index(drop=True)
    return result


def get_business_day_info(ds_series: pd.Series) -> dict[str, Any]:
    """Derive business-day metadata from a ds series.

    Parameters
    ----------
    ds_series : pd.Series
        Datetime series.

    Returns
    -------
    dict with ``business_day``, ``hour_business``, ``period``.
    """
    ds = pd.to_datetime(ds_series)
    # business_day = ds - 1h (because D+1 00:00 belongs to day D)
    shifted = ds - pd.Timedelta(hours=1)
    business_day = shifted.dt.date
    # hour_business: apply the same shift so D+1 00:00 → hour 24, D 01:00 → hour 1
    hour_business = shifted.dt.hour + 1  # 0→1, 23→24
    hour_business = hour_business.where(hour_business <= 24, 24)

    def _period(h: int) -> str:
        if 1 <= h <= 8:
            return "1_8"
        elif 9 <= h <= 16:
            return "9_16"
        return "17_24"

    period = hour_business.apply(_period)

    return {
        "business_day": business_day,
        "hour_business": hour_business,
        "period": period,
    }
