"""Feature engineering for daily sku-level demand."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .holidays import HolidayCalendar, load_calendar

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)
DEFAULT_WINDOWS: tuple[int, ...] = (7, 28)
WEEKEND_DAYS = (4, 5)  # Friday, Saturday - the Bangladeshi weekend


def calendar_features(dates: pd.Series, calendar: HolidayCalendar | None = None) -> pd.DataFrame:
    calendar = calendar or load_calendar()
    dates = pd.to_datetime(dates)
    plain = dates.dt.date
    feats = pd.DataFrame(index=dates.index)
    feats["dow"] = dates.dt.dayofweek
    feats["day_of_month"] = dates.dt.day
    feats["month"] = dates.dt.month
    feats["is_weekend"] = dates.dt.dayofweek.isin(WEEKEND_DAYS).astype(int)
    feats["is_month_start"] = (dates.dt.day <= 3).astype(int)
    feats["is_holiday"] = plain.map(calendar.is_holiday).astype(int)
    feats["is_eid"] = plain.map(calendar.is_eid).astype(int)
    feats["pre_eid"] = plain.map(calendar.is_pre_eid).astype(int)
    return feats


def lag_features(
    series: pd.Series, lags: Sequence[int] = DEFAULT_LAGS, windows: Sequence[int] = DEFAULT_WINDOWS
) -> pd.DataFrame:
    """Lags and trailing rolling statistics shifted by one day so they never see the target."""
    feats = pd.DataFrame(index=series.index)
    for lag in lags:
        feats[f"lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    for window in windows:
        feats[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=1).mean()
        feats[f"roll_std_{window}"] = shifted.rolling(window, min_periods=2).std().fillna(0.0)
    feats["roll_max_7"] = shifted.rolling(7, min_periods=1).max()
    return feats


def make_features(
    df: pd.DataFrame,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    calendar: HolidayCalendar | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Return ``df`` (single sku, daily, sorted) with feature columns appended."""
    if df["sku"].nunique() != 1:
        raise ValueError("make_features expects a single-sku frame; use series_for() first.")
    data = df.sort_values("date").reset_index(drop=True)
    feats = pd.concat(
        [
            data,
            calendar_features(data["date"], calendar),
            lag_features(data["units_sold"], lags, windows),
        ],
        axis=1,
    )
    feats["promo"] = feats["promo"].astype(int)
    if dropna:
        feats = feats.dropna().reset_index(drop=True)
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", "sku", "units_sold", "price"}
    return [c for c in df.columns if c not in excluded and np.issubdtype(df[c].dtype, np.number)]
