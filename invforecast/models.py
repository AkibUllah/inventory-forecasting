"""Forecasting models behind one small interface.

Every model implements ``fit(history)`` and ``predict(horizon, future_exog)`` where
``history`` is a cleaned single-sku frame (``date``, ``units_sold``, ``promo``) and
``future_exog`` optionally carries ``date`` and ``promo`` for the horizon days.
Baselines ignore features; the gradient-boosting model builds lag / calendar
features and forecasts recursively, feeding each prediction back as the next lag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .features import DEFAULT_LAGS, DEFAULT_WINDOWS, feature_columns, make_features
from .holidays import HolidayCalendar, load_calendar


class Forecaster(Protocol):
    name: str

    def fit(self, history: pd.DataFrame) -> Forecaster: ...

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray: ...


def _future_dates(history: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    last = pd.Timestamp(history["date"].max())
    return pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")


def forecast_frame(
    history: pd.DataFrame, predictions: np.ndarray, sku: str | None = None
) -> pd.DataFrame:
    dates = _future_dates(history, len(predictions))
    return pd.DataFrame(
        {
            "date": dates,
            "sku": sku if sku is not None else history["sku"].iloc[0],
            "forecast": np.clip(np.asarray(predictions, dtype=float), 0, None).round(2),
        }
    )


@dataclass
class NaiveForecaster:
    """Tomorrow looks like today."""

    name: str = "naive"
    last_value: float = 0.0

    def fit(self, history: pd.DataFrame) -> NaiveForecaster:
        self.last_value = float(history["units_sold"].iloc[-1])
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        return np.full(horizon, self.last_value)


@dataclass
class SeasonalNaiveForecaster:
    """Same weekday last week."""

    season: int = 7
    name: str = "seasonal_naive"
    last_season: np.ndarray = field(default_factory=lambda: np.zeros(7))

    def fit(self, history: pd.DataFrame) -> SeasonalNaiveForecaster:
        values = history["units_sold"].to_numpy(dtype=float)
        if len(values) < self.season:
            values = np.pad(values, (self.season - len(values), 0), mode="edge")
        self.last_season = values[-self.season :]
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        reps = int(np.ceil(horizon / self.season))
        return np.tile(self.last_season, reps)[:horizon]


@dataclass
class MovingAverageForecaster:
    window: int = 28
    name: str = "moving_average"
    level: float = 0.0

    def fit(self, history: pd.DataFrame) -> MovingAverageForecaster:
        self.level = float(history["units_sold"].tail(self.window).mean())
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        return np.full(horizon, self.level)


@dataclass
class GradientBoostingForecaster:
    """scikit-learn GradientBoostingRegressor on lag + calendar + promo features."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    lags: tuple[int, ...] = DEFAULT_LAGS
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    random_state: int = 42
    name: str = "gradient_boosting"
    calendar: HolidayCalendar | None = None
    model: GradientBoostingRegressor | None = None
    columns: list[str] = field(default_factory=list)
    history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> GradientBoostingForecaster:
        self.calendar = self.calendar or load_calendar()
        feats = make_features(history, lags=self.lags, windows=self.windows, calendar=self.calendar)
        if len(feats) < 14:
            raise ValueError("Need at least 14 feature rows after lagging; provide more history.")
        self.columns = feature_columns(feats)
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.9,
            random_state=self.random_state,
        )
        self.model.fit(feats[self.columns], feats["units_sold"])
        self.history = history[["date", "sku", "units_sold", "promo", "price"]].copy()
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        if self.model is None or self.history is None:
            raise RuntimeError("fit() must be called before predict()")
        working = self.history.copy()
        dates = _future_dates(working, horizon)
        promo_lookup: dict[pd.Timestamp, int] = {}
        if future_exog is not None and "promo" in future_exog.columns:
            exog = future_exog.copy()
            exog["date"] = pd.to_datetime(exog["date"]).dt.normalize()
            promo_lookup = dict(zip(exog["date"], exog["promo"].astype(int), strict=False))
        preds: list[float] = []
        for day in dates:
            row = pd.DataFrame(
                {
                    "date": [day],
                    "sku": [working["sku"].iloc[0]],
                    "units_sold": [np.nan],
                    "promo": [promo_lookup.get(day, 0)],
                    "price": [working["price"].iloc[-1]],
                }
            )
            extended = pd.concat([working, row], ignore_index=True)
            feats = make_features(
                extended, lags=self.lags, windows=self.windows, calendar=self.calendar, dropna=False
            )
            x = feats[self.columns].iloc[[-1]].fillna(0.0)
            yhat = max(0.0, float(self.model.predict(x)[0]))
            preds.append(yhat)
            row["units_sold"] = yhat
            working = pd.concat([working, row], ignore_index=True)
        return np.asarray(preds)

    def feature_importances(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("fit() must be called first")
        return pd.Series(self.model.feature_importances_, index=self.columns).sort_values(
            ascending=False
        )


@dataclass
class EnsembleForecaster:
    """Weighted mean of member forecasts (default: seasonal naive + moving average + GBM)."""

    members: list[Forecaster] = field(default_factory=list)
    weights: list[float] | None = None
    name: str = "ensemble"

    def __post_init__(self) -> None:
        if not self.members:
            self.members = [
                SeasonalNaiveForecaster(),
                MovingAverageForecaster(window=28),
                GradientBoostingForecaster(),
            ]
        if self.weights is None:
            self.weights = [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("weights must match members")

    def fit(self, history: pd.DataFrame) -> EnsembleForecaster:
        for member in self.members:
            member.fit(history)
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        stack = np.vstack([m.predict(horizon, future_exog) for m in self.members])
        weights = np.asarray(self.weights, dtype=float)
        return (stack * weights[:, None]).sum(axis=0) / weights.sum()


MODELS: dict[str, type] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "gradient_boosting": GradientBoostingForecaster,
    "ensemble": EnsembleForecaster,
}


def build_model(name: str, **kwargs) -> Forecaster:
    try:
        return MODELS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
