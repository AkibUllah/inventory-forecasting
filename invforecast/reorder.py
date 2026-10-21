"""Replenishment maths: safety stock, reorder point and order quantity.

All formulas are the textbook continuous-review model with demand variability only
(lead time treated as fixed), which is what a small shop can actually estimate:

    safety_stock  = z(service_level) * sigma_daily * sqrt(lead_time_days)
    reorder_point = mean_daily_demand * lead_time_days + safety_stock
    order_qty     = max(0, forecast_demand(review_period + lead_time) + safety_stock - position)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


def z_score(service_level: float) -> float:
    """Standard-normal quantile for a cycle service level in (0, 1)."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be strictly between 0 and 1")
    return NormalDist().inv_cdf(service_level)


def safety_stock(daily_std: float, lead_time_days: float, service_level: float) -> float:
    if daily_std < 0 or lead_time_days <= 0:
        raise ValueError("daily_std must be >= 0 and lead_time_days > 0")
    return z_score(service_level) * daily_std * math.sqrt(lead_time_days)


def reorder_point(mean_daily_demand: float, lead_time_days: float, safety: float) -> float:
    if mean_daily_demand < 0 or lead_time_days <= 0 or safety < 0:
        raise ValueError("inputs must be non-negative with a positive lead time")
    return mean_daily_demand * lead_time_days + safety


@dataclass(frozen=True)
class ReorderRecommendation:
    sku: str
    on_hand: float
    on_order: float
    lead_time_days: int
    service_level: float
    mean_daily_demand: float
    daily_std: float
    safety_stock: float
    reorder_point: float
    forecast_over_lead_time: float
    recommended_order_qty: float
    days_of_cover: float | None
    should_order: bool

    def as_dict(self) -> dict:
        return asdict(self)


def recommend(
    sku: str,
    *,
    history_units: pd.Series | np.ndarray,
    forecast_units: pd.Series | np.ndarray,
    on_hand: float,
    lead_time_days: int,
    service_level: float = 0.95,
    on_order: float = 0.0,
    review_period_days: int = 0,
    lookback_days: int = 56,
) -> ReorderRecommendation:
    """Combine recent demand variability with the forecast into an order recommendation.

    * ``history_units`` - recent daily actuals (variability and mean come from the last
      ``lookback_days``).
    * ``forecast_units`` - daily forecast; the first ``lead_time + review_period`` days are
      summed as the demand to cover.
    """
    hist = pd.Series(history_units, dtype=float).tail(lookback_days)
    fc = pd.Series(forecast_units, dtype=float)
    if hist.empty or fc.empty:
        raise ValueError("history and forecast must be non-empty")
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be positive")

    mean_daily = float(hist.mean())
    daily_std = float(hist.std(ddof=1)) if len(hist) > 1 else 0.0
    ss = safety_stock(daily_std, lead_time_days, service_level)
    rop = reorder_point(mean_daily, lead_time_days, ss)

    cover_days = lead_time_days + review_period_days
    fc_cover = fc.head(cover_days)
    if len(fc_cover) < cover_days:  # extend with the mean forecast if the horizon is short
        fc_cover = pd.concat([fc_cover, pd.Series([fc.mean()] * (cover_days - len(fc_cover)))])
    forecast_demand = float(fc_cover.sum())

    position = on_hand + on_order
    order_qty = max(0.0, forecast_demand + ss - position)
    forecast_daily = float(fc.mean()) if fc.mean() > 0 else mean_daily
    days_of_cover = (on_hand / forecast_daily) if forecast_daily > 0 else None

    return ReorderRecommendation(
        sku=sku,
        on_hand=float(on_hand),
        on_order=float(on_order),
        lead_time_days=lead_time_days,
        service_level=service_level,
        mean_daily_demand=round(mean_daily, 3),
        daily_std=round(daily_std, 3),
        safety_stock=round(ss, 2),
        reorder_point=round(rop, 2),
        forecast_over_lead_time=round(forecast_demand, 2),
        recommended_order_qty=float(math.ceil(order_qty)),
        days_of_cover=round(days_of_cover, 1) if days_of_cover is not None else None,
        should_order=position <= rop,
    )
