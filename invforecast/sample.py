"""Synthetic POS data with the shape of a Dhaka neighbourhood shop.

Each sku gets a base level, a gentle trend, weekly seasonality (Friday/Saturday peaks),
a pre-Eid ramp, a dip on public holidays, random promotion days with a lift, and
Poisson noise. The generator is seeded so the bundled ``data/sample_sales.csv`` is
reproducible.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .holidays import HolidayCalendar, load_calendar

SKU_PROFILES = [
    # (sku, base daily units, trend per year, promo lift, price BDT)
    ("RICE-MINI-5KG", 18.0, 0.10, 1.6, 395.0),
    ("OIL-SOY-1L", 25.0, 0.05, 1.8, 175.0),
    ("TEA-ISPAHANI-400G", 9.0, 0.00, 1.5, 240.0),
    ("DETERGENT-1KG", 6.0, 0.15, 2.0, 150.0),
    ("MILK-POWDER-500G", 4.0, 0.20, 1.4, 420.0),
    ("SUGAR-1KG", 14.0, -0.05, 1.3, 140.0),
]

# Multiplier per weekday (Mon=0 ... Sun=6); Fri/Sat are the Bangladeshi weekend.
WEEKDAY_FACTORS = np.array([0.95, 0.90, 0.92, 0.98, 1.25, 1.20, 0.80])


def generate_sample(
    *,
    start: date = date(2025, 1, 1),
    days: int = 365,
    n_skus: int = 4,
    seed: int = 42,
    promo_probability: float = 0.06,
    calendar: HolidayCalendar | None = None,
) -> pd.DataFrame:
    if not 1 <= n_skus <= len(SKU_PROFILES):
        raise ValueError(f"n_skus must be between 1 and {len(SKU_PROFILES)}")
    if days < 60:
        raise ValueError("days must be at least 60 to be useful for forecasting")
    rng = np.random.default_rng(seed)
    calendar = calendar or load_calendar()
    dates = [start + timedelta(days=i) for i in range(days)]
    rows = []
    for sku, base, trend, promo_lift, price in SKU_PROFILES[:n_skus]:
        promo_days = rng.random(days) < promo_probability
        for i, day in enumerate(dates):
            level = base * (1 + trend * i / 365)
            level *= WEEKDAY_FACTORS[day.weekday()]
            if calendar.is_pre_eid(day):
                level *= 1.35
            if calendar.is_eid(day):
                level *= 0.35
            elif calendar.is_holiday(day):
                level *= 0.7
            if day.day <= 3:  # salary days
                level *= 1.15
            promo = int(promo_days[i])
            if promo:
                level *= promo_lift
            units = rng.poisson(max(level, 0.1))
            rows.append(
                {
                    "date": day.isoformat(),
                    "sku": sku,
                    "units_sold": int(units),
                    "promo": promo,
                    "price": round(price * (0.9 if promo else 1.0), 2),
                }
            )
    return pd.DataFrame(rows)
