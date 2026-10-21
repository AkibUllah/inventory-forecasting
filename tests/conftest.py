from datetime import date

import numpy as np
import pandas as pd
import pytest

from invforecast.data import clean_sales, series_for
from invforecast.sample import generate_sample


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Two skus, 200 days, deterministic."""
    return clean_sales(generate_sample(start=date(2025, 1, 1), days=200, n_skus=2, seed=7))


@pytest.fixture(scope="session")
def rice(sample_df) -> pd.DataFrame:
    return series_for(sample_df, "RICE-MINI-5KG")


@pytest.fixture
def flat_series() -> pd.DataFrame:
    """Perfectly regular weekly pattern, no noise: useful for exact-value assertions."""
    dates = pd.date_range("2025-03-03", periods=70, freq="D")  # starts on a Monday
    pattern = np.array([10, 10, 10, 10, 20, 20, 5], dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "sku": "FLAT",
            "units_sold": np.tile(pattern, 10),
            "promo": 0,
            "price": 100.0,
        }
    )
