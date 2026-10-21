from datetime import date

import pandas as pd
import pytest

from invforecast.data import DataError, clean_sales, load_sales, series_for, skus
from invforecast.features import calendar_features, feature_columns, lag_features, make_features
from invforecast.holidays import load_calendar


def test_clean_sales_fills_gaps_sums_duplicates_and_clips_negatives():
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-04", "bad-date"],
            "sku": ["A", "A", "A ", "A"],
            "units_sold": [3, 2, -5, 1],
            "promo": [1, 0, None, 0],
        }
    )
    cleaned = clean_sales(raw)
    assert list(cleaned["date"].dt.strftime("%Y-%m-%d")) == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
    ]
    assert cleaned["units_sold"].tolist() == [
        5.0,
        0.0,
        0.0,
        0.0,
    ]  # dup summed, gap zero, negative clipped
    assert cleaned["promo"].tolist() == [1, 0, 0, 0]
    assert cleaned["sku"].unique().tolist() == ["A"]


def test_load_sales_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,qty\n2025-01-01,3\n")
    with pytest.raises(DataError, match="missing required column"):
        load_sales(path)


def test_series_for_and_skus(sample_df):
    assert skus(sample_df) == ["OIL-SOY-1L", "RICE-MINI-5KG"]
    rice = series_for(sample_df, "RICE-MINI-5KG")
    assert len(rice) == 200
    assert rice["date"].is_monotonic_increasing
    with pytest.raises(DataError):
        series_for(sample_df, "NOPE")


def test_calendar_features_weekend_and_holidays():
    dates = pd.Series(
        pd.to_datetime(["2025-03-21", "2025-03-26", "2025-03-28", "2025-03-31", "2025-04-02"])
    )
    feats = calendar_features(dates, load_calendar())
    # 03-21 is a Friday (BD weekend), 03-26 Independence Day, 03-31 Eid-ul-Fitr (3 days)
    assert feats["is_weekend"].tolist() == [1, 0, 1, 0, 0]
    assert feats["is_holiday"].tolist() == [0, 1, 0, 1, 1]
    assert feats["is_eid"].tolist() == [0, 0, 0, 1, 1]
    assert feats["pre_eid"].tolist() == [1, 1, 1, 0, 0]
    assert feats["dow"].tolist() == [4, 2, 4, 0, 2]


def test_holiday_calendar_expands_fixed_dates_and_eid_runup():
    cal = load_calendar()
    assert cal.is_holiday(date(2025, 12, 16)) and cal.is_holiday(date(2026, 12, 16))
    assert cal.name(date(2025, 2, 21)).startswith("Shaheed Dibosh")
    assert cal.is_eid(date(2025, 6, 9)) and not cal.is_eid(date(2025, 6, 10))
    assert cal.days_to_next_eid(date(2025, 6, 1)) == 6
    assert cal.is_pre_eid(date(2025, 6, 1)) and not cal.is_pre_eid(date(2025, 5, 20))


def test_custom_holiday_yaml(tmp_path):
    yaml_path = tmp_path / "h.yaml"
    yaml_path.write_text(
        "fixed: [{month: 1, day: 1, name: New Year}]\n"
        "moveable: {2025: [{date: 2025-07-01, name: Eid-ul-Test, span: 2}]}\n"
        "eid_runup_days: 3\ncustom: [{date: 2025-09-09, name: Stock take}]\n"
    )
    cal = load_calendar(yaml_path)
    assert cal.is_holiday(date(2025, 1, 1)) and cal.is_holiday(date(2024, 1, 1))
    assert cal.is_eid(date(2025, 7, 2)) and cal.eid_runup_days == 3
    assert cal.name(date(2025, 9, 9)) == "Stock take"
    assert cal.is_pre_eid(date(2025, 6, 29)) and not cal.is_pre_eid(date(2025, 6, 27))


def test_lag_features_never_leak_the_target(flat_series):
    feats = lag_features(flat_series["units_sold"], lags=(1, 7), windows=(7,))
    assert feats["lag_1"].iloc[1] == flat_series["units_sold"].iloc[0]
    assert feats["lag_7"].iloc[7] == flat_series["units_sold"].iloc[0]
    # rolling mean at t uses t-1..t-7 only: the full-week mean is (4*10+20+20+5)/7 = 85/7
    assert feats["roll_mean_7"].iloc[7] == pytest.approx(85 / 7)
    assert feats["roll_max_7"].iloc[7] == 20
    assert pd.isna(feats["lag_7"].iloc[6])


def test_make_features_shape_and_columns(rice):
    feats = make_features(rice)
    assert len(feats) == len(rice) - 28  # longest lag drops 28 rows
    cols = feature_columns(feats)
    for expected in (
        "lag_1",
        "lag_28",
        "roll_mean_7",
        "roll_std_28",
        "dow",
        "is_holiday",
        "pre_eid",
        "promo",
    ):
        assert expected in cols
    assert "units_sold" not in cols and "date" not in cols
    with pytest.raises(ValueError, match="single-sku"):
        make_features(pd.concat([rice, rice.assign(sku="OTHER")]))
