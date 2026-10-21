import math

import numpy as np
import pandas as pd
import pytest

from invforecast.backtest import backtest, backtest_series, make_folds, summarize, to_markdown
from invforecast.metrics import all_metrics, bias, mae, mape, rmse, smape
from invforecast.models import (
    EnsembleForecaster,
    GradientBoostingForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    build_model,
    forecast_frame,
)


def test_metrics_known_values():
    y, p = [10, 20, 30, 0], [12, 18, 30, 0]
    assert mae(y, p) == 1.0
    assert rmse(y, p) == pytest.approx(math.sqrt((4 + 4) / 4))
    assert mape(y, p) == pytest.approx((0.2 + 0.1 + 0.0) / 3 * 100)  # zero actual excluded
    assert smape(y, p) == pytest.approx((2 * 2 / 22 + 2 * 2 / 38 + 0 + 0) / 4 * 100)
    assert bias(y, p) == 0.0
    assert set(all_metrics(y, p)) == {"mae", "rmse", "mape", "smape", "bias"}


def test_metrics_edge_cases():
    assert math.isnan(mape([0, 0], [1, 2]))
    assert smape([0, 0], [0, 0]) == 0.0
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        mae([], [])


def test_baseline_models_on_regular_pattern(flat_series):
    naive = NaiveForecaster().fit(flat_series)
    assert naive.predict(3).tolist() == [5.0, 5.0, 5.0]  # last day was a Sunday (5)

    seasonal = SeasonalNaiveForecaster().fit(flat_series)
    assert seasonal.predict(9).tolist() == [10, 10, 10, 10, 20, 20, 5, 10, 10]

    ma = MovingAverageForecaster(window=7).fit(flat_series)
    assert ma.predict(2).tolist() == pytest.approx([85 / 7] * 2)


def test_gradient_boosting_learns_weekly_pattern(flat_series):
    model = GradientBoostingForecaster(n_estimators=150).fit(flat_series)
    preds = model.predict(7)
    # history ends on a Sunday, so the forecast runs Mon..Sun: 10,10,10,10,20,20,5
    assert preds[4] > preds[0] and preds[5] > preds[0]
    assert preds[6] < preds[0]
    assert np.all(preds >= 0)
    importances = model.feature_importances()
    assert importances.index[0] in {"lag_7", "lag_14", "lag_28", "dow", "is_weekend", "roll_max_7"}


def test_gradient_boosting_requires_history_and_fit():
    short = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20),
            "sku": "X",
            "units_sold": 1.0,
            "promo": 0,
            "price": 1.0,
        }
    )
    with pytest.raises(ValueError, match="at least 14"):
        GradientBoostingForecaster().fit(short)
    with pytest.raises(RuntimeError):
        GradientBoostingForecaster().predict(3)


def test_ensemble_averages_members(flat_series):
    ensemble = EnsembleForecaster(
        members=[NaiveForecaster(), MovingAverageForecaster(window=7)], weights=[1, 3]
    ).fit(flat_series)
    expected = (5.0 * 1 + (85 / 7) * 3) / 4
    assert ensemble.predict(2).tolist() == pytest.approx([expected, expected])
    with pytest.raises(ValueError):
        EnsembleForecaster(members=[NaiveForecaster()], weights=[1, 2])


def test_build_model_and_forecast_frame(flat_series):
    assert isinstance(build_model("seasonal_naive"), SeasonalNaiveForecaster)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("prophet")
    frame = forecast_frame(flat_series, np.array([1.234, -2.0]))
    assert frame["date"].iloc[0] == pd.Timestamp("2025-05-12")  # day after the last history date
    assert frame["forecast"].tolist() == [1.23, 0.0]
    assert frame["sku"].unique().tolist() == ["FLAT"]


def test_make_folds_are_contiguous_and_non_overlapping():
    dates = pd.Series(pd.date_range("2025-01-01", periods=120))
    folds = make_folds(dates, horizon=14, n_splits=3)
    assert [f.index for f in folds] == [0, 1, 2]
    assert folds[-1].test_end == pd.Timestamp("2025-04-30")
    for fold in folds:
        assert (fold.test_end - fold.test_start).days == 13
        assert fold.train_end == fold.test_start - pd.Timedelta(days=1)
    assert folds[0].test_end == folds[1].test_start - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Not enough history"):
        make_folds(dates, horizon=30, n_splits=5)


def test_backtest_series_shape(rice):
    results = backtest_series(rice, SeasonalNaiveForecaster, horizon=7, n_splits=3)
    assert len(results) == 3
    assert set(results.columns) >= {
        "sku",
        "model",
        "fold",
        "train_end",
        "mae",
        "rmse",
        "mape",
        "smape",
        "bias",
    }
    assert (results["model"] == "seasonal_naive").all()
    assert results["train_days"].is_monotonic_increasing


def test_backtest_summary_and_markdown(sample_df):
    results = backtest(sample_df, ["naive", "moving_average"], horizon=7, n_splits=2)
    assert len(results) == 2 * 2 * 2  # skus x models x folds
    summary = summarize(results)
    assert summary["model"].tolist() == sorted(
        summary["model"], key=lambda m: summary.set_index("model").loc[m, "smape"]
    )
    assert summary["folds"].tolist() == [4, 4]
    md = to_markdown(summary)
    assert md.startswith("| Model | MAE |")
    assert md.count("\n") == 3
