"""Rolling-origin backtesting.

For each of ``n_splits`` cut-off points (spaced ``step`` days apart, ending so the
last fold still has ``horizon`` days of actuals) the model is fitted on history up
to the cut-off and scored on the next ``horizon`` days. Results come back per fold
and aggregated per model, which is what the README table is built from.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import series_for, skus
from .metrics import all_metrics
from .models import Forecaster, build_model


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(
    dates: pd.Series, horizon: int, n_splits: int, step: int | None = None
) -> list[Fold]:
    dates = pd.to_datetime(pd.Series(dates)).sort_values().reset_index(drop=True)
    step = step or horizon
    last = dates.iloc[-1]
    folds: list[Fold] = []
    for i in range(n_splits):
        test_end = last - pd.Timedelta(days=step * (n_splits - 1 - i))
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        if train_end <= dates.iloc[0]:
            raise ValueError("Not enough history for the requested horizon/splits/step.")
        folds.append(Fold(index=i, train_end=train_end, test_start=test_start, test_end=test_end))
    return folds


def backtest_series(
    history: pd.DataFrame,
    model_factory: Callable[[], Forecaster],
    *,
    horizon: int = 14,
    n_splits: int = 4,
    step: int | None = None,
    min_train_days: int = 56,
) -> pd.DataFrame:
    """Per-fold metrics for one sku (sku, model, fold, train_end, mae, rmse, mape, smape, bias)."""
    history = history.sort_values("date").reset_index(drop=True)
    rows = []
    for fold in make_folds(history["date"], horizon, n_splits, step):
        train = history[history["date"] <= fold.train_end]
        test = history[(history["date"] >= fold.test_start) & (history["date"] <= fold.test_end)]
        if len(train) < min_train_days or len(test) < horizon:
            continue
        model = model_factory().fit(train)
        preds = model.predict(horizon, future_exog=test[["date", "promo"]])
        metrics = all_metrics(test["units_sold"].to_numpy(), preds)
        rows.append(
            {
                "sku": history["sku"].iloc[0],
                "model": getattr(model, "name", type(model).__name__),
                "fold": fold.index,
                "train_end": fold.train_end.date(),
                "train_days": len(train),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def backtest(
    df: pd.DataFrame,
    models: Iterable[str] = (
        "naive",
        "seasonal_naive",
        "moving_average",
        "gradient_boosting",
        "ensemble",
    ),
    *,
    horizon: int = 14,
    n_splits: int = 4,
    step: int | None = None,
    sku_list: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run every model over every sku; returns the concatenated per-fold frame."""
    frames = []
    for sku in sku_list or skus(df):
        history = series_for(df, sku)
        for name in models:
            frames.append(
                backtest_series(
                    history,
                    lambda n=name: build_model(n),
                    horizon=horizon,
                    n_splits=n_splits,
                    step=step,
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per model across skus and folds, best model first (by sMAPE)."""
    if results.empty:
        return results
    summary = (
        results.groupby("model")[["mae", "rmse", "mape", "smape", "bias"]]
        .mean()
        .assign(folds=results.groupby("model").size())
        .sort_values("smape")
        .reset_index()
    )
    return summary


def to_markdown(summary: pd.DataFrame) -> str:
    header = (
        "| Model | MAE | RMSE | MAPE % | sMAPE % | Bias | Folds |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for _, row in summary.iterrows():
        mape_text = "n/a" if np.isnan(row["mape"]) else f"{row['mape']:.1f}"
        lines.append(
            f"| {row['model']} | {row['mae']:.2f} | {row['rmse']:.2f} | {mape_text} | "
            f"{row['smape']:.1f} | {row['bias']:+.2f} | {int(row['folds'])} |"
        )
    return "\n".join(lines)
