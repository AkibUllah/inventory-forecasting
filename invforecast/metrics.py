"""Forecast accuracy metrics. All accept array-likes and return floats."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("empty input")
    return a, b


def mae(y_true, y_pred) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error over non-zero actuals (NaN if all actuals are zero)."""
    a, b = _arrays(y_true, y_pred)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent; zero-zero pairs count as perfect."""
    a, b = _arrays(y_true, y_pred)
    denom = np.abs(a) + np.abs(b)
    safe = np.where(denom == 0, 1.0, denom)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(a - b) / safe)
    return float(np.mean(ratio) * 100)


def bias(y_true, y_pred) -> float:
    """Mean signed error (positive means over-forecasting)."""
    a, b = _arrays(y_true, y_pred)
    return float(np.mean(b - a))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
