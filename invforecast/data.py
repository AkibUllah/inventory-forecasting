"""Loading and cleaning POS sales exports.

Expected columns: ``date``, ``sku``, ``units_sold``; optional ``price``, ``promo``
(0/1). Cleaning makes the series regular: one row per sku per calendar day, missing
days filled with zero sales (a closed day or no sale is still a zero-demand
observation), negatives (returns) clipped at zero, duplicates summed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("date", "sku", "units_sold")


class DataError(ValueError):
    pass


def load_sales(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"CSV is missing required column(s): {', '.join(missing)}")
    return clean_sales(df)


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"])
    out["sku"] = out["sku"].astype(str).str.strip()
    out["units_sold"] = pd.to_numeric(out["units_sold"], errors="coerce").fillna(0).clip(lower=0)
    if "promo" not in out.columns:
        out["promo"] = 0
    out["promo"] = pd.to_numeric(out["promo"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    if "price" not in out.columns:
        out["price"] = float("nan")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    aggregated = (
        out.groupby(["sku", "date"], as_index=False)
        .agg(units_sold=("units_sold", "sum"), promo=("promo", "max"), price=("price", "mean"))
        .sort_values(["sku", "date"])
    )
    if aggregated.empty:
        raise DataError("No valid rows after cleaning.")

    frames = []
    for sku, group in aggregated.groupby("sku", sort=True):
        full_index = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        reindexed = group.set_index("date").reindex(full_index).rename_axis("date").reset_index()
        reindexed["sku"] = sku
        reindexed["units_sold"] = reindexed["units_sold"].fillna(0.0)
        reindexed["promo"] = reindexed["promo"].fillna(0).astype(int)
        reindexed["price"] = reindexed["price"].ffill().bfill()
        frames.append(reindexed)
    result = pd.concat(frames, ignore_index=True)
    result["units_sold"] = result["units_sold"].astype(float)
    return result[["date", "sku", "units_sold", "promo", "price"]]


def series_for(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    sub = df[df["sku"] == sku].sort_values("date").reset_index(drop=True)
    if sub.empty:
        raise DataError(f"Unknown sku '{sku}'.")
    return sub


def skus(df: pd.DataFrame) -> list[str]:
    return sorted(df["sku"].unique().tolist())
